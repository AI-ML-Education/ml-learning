"""A tiny autoregressive "language model" toy for the RLHF lessons (files 15-18).

Real RLHF runs on transformers with billions of parameters generating text. The
ALGORITHMS (reward modeling, PPO, DPO, GRPO), however, are identical no matter how
big the model is. So we replace the giant LM with the smallest thing that still has
the two properties every RLHF method needs:

    1. We can SAMPLE sequences from a policy   pi_theta(sequence).
    2. We can compute the LOG-PROB of a sequence  log pi_theta(sequence).

Our "language": fixed-length sequences of `length` tokens over a vocab of `vocab`
symbols, generated left-to-right by a 1-layer GRU. Think of each token as a word
and a sequence as a short "sentence".

The HUMAN PREFERENCE we are aligning to is encoded by `true_reward`: humans like
sequences that contain more of the TARGET token. The optimum is "all target tokens",
but a good aligned model should raise the target frequency WITHOUT collapsing to a
single degenerate output — which is exactly what the KL-to-reference penalty buys us.

Everything is deliberately small so each lesson runs in seconds on a laptop CPU.
"""

import numpy as np
import torch
import torch.nn as nn

VOCAB = 6        # number of distinct tokens ("words")
LENGTH = 5       # tokens per sequence ("sentence length")
TARGET = 3       # the token humans prefer to see (the "good word")
BOS = 0          # beginning-of-sequence token fed in at the first step


def true_reward(seqs):
    """The hidden 'human' preference: fraction of tokens equal to TARGET, in [0, 1].

    seqs: LongTensor [B, LENGTH].  Returns FloatTensor [B].
    This stands in for "how much a human likes this response". RLHF never sees this
    directly — it only sees PAIRWISE PREFERENCES derived from it (file 15).
    """
    return (seqs == TARGET).float().mean(dim=1)


class ToyLM(nn.Module):
    """A minimal autoregressive language model: embed -> GRU -> next-token logits."""

    def __init__(self, vocab=VOCAB, emb=16, hidden=32, length=LENGTH):
        super().__init__()
        self.vocab, self.length, self.hidden = vocab, length, hidden
        self.embed = nn.Embedding(vocab, emb)
        self.gru = nn.GRUCell(emb, hidden)
        self.head = nn.Linear(hidden, vocab)

    def step_logits(self, prev_token, h):
        """One decoding step: given the previous token and hidden state, score next token."""
        h = self.gru(self.embed(prev_token), h)
        return self.head(h), h

    @torch.no_grad()
    def generate(self, n, rng=None, greedy=False):
        """Autoregressively sample `n` sequences. Returns LongTensor [n, LENGTH]."""
        gen = torch.Generator().manual_seed(int(rng.integers(2**31))) if rng is not None else None
        h = torch.zeros(n, self.hidden)
        prev = torch.full((n,), BOS, dtype=torch.long)
        out = []
        for _ in range(self.length):
            logits, h = self.step_logits(prev, h)
            probs = torch.softmax(logits, dim=1)
            if greedy:
                prev = probs.argmax(dim=1)
            else:
                prev = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            out.append(prev)
        return torch.stack(out, dim=1)

    def logprob_per_token(self, seqs):
        """Per-token log-probs (teacher forcing). Returns FloatTensor [B, LENGTH].

        Row t holds log pi_theta(token_t | tokens_<t). PPO (file 16) treats each token
        as an action and needs these per-token, not just their sum.
        """
        B = seqs.shape[0]
        h = torch.zeros(B, self.hidden)
        prev = torch.full((B,), BOS, dtype=torch.long)
        per_tok = []
        for t in range(self.length):
            logits, h = self.step_logits(prev, h)
            logp = torch.log_softmax(logits, dim=1)
            per_tok.append(logp.gather(1, seqs[:, t:t + 1]).squeeze(1))
            prev = seqs[:, t]
        return torch.stack(per_tok, dim=1)

    def logprob(self, seqs):
        """Sum of per-token log-probs. Returns FloatTensor [B] (grad-enabled).

        This is log pi_theta(sequence) = sum_t log pi_theta(token_t | tokens_<t),
        the quantity REINFORCE (file 15), DPO (file 17) and GRPO (file 18) differentiate.
        """
        return self.logprob_per_token(seqs).sum(dim=1)


class RewardModel(nn.Module):
    """Predicts a scalar 'how much a human likes this' from a full sequence.

    Trained ONLY from pairwise preferences via the Bradley-Terry model (file 15):
        loss = -log sigmoid( r(preferred) - r(rejected) ).
    It learns to imitate the ORDERING induced by `true_reward` without ever seeing it.
    """

    def __init__(self, vocab=VOCAB, emb=16, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, seqs):
        _, h = self.gru(self.embed(seqs))      # h: [1, B, hidden]
        return self.head(h.squeeze(0)).squeeze(1)


def make_preference_pairs(policy, n_pairs, rng, noise=0.05):
    """Sample sequence pairs from `policy` and label each with the human-preferred one.

    Returns (winners, losers): two LongTensors [n_pairs, LENGTH].
    Labels come from `true_reward` with a little flip noise (humans are noisy).
    """
    a = policy.generate(n_pairs, rng)
    b = policy.generate(n_pairs, rng)
    ra, rb = true_reward(a), true_reward(b)
    a_wins = ra >= rb
    flip = torch.rand(n_pairs) < noise           # occasionally the human disagrees
    a_wins = a_wins ^ flip
    winners = torch.where(a_wins[:, None], a, b)
    losers = torch.where(a_wins[:, None], b, a)
    return winners, losers


if __name__ == "__main__":
    # Smoke test: an untrained LM produces near-uniform target frequency (~1/VOCAB).
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    lm = ToyLM()
    seqs = lm.generate(2000, rng)
    print(f"vocab={VOCAB} length={LENGTH} target token={TARGET}")
    print(f"untrained LM mean true reward (target frequency): {true_reward(seqs).mean():.3f} "
          f"(uniform baseline = {1/VOCAB:.3f})")
    w, l = make_preference_pairs(lm, 5, rng)
    print("example preference pair (winner should have >= target count):")
    print("  winner:", w[0].tolist(), "rejected:", l[0].tolist())
    print("logprob shape:", tuple(lm.logprob(seqs[:4]).shape), "(grad-enabled =", lm.logprob(seqs[:4]).requires_grad, ")")
