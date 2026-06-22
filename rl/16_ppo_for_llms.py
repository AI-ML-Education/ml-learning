# 16_ppo_for_llms.py
r"""
PPO FOR LLMs — the algorithm that actually runs Stage 3 of RLHF.
================================================================

WHAT THIS FILE COVERS
    File 15 used plain REINFORCE for the RL stage. Real systems (InstructGPT,
    Llama-2-Chat, the original ChatGPT) use PPO (file 10) instead, because REINFORCE
    is too high-variance and unstable for billion-parameter models. Here we run PPO
    on the toy language model from `_toy_llm.py`, framing text generation as a
    token-level MDP — the exact setup used in libraries like TRL.

LANGUAGE GENERATION AS AN MDP
    state   s_t  = the prompt + tokens generated so far  (the context)
    action  a_t  = the next token                        (one step of decoding)
    policy  pi   = the language model                     (logits over the vocab)
    episode      = generating one full response of LENGTH tokens
    reward       = reward model score, delivered at the LAST token (sparse),
                   MINUS a per-token KL penalty to the frozen reference pi_ref.

    So a "trajectory" is a sentence, each token is an action, and we do credit
    assignment ACROSS TOKENS with GAE (file 09) just like any other RL problem.

THE PER-TOKEN REWARD (this is the standard RLHF reward shaping)
        r_t = - beta * ( log pi(a_t|s_t) - log pi_ref(a_t|s_t) )     for every token
        r_T += r_phi(full sequence)                                  at the last token
    The first part is a dense, per-token KL leash; the second is the sparse terminal
    reward from the reward model. PPO then maximizes the discounted sum.

PPO RECAP (see file 10 for the full derivation)
    ratio_t        = exp( log pi_new(a_t|s_t) - log pi_old(a_t|s_t) )
    clipped surr   = min( ratio_t * A_t,  clip(ratio_t, 1-eps, 1+eps) * A_t )
    actor loss     = - mean_t( clipped surr )
    critic loss    = mean_t( (V_theta(s_t) - return_t)^2 )
    entropy bonus  = + c * mean_t( H[pi(.|s_t)] )       (keeps exploration alive)
    The value head V_theta shares the LM trunk and estimates "expected future reward
    from this point in the sentence", which GAE uses to reduce variance.

MATH SUMMARY
    A_t (GAE):   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
                 A_t     = sum_{l>=0} (gamma*lambda)^l * delta_{t+l}
    Objective:   maximize  E[ min(ratio*A, clip(ratio,1±eps)*A) ]  - value_loss
                           + entropy_bonus
    KL control:  folded into r_t so the policy is pulled back toward pi_ref each step.

GOTCHAS
    * old_logp must be the log-prob FROM GENERATION TIME (the behavior policy), frozen.
      Recompute new_logp each PPO epoch; the ratio compares new vs old.
    * The value head predicts returns that INCLUDE the KL penalty, so it must be
      trained on the same shaped reward the actor sees.
    * Sparse terminal reward + dense KL penalty is the trick that makes per-token
      credit assignment work; GAE smears the terminal reward back over the tokens.
    * Clipping (not a hard KL constraint) is what keeps each update small and stable.

INTUITION
    Generating a sentence is a tiny episode. PPO asks, token by token: "did choosing
    this word lead to a better-than-expected final score?" If yes, make that word a
    bit more likely — but never move too far in one step (the clip), and never stray
    far from how the trusted reference talks (the KL penalty).

This file: a real (tiny) PPO-RLHF loop with value head, GAE, clipping, and KL.
We use the GROUND-TRUTH reward here to isolate the PPO mechanics; in a full system
this terminal reward is the learned reward model from file 15.
Next (file 17): DPO — skip the reward model and PPO entirely, optimize preferences directly.
"""

import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn

_here = os.path.dirname(os.path.abspath(__file__))


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_here, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


toy = _load("toy_llm", "_toy_llm.py")
ToyLM, true_reward = toy.ToyLM, toy.true_reward
VOCAB, LENGTH, TARGET, BOS = toy.VOCAB, toy.LENGTH, toy.TARGET, toy.BOS


class ActorCriticLM(nn.Module):
    """The toy LM with an added value head — shared GRU trunk, policy + value outputs."""

    def __init__(self, vocab=VOCAB, emb=16, hidden=32, length=LENGTH):
        super().__init__()
        self.vocab, self.length, self.hidden = vocab, length, hidden
        self.embed = nn.Embedding(vocab, emb)
        self.gru = nn.GRUCell(emb, hidden)
        self.policy_head = nn.Linear(hidden, vocab)
        self.value_head = nn.Linear(hidden, 1)

    def _step(self, prev_token, h):
        h = self.gru(self.embed(prev_token), h)
        return self.policy_head(h), self.value_head(h).squeeze(1), h

    @torch.no_grad()
    def rollout(self, n, rng):
        """Generate n sequences, recording per-token old log-probs and value estimates."""
        gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
        h = torch.zeros(n, self.hidden)
        prev = torch.full((n,), BOS, dtype=torch.long)
        seqs, logps, values = [], [], []
        for _ in range(self.length):
            logits, v, h = self._step(prev, h)
            probs = torch.softmax(logits, dim=1)
            tok = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            logp = torch.log_softmax(logits, dim=1).gather(1, tok[:, None]).squeeze(1)
            seqs.append(tok); logps.append(logp); values.append(v)
            prev = tok
        return (torch.stack(seqs, 1), torch.stack(logps, 1), torch.stack(values, 1))

    def evaluate(self, seqs):
        """Teacher-force seqs; return per-token (new_logp, entropy, value). Grad-enabled."""
        B = seqs.shape[0]
        h = torch.zeros(B, self.hidden)
        prev = torch.full((B,), BOS, dtype=torch.long)
        logps, ents, vals = [], [], []
        for t in range(self.length):
            logits, v, h = self._step(prev, h)
            logp_all = torch.log_softmax(logits, dim=1)
            logps.append(logp_all.gather(1, seqs[:, t:t + 1]).squeeze(1))
            ents.append(-(logp_all.exp() * logp_all).sum(1))
            vals.append(v)
            prev = seqs[:, t]
        return torch.stack(logps, 1), torch.stack(ents, 1), torch.stack(vals, 1)


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """GAE over the TOKEN axis. rewards, values: [B, L]. Returns advantages, returns [B, L]."""
    B, L = rewards.shape
    adv = torch.zeros(B, L)
    last = torch.zeros(B)
    for t in reversed(range(L)):
        next_v = values[:, t + 1] if t + 1 < L else torch.zeros(B)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    returns = adv + values
    return adv, returns


def train_ppo_rlhf(policy, ref, beta=0.05, iters=120, batch=256, epochs=4,
                   clip_eps=0.2, lr=3e-3, ent_coef=0.005, rng=None):
    """PPO Stage-3 RLHF loop on the toy LM. Returns the policy and a history list."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for _ in range(iters):
        # ---------- ROLLOUT (behavior policy, frozen) ----------
        seqs, old_logp, values = policy.rollout(batch, rng)
        with torch.no_grad():
            ref_logp = ref.logprob_per_token(seqs)             # [B, L] reference log-probs
            kl_per_tok = old_logp - ref_logp                   # per-token KL estimate
            rewards = -beta * kl_per_tok                        # dense KL-penalty reward
            term = true_reward(seqs)                            # terminal RM score (sparse)
            rewards[:, -1] = rewards[:, -1] + term             # add it at the last token
            adv, returns = compute_gae(rewards, values)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)       # normalize advantages

        # ---------- PPO UPDATE (several epochs over the same rollout) ----------
        for _ in range(epochs):
            new_logp, ent, new_v = policy.evaluate(seqs)
            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            actor_loss = -torch.min(unclipped, clipped).mean()
            critic_loss = nn.functional.mse_loss(new_v, returns)
            loss = actor_loss + 0.5 * critic_loss - ent_coef * ent.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        with torch.no_grad():
            history.append((term.mean().item(), kl_per_tok.sum(1).mean().item()))
    return policy, history


def make_demonstrations(n, rng):
    """'Human' demos: ~70% target token per position (good but varied)."""
    probs = np.full(VOCAB, 0.3 / (VOCAB - 1)); probs[TARGET] = 0.7
    return torch.tensor(rng.choice(VOCAB, size=(n, LENGTH), p=probs), dtype=torch.long)


def sft(model, demos, epochs=150, lr=5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = -model.logprob_per_token(demos).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def mean_true_reward(policy, rng, n=3000):
    with torch.no_grad():
        return true_reward(policy.rollout(n, rng)[0]).mean().item()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    print("=" * 72)
    print("PPO-RLHF on a toy language model (text generation as a token-level MDP)")
    print(f"(terminal reward = frequency of token {TARGET}; random baseline = {1/VOCAB:.3f})")
    print("=" * 72)

    # Reference policy pi_ref = a quickly SFT'd ToyLM (frozen). The actor is initialized
    # from the same weights but carries a fresh value head.
    demos = make_demonstrations(800, rng)
    ref = sft(ToyLM(), demos)
    for p in ref.parameters():
        p.requires_grad_(False)

    actor = ActorCriticLM()
    # copy the shared trunk + policy head from the SFT reference into the actor
    actor.embed.load_state_dict(ref.embed.state_dict())
    actor.gru.load_state_dict(ref.gru.state_dict())
    actor.policy_head.load_state_dict(ref.head.state_dict())

    print(f"\nReference (SFT) policy   true reward = {true_reward(ref.generate(3000, rng)).mean():.3f}")
    print(f"Actor before PPO         true reward = {mean_true_reward(actor, rng):.3f}")

    actor, hist = train_ppo_rlhf(actor, ref, beta=0.05, rng=rng)

    print("\nPPO progress (every 30 iters):  term-reward     KL(pi||ref)")
    for i in range(29, len(hist), 30):
        print(f"  iter {i + 1:3d}                 {hist[i][0]:.3f}          {hist[i][1]:.2f}")

    print(f"\nActor after PPO          true reward = {mean_true_reward(actor, rng):.3f}")
    with torch.no_grad():
        print("sample generations:", actor.rollout(6, rng)[0].tolist())

    print("\nTakeaways")
    print("  • Each token is an action; a sentence is an episode; reward lands at the end.")
    print("  • The KL penalty is added PER TOKEN; the reward-model score is added once, last.")
    print("  • GAE smears that terminal reward back over tokens; clipping keeps updates small.")
    print("  • This is exactly file 10's PPO — only the 'environment' is a language model.")
    print("\nNext (file 17): DPO — get the same alignment with NO reward model and NO RL loop.")
