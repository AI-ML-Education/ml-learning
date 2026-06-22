# 15_rlhf_pipeline.py
r"""
RLHF: REINFORCEMENT LEARNING FROM HUMAN FEEDBACK — the alignment pipeline.
==========================================================================

WHAT THIS FILE COVERS
    The three-stage recipe that turned a raw next-token predictor (GPT-3) into a
    helpful assistant (InstructGPT / ChatGPT), and the same recipe that aligns
    Llama, Claude, Gemini, etc. We build a tiny but FAITHFUL end-to-end version on
    the toy language model from `_toy_llm.py`.

WHY RL AT ALL FOR LANGUAGE MODELS?
    A pretrained LM (built in llms/) only knows P(next token | text).
    "Be helpful, honest, harmless" is not a next-token label — there is no dataset
    of "the single correct response". Instead humans can COMPARE two responses and
    say which is better. RLHF turns those comparisons into a reward signal and
    OPTIMIZES the LM against it. The LM is the policy; generating a response is an
    episode; the reward arrives at the end (a classic sparse-reward RL problem).

THE THREE STAGES
    Stage 1 — SFT (Supervised Fine-Tuning):
        Imitate human-written demonstrations with ordinary cross-entropy. This gives
        a decent starting policy and, crucially, the REFERENCE model pi_ref that the
        RL stage must stay close to.
    Stage 2 — Reward Modeling (RM):
        Collect pairs (response A, response B) with a human label "A > B". Fit a
        reward model r_phi with the Bradley-Terry loss so that
            r_phi(preferred) > r_phi(rejected).
        The RM is a learned PROXY for human preference — we never get a numeric score
        from a human, only comparisons.
    Stage 3 — RL fine-tuning (this file uses REINFORCE; file 16 uses PPO):
        Maximize the reward model's score, but PENALIZE drifting away from pi_ref:
            maximize  E_{y ~ pi}[ r_phi(y) ]  -  beta * KL( pi(.|x) || pi_ref(.|x) ).
        The KL term is the safety belt: without it the policy "reward hacks" the
        imperfect RM and collapses to degenerate text (demonstrated below).

MATH SUMMARY
    Bradley-Terry preference model:
        P(y_w > y_l) = sigmoid( r_phi(y_w) - r_phi(y_l) )
        RM loss      = - E[ log sigmoid( r_phi(y_w) - r_phi(y_l) ) ]
    KL-regularized RL objective (the "RLHF objective"):
        J(theta) = E_{y~pi_theta}[ r_phi(y) - beta * ( log pi_theta(y) - log pi_ref(y) ) ]
    REINFORCE gradient (score-function estimator, baseline b):
        grad J = E[ (R(y) - b) * grad log pi_theta(y) ],
        where R(y) = r_phi(y) - beta*( log pi_theta(y) - log pi_ref(y) ).

GOTCHAS
    * The RM is imperfect; over-optimizing it is GOODHART'S LAW in action — the score
      goes up while real quality goes down. The KL penalty (beta) is what keeps the
      policy honest. Too-large beta = no learning; too-small beta = reward hacking.
    * pi_ref is frozen. It is the SFT model, not the current policy.
    * Reward is per-SEQUENCE and sparse (only at the end), so credit assignment is
      done by the log-prob of the whole sequence — exactly REINFORCE (file 08).

INTUITION
    SFT teaches the model to TALK like the demonstrations. The reward model is a
    cheap stand-in judge trained from human taste. RL then nudges the talker to score
    better with the judge while a leash (KL to pi_ref) stops it from running off into
    gibberish that only fools the judge.

This file: full pipeline end-to-end + a side-by-side of "with KL" vs "no leash".
Next (file 16): the same Stage 3, done properly with PPO (clipping + value head).
"""

import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn

# --- load the shared toy LM helper (digit/underscore-prefixed → import via path) ---
_here = os.path.dirname(os.path.abspath(__file__))


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_here, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


toy = _load("toy_llm", "_toy_llm.py")
ToyLM, RewardModel, true_reward = toy.ToyLM, toy.RewardModel, toy.true_reward
make_preference_pairs, TARGET, VOCAB = toy.make_preference_pairs, toy.TARGET, toy.VOCAB


# ----------------------------------------------------------------------------------
# STAGE 1 — Supervised Fine-Tuning (SFT)
# ----------------------------------------------------------------------------------
def supervised_finetune(policy, demos, epochs=150, lr=5e-3):
    """Imitate human demonstrations with plain cross-entropy (maximize log pi(demo))."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(epochs):
        loss = -policy.logprob(demos).mean()       # maximize log-likelihood of demos
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy


def make_demonstrations(n, rng):
    """Synthesize 'human-written' demos: sequences that are good but not perfect.

    Each demo independently picks the TARGET token ~70% of the time. This mimics
    real demonstrations: high quality, varied, never a single canned answer.
    """
    probs = np.full(VOCAB, 0.3 / (VOCAB - 1))
    probs[TARGET] = 0.7
    data = rng.choice(VOCAB, size=(n, toy.LENGTH), p=probs)
    return torch.tensor(data, dtype=torch.long)


# ----------------------------------------------------------------------------------
# STAGE 2 — Reward Modeling (Bradley-Terry)
# ----------------------------------------------------------------------------------
def train_reward_model(rm, winners, losers, epochs=300, lr=5e-3):
    """Fit r_phi so r(winner) > r(loser) via the Bradley-Terry / pairwise-logistic loss."""
    opt = torch.optim.Adam(rm.parameters(), lr=lr)
    for _ in range(epochs):
        margin = rm(winners) - rm(losers)
        loss = -nn.functional.logsigmoid(margin).mean()   # -log sigmoid(r_w - r_l)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return rm


def reward_model_accuracy(rm, policy, rng, n=400):
    """Fraction of fresh pairs the RM ranks the same way the true preference does."""
    w, l = make_preference_pairs(policy, n, rng, noise=0.0)
    with torch.no_grad():
        agree = (rm(w) > rm(l)).float().mean().item()
    return agree


# ----------------------------------------------------------------------------------
# STAGE 3 — RL fine-tuning against the reward model (REINFORCE + KL leash)
# ----------------------------------------------------------------------------------
def rlhf_finetune(policy, ref, rm, beta, iters=120, batch=256, lr=3e-3, rng=None):
    """Maximize E[r_phi(y)] - beta * KL(pi || pi_ref) with REINFORCE and a mean baseline."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for _ in range(iters):
        seqs = policy.generate(batch, rng)                 # sample responses (no grad)
        logp = policy.logprob(seqs)                        # grad-enabled log pi(y)
        with torch.no_grad():
            logref = ref.logprob(seqs)                     # frozen reference log pi_ref(y)
            rm_score = rm(seqs)
            approx_kl = (logp - logref)                    # per-sequence KL estimate
            reward = rm_score - beta * approx_kl           # the RLHF per-sequence reward
            advantage = reward - reward.mean()             # baseline = batch mean
        loss = -(advantage * logp).mean()                  # REINFORCE objective
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            history.append((true_reward(seqs).mean().item(), approx_kl.mean().item()))
    return policy, history


def mean_true_reward(policy, rng, n=3000):
    with torch.no_grad():
        return true_reward(policy.generate(n, rng)).mean().item()


def avg_distinct_tokens(policy, rng, n=500):
    """Diversity probe: average number of distinct tokens per generated sequence.

    1.0 means the model emits the SAME token every position (fully collapsed);
    higher means more varied, on-distribution text.
    """
    with torch.no_grad():
        seqs = policy.generate(n, rng)
    return float(np.mean([len(set(s.tolist())) for s in seqs]))


# ----------------------------------------------------------------------------------
# DEMO — run the whole pipeline
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    print("=" * 72)
    print("RLHF PIPELINE on a toy language model")
    print(f"(reward = frequency of token {TARGET}; random LM baseline = {1/VOCAB:.3f})")
    print("=" * 72)

    base = ToyLM()
    print(f"\n[0] Pretrained base LM        true reward = {mean_true_reward(base, rng):.3f}")

    # --- Stage 1: SFT ---
    demos = make_demonstrations(800, rng)
    sft = supervised_finetune(ToyLM(), demos)
    print(f"[1] After SFT (imitate demos)  true reward = {mean_true_reward(sft, rng):.3f}")

    # Freeze a copy of the SFT model as the reference policy pi_ref.
    ref = ToyLM()
    ref.load_state_dict(sft.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    # --- Stage 2: Reward model from preferences (collected from the SFT policy) ---
    winners, losers = make_preference_pairs(sft, 3000, rng, noise=0.05)
    rm = train_reward_model(RewardModel(), winners, losers)
    acc = reward_model_accuracy(rm, sft, rng)
    print(f"[2] Reward model agreement with true preference: {acc:.1%}")

    # --- Stage 3: RL fine-tuning — sweep the KL strength `beta` to see the trade-off ---
    # beta is a DIAL: large beta = stay glued to pi_ref (safe, little gain);
    # small beta = chase reward hard (big gain, large drift, eventually collapse).
    print("\n[3] RL fine-tuning against the reward model (REINFORCE + KL leash):")
    print(f"    {'beta':>6} | {'true reward':>11} | {'KL to ref':>9} | {'distinct tok/seq':>16}")
    print("    " + "-" * 54)
    for beta in (5.0, 2.0, 0.5, 0.0):
        policy = ToyLM()
        policy.load_state_dict(sft.state_dict())
        policy, hist = rlhf_finetune(policy, ref, rm, beta=beta, rng=rng)
        tr = mean_true_reward(policy, rng)
        kl = hist[-1][1]
        div = avg_distinct_tokens(policy, rng)
        flag = "  ← collapsed" if div < 1.05 else ""
        print(f"    {beta:>6.2f} | {tr:>11.3f} | {kl:>9.2f} | {div:>16.2f}{flag}")
    with torch.no_grad():
        samples = policy.generate(6, rng).tolist()      # beta=0 samples
    print(f"    (beta=0 example samples, note the collapse toward token {TARGET}): {samples}")

    print("\nTakeaways")
    print("  • SFT alone already lifts quality by imitation.")
    print("  • The reward model turns pairwise human taste into a trainable signal.")
    print("  • beta trades reward against staying near the trusted reference (pi_ref):")
    print("      high beta → diverse but timid; low beta → high reward but collapses.")
    print("  • Real RLHF lives in the middle — improve reward while the KL leash holds")
    print("    the policy on-distribution (this is also what stops reward-model hacking).")
    print("\nNext (file 16): replace REINFORCE with PPO — the algorithm real RLHF uses.")
