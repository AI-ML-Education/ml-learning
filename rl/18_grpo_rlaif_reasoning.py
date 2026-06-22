# 18_grpo_rlaif_reasoning.py
r"""
GRPO, RLAIF, and REASONING RL — the 2024-2025 frontier (DeepSeek-R1 & friends).
==============================================================================

WHAT THIS FILE COVERS
    The newest wave of RL-for-LLMs, and one important myth-buster:
      1. GRPO (Group Relative Policy Optimization) — the critic-free PPO variant
         behind DeepSeek-R1 / R1-Zero. Runnable here.
      2. RLAIF (RL from AI Feedback) — replace human labels with an LLM judge
         (Anthropic's "Constitutional AI"). Runnable twist on GRPO here.
      3. Reasoning RL — using a VERIFIABLE reward (is the answer correct?) to grow
         chain-of-thought, no reward model at all. Explained.
      4. AlphaFold — the honest note: AlphaFold is NOT reinforcement learning.

------------------------------------------------------------------------------
1) GRPO — PPO without the value network
------------------------------------------------------------------------------
    PPO (file 16) needs a CRITIC (value head) to compute advantages. For a 70B LLM
    that critic is another 70B-ish network — expensive memory and a second thing to
    train. GRPO removes it with a beautifully simple idea borrowed from old-school
    policy gradients: estimate the baseline EMPIRICALLY from a GROUP of samples.

    For each prompt x, sample a GROUP of G responses y_1..y_G from the current policy,
    score each with the reward r_i, and define a GROUP-RELATIVE advantage:

        A_i = ( r_i - mean(r_1..r_G) ) / ( std(r_1..r_G) + eps ).

    "Was this response better or worse than my other tries on the SAME prompt?"
    That standardized score replaces the critic. Every token of response i gets the
    same advantage A_i. Then optimize a PPO-style clipped objective plus a KL penalty
    to the frozen reference:

        ratio_t = exp( log pi(a_t) - log pi_old(a_t) )
        L = - E[ min(ratio_t * A_i, clip(ratio_t, 1±eps) * A_i) - beta * KL_t ]

    GRPO is ideal for VERIFIABLE-REWARD tasks (math, code): the reward is just
    "+1 if the final answer is correct, else 0" — no reward model needed.

    KL estimator (the "k3" unbiased estimator DeepSeek uses, always >= 0):
        KL_t ≈ exp(log pi_ref - log pi) - (log pi_ref - log pi) - 1.

------------------------------------------------------------------------------
2) RLAIF — the labels come from an AI, not a human
------------------------------------------------------------------------------
    Human preference labels are slow and costly. RLAIF asks a capable LLM to be the
    JUDGE: "which response better follows these principles (a 'constitution')?" Those
    AI labels train the reward / give the reward signal, and the rest of the pipeline
    (PPO or GRPO) is UNCHANGED. Below we swap our reward function from a raw
    "correctness" score to an AI-judge score that also values diversity — the only
    line that changes — and watch behavior change accordingly.

------------------------------------------------------------------------------
3) Reasoning RL (R1-Zero) — RL grows chain-of-thought from scratch
------------------------------------------------------------------------------
    DeepSeek-R1-Zero applied GRPO to a BASE model with ONLY a verifiable reward
    (correct/incorrect on math) and a format reward — no SFT, no reward model, no
    human preferences. The model spontaneously learned to write long reasoning
    traces, re-check itself, and backtrack (the famous "aha moment"). Lesson: if you
    can VERIFY the answer, RL can discover the reasoning process that gets there.
    This is policy improvement (files 03-10) applied to "thinking", at scale.

------------------------------------------------------------------------------
4) AlphaFold — NOT reinforcement learning (myth-buster)
------------------------------------------------------------------------------
    AlphaFold2/3 are frequently lumped in with AlphaGo/AlphaZero because of the
    "Alpha" branding, but they are SUPERVISED / geometric deep learning, NOT RL:
      * Trained by gradient descent on the Protein Data Bank (known structures) to
        predict 3D coordinates from a sequence + multiple-sequence alignment.
      * Core pieces are the Evoformer and a structure module (AF2) or a diffusion
        decoder (AF3) — there is no agent, no environment, no reward, no policy,
        no exploration. The loss is a structural/geometric error vs. ground truth.
    Contrast with AlphaZero (file 14), which is GENUINE RL: self-play generates its
    own data and improves a policy from outcomes. Knowing the difference matters:
    "Alpha" is a brand, not a synonym for reinforcement learning.

MATH SUMMARY
    GRPO advantage    A_i = (r_i - mean_group) / (std_group + eps)   (no critic)
    GRPO loss         -E[ min(rho*A, clip(rho,1±eps)*A) - beta*KL_k3 ]
    RLAIF             same loss; reward r comes from an AI judge, not a human
    Reasoning RL      r = verifiable correctness; CoT emerges from optimization
    AlphaFold         L = geometric error vs. PDB; supervised, NOT RL

GOTCHAS
    * GRPO's group standardization needs G >= 2; with G=1 the advantage is undefined.
    * No critic means GRPO leans on having SEVERAL samples per prompt for a stable
      baseline — more samples, less variance.
    * RLAIF inherits the judge's biases; the "constitution"/prompt is now part of the
      alignment target. Garbage judge in, garbage policy out.
    * Verifiable rewards are powerful but only exist when correctness is checkable
      (math/code/games) — open-ended chat still leans on preference models.

INTUITION
    GRPO: take a few swings at the same problem, keep doing more of what beat your own
    average, less of what lagged it — no separate coach (critic) required. RLAIF: let a
    smart model grade the homework instead of a person. Reasoning RL: reward only the
    right final answer and the model figures out the working-out on its own.

This file: runnable GRPO (critic-free) + an RLAIF variant on the toy LM; reasoning RL
and AlphaFold are explained above. This is the end of the RL ladder — see the recap.
"""

import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_here, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


toy = _load("toy_llm", "_toy_llm.py")
ToyLM, true_reward = toy.ToyLM, toy.true_reward
VOCAB, LENGTH, TARGET = toy.VOCAB, toy.LENGTH, toy.TARGET


def correctness_reward(seqs):
    """A VERIFIABLE reward: fraction of target tokens (stands in for 'is it correct?')."""
    return true_reward(seqs)


def ai_judge_reward(seqs):
    """An AI JUDGE's score (RLAIF stand-in): rewards target tokens BUT wants variety too.

    A real RLAIF judge is another LLM grading against a 'constitution'. Here the judge
    has richer taste than raw correctness: it likes target tokens yet PENALIZES the
    degenerate all-same-token answer, preferring a healthy mix (~3 distinct tokens).
    Result: GRPO under this judge stays target-heavy WITHOUT collapsing.
    """
    counts = F.one_hot(seqs, VOCAB).float().sum(1)          # [B, VOCAB] token counts
    freq_target = counts[:, TARGET] / LENGTH                # how 'correct'
    distinct = (counts > 0).float().sum(1)                  # how many distinct tokens
    return freq_target - 0.25 * (distinct - 3.0).abs()      # peak ≈ 3 distinct, target-rich


def grpo_train(reward_fn, ref, n_iters=200, n_groups=32, group_size=8, beta=0.02,
               clip_eps=0.2, epochs=2, lr=3e-3, rng=None):
    """Critic-free GRPO on the toy LM. Returns the trained policy and reward history."""
    policy = ToyLM()
    policy.load_state_dict(ref.state_dict())               # start from SFT reference
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    B = n_groups * group_size
    history = []
    for _ in range(n_iters):
        # ---- sample a GROUP of responses per prompt (here one shared prompt) ----
        seqs = policy.generate(B, rng)
        with torch.no_grad():
            old_logp = policy.logprob_per_token(seqs)       # behavior log-probs [B, L]
            ref_logp = ref.logprob_per_token(seqs)          # frozen reference [B, L]
            r = reward_fn(seqs)                             # [B]
            rg = r.view(n_groups, group_size)
            # GROUP-RELATIVE advantage: standardize within each group (no critic!)
            adv = (rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-6)
            adv = adv.reshape(B, 1)                          # same advantage for all tokens

        # ---- PPO-style clipped update with k3 KL penalty, several epochs ----
        for _ in range(epochs):
            new_logp = policy.logprob_per_token(seqs)
            ratio = torch.exp(new_logp - old_logp)
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
            # k3 unbiased KL estimator (always >= 0):
            diff = ref_logp - new_logp
            kl = torch.exp(diff) - diff - 1.0
            loss = -(surr - beta * kl).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        history.append(r.mean().item())
    return policy, history


def make_demonstrations(n, rng):
    probs = np.full(VOCAB, 0.3 / (VOCAB - 1)); probs[TARGET] = 0.7
    return torch.tensor(rng.choice(VOCAB, size=(n, LENGTH), p=probs), dtype=torch.long)


def sft(model, demos, epochs=150, lr=5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = -model.logprob(demos).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def report(policy, rng, n=3000):
    with torch.no_grad():
        seqs = policy.generate(n, rng)
        freq = true_reward(seqs).mean().item()
        div = float(np.mean([len(set(s.tolist())) for s in seqs]))
    return freq, div


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    print("=" * 72)
    print("GRPO (critic-free) + RLAIF on a toy language model")
    print(f"(target token = {TARGET}; random target-frequency baseline = {1/VOCAB:.3f})")
    print("=" * 72)

    demos = make_demonstrations(800, rng)
    ref = sft(ToyLM(), demos)
    for p in ref.parameters():
        p.requires_grad_(False)
    f0, d0 = report(ref, rng)
    print(f"\nReference (SFT)             target-freq = {f0:.3f}   distinct tok/seq = {d0:.2f}")

    # --- GRPO with a VERIFIABLE 'correctness' reward (math/code style) ---
    grpo_correct, hist_c = grpo_train(correctness_reward, ref, rng=rng)
    fc, dc = report(grpo_correct, rng)
    print(f"GRPO + correctness reward  target-freq = {fc:.3f}   distinct tok/seq = {dc:.2f}")

    # --- RLAIF: identical GRPO, but the reward comes from an AI judge (values diversity) ---
    grpo_ai, hist_a = grpo_train(ai_judge_reward, ref, rng=rng)
    fa, da = report(grpo_ai, rng)
    print(f"RLAIF (GRPO + AI judge)    target-freq = {fa:.3f}   distinct tok/seq = {da:.2f}")

    print("\nGRPO learning curve (correctness reward), every 40 iters:")
    for i in range(39, len(hist_c), 40):
        print(f"  iter {i + 1:3d}  group-mean reward = {hist_c[i]:.3f}")

    with torch.no_grad():
        print("\nGRPO+correctness samples:", grpo_correct.generate(4, rng).tolist())
        print("RLAIF (diversity-aware) samples:", grpo_ai.generate(4, rng).tolist())

    print("\nTakeaways")
    print("  • GRPO drops PPO's critic: the baseline is the GROUP MEAN of several samples.")
    print("  • Verifiable 'correctness' reward maximizes target tokens — and collapses.")
    print("  • RLAIF swaps in an AI judge that also values diversity → stays varied.")
    print("  • Only the REWARD FUNCTION changed between the two runs; GRPO itself is identical.")
    print("  • Reasoning RL (R1-Zero) is this idea with a verifiable reward → emergent CoT.")
    print("  • AlphaFold is NOT RL — it is supervised geometric deep learning on the PDB.")
    print("\nThat's the full ladder: bandits → MDPs → DP/MC/TD → deep RL → AlphaZero → RLHF/DPO/GRPO.")
