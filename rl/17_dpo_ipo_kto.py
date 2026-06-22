# 17_dpo_ipo_kto.py
r"""
DPO and friends — preference optimization WITHOUT a reward model or RL loop.
===========================================================================

WHAT THIS FILE COVERS
    DPO (Direct Preference Optimization, 2023) and its relatives IPO and KTO. These
    methods get the SAME alignment goal as RLHF (files 15-16) but throw away both the
    separate reward model AND the PPO loop, replacing the whole thing with one
    supervised-style classification loss on preference pairs. This is why DPO took
    over open-source alignment (Zephyr, many Llama/Mistral fine-tunes).

THE KEY INSIGHT (why a reward model is unnecessary)
    The KL-regularized RLHF objective from file 15 has a KNOWN closed-form optimum:
        pi*(y|x)  =  (1/Z(x)) * pi_ref(y|x) * exp( r(y|x) / beta ).
    Invert it to express the reward in terms of the optimal policy:
        r(y|x)  =  beta * log( pi*(y|x) / pi_ref(y|x) )  +  beta*log Z(x).
    Substitute THIS into the Bradley-Terry preference likelihood (file 15). The
    intractable log Z(x) CANCELS (it is the same for y_w and y_l), leaving a loss that
    depends only on the policy and the reference — no reward model at all:

        L_DPO = - E_{(x, y_w, y_l)} log sigmoid(
                    beta * [ (log pi(y_w) - log pi_ref(y_w))
                           - (log pi(y_l) - log pi_ref(y_l)) ] ).

    In words: raise the policy's log-prob on the PREFERRED response and lower it on
    the REJECTED one, each measured RELATIVE to the frozen reference, with strength
    beta. The bracket is an "implicit reward" the policy defines for itself.

VARIANTS
    IPO (Identity-PO): DPO's log-sigmoid can over-fit when preferences are nearly
        deterministic (it keeps pushing the margin to infinity). IPO replaces it with
        a SQUARED loss that targets a FINITE margin, regularizing better:
            L_IPO = E [ ( (log pi(y_w)-log pi_ref(y_w)) - (log pi(y_l)-log pi_ref(y_l))
                         - 1/(2*beta) )^2 ].
    KTO (Kahneman-Tversky Optimization): needs only UNPAIRED labels — each example is
        just "good" or "bad", no matched pair. Uses a prospect-theory value function
        around a reference point (the average implicit reward). Cheaper to label;
        explained here, not coded (it needs a running KL reference term).

MATH SUMMARY
    implicit reward    h(y) = beta * ( log pi(y) - log pi_ref(y) )
    DPO loss           -log sigmoid( h(y_w) - h(y_l) )
    IPO loss           ( (h(y_w)-h(y_l))/beta - 1/(2*beta) )^2     [margin form]
    Optimum policy     pi* ∝ pi_ref * exp(r/beta)   (the bridge that makes it all work)

GOTCHAS
    * pi_ref is frozen and appears in EVERY term — DPO is meaningless without it; it is
      the anchor that defines the implicit reward and supplies the KL regularization.
    * beta plays the same role as in PPO-RLHF: bigger beta = stay closer to pi_ref.
    * DPO optimizes preferences DIRECTLY on a fixed dataset — it is OFFLINE (file 12):
      no sampling from the policy, no environment, no reward model. Just gradient steps.
    * Because it never samples, DPO can't discover new behaviors beyond the dataset's
      support — a known limitation versus online PPO/GRPO.

INTUITION
    RLHF: train a judge (reward model), then let the student practice against the judge
    (PPO). DPO: skip the judge entirely — the comparisons themselves tell the student
    "make this answer more likely than that one", and the math proves that's equivalent
    to the RLHF optimum. One loss, one dataset, done.

This file: runnable DPO and IPO on the toy LM's preference data; compare to the SFT
reference and to each other. No reward model, no rollouts.
Next (file 18): GRPO — bring sampling back, but ditch the value head (DeepSeek-R1's method).
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
make_preference_pairs = toy.make_preference_pairs
VOCAB, LENGTH, TARGET = toy.VOCAB, toy.LENGTH, toy.TARGET


def implicit_reward(policy, ref, seqs):
    """h(y) = log pi(y) - log pi_ref(y)  (the per-sequence implicit reward, sans beta)."""
    lp = policy.logprob(seqs)
    with torch.no_grad():
        lr = ref.logprob(seqs)
    return lp - lr


def dpo_loss(policy, ref, winners, losers, beta):
    """-log sigmoid( beta * (h(y_w) - h(y_l)) ) — the Direct Preference Optimization loss."""
    margin = implicit_reward(policy, ref, winners) - implicit_reward(policy, ref, losers)
    return -nn.functional.logsigmoid(beta * margin).mean()


def ipo_loss(policy, ref, winners, losers, beta):
    """( (h(y_w) - h(y_l)) - 1/(2*beta) )^2 — IPO's bounded-margin alternative."""
    margin = implicit_reward(policy, ref, winners) - implicit_reward(policy, ref, losers)
    return ((margin - 1.0 / (2.0 * beta)) ** 2).mean()


def train_preference(loss_fn, ref, winners, losers, beta=0.3, epochs=400, batch=256,
                     lr=3e-3, rng=None):
    """Generic offline preference-optimization loop (works for DPO or IPO)."""
    policy = ToyLM()
    policy.load_state_dict(ref.state_dict())          # start from the SFT reference
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = winners.shape[0]
    for _ in range(epochs):
        idx = rng.integers(0, n, size=batch)
        w = winners[idx]; l = losers[idx]
        loss = loss_fn(policy, ref, w, l, beta)
        opt.zero_grad(); loss.backward(); opt.step()
    return policy


def make_demonstrations(n, rng):
    probs = np.full(VOCAB, 0.3 / (VOCAB - 1)); probs[TARGET] = 0.7
    return torch.tensor(rng.choice(VOCAB, size=(n, LENGTH), p=probs), dtype=torch.long)


def sft(model, demos, epochs=150, lr=5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = -model.logprob(demos).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def mean_true_reward(policy, rng, n=3000):
    with torch.no_grad():
        return true_reward(policy.generate(n, rng)).mean().item()


def preference_accuracy(policy, winners, losers):
    """Fraction of held-out pairs the policy ranks correctly: log pi(w) > log pi(l)."""
    with torch.no_grad():
        m = policy.logprob(winners) - policy.logprob(losers)
    return (m > 0).float().mean().item()


def logp_margin(policy, winners, losers):
    """Mean log-prob gap the policy puts between winners and losers (nats).

    This is what DPO/IPO sculpt: DPO (log-sigmoid) pushes it as large as it can;
    IPO (squared loss) targets a FINITE margin, so it stays bounded.
    """
    with torch.no_grad():
        return (policy.logprob(winners) - policy.logprob(losers)).mean().item()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    print("=" * 72)
    print("DPO / IPO on a toy language model — alignment with NO reward model, NO RL")
    print(f"(true reward = frequency of token {TARGET}; random baseline = {1/VOCAB:.3f})")
    print("=" * 72)

    # Reference = SFT model (frozen). DPO/IPO both optimize relative to it.
    demos = make_demonstrations(800, rng)
    ref = sft(ToyLM(), demos)
    ref_frozen = ToyLM(); ref_frozen.load_state_dict(ref.state_dict())
    for p in ref_frozen.parameters():
        p.requires_grad_(False)
    print(f"\nReference (SFT) policy    true reward = {mean_true_reward(ref_frozen, rng):.3f}")

    # One fixed OFFLINE preference dataset, reused by both methods (no sampling during training).
    winners, losers = make_preference_pairs(ref_frozen, 4000, rng, noise=0.05)
    held_w, held_l = make_preference_pairs(ref_frozen, 1000, rng, noise=0.0)

    dpo = train_preference(dpo_loss, ref_frozen, winners, losers, beta=0.3, rng=rng)
    ipo = train_preference(ipo_loss, ref_frozen, winners, losers, beta=0.3, rng=rng)

    print(f"\n{'method':>10} | {'true reward':>11} | {'pref-acc':>8} | {'logp margin (nats)':>18}")
    print("-" * 54)
    for name, pol in (("SFT ref", ref_frozen), ("DPO", dpo), ("IPO", ipo)):
        print(f"{name:>10} | {mean_true_reward(pol, rng):>11.3f} | "
              f"{preference_accuracy(pol, held_w, held_l):>8.1%} | "
              f"{logp_margin(pol, held_w, held_l):>18.2f}")

    with torch.no_grad():
        print("\nDPO sample generations:", dpo.generate(6, rng).tolist())

    print("\nTakeaways")
    print("  • DPO turns RLHF into a single classification-style loss on preference pairs.")
    print("  • The frozen reference defines the implicit reward and supplies KL regularization.")
    print("  • No reward model, no sampling, no PPO — it is OFFLINE preference learning.")
    print("  • IPO swaps the log-sigmoid for a bounded squared margin to curb over-fitting.")
    print("  • KTO needs only good/bad labels (no pairs) via a prospect-theory value function.")
    print("\nNext (file 18): GRPO — online sampling like PPO, but value-head-free (DeepSeek-R1).")
