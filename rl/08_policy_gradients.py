# Policy Gradients — Learning the Policy Directly (REINFORCE)
"""
WHAT/WHY:
Value-based methods (DQN) learn Q, then act greedily. Policy-gradient methods skip the
middleman: they PARAMETERIZE THE POLICY π_θ(a|s) directly and adjust θ by gradient ASCENT
on expected return. This is a different, complementary branch of RL — and it's the branch
that powers RLHF (file 16): PPO is a policy-gradient method.

Why learn the policy directly?
  - Natural for CONTINUOUS or high-dimensional action spaces (DQN's max over actions is
    awkward or impossible there).
  - Can learn STOCHASTIC policies (sometimes optimal, e.g., bluffing in poker, or under
    partial observability). A greedy value policy is always deterministic.
  - Smooth: small θ changes → small policy changes (good for stability/convergence).
Trade-off: high variance, and it converges to a LOCAL optimum.

---

1. THE OBJECTIVE

Maximize expected return of trajectories τ sampled by following π_θ:

    J(θ) = E_{τ ~ π_θ} [ R(τ) ]          where R(τ) = Σ_t γ^t r_t

We want ∇_θ J(θ) to do gradient ASCENT: θ ← θ + α ∇_θ J(θ).

---

2. THE POLICY GRADIENT THEOREM (the one equation to know)

The gradient looks impossible — the environment dynamics depend on θ through which states
we visit. The "log-derivative trick" (a.k.a. REINFORCE trick) makes it tractable:

    ∇_θ J(θ) = E_{τ ~ π_θ} [ Σ_t  ∇_θ log π_θ(a_t | s_t) · G_t ]

In words: increase the log-probability of actions that led to HIGH return, decrease it for
actions that led to LOW return — weighted by how good the outcome was.

DERIVATION SKETCH (why the log shows up):
  ∇_θ E[R] = ∇_θ ∫ p_θ(τ) R(τ) dτ = ∫ p_θ(τ) ∇_θ log p_θ(τ) R(τ) dτ = E[ ∇log p_θ(τ) · R ]
  using the identity  ∇p = p · ∇log p. The environment terms in log p_θ(τ) don't depend on
  θ, so only Σ_t ∇log π_θ(a_t|s_t) survives. Beautiful: no model needed.

---

3. THE LOSS YOU ACTUALLY CODE

Autodiff does gradient DESCENT, so we minimize the NEGATIVE:

    L(θ) = - (1/N) Σ_episodes Σ_t  log π_θ(a_t|s_t) · Ĝ_t

where Ĝ_t is an estimate of the return-from-t (the "weight" / credit for that action).
This is sometimes called the "surrogate" or "pseudo-loss" — its gradient equals the policy
gradient; its value is not a real loss to interpret.

---

4. REDUCING VARIANCE (the make-or-break detail)

Raw REINFORCE is correct but VERY noisy. Two standard fixes, both used below:

(a) REWARD-TO-GO: action a_t can only affect FUTURE rewards, so weight it by the return
    FROM t onward (G_t), not the whole-episode return. Less variance, still unbiased.

(b) BASELINE SUBTRACTION: subtract a state-dependent baseline b(s_t) that doesn't depend on
    the action:
        ∇_θ J = E[ Σ_t ∇log π_θ(a_t|s_t) · (G_t - b(s_t)) ]
    Subtracting b doesn't change the expectation (it's unbiased) but slashes variance.
    The best baseline is V(s) → then (G_t - V(s_t)) is the ADVANTAGE A(s,a): "how much
    better than average was this action?" Using a learned V as baseline is the bridge to
    Actor-Critic (file 09). Below we use a simple, effective baseline: the batch mean
    return (and also show a learned-value baseline).

Also standard: normalize advantages to zero mean / unit std per batch for stable steps.

---

5. THE REINFORCE ALGORITHM

    for each batch of episodes:
        run π_θ to collect trajectories (s_t, a_t, r_t)
        compute returns-to-go G_t (discounted)
        compute advantages  Â_t = G_t - baseline
        loss = - mean_t [ log π_θ(a_t|s_t) · Â_t ]
        θ ← θ - α ∇_θ loss        (i.e., ascent on J)

It's "Monte Carlo" (file 04 lineage): updates after FULL episodes, uses real returns, no
bootstrapping. That's why it's unbiased but high-variance — and why Actor-Critic next swaps
the MC return for a bootstrapped TD estimate.

---

MATH SUMMARY:
  ∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · Â_t ]
  Â_t      = G_t - b(s_t)   (advantage; b ≈ V is best)
  loss     = - mean_t [ log π_θ(a_t|s_t) · Â_t ]   (minimize → ascent on J)

GOTCHAS:
- Sign! You ASCEND J, so MINIMIZE the negative surrogate. Flip it and the agent gets worse.
- Use reward-TO-GO, not full-episode return, for each action (much lower variance).
- A baseline that depends on the ACTION would bias the gradient — baseline = b(state) only.
- Don't backprop through the returns/advantages (treat them as constants → .detach()).
- High variance: average over several episodes per update; normalize advantages.
- Exploration comes from the policy's own stochasticity; add an ENTROPY bonus to keep it
  from collapsing to a near-deterministic policy too early.

INTUITION:
"Make good actions more likely and bad actions less likely, in proportion to how much
better than expected they turned out." That single sentence is the whole of policy gradients
— and the conceptual core of RLHF.
"""

import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("envs", os.path.join(_here, "_environments.py"))
_envs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_envs)
CartPole = _envs.CartPole


class PolicyNet(nn.Module):
    """Maps a state to a categorical distribution over discrete actions (the policy π_θ)."""

    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return Categorical(logits=logits)        # a distribution we can sample + score


class ValueNet(nn.Module):
    """Optional learned baseline b(s) ≈ V(s) to reduce variance (advantage = G - V)."""

    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def discounted_returns_to_go(rewards, gamma):
    """G_t = r_t + γ r_{t+1} + ... computed backward in O(T)."""
    G = 0.0
    out = np.zeros(len(rewards))
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


def run_episode(env, policy, max_steps=500):
    """Collect one trajectory under the CURRENT stochastic policy."""
    states, actions, rewards = [], [], []
    s = env.reset()
    for _ in range(max_steps):
        st = torch.tensor(s, dtype=torch.float32)
        dist = policy(st)
        a = dist.sample()
        s_next, r, done = env.step(int(a.item()))
        states.append(s); actions.append(int(a.item())); rewards.append(r)
        s = s_next
        if done:
            break
    return states, actions, rewards


def train_reinforce(n_iters=120, episodes_per_iter=8, gamma=0.99,
                    lr_pi=3e-3, lr_v=1e-2, use_value_baseline=True,
                    entropy_coef=0.01, seed=0):
    """REINFORCE with reward-to-go + (learned value or batch-mean) baseline on CartPole."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = CartPole(seed=seed)

    policy = PolicyNet(env.state_dim, env.n_actions)
    value = ValueNet(env.state_dim)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=lr_pi)
    opt_v = torch.optim.Adam(value.parameters(), lr=lr_v)

    history = []
    for it in range(n_iters):
        batch_states, batch_actions, batch_returns = [], [], []
        ep_returns = []

        # --- collect a batch of full episodes ---
        for _ in range(episodes_per_iter):
            s, a, r = run_episode(env, policy)
            G = discounted_returns_to_go(r, gamma)
            batch_states.extend(s); batch_actions.extend(a); batch_returns.extend(G)
            ep_returns.append(sum(r))

        S = torch.tensor(np.array(batch_states), dtype=torch.float32)
        A = torch.tensor(batch_actions, dtype=torch.int64)
        G = torch.tensor(np.array(batch_returns), dtype=torch.float32)

        # --- advantage = return - baseline ---
        if use_value_baseline:
            baseline = value(S).detach()
            adv = G - baseline
        else:
            adv = G - G.mean()                          # simple batch-mean baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # normalize for stable steps

        # --- policy (actor) loss: -E[ logπ(a|s) · Â ] - entropy bonus ---
        dist = policy(S)
        logp = dist.log_prob(A)
        entropy = dist.entropy().mean()
        pi_loss = -(logp * adv).mean() - entropy_coef * entropy
        opt_pi.zero_grad(); pi_loss.backward(); opt_pi.step()

        # --- fit the value baseline toward the observed returns (regression) ---
        if use_value_baseline:
            v_loss = nn.functional.mse_loss(value(S), G)
            opt_v.zero_grad(); v_loss.backward(); opt_v.step()

        history.append(np.mean(ep_returns))
        if (it + 1) % 15 == 0:
            print(f"  iter {it + 1:3d} | avg episode return: {np.mean(ep_returns):6.1f}")

    return history, policy, env


if __name__ == "__main__":
    print("=== REINFORCE (policy gradient) on CartPole ===\n")
    print("Training the policy DIRECTLY by gradient ascent on expected return...\n")
    history, policy, env = train_reinforce()

    print(f"\nAvg return  first 15 iters: {np.mean(history[:15]):6.1f}")
    print(f"Avg return   last 15 iters: {np.mean(history[-15:]):6.1f}")

    def greedy(s):
        with torch.no_grad():
            return int(torch.argmax(policy(torch.tensor(s, dtype=torch.float32)).logits).item())

    evals = [_envs.run_greedy_episode(CartPole(seed=2000 + i), greedy) for i in range(10)]
    print(f"\nGreedy policy over 10 fresh episodes: mean {np.mean(evals):.1f}, "
          f"min {min(evals):.0f}, max {max(evals):.0f}")
    print("\nREINFORCE uses FULL Monte-Carlo returns (unbiased, high variance).")
    print("Next (file 09): replace the MC return with a bootstrapped critic = Actor-Critic.")
