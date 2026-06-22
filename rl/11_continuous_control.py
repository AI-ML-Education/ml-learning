# Continuous Control — DDPG, TD3, and SAC
"""
WHAT/WHY:
DQN (file 07) needs `max_a Q(s,a)` — fine for a handful of discrete actions, impossible for
CONTINUOUS actions (a robot joint torque is a real number; there are infinitely many). Yet
most of the physical world — robotics, locomotion, autonomous driving, drug-dose control —
is continuous. This file covers the three algorithms that dominate continuous control:

  - DDPG : Deep Deterministic Policy Gradient (the foundation). Implemented & run below.
  - TD3  : Twin Delayed DDPG (DDPG + 3 stability fixes). Explained, with the fixes noted.
  - SAC  : Soft Actor-Critic (maximum-entropy RL, the current default). Explained.

All three are OFF-POLICY actor-critics with a replay buffer (like DQN), so they're sample
efficient — important when real-robot data is expensive.

---

1. THE CORE TRICK: A DETERMINISTIC ACTOR REPLACES THE max

DQN's bottleneck is `max_a Q(s,a)`. DDPG learns a deterministic actor μ_φ(s) that OUTPUTS
the maximizing action directly, so:

    max_a Q(s,a)   ≈   Q(s, μ_φ(s))

Now everything is differentiable. Two networks:
  - Critic Q_w(s,a): same TD/Bellman regression as DQN, but a takes a continuous vector.
  - Actor  μ_φ(s):   trained to output actions that MAXIMIZE the critic.

The actor's gradient is the DETERMINISTIC POLICY GRADIENT — just backprop the critic's
value w.r.t. the action, through the actor (the chain rule):

    ∇_φ J = E_s [ ∇_a Q_w(s,a)|_{a=μ_φ(s)} · ∇_φ μ_φ(s) ]

INTUITION: "Nudge the actor's output in whatever direction the critic says increases value."
The critic is a differentiable model of "how good"; the actor does gradient ascent on it.

---

2. DDPG'S MACHINERY (mostly borrowed from DQN)

  - Replay buffer: off-policy, decorrelated minibatches (file 07).
  - TARGET networks for BOTH actor and critic, updated by slow POLYAK averaging (soft update)
    instead of hard copies:   w⁻ ← τ·w + (1-τ)·w⁻,  with τ small (e.g., 0.005).
  - Critic target:  y = r + γ·(1-done)·Q⁻(s', μ⁻(s'))
  - EXPLORATION: a deterministic actor never explores on its own, so add NOISE to actions
    during data collection (Gaussian noise, or the original Ornstein-Uhlenbeck noise).

---

3. WHY TD3 (DDPG is fragile — three targeted fixes)

DDPG is notoriously unstable and OVERESTIMATES Q (the same maximization bias from file 05,
now through the actor). TD3 = DDPG + three fixes:

  (1) CLIPPED DOUBLE-Q: learn TWO critics, use the SMALLER for the target → curbs
      overestimation.       y = r + γ·min(Q⁻_1, Q⁻_2)(s', ã)
  (2) TARGET POLICY SMOOTHING: add clipped noise to the target action ã = μ⁻(s') + clip(ε)
      → prevents the actor from exploiting sharp critic errors (a regularizer).
  (3) DELAYED POLICY UPDATES: update the actor (and targets) less often than the critics
      (e.g., every 2 critic steps) → lets the critic settle before the actor chases it.

These three small changes turn DDPG from finicky to robust; TD3 is a great default.

---

4. WHY SAC (the modern go-to)

Soft Actor-Critic optimizes a MAXIMUM-ENTROPY objective: maximize reward AND policy entropy:

    J = E[ Σ_t r_t + α·H(π(·|s_t)) ]

  - The entropy bonus α·H rewards keeping the policy as random as possible while still
    succeeding → better EXPLORATION and far more ROBUST, stable training.
  - The actor is STOCHASTIC (a squashed Gaussian: mean+std passed through tanh to bound it),
    re-parameterized so you can backprop through sampling (the "reparameterization trick").
  - Uses clipped double-Q like TD3. The temperature α (entropy weight) can be auto-tuned.
SAC is usually the most sample-efficient and reliable for continuous control today.

---

5. DDPG/TD3/SAC vs PPO — when to use which

  - PPO (file 10): on-policy, robust, easy to tune, no replay buffer. Great default,
    especially with cheap/parallel simulators. The RLHF choice.
  - DDPG/TD3/SAC: off-policy, replay buffer, much more SAMPLE-EFFICIENT. Preferred when
    environment interactions are expensive (real robots) and actions are continuous.

---

MATH SUMMARY (DDPG):
  critic target:  y = r + γ(1-done) Q⁻(s', μ⁻(s'))
  critic loss:    L_Q = ( Q_w(s,a) - y )^2
  actor loss:     L_μ = - Q_w(s, μ_φ(s))          (maximize critic → minimize its negative)
  soft update:    θ⁻ ← τθ + (1-τ)θ⁻
  TD3 adds: twin critics + target min, target-action noise, delayed actor updates.
  SAC adds: stochastic actor + entropy bonus αH + reparameterized sampling.

GOTCHAS:
- No exploration noise → deterministic actor gets stuck; too much → never converges.
- Overestimation: single-critic DDPG inflates Q; use TD3's min of twin critics.
- Hard target copies cause instability; use small-τ Polyak (soft) updates.
- Unsquashed actions can exceed bounds; squash with tanh and scale to the action range.
- Replay + bootstrapping + approximation = the deadly triad (file 06); targets tame it.

INTUITION:
Continuous control = "DQN where a learned actor stands in for the impossible max, trained by
pushing its output uphill on the critic." TD3 makes that stable; SAC adds an entropy bonus
that makes exploration and robustness almost automatic.
"""

import importlib.util
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("envs", os.path.join(_here, "_environments.py"))
_envs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_envs)
PointMass1D = _envs.PointMass1D


class Actor(nn.Module):
    """Deterministic policy μ(s) → action in [-1, 1] (tanh-squashed)."""

    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh())

    def forward(self, s):
        return self.net(s)


class Critic(nn.Module):
    """Q(s, a) for continuous a: concatenate state and action."""

    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(np.array(a), dtype=torch.float32),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


def train_ddpg(n_episodes=120, gamma=0.99, tau=0.005, lr=1e-3,
               batch_size=128, noise_std=0.1, seed=0):
    """DDPG on the continuous PointMass1D task. Returns per-episode return history."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = PointMass1D(seed=seed)
    sd, ad = env.state_dim, env.action_dim

    actor, actor_t = Actor(sd, ad), Actor(sd, ad)
    critic, critic_t = Critic(sd, ad), Critic(sd, ad)
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr)
    buffer = ReplayBuffer()

    returns = []
    for ep in range(n_episodes):
        s = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            with torch.no_grad():
                a = actor(torch.tensor(s, dtype=torch.float32)).numpy()
            a = np.clip(a + np.random.normal(0, noise_std, size=ad), -1, 1)  # explore
            s2, r, done = env.step(a)
            buffer.push(s, a, r, s2, float(done))
            s = s2
            ep_ret += r

            if len(buffer) >= batch_size:
                bs, ba, br, bs2, bd = buffer.sample(batch_size)
                # --- critic update: regress to the Bellman target ---
                with torch.no_grad():
                    a2 = actor_t(bs2)
                    y = br + gamma * (1 - bd) * critic_t(bs2, a2)
                q = critic(bs, ba)
                critic_loss = nn.functional.mse_loss(q, y)
                opt_c.zero_grad(); critic_loss.backward(); opt_c.step()

                # --- actor update: push actions uphill on the critic ---
                actor_loss = -critic(bs, actor(bs)).mean()
                opt_a.zero_grad(); actor_loss.backward(); opt_a.step()

                # --- slow (Polyak) target tracking ---
                soft_update(actor_t, actor, tau)
                soft_update(critic_t, critic, tau)

        returns.append(ep_ret)
        if (ep + 1) % 20 == 0:
            print(f"  episode {ep + 1:3d} | avg return (last 20): {np.mean(returns[-20:]):7.2f}")

    return returns, actor, env


if __name__ == "__main__":
    print("=== DDPG on continuous PointMass1D (push the mass to the origin) ===\n")
    print("Reward is negative distance-energy; closer to 0 is better (max possible ~ 0)...\n")
    returns, actor, env = train_ddpg()

    print(f"\nAvg return  first 20 episodes: {np.mean(returns[:20]):7.2f}")
    print(f"Avg return   last 20 episodes: {np.mean(returns[-20:]):7.2f}")

    def greedy(s):
        with torch.no_grad():
            return actor(torch.tensor(s, dtype=torch.float32)).numpy()

    evals = [_envs.run_greedy_episode(PointMass1D(seed=5000 + i), greedy) for i in range(10)]
    rand = [_envs.run_greedy_episode(PointMass1D(seed=5000 + i),
                                     lambda s: np.random.uniform(-1, 1, 1)) for i in range(10)]
    print(f"\nLearned controller return (10 eps): {np.mean(evals):7.2f}")
    print(f"Random controller return  (10 eps): {np.mean(rand):7.2f}")
    print(f"DDPG is {np.mean(evals) - np.mean(rand):+.2f} better than random.")
    print("\nTD3 would add twin critics + target smoothing + delayed updates for stability;")
    print("SAC would add a stochastic actor + entropy bonus. Next (file 12): exploration & offline RL.")
