# Actor-Critic & GAE — Bootstrapping Meets Policy Gradients
"""
WHAT/WHY:
REINFORCE (file 08) is unbiased but high-variance because it waits for the FULL Monte-Carlo
return G_t. Actor-Critic borrows the bootstrapping idea from TD (file 05): replace the noisy
return with a learned VALUE estimate. This is the synthesis of the two great RL families —
and the template that PPO (file 10) and RLHF (file 16) build on directly.

Two networks working together:
  - ACTOR  π_θ(a|s): chooses actions (the policy). Updated by policy gradient.
  - CRITIC V_w(s):   estimates how good states are. Updated by TD regression.
The critic "criticizes" the actor's choices by providing a low-variance advantage signal;
the actor improves toward actions the critic rates above average.

Analogy: a performer (actor) and a coach (critic). The coach doesn't act, but gives instant,
informed feedback ("that was better than your average move"), so the performer improves
faster than if they had to wait for the final score each time.

---

1. THE ADVANTAGE (the actor's learning signal)

Advantage = how much better an action was than the state's average:
    A(s,a) = Q(s,a) - V(s)

We don't have Q directly, but the TD error is a one-sample, bootstrapped estimate of it:
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)     ≈  A(s_t, a_t)

Policy gradient with this advantage (the actor update):
    ∇_θ J = E[ ∇_θ log π_θ(a_t|s_t) · A_t ]

Compare the "weight" on each action across methods:
    REINFORCE:        G_t                      (full MC return — unbiased, high variance)
    Actor-Critic(TD): r_t + γV(s_{t+1}) - V(s_t)  (bootstrapped — biased, low variance)

---

2. THE BIAS–VARIANCE KNOB: n-step and GAE

One-step TD has low variance but high BIAS (it trusts V, which is imperfect). The full MC
return has no bias but high variance. n-step returns interpolate:
    A_t^(n) = r_t + γ r_{t+1} + ... + γ^n V(s_{t+n}) - V(s_t)

GENERALIZED ADVANTAGE ESTIMATION (GAE) elegantly averages ALL n-step advantages with an
exponential weight λ ∈ [0,1]:
    A_t^GAE = Σ_{l≥0} (γλ)^l · δ_{t+l}        where δ_t = r_t + γV(s_{t+1}) - V(s_t)

  - λ = 0  → A_t = δ_t            (one-step TD: low variance, more bias)
  - λ = 1  → A_t = G_t - V(s_t)   (Monte-Carlo: no bias, high variance)
  - λ ≈ 0.95 is the sweet spot used almost everywhere (incl. PPO/RLHF).

Computed efficiently backward in O(T):
    A_t = δ_t + γλ · A_{t+1}      (and the value target is  V_target_t = A_t + V(s_t))

GAE is THE advantage estimator you'll see in PPO and RLHF — this file is where it's born.

---

3. THE A2C UPDATE (one synchronous step)

    collect a batch of transitions with the current actor
    compute δ_t and A_t^GAE using the critic
    actor loss  = - mean_t [ log π_θ(a_t|s_t) · A_t ]      (ascent on J)
    critic loss =   mean_t [ (V_w(s_t) - V_target_t)^2 ]   (regression to TD target)
    entropy bonus encourages exploration (subtract β·H from the loss)
    total loss  = actor_loss + c1·critic_loss - c2·entropy
    one gradient step on θ and w

"A2C" = Advantage Actor-Critic (synchronous). "A3C" is the older ASYNCHRONOUS version that
ran many actors in parallel threads; A2C is the simpler, equally-good synchronous variant.

---

4. WHY THIS MATTERS FOR EVERYTHING AFTER

PPO (file 10) = A2C + GAE + a clipped objective that prevents destructive big updates.
RLHF (file 16) = PPO where the "environment" is a language model generating tokens and the
"reward" comes from a learned human-preference model. So the actor-critic structure you
build here is literally the skeleton of LLM alignment.

---

MATH SUMMARY:
  δ_t      = r_t + γ V(s_{t+1}) - V(s_t)
  A_t^GAE  = δ_t + γλ A_{t+1}^GAE
  V_target = A_t^GAE + V(s_t)
  actor:   L_π = - mean[ logπ(a|s) A_t ]
  critic:  L_V =   mean[ (V(s) - V_target)^2 ]

GOTCHAS:
- Detach the advantage when updating the actor (it's a weight, not a path for gradients).
- Detach the critic's bootstrap target (semi-gradient, like file 06).
- Critic must be "warm" enough; a wildly wrong V gives a misleading advantage.
- Balance actor vs critic learning rates / loss weights, or one dominates.
- Forgetting the (1-done) mask on the bootstrap at episode end.
- Too little entropy → premature determinism → stuck in a bad policy.

INTUITION:
Actor-Critic = "policy gradients, but instead of waiting for the final score, ask a learned
coach (the critic) for instant feedback on whether each action beat the average." Lower
variance, faster learning, and the foundation of modern on-policy RL.
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


class ActorCritic(nn.Module):
    """Shared trunk with two heads: a policy (actor) and a value (critic)."""

    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return Categorical(logits=self.actor(h)), self.critic(h).squeeze(-1)


def compute_gae(rewards, values, dones, gamma, lam, last_value):
    """Backward GAE: returns advantages A_t and value targets (A_t + V_t)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


def train_a2c(n_updates=300, rollout_len=256, gamma=0.99, lam=0.95,
              lr=7e-4, value_coef=0.5, entropy_coef=0.01, seed=0):
    """A2C with GAE on CartPole. A single environment, fixed-length rollouts."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = CartPole(seed=seed)
    net = ActorCritic(env.state_dim, env.n_actions)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    s = env.reset()
    ep_return, completed = 0.0, []

    for update in range(n_updates):
        states, actions, rewards, dones, values, logps = [], [], [], [], [], []

        # --- collect a fixed-length rollout (may span multiple episodes) ---
        for _ in range(rollout_len):
            st = torch.tensor(s, dtype=torch.float32)
            dist, v = net(st)
            a = dist.sample()
            s_next, r, done = env.step(int(a.item()))

            states.append(s); actions.append(int(a.item())); rewards.append(r)
            dones.append(float(done)); values.append(v.item())
            logps.append(dist.log_prob(a).item())

            ep_return += r
            s = s_next
            if done:
                completed.append(ep_return); ep_return = 0.0
                s = env.reset()

        # bootstrap value of the state we stopped at
        with torch.no_grad():
            _, last_v = net(torch.tensor(s, dtype=torch.float32))
        adv, returns = compute_gae(rewards, values, dones, gamma, lam, last_v.item())

        S = torch.tensor(np.array(states), dtype=torch.float32)
        A = torch.tensor(actions, dtype=torch.int64)
        ADV = torch.tensor(adv, dtype=torch.float32)
        RET = torch.tensor(returns, dtype=torch.float32)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        # --- joint actor + critic update ---
        dist, v = net(S)
        logp = dist.log_prob(A)
        actor_loss = -(logp * ADV).mean()
        critic_loss = nn.functional.mse_loss(v, RET)
        entropy = dist.entropy().mean()
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

        if (update + 1) % 30 == 0 and completed:
            recent = np.mean(completed[-20:])
            print(f"  update {update + 1:3d} | episodes done {len(completed):4d} | "
                  f"avg return (last 20): {recent:6.1f}")

    return completed, net, env


if __name__ == "__main__":
    print("=== A2C (Advantage Actor-Critic) with GAE on CartPole ===\n")
    print("Actor proposes actions; critic scores states; GAE forms the advantage...\n")
    completed, net, env = train_a2c()

    if completed:
        print(f"\nAvg return  first 20 episodes: {np.mean(completed[:20]):6.1f}")
        print(f"Avg return   last 20 episodes: {np.mean(completed[-20:]):6.1f}")

    def greedy(s):
        with torch.no_grad():
            dist, _ = net(torch.tensor(s, dtype=torch.float32))
            return int(torch.argmax(dist.logits).item())

    evals = [_envs.run_greedy_episode(CartPole(seed=3000 + i), greedy) for i in range(10)]
    print(f"\nGreedy policy over 10 fresh episodes: mean {np.mean(evals):.1f}, "
          f"min {min(evals):.0f}, max {max(evals):.0f}")
    print("\nGAE (λ) traded bias vs variance; the critic cut REINFORCE's noise.")
    print("Next (file 10): TRPO → PPO add a CLIPPED objective so updates never blow up.")
