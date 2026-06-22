# TRPO → PPO — The Clipped Objective That Tamed Policy Gradients
"""
WHAT/WHY:
You saw in file 09 that vanilla actor-critic can collapse: one over-large policy update
destroys a good policy. The reason is that policy gradients are only valid LOCALLY — the
advantages were estimated under the OLD policy, so if you move too far, your data no longer
describes the policy you've created. PPO (Proximal Policy Optimization) fixes this with a
brilliantly simple trick: keep each update PROXIMAL (close) to the old policy.

PPO is the most important on-policy algorithm in practice. It's the default for robotics,
game-playing, and — crucially for you — it is the RL engine of RLHF (file 16). Master this
file and the LLM-alignment files become "PPO where the environment is a language model."

---

1. THE TRUST-REGION IDEA (TRPO, PPO's predecessor)

TRPO maximizes the policy improvement while CONSTRAINING how far the new policy can move
from the old, measured by KL divergence:

    maximize_θ   E[ (π_θ(a|s) / π_old(a|s)) · A ]
    subject to   E[ KL(π_old(·|s) || π_θ(·|s)) ] ≤ δ

The ratio  r(θ) = π_θ(a|s) / π_old(a|s)  is IMPORTANCE SAMPLING: it reweights data
collected under π_old to estimate the new policy's objective. TRPO enforces the trust region
exactly using second-order optimization (Fisher matrix, conjugate gradient) — correct but
complex and expensive.

---

2. PPO'S CLIPPED SURROGATE (first-order, dead simple, just as good)

PPO replaces the hard KL constraint with a CLIPPED objective that removes the incentive to
move the ratio too far:

    L_CLIP(θ) = E[ min( r(θ)·A ,  clip(r(θ), 1-ε, 1+ε)·A ) ]

with ε ≈ 0.2. Read it case by case (this is the heart of PPO):
  - A > 0 (good action): we want to increase its probability → r grows. But once r > 1+ε,
    the clip caps the objective: no extra reward for pushing further. Stops overshooting.
  - A < 0 (bad action): we want to decrease its probability → r shrinks. Once r < 1-ε, the
    clip caps it: no extra reward for pushing further down.
  - The `min` makes the bound PESSIMISTIC: clipping only ever removes incentive, never adds
    it, so the update can't be gamed by a large ratio.

INTUITION: "Improve the policy, but if a single update tries to change an action's
probability by more than ~20%, stop giving credit for going further." A self-imposed speed
limit that prevents the file-09 collapse.

---

3. WHY PPO CAN REUSE DATA (sample efficiency)

Because of importance sampling + clipping, PPO can take SEVERAL gradient EPOCHS over the
SAME batch of collected experience (e.g., 4–10 passes in minibatches), instead of one step
then throwing the data away. The clip keeps those repeated updates from drifting too far.
This makes PPO far more sample-efficient than vanilla policy gradients.

---

4. THE FULL PPO LOOP

    repeat:
        collect a rollout with π_old (store states, actions, rewards, values, logπ_old)
        compute advantages with GAE (file 09); compute value targets
        for K epochs:
            for each minibatch:
                r(θ)   = exp( logπ_θ(a|s) - logπ_old(a|s) )        # the ratio
                L_clip = -mean( min(r·Â, clip(r,1-ε,1+ε)·Â) )      # clipped actor loss
                L_v    =  mean( (V_θ(s) - V_target)^2 )            # critic loss
                L_ent  = -mean( entropy )                          # bonus (exploration)
                loss   = L_clip + c1·L_v + c2·L_ent
                gradient step on θ
        π_old ← π_θ

Common extras (all standard in real PPO): advantage normalization, value-function clipping,
gradient clipping, an early-stop if KL exceeds a threshold, and orthogonal init.

---

5. PPO IN ONE PICTURE OF THE FAMILY TREE

    Policy Gradient (REINFORCE, file 08)
        + learned critic + GAE        → Actor-Critic / A2C (file 09)
        + clipped trust region        → PPO (this file)
        + LM as policy, reward model  → RLHF / PPO-for-LLMs (file 16)

---

MATH SUMMARY:
  ratio:    r(θ) = π_θ(a|s) / π_old(a|s) = exp(logπ_θ - logπ_old)
  clip:     L_CLIP = E[ min( r·A, clip(r, 1-ε, 1+ε)·A ) ]
  total:    L = -L_CLIP + c1·MSE(V, V_target) - c2·H(π)

GOTCHAS:
- Compute logπ_old ONCE at collection time and DETACH it; the ratio's denominator is fixed.
- Recompute logπ_θ fresh each epoch (that's what carries the gradient).
- Forgetting the `min`/pessimism turns clipping into a no-op.
- ε too large → PPO ≈ unstable A2C; too small → painfully slow. 0.1–0.3 is typical.
- Too many epochs over one batch → policy drifts off-data despite clipping (watch the KL).
- Normalize advantages per batch; mismatched value/advantage scaling hurts.

INTUITION:
PPO = actor-critic with a "don't change too fast" guardrail implemented as a clip. That one
guardrail turned policy gradients from finicky to reliable — which is why PPO became the
backbone of both game-playing agents and instruction-tuned LLMs.
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
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return Categorical(logits=self.actor(h)), self.critic(h).squeeze(-1)


def compute_gae(rewards, values, dones, gamma, lam, last_value):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae, next_value = 0.0, last_value
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


def train_ppo(n_updates=60, rollout_len=2048, gamma=0.99, lam=0.95,
              clip_eps=0.2, epochs=10, minibatch=256, lr=3e-4,
              value_coef=0.5, entropy_coef=0.01, seed=0):
    """Proximal Policy Optimization on CartPole."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = CartPole(seed=seed)
    net = ActorCritic(env.state_dim, env.n_actions)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    s = env.reset()
    ep_return, completed = 0.0, []

    for update in range(n_updates):
        states, actions, rewards, dones, values, logps = [], [], [], [], [], []

        # --- collect a rollout under the CURRENT (soon-to-be-old) policy ---
        for _ in range(rollout_len):
            st = torch.tensor(s, dtype=torch.float32)
            with torch.no_grad():
                dist, v = net(st)
                a = dist.sample()
                logp = dist.log_prob(a)
            s_next, r, done = env.step(int(a.item()))
            states.append(s); actions.append(int(a.item())); rewards.append(r)
            dones.append(float(done)); values.append(v.item()); logps.append(logp.item())
            ep_return += r
            s = s_next
            if done:
                completed.append(ep_return); ep_return = 0.0; s = env.reset()

        with torch.no_grad():
            _, last_v = net(torch.tensor(s, dtype=torch.float32))
        adv, returns = compute_gae(rewards, values, dones, gamma, lam, last_v.item())

        S = torch.tensor(np.array(states), dtype=torch.float32)
        A = torch.tensor(actions, dtype=torch.int64)
        OLD_LOGP = torch.tensor(logps, dtype=torch.float32)         # fixed denominator
        ADV = torch.tensor(adv, dtype=torch.float32)
        RET = torch.tensor(returns, dtype=torch.float32)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)

        # --- K epochs of minibatch updates over the SAME data (PPO's reuse) ---
        n = len(S)
        idx = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, n, minibatch):
                mb = idx[start:start + minibatch]
                dist, v = net(S[mb])
                new_logp = dist.log_prob(A[mb])
                ratio = torch.exp(new_logp - OLD_LOGP[mb])          # r(θ)

                surr1 = ratio * ADV[mb]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * ADV[mb]
                actor_loss = -torch.min(surr1, surr2).mean()        # clipped, pessimistic
                critic_loss = nn.functional.mse_loss(v, RET[mb])
                entropy = dist.entropy().mean()
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if (update + 1) % 6 == 0 and completed:
            print(f"  update {update + 1:2d} | episodes {len(completed):4d} | "
                  f"avg return (last 20): {np.mean(completed[-20:]):6.1f}")

    return completed, net, env


if __name__ == "__main__":
    print("=== PPO (clipped surrogate) on CartPole ===\n")
    print("Same actor-critic as file 09, now with a clip that prevents destructive steps...\n")
    completed, net, env = train_ppo()

    if completed:
        print(f"\nAvg return  first 20 episodes: {np.mean(completed[:20]):6.1f}")
        print(f"Avg return   last 20 episodes: {np.mean(completed[-20:]):6.1f}")

    def greedy(s):
        with torch.no_grad():
            dist, _ = net(torch.tensor(s, dtype=torch.float32))
            return int(torch.argmax(dist.logits).item())

    evals = [_envs.run_greedy_episode(CartPole(seed=4000 + i), greedy) for i in range(10)]
    print(f"\nGreedy policy over 10 fresh episodes: mean {np.mean(evals):.1f}, "
          f"min {min(evals):.0f}, max {max(evals):.0f}")
    print("\nNotice PPO's smooth, monotonic-ish improvement vs A2C's volatility.")
    print("Next (file 11): continuous actions — DDPG, TD3, SAC. Then RLHF uses THIS PPO.")
