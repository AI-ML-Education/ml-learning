# Deep Q-Networks (DQN) — Q-Learning Meets Neural Networks
"""
WHAT/WHY:
DQN is the algorithm that ignited modern Deep RL: in 2015 it learned to play dozens of
Atari games from raw pixels, at human level, using ONE architecture and ONE set of
hyperparameters. It is, at its core, exactly the Q-learning of file 05 with a neural net
as the function approximator from file 06 — plus two stabilizers that make that marriage
actually work.

Recall the deadly triad (file 06): approximation + bootstrapping + off-policy can diverge.
Naively training a neural net with the Q-learning target is unstable because:
  1. Consecutive samples are highly correlated (you walk through one trajectory).
  2. The target r + γ max Q(s') uses the SAME network you're updating → a moving target
     chasing itself ("you're trying to hit a target that jumps every time you shoot").

DQN fixes both:

---

1. EXPERIENCE REPLAY (fixes correlation + data efficiency)

Store transitions (s, a, r, s', done) in a big REPLAY BUFFER. Train on RANDOM MINIBATCHES
sampled from it.
  - Breaks temporal correlation → gradients behave like i.i.d. supervised learning.
  - Reuses each transition many times → sample efficient.
Analogy: instead of cramming chapters in order, you study from randomly shuffled flashcards
covering everything you've seen.

---

2. TARGET NETWORK (fixes the moving target)

Keep a SECOND, frozen copy of the network, Q_target with weights w⁻, used ONLY to compute
the bootstrap target. Periodically (every C steps) copy w → w⁻.

    target  y = r + γ max_{a'} Q_target(s', a'; w⁻)     # w⁻ held fixed for a while
    loss      = ( y - Q(s, a; w) )^2                     # update only w
    every C steps:  w⁻ ← w

The target stays still long enough for the online network to converge toward it.

---

3. THE FULL DQN LOOP

    initialize Q (weights w), target Q⁻ (w⁻ = w), replay buffer D
    for each step:
        a = ε-greedy(Q(s))                      # explore (ε decays over time)
        observe (r, s', done); store (s,a,r,s',done) in D
        sample minibatch from D
        y = r + (1-done)·γ·max_a' Q⁻(s',a')     # target net, no grad
        gradient step on (y - Q(s,a))^2 w.r.t. w
        occasionally  w⁻ ← w

Loss is usually HUBER (smooth L1): quadratic for small errors, linear for large ones, so
the occasional huge TD error doesn't blow up the gradients.

---

4. KEY VARIANTS (each fixes one weakness — together they form "Rainbow")

- DOUBLE DQN: vanilla DQN's max overestimates Q (maximization bias, file 05 gotcha).
  Decouple action SELECTION from EVALUATION:
      a* = argmax_a Q_online(s', a)             # online net picks the action
      y  = r + γ Q_target(s', a*)               # target net scores it
  Cheap, one-line change, noticeably more stable. (Implemented below as a toggle.)

- DUELING DQN: split the head into V(s) and Advantage A(s,a), recombine
      Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
  Learns "which states are valuable" separately from "which action is better" → faster.

- PRIORITIZED EXPERIENCE REPLAY (PER): sample surprising transitions (big |TD error|)
  more often, with importance-sampling correction. Learns from rare, informative events.

- N-STEP returns, NOISY NETS (exploration), DISTRIBUTIONAL RL (C51/QR-DQN: predict the
  whole return distribution, not just its mean). Rainbow = all of these combined.

---

MATH SUMMARY:
  target:  y = r + (1-done)·γ·max_{a'} Q⁻(s', a')        (Double: use online argmax)
  loss:    L(w) = E_{(s,a,r,s')~D} [ Huber( y - Q_w(s,a) ) ]
  sync:    w⁻ ← w  every C steps

GOTCHAS:
- Forgetting (1-done): bootstrapping past a terminal state poisons learning.
- Updating the target net every step (defeats its purpose) or never (stale).
- Replay buffer too small → still correlated; too small batch → noisy.
- Not decaying ε → never exploits; decaying too fast → premature convergence.
- DQN is for DISCRETE actions only. Continuous actions need file 11 (DDPG/SAC).
- Reward/observation scaling matters a lot for neural stability.

INTUITION:
DQN = tabular Q-learning where the "table" is a neural net, made stable by (1) learning
from a shuffled memory of past experience and (2) chasing a target that holds still.
Those two ideas unlocked RL from raw pixels.
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
CartPole = _envs.CartPole


class QNetwork(nn.Module):
    """Maps a state vector to one Q-value per discrete action: Q(s, ·)."""

    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular buffer of transitions; sample uniform random minibatches."""

    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.int64).unsqueeze(1),
                torch.tensor(r, dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.buf)


def train_dqn(n_episodes=200, gamma=0.99, lr=1e-3, batch_size=64,
              target_sync=500, double=True, seed=0):
    """Train DQN (Double DQN by default) on CartPole. Returns episode-return history."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = CartPole(seed=seed)
    online = QNetwork(env.state_dim, env.n_actions)
    target = QNetwork(env.state_dim, env.n_actions)
    target.load_state_dict(online.state_dict())          # start identical
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer()

    eps_start, eps_end, eps_decay = 1.0, 0.02, 0.995
    eps = eps_start
    global_step = 0
    returns = []

    for ep in range(n_episodes):
        s = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            # --- act: ε-greedy ---
            if random.random() < eps:
                a = random.randrange(env.n_actions)
            else:
                with torch.no_grad():
                    q = online(torch.tensor(s, dtype=torch.float32))
                    a = int(torch.argmax(q).item())

            s_next, r, done = env.step(a)
            buffer.push(s, a, r, s_next, float(done))
            s = s_next
            ep_return += r
            global_step += 1

            # --- learn from a random minibatch ---
            if len(buffer) >= batch_size:
                bs, ba, br, bs2, bd = buffer.sample(batch_size)
                q_sa = online(bs).gather(1, ba)                       # Q(s,a)

                with torch.no_grad():
                    if double:
                        next_a = online(bs2).argmax(1, keepdim=True)  # online selects
                        q_next = target(bs2).gather(1, next_a)        # target evaluates
                    else:
                        q_next = target(bs2).max(1, keepdim=True)[0]  # vanilla max
                    y = br + (1.0 - bd) * gamma * q_next              # (1-done) guard

                loss = nn.functional.smooth_l1_loss(q_sa, y)         # Huber loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()

            # --- periodically refresh the frozen target network ---
            if global_step % target_sync == 0:
                target.load_state_dict(online.state_dict())

        eps = max(eps_end, eps * eps_decay)
        returns.append(ep_return)
        if (ep + 1) % 20 == 0:
            avg = np.mean(returns[-20:])
            print(f"  episode {ep + 1:3d} | eps {eps:.2f} | avg return (last 20): {avg:6.1f}")

    return returns, online, env


if __name__ == "__main__":
    print("=== Deep Q-Network (Double DQN) on CartPole ===\n")
    print("Training... (return = how many steps the pole stayed up; max 500)\n")
    returns, net, env = train_dqn(n_episodes=200, double=True)

    first20 = np.mean(returns[:20])
    last20 = np.mean(returns[-20:])
    best = max(returns)
    print(f"\nAverage return  first 20 episodes: {first20:6.1f}")
    print(f"Average return   last 20 episodes: {last20:6.1f}")
    print(f"Best single episode return:        {best:6.1f}")
    print(f"Improvement factor: {last20 / max(first20, 1):.1f}x")

    # Greedy evaluation of the learned policy (no exploration).
    def greedy(s):
        with torch.no_grad():
            return int(torch.argmax(net(torch.tensor(s, dtype=torch.float32))).item())

    eval_returns = [_envs.run_greedy_episode(CartPole(seed=1000 + i), greedy) for i in range(10)]
    print(f"\nGreedy policy over 10 fresh episodes: mean {np.mean(eval_returns):.1f}, "
          f"min {min(eval_returns):.0f}, max {max(eval_returns):.0f}")
    print("\nDQN learns DISCRETE-action control. Next (file 08): learn the policy")
    print("DIRECTLY with policy gradients instead of going through Q-values.")
