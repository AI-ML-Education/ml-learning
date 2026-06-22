# Temporal Difference Learning — SARSA & Q-Learning (The RL Workhorses)
"""
WHAT/WHY:
Temporal Difference (TD) learning is the heart of modern RL. It combines the best of the
two methods you've seen:
  - Like Monte Carlo (file 04): it learns from raw experience, NO model required.
  - Like Dynamic Programming (file 03): it BOOTSTRAPS — it updates a value estimate using
    other current estimates, without waiting for the episode to end.

That second point is the breakthrough: TD updates EVERY step, online, even in never-ending
tasks. This is why TD (and its deep descendants like DQN) dominate practical RL.

---

1. THE TD(0) UPDATE (prediction)

After observing a single transition (s, a, r, s'):

    TD target  =  r + γ·V(s')          # one real reward + estimate of the rest
    TD error δ =  [r + γ·V(s')] - V(s)  # surprise: target minus current estimate
    V(s)       ←  V(s) + α·δ            # nudge estimate toward the target

α (alpha) ∈ (0, 1] is the LEARNING RATE (step size).

Compare the targets:
    Monte Carlo target:  G_t              (the FULL actual return — needs episode end)
    TD(0) target:        r + γ·V(s')      (one step real, rest BOOTSTRAPPED)

INTUITION of bootstrapping: you don't replay the whole game to value a chess move — you
glance one move ahead and trust your gut feel ("estimate") for the resulting position.

BIAS/VARIANCE: MC is unbiased but high-variance (full return is noisy). TD is biased
(it trusts imperfect estimates) but low-variance and updates immediately → usually learns
much faster.

---

2. SARSA (on-policy control)

Name = the tuple it uses: (S, A, R, S', A'). It updates Q using the action a' that the
policy ACTUALLY takes next:

    Q(s,a) ← Q(s,a) + α [ r + γ·Q(s', a') - Q(s,a) ]
                                     ^^^^ a' chosen by the SAME (ε-greedy) policy

"On-policy": it learns the value of the policy it is following — including its exploration.
Effect: SARSA learns a SAFER policy, because it accounts for the fact that it sometimes
explores (and might step into the pit).

---

3. Q-LEARNING (off-policy control)

Updates Q using the BEST next action (max), regardless of what the policy actually does:

    Q(s,a) ← Q(s,a) + α [ r + γ·max_{a'} Q(s', a') - Q(s,a) ]
                                     ^^^^^^^^^^^^^ greedy target, not the action taken

"Off-policy": it directly learns the OPTIMAL (greedy) policy Q* while still EXPLORING with
an ε-greedy behavior policy. This is the famous decoupling of "behave" vs "learn about".
Q-learning is a sampled version of the Bellman OPTIMALITY equation (file 02).

---

4. SARSA vs Q-LEARNING — the classic "Cliff" intuition

Imagine a shortcut path running right along a cliff edge (a pit). Off the cliff = big
negative reward.
  - Q-learning learns the OPTIMAL path: walk right along the edge (greedy/optimal).
  - SARSA learns a SAFER path: stay a row away from the edge, because while exploring
    (ε-greedy) it occasionally slips off the cliff, so it values edge-states lower.
Neither is "wrong" — they optimize different objectives (optimal vs. on-policy-with-
exploration). As ε → 0, SARSA's policy approaches Q-learning's optimal one.

---

5. THE GENERAL PATTERN (you'll see this everywhere, including deep RL)

    new_estimate ← old_estimate + step_size · (target - old_estimate)

DQN is literally Q-learning with a neural network approximating Q and this exact target.
Internalize this line and you understand the engine of value-based RL.

---

MATH (side by side):
  TD(0):      V(s)   ← V(s)   + α[ r + γ V(s')               - V(s)   ]
  SARSA:      Q(s,a) ← Q(s,a) + α[ r + γ Q(s', a')           - Q(s,a) ]   (a' ~ policy)
  Q-learning: Q(s,a) ← Q(s,a) + α[ r + γ max_{a'} Q(s', a')  - Q(s,a) ]   (greedy target)

GOTCHAS:
- Terminal next-state: the target is just r (no γ·Q(s',·) term, since there's no future).
  Bootstrapping past a terminal state is a classic bug.
- Learning rate α too high → unstable/oscillates; too low → crawls. Try 0.1–0.5 here.
- Exploration: a purely greedy policy can get stuck; keep ε > 0 (often decay it).
- Q-learning's max introduces "maximization bias" (overestimation) — Double Q-learning
  fixes it (a great next topic).
- On-policy (SARSA) value depends on ε; off-policy (Q-learning) targets the optimal policy.

INTUITION:
TD = "learn a little from every single step by comparing what you expected to what just
happened." It's how you'd actually coach someone in real time, not just after the game.
"""

import importlib.util
import os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_here, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mdp = _load("mdp_module", "01_markov_decision_processes.py")
_dp = _load("dp_module", "03_dynamic_programming.py")
GridWorld = _mdp.GridWorld


def epsilon_greedy(Q, s, eps, n_actions, rng):
    """Pick a random action with prob eps, else the greedy action argmax_a Q[s,a]."""
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    return int(np.argmax(Q[s]))


def td0_prediction(env, policy, n_episodes=5000, alpha=0.1, gamma=None, max_steps=200):
    """TD(0) prediction: estimate V_π for a fixed stochastic policy, online, no model."""
    gamma = env.gamma if gamma is None else gamma
    V = np.zeros(env.n_states)
    rng = np.random.default_rng(0)
    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = rng.choice(env.n_actions, p=policy[s])
            s_next, r, done = env.step(a)
            target = r if done else r + gamma * V[s_next]   # no bootstrap past terminal
            V[s] += alpha * (target - V[s])                 # TD error update
            s = s_next
            if done:
                break
    return V


def sarsa(env, n_episodes=8000, alpha=0.5, gamma=None,
          eps_start=1.0, eps_end=0.05, max_steps=200):
    """On-policy TD control. Returns Q and greedy policy."""
    gamma = env.gamma if gamma is None else gamma
    Q = np.zeros((env.n_states, env.n_actions))
    rng = np.random.default_rng(0)
    for ep in range(n_episodes):
        eps = eps_start + (eps_end - eps_start) * (ep / n_episodes)
        s = env.reset()
        a = epsilon_greedy(Q, s, eps, env.n_actions, rng)
        for _ in range(max_steps):
            s_next, r, done = env.step(a)
            if done:
                Q[s, a] += alpha * (r - Q[s, a])            # terminal: target = r
                break
            a_next = epsilon_greedy(Q, s_next, eps, env.n_actions, rng)  # the second 'A'
            target = r + gamma * Q[s_next, a_next]          # bootstrap on action TAKEN
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s_next, a_next
    return Q, np.argmax(Q, axis=1)


def q_learning(env, n_episodes=8000, alpha=0.5, gamma=None,
               eps_start=1.0, eps_end=0.05, max_steps=200):
    """Off-policy TD control. Returns Q and greedy policy."""
    gamma = env.gamma if gamma is None else gamma
    Q = np.zeros((env.n_states, env.n_actions))
    rng = np.random.default_rng(0)
    for ep in range(n_episodes):
        eps = eps_start + (eps_end - eps_start) * (ep / n_episodes)
        s = env.reset()
        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, eps, env.n_actions, rng)   # behave: ε-greedy
            s_next, r, done = env.step(a)
            if done:
                Q[s, a] += alpha * (r - Q[s, a])                # terminal: target = r
                break
            target = r + gamma * np.max(Q[s_next])              # learn: greedy target
            Q[s, a] += alpha * (target - Q[s, a])
            s = s_next
    return Q, np.argmax(Q, axis=1)


def greedy_success_rate(env, policy, n_trials=2000, max_steps=200):
    """Run the greedy policy and report the fraction of episodes that reach the goal."""
    rng = np.random.default_rng(123)
    env_eval = type(env)(slip=env.slip, gamma=env.gamma, seed=0)
    env_eval.rng = rng
    goal = env_eval._to_s(env_eval.goal)
    wins = 0
    for _ in range(n_trials):
        s = env_eval.reset()
        for _ in range(max_steps):
            s, r, done = env_eval.step(int(policy[s]))
            if done:
                if s == goal:
                    wins += 1
                break
    return wins / n_trials


if __name__ == "__main__":
    print("=== Temporal Difference learning on GridWorld ===\n")
    env = GridWorld(slip=0.0)

    # --- TD(0) prediction vs exact DP ---
    rand_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
    V_dp = _dp.policy_evaluation(env, rand_policy)
    V_td = td0_prediction(env, rand_policy, n_episodes=5000)
    print(f"TD(0) prediction error vs exact DP: "
          f"max|V_DP - V_TD| = {np.max(np.abs(V_dp - V_td)):.3f}\n")

    # --- SARSA vs Q-learning control ---
    Q_s, pi_sarsa = sarsa(env)
    Q_q, pi_qlearn = q_learning(env)
    _, pi_star, _ = _dp.value_iteration(env)

    env.render_policy(pi_sarsa, title="SARSA learned policy (on-policy):")
    env.render_policy(pi_qlearn, title="Q-learning learned policy (off-policy):")
    env.render_policy(pi_star, title="Optimal policy from Value Iteration (ground truth):")

    non_terminal = [s for s in range(env.n_states) if not env.is_terminal(s)]
    sarsa_match = sum(pi_sarsa[s] == pi_star[s] for s in non_terminal)
    q_match = sum(pi_qlearn[s] == pi_star[s] for s in non_terminal)
    print(f"SARSA      matches optimal on {sarsa_match}/{len(non_terminal)} states")
    print(f"Q-learning matches optimal on {q_match}/{len(non_terminal)} states\n")

    # --- On an ICY world, SARSA's caution vs Q-learning's optimism shows up ---
    icy = GridWorld(slip=0.3)
    _, pi_sarsa_icy = sarsa(icy)
    _, pi_q_icy = q_learning(icy)
    print(f"ICY world (slip=0.3) greedy-policy success rate (reaching the goal):")
    print(f"  SARSA      : {greedy_success_rate(icy, pi_sarsa_icy):.1%}")
    print(f"  Q-learning : {greedy_success_rate(icy, pi_q_icy):.1%}")
    print("\nYou've now built tabular RL from scratch: DP → Monte Carlo → TD.")
    print("Next steps: function approximation & Deep Q-Networks, then policy gradients.")
