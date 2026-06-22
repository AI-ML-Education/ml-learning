# Dynamic Programming — Solving an MDP When You KNOW the Model
"""
WHAT/WHY:
Dynamic Programming (DP) solves an MDP *exactly* when we have full knowledge of the model
(the transition probabilities P and rewards R). It's the "cheat mode" of RL: because we
know how the world works, we can compute the optimal policy by directly turning the Bellman
equations (file 02) into iterative updates.

You rarely have the true model in real problems — so why learn DP first?
  1. It builds rock-solid intuition for value functions and policies.
  2. Every model-free method (Monte Carlo, TD, DQN) is an *approximation* of these exact
     updates using samples instead of the known model.
  3. It gives us a GROUND-TRUTH optimal policy to compare the later methods against.

We solve the GridWorld defined in file 01 three ways:
  A) Policy Evaluation  — given a fixed policy, compute its V (prediction).
  B) Policy Iteration   — alternate evaluation + greedy improvement until optimal (control).
  C) Value Iteration    — collapse the two into one update; fastest to code (control).

---

1. POLICY EVALUATION (prediction): "How good is THIS policy?"

Repeatedly apply the Bellman EXPECTATION equation as an assignment until V stops changing:

    V(s) ← Σ_a π(a|s) Σ_{s'} P(s'|s,a) [ R + γ V(s') ]

This is a fixed-point iteration. Because the Bellman operator is a γ-contraction (file 02),
V is guaranteed to converge to V_π regardless of where you start.

---

2. POLICY IMPROVEMENT: "Can I do better by acting greedily?"

Given V_π, compute Q_π(s,a) for every action, then make a NEW policy that picks the best:

    π'(s) = argmax_a  Σ_{s'} P(s'|s,a) [ R + γ V_π(s') ]

The Policy Improvement Theorem guarantees π' is at least as good as π. If π' == π, we've
hit the optimal policy π*.

---

3. POLICY ITERATION (control): evaluate ↔ improve until stable.

    repeat:
        V ← PolicyEvaluation(π)        # make V consistent with current π
        π ← greedy(V)                  # improve π using that V
    until π stops changing

Converges to the optimal π* in a finite number of iterations (often very few).

---

4. VALUE ITERATION (control): why wait for full evaluation?

Fold improvement directly into evaluation using the Bellman OPTIMALITY equation:

    V(s) ← max_a Σ_{s'} P(s'|s,a) [ R + γ V(s') ]

Run until V stops changing, then read off the greedy policy once at the end. Simpler to
implement and usually faster in practice.

---

POLICY ITERATION vs VALUE ITERATION:
  - Policy Iteration: fully evaluates each policy (expensive inner loop) but needs few
    outer iterations. Exact π each round.
  - Value Iteration:  one Bellman-optimality sweep per iteration (cheap) but needs more
    sweeps. Equivalent to policy iteration with a single evaluation sweep.
  - Both converge to the SAME optimal V* and π*.

---

MATH (the three update rules in one place):
  Eval:   V(s) ← Σ_a π(a|s) Σ_{s'} P[ R + γ V(s') ]
  Improve:π(s) ← argmax_a Σ_{s'} P[ R + γ V(s') ]
  VI:     V(s) ← max_a Σ_{s'} P[ R + γ V(s') ]

GOTCHAS:
- Terminal states must keep V = 0 (don't bootstrap past them). Our env's transitions()
  returns (1.0, s, 0.0) for terminals, so the math handles this automatically.
- "Synchronous" sweeps use V from the previous iteration; "in-place" uses the freshest
  values (often faster). We use a fresh copy per sweep here for clarity.
- The convergence threshold (theta) trades accuracy for speed.
- DP cost is O(|S|²|A|) per sweep — fine for grids, infeasible for huge state spaces
  (which is exactly why we move to sampling methods in files 04–05).

INTUITION:
DP "floods" value backward from the goal: the goal is worth +1, its neighbors a bit less
(due to γ and step cost), and so on. The optimal policy is just "always step uphill" on
this value landscape.
"""

import importlib.util
import os
import numpy as np

# --- import GridWorld from the digit-prefixed file 01 (normal import won't work) ---
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "mdp_module", os.path.join(_here, "01_markov_decision_processes.py")
)
_mdp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mdp)
GridWorld = _mdp.GridWorld


def policy_evaluation(env, policy, gamma=None, theta=1e-8, max_sweeps=1000):
    """Compute V_π for a (possibly stochastic) policy.

    policy: array [n_states, n_actions] giving π(a|s).
    Returns V: array [n_states].
    """
    gamma = env.gamma if gamma is None else gamma
    V = np.zeros(env.n_states)
    for _ in range(max_sweeps):
        delta = 0.0
        V_new = V.copy()
        for s in range(env.n_states):
            if env.is_terminal(s):
                continue
            v = 0.0
            for a in range(env.n_actions):
                pa = policy[s, a]
                if pa == 0.0:
                    continue
                # expected one-step backup for this action
                q = sum(p * (r + gamma * V[s_next])
                        for p, s_next, r in env.transitions(s, a))
                v += pa * q
            V_new[s] = v
            delta = max(delta, abs(v - V[s]))
        V = V_new
        if delta < theta:
            break
    return V


def greedy_policy_from_V(env, V, gamma=None):
    """Derive the deterministic greedy policy (array of best action per state) from V."""
    gamma = env.gamma if gamma is None else gamma
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        if env.is_terminal(s):
            continue
        q_values = [
            sum(p * (r + gamma * V[s_next]) for p, s_next, r in env.transitions(s, a))
            for a in range(env.n_actions)
        ]
        policy[s] = int(np.argmax(q_values))
    return policy


def policy_iteration(env, gamma=None, theta=1e-8):
    """Alternate full policy evaluation and greedy improvement until the policy is stable."""
    gamma = env.gamma if gamma is None else gamma
    # start from a uniform random stochastic policy
    policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
    iterations = 0
    while True:
        iterations += 1
        V = policy_evaluation(env, policy, gamma, theta)
        new_greedy = greedy_policy_from_V(env, V, gamma)

        # convert greedy (deterministic) into the same [S, A] one-hot format to compare
        new_policy = np.zeros_like(policy)
        for s in range(env.n_states):
            new_policy[s, new_greedy[s]] = 1.0

        if np.array_equal(new_policy, policy):
            return V, new_greedy, iterations
        policy = new_policy


def value_iteration(env, gamma=None, theta=1e-8, max_sweeps=1000):
    """Iterate the Bellman OPTIMALITY update until V converges; return V*, π*, sweeps."""
    gamma = env.gamma if gamma is None else gamma
    V = np.zeros(env.n_states)
    sweeps = 0
    for _ in range(max_sweeps):
        sweeps += 1
        delta = 0.0
        V_new = V.copy()
        for s in range(env.n_states):
            if env.is_terminal(s):
                continue
            q_values = [
                sum(p * (r + gamma * V[s_next]) for p, s_next, r in env.transitions(s, a))
                for a in range(env.n_actions)
            ]
            V_new[s] = max(q_values)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < theta:
            break
    policy = greedy_policy_from_V(env, V, gamma)
    return V, policy, sweeps


if __name__ == "__main__":
    print("=== Dynamic Programming on GridWorld ===\n")
    env = GridWorld(slip=0.0)

    # --- A) Policy Evaluation of a fixed (uniform random) policy ---
    rand_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
    V_rand = policy_evaluation(env, rand_policy)
    env.render_values(V_rand, title="A) V for the UNIFORM RANDOM policy (prediction):")

    # --- B) Policy Iteration ---
    V_pi, pi_pi, n_iter = policy_iteration(env)
    print(f"B) Policy Iteration converged in {n_iter} policy updates.")
    env.render_values(V_pi, title="   Optimal V* (policy iteration):")
    env.render_policy(pi_pi, title="   Optimal policy π* (policy iteration):")

    # --- C) Value Iteration ---
    V_vi, pi_vi, n_sweeps = value_iteration(env)
    print(f"C) Value Iteration converged in {n_sweeps} sweeps.")
    env.render_policy(pi_vi, title="   Optimal policy π* (value iteration):")

    agree = np.array_equal(pi_pi, pi_vi)
    print(f"Policy iteration and value iteration agree on π*: {agree}")
    print(f"Max |V_PI - V_VI| = {np.max(np.abs(V_pi - V_vi)):.2e}  (should be ~0)\n")

    # --- Bonus: how stochasticity (slippery floor) changes the optimal policy ---
    icy = GridWorld(slip=0.4)
    _, pi_icy, _ = value_iteration(icy)
    icy.render_policy(pi_icy, title="Bonus) Optimal policy on an ICY world (slip=0.4):")
    print("Notice how the icy-world policy steers AWAY from the pit to avoid slipping in.")
