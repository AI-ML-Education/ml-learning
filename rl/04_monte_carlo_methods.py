# Monte Carlo Methods — Learning From Complete Episodes (No Model Needed)
"""
WHAT/WHY:
Dynamic Programming (file 03) needed the full model P(s'|s,a). In the real world we usually
DON'T have it. Monte Carlo (MC) methods drop that requirement: they learn purely from
EXPERIENCE — by running episodes and averaging the actual returns observed.

Core idea (almost embarrassingly simple):
    The value of a state = the average return that actually followed it.
Run many episodes, record the return after each state visit, average them. By the law of
large numbers, the average converges to the true expected return V_π(s).

"Monte Carlo" = estimate an expectation by averaging random samples (like estimating π by
throwing darts). Here the samples are full episode returns.

---

1. MC PREDICTION (evaluate a fixed policy)

For each episode:
  1. Generate a full trajectory: s_0, a_0, r_1, s_1, ..., s_T (terminal).
  2. Walk BACKWARD computing the return G at each step:  G ← r_{t+1} + γ·G.
  3. For each state visited, append G to that state's list of returns.
V(s) = average of all returns recorded for s.

First-visit MC:  only count the return from the FIRST time a state is visited per episode.
Every-visit MC:  count it every time the state is visited. (Both converge; first-visit is
                 the classic textbook version.)

KEY PROPERTY: MC does NOT bootstrap. It uses the *real* return all the way to episode end,
not an estimate of future value. (Contrast with TD in file 05, which bootstraps.)

---

2. WHY BACKWARD? (efficient return computation)

The return satisfies G_t = r_{t+1} + γ·G_{t+1}. Computing it back-to-front means each step
is O(1) and you build returns for the whole episode in a single reverse pass.

---

3. MC CONTROL (find a good policy, not just evaluate one)

To improve the policy we need ACTION values Q(s,a) (you can't act greedily on V without a
model). So:
  - Estimate Q(s,a) by averaging returns following each (s,a) pair.
  - Improve the policy greedily: π(s) = argmax_a Q(s,a).

THE EXPLORATION PROBLEM: if the policy is greedy, many (s,a) pairs are never tried, so
their Q stays unknown — you can't improve toward actions you never sample.

Fix used here: ε-greedy exploration.
    With prob (1-ε): take the greedy action (exploit).
    With prob ε:     take a random action     (explore).
This guarantees every action keeps getting sampled. Decaying ε over time lets the agent
explore early and exploit later. This algorithm is "on-policy first-visit MC control".

---

4. INCREMENTAL MEAN (so we don't store every return)

Instead of keeping all returns, update the average online:
    N(s,a) ← N(s,a) + 1
    Q(s,a) ← Q(s,a) + (1 / N(s,a)) · [ G - Q(s,a) ]
This is the same "estimate ← estimate + step·(target - estimate)" pattern that powers TD.

---

MONTE CARLO vs DYNAMIC PROGRAMMING:
  - DP needs the model; MC needs only sampled episodes.
  - DP bootstraps (uses V(s') estimates); MC uses real full returns (no bootstrap).
  - MC has zero bias but HIGH variance (a full return depends on many random steps).
  - MC only works for EPISODIC tasks (you must reach a terminal state to get a return).
  - MC updates only at episode END; TD (file 05) updates every step.

MATH:
  Return:    G_t = r_{t+1} + γ G_{t+1}
  Estimate:  V(s) = average of returns observed from s
  Control:   Q(s,a) ← Q(s,a) + (1/N)·[G - Q(s,a)],  π = ε-greedy(Q)

GOTCHAS:
- Must reach a terminal state, or the return is undefined (cap episode length as a guard).
- First-visit vs every-visit: don't double-count if you intend first-visit.
- Greedy-only policies starve exploration → use ε-greedy (and/or exploring starts).
- High variance means MC can need MANY episodes to converge.

INTUITION:
"Don't reason about the dynamics — just try it many times and average what happened."
MC is the most direct embodiment of learning from experience.
"""

import importlib.util
import os
import numpy as np

# import GridWorld (file 01) and the DP ground-truth solver (file 03) for comparison
_here = os.path.dirname(os.path.abspath(__file__))


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_here, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mdp = _load("mdp_module", "01_markov_decision_processes.py")
_dp = _load("dp_module", "03_dynamic_programming.py")
GridWorld = _mdp.GridWorld


def generate_episode(env, policy_fn, max_steps=200):
    """Roll out one episode. policy_fn(state) -> action. Returns list of (s, a, r)."""
    episode = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy_fn(s)
        s_next, r, done = env.step(a)
        episode.append((s, a, r))
        s = s_next
        if done:
            break
    return episode


def mc_prediction(env, policy, n_episodes=20000, gamma=None, first_visit=True):
    """First-visit MC prediction: estimate V_π for a fixed stochastic policy.

    policy: array [n_states, n_actions].
    """
    gamma = env.gamma if gamma is None else gamma
    V = np.zeros(env.n_states)
    counts = np.zeros(env.n_states)
    rng = np.random.default_rng(0)

    def policy_fn(s):
        return rng.choice(env.n_actions, p=policy[s])

    for _ in range(n_episodes):
        episode = generate_episode(env, policy_fn)
        G = 0.0
        visited = set()
        # walk backward to accumulate returns
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = r + gamma * G
            if first_visit and s in visited:
                continue
            visited.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]   # incremental mean
    return V


def mc_control(env, n_episodes=50000, gamma=None,
               eps_start=1.0, eps_end=0.05, decay=True):
    """On-policy first-visit MC control with ε-greedy exploration. Returns Q, greedy π."""
    gamma = env.gamma if gamma is None else gamma
    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros((env.n_states, env.n_actions))
    rng = np.random.default_rng(0)

    for ep in range(n_episodes):
        # linearly decay epsilon from eps_start to eps_end
        if decay:
            eps = eps_start + (eps_end - eps_start) * (ep / n_episodes)
        else:
            eps = eps_start

        def policy_fn(s, _eps=eps):
            if rng.random() < _eps:
                return rng.integers(env.n_actions)        # explore
            return int(np.argmax(Q[s]))                   # exploit

        episode = generate_episode(env, policy_fn)
        G = 0.0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = r + gamma * G
            if (s, a) in visited:
                continue
            visited.add((s, a))
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]            # incremental mean

    greedy = np.argmax(Q, axis=1)
    return Q, greedy


if __name__ == "__main__":
    print("=== Monte Carlo methods on GridWorld ===\n")
    env = GridWorld(slip=0.0)

    # --- Prediction: evaluate the uniform random policy, compare to exact DP ---
    rand_policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
    V_dp = _dp.policy_evaluation(env, rand_policy)        # ground truth (model-based)
    V_mc = mc_prediction(env, rand_policy, n_episodes=20000)

    env.render_values(V_dp, title="Ground-truth V (DP, exact) for random policy:")
    env.render_values(V_mc, title="MC estimate of V (20k episodes, no model):")
    print(f"Max |V_DP - V_MC| = {np.max(np.abs(V_dp - V_mc)):.3f}  "
          f"(shrinks as episodes ↑; MC has high variance)\n")

    # --- Control: learn an optimal policy from experience alone ---
    Q, pi_mc = mc_control(env, n_episodes=50000)
    env.render_policy(pi_mc, title="MC control learned policy (ε-greedy, 50k episodes):")

    # Compare against the exact optimal policy from value iteration.
    _, pi_star, _ = _dp.value_iteration(env)
    # compare only on non-terminal states (terminals have no action)
    non_terminal = [s for s in range(env.n_states) if not env.is_terminal(s)]
    matches = sum(pi_mc[s] == pi_star[s] for s in non_terminal)
    print(f"MC policy matches the optimal DP policy on "
          f"{matches}/{len(non_terminal)} non-terminal states.")
    print("Mismatches (if any) are usually states far from the goal that are rarely visited.")
