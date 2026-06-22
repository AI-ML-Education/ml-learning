# Markov Decision Processes (MDPs) — The Formal RL Problem
"""
WHAT/WHY:
Almost every RL problem is formalized as a Markov Decision Process (MDP). An MDP is the
mathematical "container" that defines the world the agent lives in. Once a problem is
written as an MDP, we can apply the whole toolbox of RL algorithms to it.

If file 00 was the intuition, this file is the precise definition — plus a real, runnable
GridWorld environment we will reuse in files 03–05.

---

1. THE MARKOV PROPERTY (why "Markov")

"The future depends only on the present, not the past."

Formally:
    P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0) = P(s_{t+1} | s_t, a_t)

Meaning: the current state captures everything relevant about the history. You don't need
to remember how you got here, only where you are.

Analogy: a chess position is Markov — the board tells you everything; you don't need the
move history to decide the best next move. A single video frame is NOT Markov (you can't
tell which way a ball is moving from one frame).

GOTCHA: if your state lacks information (e.g., velocity), the problem becomes a POMDP
(Partially Observable MDP) and is much harder. Designing the state representation is half
the battle.

---

2. THE 5 INGREDIENTS OF AN MDP:  (S, A, P, R, γ)

- S : set of states.
- A : set of actions (sometimes A(s), the actions available in state s).
- P : transition function. P(s' | s, a) = probability of landing in s' after doing a in s.
- R : reward function. R(s, a, s') = expected reward for that transition.
       (Common simpler forms: R(s, a) or R(s).)
- γ : discount factor in [0, 1] (see file 00).

Together these define the *dynamics*: how the world responds to actions.

---

3. POLICY, RETURN, AND THE GOAL

- Policy π(a | s): the agent's behavior (a distribution over actions per state).
- Return  G_t = Σ_{k=0}^∞ γ^k · r_{t+k+1}  (the discounted future reward).
- GOAL: find a policy π* that maximizes the EXPECTED return from every state:

        π*  =  argmax_π  E_π [ G_t ]

The expectation E_π is over the randomness in both the policy and the environment.

---

4. DETERMINISTIC vs. STOCHASTIC ENVIRONMENTS

- Deterministic: action a in state s always leads to the same s'.  (P is 0/1.)
- Stochastic:    a in s leads to different s' with some probabilities.
  Example: "move right" succeeds 80% of the time, but slips up/down 10% each (icy floor).

Real environments are usually stochastic. Our GridWorld below supports a "slip"
probability so you can experiment with both.

---

5. EPISODIC vs. CONTINUING TASKS

- Episodic:   has terminal states; runs in episodes (a maze, a game). G_t is a finite sum.
- Continuing: no natural end (a thermostat controller); γ < 1 keeps the return finite.

Terminal states are "absorbing": once entered, the episode ends (value 0 afterward).

---

6. THE GRIDWORLD WE'LL USE EVERYWHERE

A 4x4 grid. The agent starts top-left, wants to reach a goal. One cell is a trap (pit).

  Layout (row, col), S = start, G = goal (+1), X = pit (-1), . = normal cell:

        col:  0    1    2    3
   row 0:    S    .    .    .
   row 1:    .    .    X    .
   row 2:    .    .    .    .
   row 3:    .    .    .    G

  - Actions: 0=up, 1=right, 2=down, 3=left.
  - Reward: -0.04 per step (a small "cost of living" that encourages short paths),
            +1.0 for reaching the goal, -1.0 for falling in the pit.
  - Goal and pit are terminal.
  - Optional slip probability: with prob `slip`, the agent moves in a random
    perpendicular direction instead of the intended one (a stochastic, "icy" world).
  - Bumping into a wall keeps you in place (but you still pay the step cost).

This is a classic and is the perfect sandbox: small enough to print every value, rich
enough to show off exploration, discounting, and stochasticity.

---

MATH RECAP (one box to remember):
    MDP            = (S, A, P, R, γ)
    Policy         π(a|s)
    Return         G_t = Σ γ^k r_{t+k+1}
    Objective      max_π  E_π[G_t]

GOTCHAS:
- Off-by-one on rewards: r_{t+1} is the reward for action a_t (the NEXT reward).
- Forgetting terminal states have value 0 (no future reward).
- Treating a non-Markov observation as if it were Markov.
- Reward shaping that accidentally rewards loops instead of reaching the goal.

INTUITION:
An MDP turns a fuzzy "learn to behave well" goal into a crisp optimization problem:
choose the policy that earns the most expected discounted reward in this defined world.
"""

import numpy as np


class GridWorld:
    """A small, fully-defined MDP used throughout the RL files.

    States are integers 0..(rows*cols - 1), numbered row-major.
    This class exposes BOTH:
      - a model (transition probabilities + rewards) for Dynamic Programming (file 03), and
      - a step()/reset() interface for sample-based learning (files 04, 05).
    """

    # action index -> (d_row, d_col)
    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}

    def __init__(self, rows=4, cols=4, start=(0, 0), goal=(3, 3), pit=(1, 2),
                 step_cost=-0.04, goal_reward=1.0, pit_reward=-1.0,
                 slip=0.0, gamma=0.9, seed=0):
        self.rows, self.cols = rows, cols
        self.start = start
        self.goal = goal
        self.pit = pit
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.pit_reward = pit_reward
        self.slip = slip            # prob of slipping perpendicular to intended move
        self.gamma = gamma
        self.n_states = rows * cols
        self.n_actions = len(self.ACTIONS)
        self.rng = np.random.default_rng(seed)

        self.terminals = {self._to_s(goal), self._to_s(pit)}
        self.agent_state = self._to_s(start)

    # ---- coordinate helpers ----
    def _to_s(self, rc):
        r, c = rc
        return r * self.cols + c

    def _to_rc(self, s):
        return divmod(s, self.cols)

    def is_terminal(self, s):
        return s in self.terminals

    # ---- the deterministic "intended" move (ignores slip) ----
    def _move(self, s, a):
        if self.is_terminal(s):
            return s  # absorbing: no movement from terminal states
        r, c = self._to_rc(s)
        dr, dc = self.ACTIONS[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return nr * self.cols + nc
        return s  # bumped into a wall -> stay put

    def _reward(self, s_next):
        if s_next == self._to_s(self.goal):
            return self.goal_reward
        if s_next == self._to_s(self.pit):
            return self.pit_reward
        return self.step_cost

    # ---- MODEL: P(s', r | s, a) as a list of (prob, s_next, reward) ----
    def transitions(self, s, a):
        """Return list of (probability, next_state, reward) for taking a in s.

        With slip>0, the intended action happens with prob (1-slip); the two
        perpendicular actions each happen with prob slip/2 (a stochastic world).
        """
        if self.is_terminal(s):
            return [(1.0, s, 0.0)]  # terminal: stay, no reward

        # which actions can actually occur, and with what probability
        if self.slip <= 0.0:
            outcomes = [(1.0, a)]
        else:
            perp = {0: (3, 1), 1: (0, 2), 2: (1, 3), 3: (2, 0)}  # perpendicular pairs
            left_a, right_a = perp[a]
            outcomes = [
                (1.0 - self.slip, a),
                (self.slip / 2.0, left_a),
                (self.slip / 2.0, right_a),
            ]

        # collapse identical next-states (e.g., two slips both hit the same wall)
        merged = {}
        for prob, act in outcomes:
            s_next = self._move(s, act)
            reward = self._reward(s_next)
            key = (s_next, reward)
            merged[key] = merged.get(key, 0.0) + prob
        return [(p, s_next, r) for (s_next, r), p in merged.items()]

    # ---- SAMPLE INTERFACE: reset() / step() for model-free learning ----
    def reset(self):
        self.agent_state = self._to_s(self.start)
        return self.agent_state

    def step(self, a):
        """Sample one transition from the current state. Returns (s_next, reward, done)."""
        transitions = self.transitions(self.agent_state, a)
        probs = [p for p, _, _ in transitions]
        idx = self.rng.choice(len(transitions), p=probs)
        _, s_next, reward = transitions[idx]
        self.agent_state = s_next
        done = self.is_terminal(s_next)
        return s_next, reward, done

    # ---- pretty-printing helpers (used by later files) ----
    def render_values(self, V, title="State values V(s)"):
        print(title)
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                s = r * self.cols + c
                row_cells.append(f"{V[s]:+6.2f}")
            print("  " + " ".join(row_cells))
        print()

    def render_policy(self, policy, title="Greedy policy"):
        """policy: array of shape [n_states] giving the chosen action per state."""
        arrows = {0: "  ↑  ", 1: "  →  ", 2: "  ↓  ", 3: "  ←  "}
        print(title)
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                s = r * self.cols + c
                if s == self._to_s(self.goal):
                    row_cells.append("  G  ")
                elif s == self._to_s(self.pit):
                    row_cells.append("  X  ")
                else:
                    row_cells.append(arrows[int(policy[s])])
            print("  " + " ".join(row_cells))
        print()


if __name__ == "__main__":
    print("=== GridWorld MDP demo ===\n")
    env = GridWorld(slip=0.0)

    print(f"States: {env.n_states}, Actions: {env.n_actions}, gamma: {env.gamma}")
    print(f"Start: {env.start}  Goal(+1): {env.goal}  Pit(-1): {env.pit}\n")

    # Show the model for a single (state, action): deterministic vs stochastic.
    s = env._to_s((2, 3))  # one cell above the goal
    print(f"Deterministic world — transitions from state {s} (row2,col3) taking 'down':")
    for p, s_next, r in env.transitions(s, 2):
        print(f"   prob={p:.2f} -> state {s_next} (row{s_next // env.cols},col{s_next % env.cols}), reward={r:+.2f}")

    env_icy = GridWorld(slip=0.2)
    print(f"\nStochastic world (slip=0.2) — transitions from state {s} taking 'down':")
    for p, s_next, r in env_icy.transitions(s, 2):
        print(f"   prob={p:.2f} -> state {s_next} (row{s_next // env.cols},col{s_next % env.cols}), reward={r:+.2f}")

    # Run one episode under a random policy using the sample interface.
    print("\nOne episode under a RANDOM policy (sample interface):")
    s = env.reset()
    total = 0.0
    for t in range(50):
        a = env.rng.integers(env.n_actions)
        s_next, r, done = env.step(a)
        total += r
        if done:
            outcome = "GOAL" if s_next == env._to_s(env.goal) else "PIT"
            print(f"   reached {outcome} after {t + 1} steps, total reward = {total:+.2f}")
            break
    else:
        print(f"   did not terminate in 50 steps, total reward = {total:+.2f}")

    print("\nThis GridWorld is imported by files 03–05 to learn optimal behavior.")
