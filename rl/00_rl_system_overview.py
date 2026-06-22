# Reinforcement Learning — System Overview (The Big Picture)
"""
WHAT/WHY:
Reinforcement Learning (RL) is learning *by interaction*. An agent takes actions in
an environment, receives rewards, and tries to learn a behavior (a "policy") that
maximizes the total reward it collects over time.

Unlike supervised learning (where every input has a correct label), RL has no answer
key. The agent only gets a scalar reward signal and must figure out which actions —
possibly many steps in the past — were responsible for good or bad outcomes.

Analogy: training a dog. You don't explain the rules. You let it act, give treats for
good behavior, and over time it learns a strategy that earns the most treats.

---

1. THE AGENT-ENVIRONMENT LOOP (the heartbeat of all RL)

        +-------------------+
        |       Agent       |
        +-------------------+
           |            ^
   action  | a_t        | state s_{t+1}, reward r_{t+1}
           v            |
        +-------------------+
        |    Environment    |
        +-------------------+

At each timestep t:
  1. Agent observes state   s_t
  2. Agent picks an action  a_t  (using its policy π)
  3. Environment returns    next state s_{t+1}  and  reward r_{t+1}
  4. Repeat until the episode ends (or forever, for continuing tasks)

This stream — s_0, a_0, r_1, s_1, a_1, r_2, ... — is called a TRAJECTORY (or rollout).

---

2. CORE VOCABULARY (memorize these — everything builds on them)

- State (s):        a snapshot of the situation the agent is in.
- Action (a):       a choice the agent can make.
- Reward (r):       a scalar number; immediate feedback for the last action.
- Policy (π):       the agent's behavior. π(a|s) = probability of taking action a in state s.
                    A *deterministic* policy is written a = π(s).
- Return (G):       total (discounted) reward from a timestep onward. THIS is what we maximize.
- Value (V, Q):     expected return. "How good" a state (V) or a state-action pair (Q) is,
                    assuming we follow policy π afterward.
- Model:            the agent's (optional) knowledge of how the environment works
                    (transition probabilities + rewards). Many RL methods are "model-free".
- Episode:          one full run from start to a terminal state (e.g., one game of chess).

---

3. RETURN AND DISCOUNTING (why "future reward" needs a knob)

The return is the cumulative future reward:

    G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...
        = Σ_{k=0}^∞  γ^k · r_{t+k+1}

γ (gamma) ∈ [0, 1] is the DISCOUNT FACTOR:
  - γ = 0   → myopic; only the next reward matters.
  - γ → 1   → far-sighted; the agent cares about long-term reward.
  - γ < 1   also keeps the infinite sum finite for continuing tasks.

INTUITION: a reward today is worth more than the same reward in 100 steps, both because
of uncertainty and because we want to encourage reaching goals sooner.

---

4. THE CENTRAL CHALLENGES (what makes RL hard)

- Exploration vs. Exploitation:
  Exploit what you know works, OR explore to discover something better?
  Too much exploitation → stuck in a rut. Too much exploration → never cash in.
  Analogy: always ordering your favorite dish vs. trying new menu items.

- Credit Assignment:
  A reward may arrive long after the action that caused it. Which past action gets credit?
  Analogy: you win a chess game — which of your 40 moves actually mattered?

- Delayed consequences:
  The best action now might give a low immediate reward but set up a big future one.

- Non-stationarity:
  As the policy improves, the data distribution it sees changes. The target moves.

---

5. A MAP OF RL METHODS (where each topic in this folder fits)

                          Reinforcement Learning
                                   |
            +----------------------+----------------------+
            |                                             |
       Model-Based                                   Model-Free
   (know/learn the env)                          (learn from samples)
            |                                             |
   Dynamic Programming                  +-----------------+-----------------+
   (file 03: needs full model)          |                                   |
                                   Value-Based                        Policy-Based
                          (learn V/Q, act greedily)            (learn π directly)
                                   |                                   |
                    +--------------+--------------+            REINFORCE, Actor-Critic,
                    |                             |            PPO   (later files)
            Monte Carlo (file 04)        Temporal Difference
            (learn from full          (file 05: SARSA, Q-learning,
             episode returns)          bootstrapping each step)

This folder walks from the foundations (MDPs, value functions) through the classical
tabular algorithms (DP, Monte Carlo, TD), which is the right "from scratch" order:
every deep RL method is a generalization of these ideas.

---

6. KEY DISTINCTIONS YOU'LL KEEP HEARING

- Prediction vs. Control:
  * Prediction (evaluation): given a fixed policy π, how good is it? (estimate V_π / Q_π)
  * Control: find the *best* policy π* that maximizes return.

- On-policy vs. Off-policy:
  * On-policy:  learn about the policy you're currently using to act (e.g., SARSA).
  * Off-policy: learn about a different ("target") policy than the one collecting data
                (e.g., Q-learning learns the greedy policy while exploring).

- Tabular vs. Function Approximation:
  * Tabular:    store one value per state (small, discrete problems). We start here.
  * Approx.:    use a function (linear model, neural net) when states are too many to
                enumerate. This is where Deep RL begins.

---

7. THE LEARNING PATH IN THIS FOLDER

  00  This overview                         (you are here)
  01  Markov Decision Processes + GridWorld (the formal problem + a real environment)
  02  Value functions & the Bellman equations (the math RL is built on)
  03  Dynamic Programming                   (solve an MDP when you KNOW the model)
  04  Monte Carlo methods                   (learn from complete episodes, no model)
  05  Temporal Difference learning          (SARSA & Q-learning — the RL workhorses)

Run each file directly (e.g. `python 03_dynamic_programming.py`) to see the algorithms
learn in real time on a small grid world. Everything is pure NumPy — no gym, no torch —
so you can see exactly how every number is produced.

---

GOTCHAS (common beginner misconceptions):
- Reward != Return. Reward is immediate; return is the long-term sum we optimize.
- The agent does NOT get the value function for free — it must estimate it.
- A "good" reward function is hard to design; agents exploit loopholes ("reward hacking").
- High reward on a single episode can be luck; RL cares about EXPECTED return.

INTUITION (one sentence):
RL = trial-and-error search for a policy that maximizes expected discounted reward,
under uncertainty about both the world and the consequences of your own actions.
"""

if __name__ == "__main__":
    # A tiny, self-contained illustration of the agent-environment loop with a
    # random policy. No real learning yet — just to make the loop concrete.
    import random

    # A trivial "corridor" environment: states 0..4, goal at state 4.
    # Actions: 0 = left, 1 = right. Reward +1 on reaching the goal, else 0.
    N_STATES = 5
    GOAL = N_STATES - 1

    def step(state, action):
        nxt = max(0, min(GOAL, state + (1 if action == 1 else -1)))
        reward = 1.0 if nxt == GOAL else 0.0
        done = nxt == GOAL
        return nxt, reward, done

    random.seed(0)
    state = 0
    trajectory = [state]
    total_reward = 0.0
    gamma = 0.9
    discounted_return = 0.0

    for t in range(50):
        action = random.choice([0, 1])          # random policy: explore blindly
        state, reward, done = step(state, action)
        discounted_return += (gamma ** t) * reward
        total_reward += reward
        trajectory.append(state)
        if done:
            break

    print("Trajectory (states visited):", trajectory)
    print(f"Steps taken: {t + 1}")
    print(f"Total (undiscounted) reward: {total_reward}")
    print(f"Discounted return G_0 (gamma={gamma}): {discounted_return:.4f}")
    print("\nNotice: a random policy eventually stumbles to the goal, but wastes steps.")
    print("The rest of this folder is about learning a policy that does this efficiently.")
