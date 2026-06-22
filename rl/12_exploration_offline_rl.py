# Exploration & Offline RL — Smarter Data Collection, and Learning Without Interaction
"""
WHAT/WHY:
Two practical pillars that the earlier files glossed over:

  PART A — EXPLORATION: ε-greedy (files 04–07) is the dumbest possible strategy: it explores
  uniformly at random, ignoring what it already knows. Smarter exploration learns faster with
  far less data. We compare ε-greedy vs UCB vs Thompson sampling on a bandit, then sketch how
  these ideas scale to deep RL (intrinsic motivation / curiosity).

  PART B — OFFLINE RL: all earlier methods needed to INTERACT with the environment. But often
  interaction is dangerous or expensive (healthcare, self-driving, recommender systems). Offline
  (a.k.a. "batch") RL learns the best possible policy from a FIXED, pre-collected dataset — no
  environment access at all. This is conceptually close to how LLMs are trained on fixed corpora,
  and DPO (file 17) is essentially offline preference-based RL.

================================ PART A: EXPLORATION ================================

1. THE EXPLORATION–EXPLOITATION DILEMMA (cleanest in a BANDIT)

A k-armed bandit: k slot machines, each with an unknown mean payout. Pull arms over T rounds
to maximize total reward. Pure exploitation locks onto the first decent arm; pure exploration
wastes pulls. REGRET = (reward of always-best arm) − (your reward). Good strategies have regret
that grows only LOGARITHMICALLY in T.

2. THREE STRATEGIES (all runnable below)

- ε-GREEDY: with prob ε pull a random arm, else the best estimate. Simple, but keeps exploring
  forever at a constant rate and explores arms uniformly (even obviously-bad ones).

- UCB (Upper Confidence Bound) — "optimism in the face of uncertainty":
      pull argmax_a [ Q(a) + c·sqrt( ln t / N(a) ) ]
  The bonus is large for rarely-tried arms (small N(a)) → it explores arms it's UNCERTAIN about,
  not random ones. As an arm is pulled more, its bonus shrinks. Provably logarithmic regret.

- THOMPSON SAMPLING (Bayesian / posterior sampling):
  Keep a posterior over each arm's mean; each round SAMPLE a value from each arm's posterior and
  pull the argmax. Naturally explores in proportion to the probability an arm is best. Often the
  best in practice, and elegant.

3. SCALING EXPLORATION TO DEEP RL (conceptual)

In huge state spaces you can't count visits directly. Modern approaches:
  - COUNT-BASED / pseudo-counts: bonus ~ 1/sqrt(N(s)) using density models or hashing.
  - CURIOSITY / INTRINSIC MOTIVATION (ICM, RND): reward the agent for SURPRISE — states where a
    learned predictor has high error (i.e., novel states). Drives exploration with no external
    reward (great for sparse-reward games like Montezuma's Revenge).
  - NOISY NETS / parameter-space noise; entropy bonuses (you used these in files 08–11).
  - Optimistic initialization (start Q high so untried actions look attractive).

============================== PART B: OFFLINE RL ==============================

4. THE OFFLINE SETTING

Given a fixed dataset D = {(s, a, r, s')} collected by some unknown BEHAVIOR policy, find the
best policy WITHOUT any new interaction. The naive idea — "just run Q-learning on D" — fails.

5. WHY NAIVE OFFLINE Q-LEARNING BREAKS: DISTRIBUTIONAL SHIFT

The Bellman target uses max_{a'} Q(s', a'). That max often picks an action NOT in the dataset
("out-of-distribution"). Q is wildly wrong there (never trained on it), the error gets
bootstrapped and AMPLIFIED, and Q EXPLODES toward fantasy actions it can never verify. With no
environment to correct the delusion, the policy is garbage.

6. THE FIX: STAY CLOSE TO THE DATA (the unifying theme of offline RL)

- BEHAVIOR CLONING (BC): pure supervised learning — imitate the dataset's actions. No
  bootstrapping, so it's safe, but it can only be as good as the data and can't "stitch" better
  trajectories together.
- BCQ / BEAR: only consider actions similar to those in the dataset.
- CQL (Conservative Q-Learning): add a penalty that PUSHES DOWN Q on out-of-distribution actions
  and up on in-data actions → a conservative, pessimistic Q that won't chase fantasies.
- IQL (Implicit Q-Learning): learn values using only in-data actions (expectile regression),
  avoiding the OOD max entirely. A strong, simple default.
- DECISION TRANSFORMER: reframe RL as sequence modeling — predict actions conditioned on desired
  return; pure supervised learning, no Bellman backups. (Directly connects RL to the LLM world.)

KEY PRINCIPLE: offline RL = "improve on the data, but stay within its support." That same
pessimism-toward-the-unseen principle reappears in DPO (file 17).

---

MATH SUMMARY:
  UCB:        a_t = argmax_a [ Q(a) + c·sqrt(ln t / N(a)) ]
  Thompson:   sample θ_a ~ posterior(a); a_t = argmax_a θ_a
  Offline shift: target max_{a'} Q(s',a') queries OOD actions → overestimation/blow-up
  CQL idea:   minimize  TD-error + β·( E_{a~OOD}[Q] − E_{a~data}[Q] )   (penalize OOD Q)

GOTCHAS:
- ε-greedy explores forever at rate ε and ignores uncertainty; decay it or use UCB/Thompson.
- UCB's constant c trades exploration vs exploitation; tune it.
- Offline RL: never assume you can "just try the action" — you can't. Distributional shift is THE
  problem; every offline method is some answer to it.
- BC can't exceed the data; value-based offline methods can "stitch" but risk OOD blow-up.

INTUITION:
Exploration = "be smart about what you don't know" (try uncertain things, not random things).
Offline RL = "squeeze the best policy out of a fixed log without fooling yourself about actions
you never actually tried."
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


_envs = _load("envs", "_environments.py")
_mdp = _load("mdp_module", "01_markov_decision_processes.py")
_dp = _load("dp_module", "03_dynamic_programming.py")
Bandit = _envs.Bandit
GridWorld = _mdp.GridWorld


# ============================ PART A: bandit strategies ============================
def run_bandit(strategy, k=10, steps=2000, seed=0, eps=0.1, c=2.0):
    """Run one bandit strategy; return (total_reward, optimal_action_rate)."""
    bandit = Bandit(k=k, seed=seed)
    rng = np.random.default_rng(seed + 1)
    Q = np.zeros(k)         # estimated mean per arm
    N = np.zeros(k)         # pull counts
    # Thompson (Gaussian posterior) running stats
    total_reward = 0.0
    optimal_pulls = 0

    for t in range(1, steps + 1):
        if strategy == "epsilon":
            a = rng.integers(k) if rng.random() < eps else int(np.argmax(Q))
        elif strategy == "ucb":
            # untried arms get infinite priority; else Q + exploration bonus
            if np.any(N == 0):
                a = int(np.argmin(N))
            else:
                a = int(np.argmax(Q + c * np.sqrt(np.log(t) / N)))
        elif strategy == "thompson":
            # sample each arm's mean from a Normal posterior (variance ~ 1/N)
            samples = Q + rng.normal(0, 1, size=k) / np.sqrt(N + 1)
            a = int(np.argmax(samples))
        else:
            raise ValueError(strategy)

        r = bandit.pull(a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]          # incremental mean
        total_reward += r
        optimal_pulls += int(a == bandit.optimal_action)

    return total_reward, optimal_pulls / steps


# ============================ PART B: offline RL demo ============================
def collect_dataset(env, behavior_policy, n_episodes=400, seed=0):
    """Log transitions (s,a,r,s',done) by running a fixed behavior policy. No labels."""
    env.rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(100):
            a = behavior_policy(s)
            s2, r, done = env.step(a)
            data.append((s, a, r, s2, float(done)))
            s = s2
            if done:
                break
    return data


def offline_q_learning(env, data, gamma=None, alpha=0.1, sweeps=200):
    """Tabular Q-learning purely from a fixed dataset (no environment interaction).

    Works well here because the logged data covers the state space. The danger it can hit
    — distributional shift / OOD overestimation — appears when coverage is NARROW; CQL/IQL
    (described above) add pessimism to handle that case.
    """
    gamma = env.gamma if gamma is None else gamma
    Q = np.zeros((env.n_states, env.n_actions))
    for _ in range(sweeps):
        for (s, a, r, s2, done) in data:
            target = r if done else r + gamma * np.max(Q[s2])
            Q[s, a] += alpha * (target - Q[s, a])
    return np.argmax(Q, axis=1), Q


def behavior_cloning(env, data):
    """Pure supervised imitation: in each state, copy the action the log took most often.

    No rewards, no bootstrapping — just mimic the data. Safe but capped by the data's quality
    (it can't 'stitch' together a better path than the behavior policy demonstrated).
    """
    counts = np.zeros((env.n_states, env.n_actions))
    for (s, a, r, s2, done) in data:
        counts[s, a] += 1
    policy = np.argmax(counts, axis=1)         # most-frequent action per state
    return policy


def eval_policy(env, policy, n=500, seed=999):
    rng = np.random.default_rng(seed)
    e = GridWorld(slip=env.slip, gamma=env.gamma, seed=0)
    e.rng = rng
    goal = e._to_s(e.goal)
    wins = 0
    for _ in range(n):
        s = e.reset()
        for _ in range(100):
            s, r, done = e.step(int(policy[s]))
            if done:
                wins += int(s == goal)
                break
    return wins / n


if __name__ == "__main__":
    print("=== PART A: Exploration strategies on a 10-armed bandit ===\n")
    print(f"{'strategy':<12}{'avg reward/pull':>18}{'optimal-arm rate':>20}")
    for strat in ["epsilon", "ucb", "thompson"]:
        # average over several bandit instances for a fair comparison
        rewards, opt = [], []
        for seed in range(20):
            tr, orate = run_bandit(strat, steps=1000, seed=seed)
            rewards.append(tr / 1000); opt.append(orate)
        print(f"{strat:<12}{np.mean(rewards):>18.3f}{np.mean(opt):>19.1%}")
    print("\nUCB and Thompson beat ε-greedy by exploring UNCERTAIN arms, not random ones.\n")

    print("=== PART B: Offline RL on GridWorld (learn from a fixed log) ===\n")
    env = GridWorld(slip=0.1)
    _, pi_star, _ = _dp.value_iteration(env)        # the best achievable (uses the model)
    rng = np.random.default_rng(0)

    # behavior policy: a NOISY EXPERT — follows the optimal action 60% of the time, else
    # random. This gives a realistic log: decent but imperfect, with broad coverage.
    def behavior(s):
        if rng.random() < 0.6:
            return int(pi_star[s])
        return int(rng.integers(env.n_actions))

    dataset = collect_dataset(env, behavior, n_episodes=500)
    print(f"Collected {len(dataset)} transitions from a noisy-expert behavior policy.\n")

    pi_bc = behavior_cloning(env, dataset)          # imitate the log (supervised)
    pi_offline, _ = offline_q_learning(env, dataset)  # optimize from the log (value-based)

    print("Goal-reaching success rate (greedy evaluation):")
    print(f"  Behavior Cloning (imitate log): {eval_policy(env, pi_bc):.1%}")
    print(f"  Offline Q-learning (optimize) : {eval_policy(env, pi_offline):.1%}")
    print(f"  Optimal (value iteration)     : {eval_policy(env, pi_star):.1%}")
    print("\nBoth recover a strong policy from logged data alone — no new interaction.")
    print("Offline Q-learning can 'stitch' better-than-data paths; BC is capped by the data.")
    print("Next (file 13): planning with a MODEL — Dyna and Monte Carlo Tree Search.")
