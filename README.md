# ml-learning

My personal practice and learning journal. I use this repo to collect the material
I work through while prepping for interviews and while learning new topics — coding
patterns, ML/LLM fundamentals, data science, system design, and agentic AI.

Whenever I learn something new from practice or study, I add it here so I can revisit
it later, build on it, and keep growing the collection over time.

## How I use this

- **Practice as I prep** — drop in new exercises, patterns, and notes as I encounter them.
- **Revisit and build on** — come back to earlier material before the next round of prep.
- **Keep it cumulative** — every lesson learned stays here so future-me benefits.

## Contents

### `coding_practice/`
One home for all coding/DSA practice.

- `patterns/` — core interview patterns, numbered for ordered practice:
  `01_hashmap.py`, `02_sliding_window.py`, `03_prefix_sums.py`, `04_two_pointers.py`,
  `05_intervals.py`, `06_stacks.py`, `07_greedy_binary_search.py`, `08_heaps.py`,
  `09_graphs.py`, `10_dp.py`
  - `linear_data_structures/` — binary search, sliding window, two pointers.
  - `non_linear_data_structures/` — backtracking, BFS, DFS, dynamic programming,
    priority queue.
- `data_structures/` — binary search tree, binary tree, and an `overview.py` catalog
  of Python data structures.

### `ml/`
Machine learning foundations and LLM internals.

- `llm_fundamentals/` — a guided walk through how LLMs work, from tokenization and
  attention to KV cache, sampling, LoRA/QLoRA, training mechanics, and evaluation.
  These are the **narrated theory** (runnable `.py` lessons); the **hands-on build**
  of the same ideas lives in `llms/`.
- `data_science/` — math and ML foundations (e.g. `linear_regression_math.py`).

### `llms/`
Hands-on "build a GPT from scratch" in notebooks — I tear the architecture down and
re-implement each piece (tokenization, attention, the transformer block, the full
`GPTModel`, decoding, and classification fine-tuning). Foundation notebooks cover the
supporting math (layers, backprop, cross-entropy). Shared code lives in `llms/utils/`.
See [`llms/README.md`](llms/README.md) for the suggested reading order.

### `pytorch/`
PyTorch basics and warm-ups.

- `intro_to_pytorch.ipynb` — tensors, autograd, and the core PyTorch workflow.

### `rl/`
Reinforcement learning from scratch — the full ladder from tabular methods to the
frontier techniques that align large language models. Files 00–05 (and 12–13) are
pure NumPy (no gym, no torch); the deep-RL files (06–18) use PyTorch. Each file is a
self-contained lesson with explanation plus runnable code; run any directly to watch
the algorithm learn. Shared helpers live in `_environments.py` (CartPole, PointMass,
bandit, Tic-Tac-Toe) and `_toy_llm.py` (a tiny autoregressive LM for the RLHF files).

Classical / tabular (pure NumPy, on a small GridWorld):
- `00_rl_system_overview.py` — the agent–environment loop, return/discounting, and a
  map of RL methods.
- `01_markov_decision_processes.py` — the MDP formalism and a reusable `GridWorld`
  environment (deterministic or "icy"/stochastic).
- `02_value_functions_and_bellman.py` — V/Q value functions and the Bellman
  expectation & optimality equations.
- `03_dynamic_programming.py` — policy evaluation, policy iteration, and value
  iteration (the model-based ground truth).
- `04_monte_carlo_methods.py` — learning from complete episodes (MC prediction and
  ε-greedy MC control), no model required.
- `05_temporal_difference_learning.py` — TD(0), SARSA (on-policy), and Q-learning
  (off-policy) — the workhorses of model-free RL.

Deep RL (PyTorch):
- `06_function_approximation.py` — from tables to features and neural nets (linear
  semi-gradient TD and an MLP value approximator).
- `07_deep_q_networks.py` — DQN with replay buffer, target network, and Double DQN
  on CartPole.
- `08_policy_gradients.py` — REINFORCE with reward-to-go and a value baseline.
- `09_actor_critic.py` — A2C with Generalized Advantage Estimation (GAE).
- `10_trpo_ppo.py` — the clipped-surrogate PPO objective (and the TRPO idea behind it).
- `11_continuous_control.py` — DDPG on a continuous task, with TD3/SAC explained.
- `12_exploration_offline_rl.py` — bandit exploration (ε-greedy, UCB, Thompson) and
  offline RL (behavior cloning vs. offline Q-learning).
- `13_planning_dyna_mcts.py` — model-based planning: Dyna-Q and Monte Carlo Tree
  Search (UCT) on Tic-Tac-Toe.

Frontier: self-play and LLM alignment (PyTorch):
- `14_alphazero_muzero.py` — AlphaZero self-play (PUCT MCTS + policy/value net) on
  Tic-Tac-Toe, with MuZero explained.
- `15_rlhf_pipeline.py` — the RLHF recipe end-to-end: SFT → reward model → RL
  fine-tuning, with the KL-vs-reward trade-off swept.
- `16_ppo_for_llms.py` — text generation as a token-level MDP; PPO with a value head,
  per-token KL penalty, and GAE.
- `17_dpo_ipo_kto.py` — DPO (and IPO) preference optimization with no reward model
  and no RL loop; KTO explained.
- `18_grpo_rlaif_reasoning.py` — GRPO (critic-free, behind DeepSeek-R1), RLAIF, and
  reasoning RL — plus why AlphaFold is *not* reinforcement learning.

### `python_fundamentals/`
Python refreshers and hands-on practice.

- `dictionaries_syntax.py` — core dict syntax (get/pop, loops, comprehensions).
- `native_ds_practice.py` — practice problems on Python's native data structures.
- `numpy_practice.py`, `pandas_practice.py`

### `agentic_ai/`
Experiments with agent frameworks.

- `ms_agent_framework/` — hello-world and host agent examples.

### `system_design/`
System design practice.

- `fde_practice.py`

## Getting started

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Then run any file directly, for example:

```bash
python coding_practice/patterns/01_hashmap.py
```

## Notes

This is a private repo for my own learning. It grows continuously — structure and
contents will keep changing as I add new material.
