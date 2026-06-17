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
  `05_intervals.py`, `06_stacks.py`, `07_greedy_binarysearch.py`, `08_heaps.py`,
  `09_graphs.py`, `10_dp.py`
  - `linear_data_structure/` — binary search, sliding window, two pointers.
  - `non_linear_data_structure/` — backtracking, BFS, DFS, dynamic programming,
    priority queue.
- `data_structures/` — binary search tree, binary tree, and general data structures.

### `ml/`
Machine learning foundations and LLM internals.

- `llm_fundamentals/` — a guided walk through how LLMs work, from tokenization and
  attention to KV cache, sampling, LoRA/QLoRA, training mechanics, and evaluation.
- `data_science/` — math and ML foundations (e.g. `linear_regression_math.py`).

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

- `system_design_fde_practice.py`

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
