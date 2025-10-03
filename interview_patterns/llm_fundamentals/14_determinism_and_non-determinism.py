# Determinism & Non-Determinism in LLMs
#TODO: Add context from recent Thinking Machine Lab paper
"""
WHAT/WHY:
When generating text, we often expect reproducibility (same input, same seed →
same output). However, LLMs are not always deterministic, even at temperature=0.
Understanding sources of non-determinism is crucial for debugging, reproducibility,
and benchmarking.

---

SOURCES OF NON-DETERMINISM:

1) Floating-Point Arithmetic (Non-Associativity):
- Floating-point addition is not strictly associative:
  (a + b) + c ≠ a + (b + c) due to rounding errors.
- On GPU, reduction operations (sum, mean) may be parallelized in different orders.
- Different kernel launches or threading strategies → tiny numeric differences.
- These small differences can amplify over many layers.

2) GPU Kernels & Atomic Operations:
- Many CUDA/cuDNN kernels use atomic ops or parallel reductions.
- Execution order of threads is not guaranteed → small numerical variation.
- Non-deterministic kernel algorithms may be chosen for speed.

3) Randomness in Sampling:
- Stochastic decoding methods (temperature, top-k, top-p) rely on sampling.
- Even with fixed seeds, if PRNG state is not properly managed, results differ.
- Batch decoding can change sampling order → different results.

4) Batch Invariance:
- Running a single input vs the same input in a batch can produce slightly different outputs.
- Why? Padding, mask broadcasting, or kernel-level optimization differences.
- Can affect determinism in subtle ways.

5) Different Hardware / Drivers:
- Same model on A100 vs H100 vs CPU may yield slightly different outputs.
- Driver/library versions (CUDA, cuDNN, PyTorch) affect kernel selection.

---

WHY ZERO-TEMPERATURE ≠ FULLY DETERMINISTIC:
- At temperature=0, decoding becomes greedy (argmax).
- But due to tiny numeric differences in logits (e.g., 1.000000 vs 0.9999999),
  different tokens can win the argmax.
- Over long sequences, this diverges into completely different outputs.

---

PYTORCH NOTES (Mitigations):
- Set seeds:
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  np.random.seed(42)
  random.seed(42)

- Control cudnn:
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

- Use torch.use_deterministic_algorithms(True) for stricter guarantees (slower).
- Still, some ops may remain nondeterministic on GPU.

---

GOTCHAS:
- Even with fixed seeds, results can differ across runs/hardware.
- Benchmarking must account for statistical variance.
- Batch order can change outputs (non-batch-invariant decoding).
- Mixed precision (fp16/bf16) increases rounding errors, amplifying drift.

---

INTUITION:
- GPUs trade determinism for speed via parallelization.
- Tiny differences in floating-point ops get magnified layer by layer.
- Decoding decisions (argmax, sampling) can flip due to 1e-6 changes.

---

REAL-WORLD NOTES:
- Research papers (e.g., “LLM Non-Determinism” by Thinking Machines Lab)
  highlight these issues for reproducibility.
- vLLM and TensorRT-LLM optimize for throughput; determinism is secondary.
- Production systems often accept non-determinism, but for evaluation,
  multiple runs and statistical reporting are best practice.
"""
