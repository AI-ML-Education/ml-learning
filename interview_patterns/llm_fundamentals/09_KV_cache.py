# KV Cache (Key-Value Cache for Autoregressive Inference)
"""
WHAT/WHY:
In training, we process entire sequences at once (teacher forcing).
But in **inference (generation)**, we produce tokens one by one.

Naïve approach:
- At each step t, recompute Q, K, V for all tokens 1…t
- Complexity: O(t²) for sequence length t
- Extremely wasteful, especially for long outputs (e.g., 1k+ tokens)

**KV Cache** solves this by storing past Keys (K) and Values (V),
so we only need to compute new entries at each decoding step.

This reduces per-step cost from O(t²) → O(t).

---

HOW IT WORKS:
1. At step 1:
   - Compute Q₁, K₁, V₁ for the first token.
   - Store K₁, V₁ in cache.

2. At step 2:
   - Compute Q₂, K₂, V₂ for the new token.
   - Append K₂, V₂ to cache: cache = [K₁,K₂], [V₁,V₂].
   - Perform attention for Q₂ over full cached K,V.

3. At step t:
   - Compute Q_t, K_t, V_t for new token only.
   - Concatenate into cache.
   - Perform attention with (Q_t, K_1…K_t, V_1…V_t).

---

MATH:
At time t, attention output for the new token:

out_t = softmax( (Q_t K_{≤t}ᵀ) / √D )  V_{≤t}

Where:
- Q_t ∈ ℝ[B, H, 1, D]  (only current query)
- K_{≤t} ∈ ℝ[B, H, t, D]
- V_{≤t} ∈ ℝ[B, H, t, D]

Instead of recomputing all K_{≤t}, V_{≤t} every time,
we just append the new K_t, V_t.

---

SHAPES:
- Cache for each layer stores:
  K_cache: [B, H, T, D]
  V_cache: [B, H, T, D]

- At each step, extend along time dimension (T).

- During batch decoding (beam search, etc.), cache may be reused across beams.

---

PYTORCH NOTES:
- Implement cache as dict per layer:
  cache[layer] = { "k": K_cache, "v": V_cache }

- To update:
def update_kv_cache(cache, new_k, new_v, layer):
    if cache is None:
        return {"k": new_k, "v": new_v}
    else:
        cache["k"] = torch.cat([cache["k"], new_k], dim=2)
        cache["v"] = torch.cat([cache["v"], new_v], dim=2)
        return cache

In practice: preallocate [B, H, max_T, D] for speed, then fill positions.

---

COMPLEXITY:
- Without cache: O(T²) per generated sequence.
- With cache: O(T) total for incremental decoding.
- Critical for long generations (chatbots, summarization).

---

GOTCHAS:
- Memory: KV cache grows linearly with T. Long outputs can exhaust GPU memory.
- Multi-GPU serving: must shard or offload cache.
- Sequence parallelism (e.g., FlashAttention-2) changes implementation details.
- Need to reset cache between sequences, or truncate for sliding-window attention.

---

INTUITION:
Think of KV cache as a “notebook”:
- Once the model writes down keys and values for past tokens,
  it doesn't need to recompute them.
- At each new token, it just appends one more line and looks back into the notebook.

---

REAL-WORLD NOTES:
- All modern autoregressive LLMs (GPT-3, LLaMA, Mistral) use KV cache.
- Hugging Face generate() API automatically manages KV cache internally.
- Efficient serving frameworks (vLLM, TensorRT-LLM, FasterTransformer)
  implement advanced cache management (paged cache, continuous batching).
"""
