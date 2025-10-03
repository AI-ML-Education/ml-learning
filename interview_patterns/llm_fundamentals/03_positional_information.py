# Positional Information (Encodings / Embeddings)
"""
WHAT/WHY:
Transformers are permutation-invariant: they treat input tokens as a "set" rather
than a sequence. Without additional information, the model cannot know whether
"dog bites man" is different from "man bites dog".

Therefore, we need to inject **positional information** into the model so it
can understand the order of tokens.

APPROACHES:
1. Absolute Positional Encodings (fixed sinusoidal, learned embeddings).
2. Relative Positional Encodings (rotary, attention bias).
3. Hybrid methods.

---

MATH / VARIANTS:

1) SINUSOIDAL (absolute, fixed, no learned parameters)
- Defined by sine and cosine waves at different frequencies.
- Formula for position = pos, dimension index = i:

  PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

This creates smooth, periodic signals across dimensions.
Different frequencies allow the model to distinguish nearby and far positions.

- Advantage: extrapolates beyond training length.
- Disadvantage: rigid, not task-optimized.

2) LEARNED POSITION EMBEDDINGS
- A trainable lookup table of shape [max_seq_len, d_model].
- Each position index has a learned vector.
- Model learns optimal encodings for its training data.

- Advantage: often empirically better in-distribution.
- Disadvantage: does not extrapolate beyond trained length.

3) ROTARY POSITIONAL EMBEDDINGS (RoPE, relative)
- Instead of adding positions to token embeddings, RoPE rotates Q and K vectors
  in attention space by a position-dependent angle.
- Effect: Dot product between Q_i and K_j depends only on relative distance (i - j),
  which better captures translation-invariance of language.

Implementation idea:
- For each head dimension (must be even), group into pairs (x, y).
- Apply 2D rotation with angle θ = pos * freq:
    x' =  x * cosθ - y * sinθ
    y' =  x * sinθ + y * cosθ

- Advantage: generalizes to longer contexts; used in modern LLMs (GPT-NeoX, LLaMA, etc.).
- Disadvantage: slightly more complex implementation.

---

SHAPES:
- Position embedding (absolute): [T, d_model]
- Applied to token embeddings (x: [B, T, d_model]):
  Output = x + pos_emb[0:T]

- RoPE:
  Q, K after projection: [B, H, T, D]
  → apply rotation along the last dimension (D), pairwise.

---

EXAMPLE:
Input tokens (ids): [101, 2054, 2003, 102]
Token embeddings: [4, d_model]
Position embeddings: [4, d_model]
Final embeddings = token_emb + pos_emb

---

PYTORCH NOTES:
- Sinusoidal: precompute a matrix of [max_seq_len, d_model] and slice.
- Learned: use nn.Embedding(max_seq_len, d_model).
- RoPE: implement a helper `apply_rope(q, k, freqs)`.

---

GOTCHAS:
- With absolute learned embeddings: extrapolation fails if you try sequences longer than training max length.
- With RoPE: apply only to Q and K, not V.
- Ensure head dimension is even for RoPE (because we rotate pairs).
- Watch dtype/device when precomputing sinusoidal/RoPE tables.

---

SUMMARY:
- Absolute encodings (sinusoidal or learned) add position info at embedding level.
- RoPE modifies attention mechanism to encode relative distances.
- Modern LLMs (GPT-3, LLaMA, Mistral) use RoPE or similar relative schemes for better generalization.
"""
