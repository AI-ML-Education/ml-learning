# Multi-Head Attention (MHA)
"""
WHAT/WHY:
Scaled Dot-Product Attention (previous section) is powerful, but a single head
may not be sufficient to capture different types of relationships.
For example:
- One head might focus on short-term dependencies (next word).
- Another might capture long-range dependencies (topic consistency).
- Yet another might track structural signals (e.g., punctuation or syntax).

**Multi-Head Attention (MHA)** solves this by projecting the input into
multiple subspaces ("heads"), performing attention in parallel, and then
combining results.

This enables the model to learn diverse attention patterns simultaneously.

---

MATH:

Let:
- x ∈ ℝ[B, T, d_model]  (input embeddings)
- H = number of heads
- D = head dimension = d_model / H

Step 1. Linear projections for Q, K, V:
Q = x W_Q,   K = x W_K,   V = x W_V

Where:
- W_Q, W_K, W_V ∈ ℝ[d_model, d_model]
- Each projects from full hidden size to a new hidden space.

Step 2. Reshape into heads:
Q, K, V → [B, H, T, D]
(i.e., split d_model into H heads of size D)

Step 3. Apply scaled dot-product attention (per head):
head_i = Attention(Q_i, K_i, V_i)
Shape: [B, T, D]

Step 4. Concatenate all heads:
concat = [head₁; head₂; …; head_H] 
Shape: [B, T, H*D] = [B, T, d_model]

Step 5. Final projection:
output = concat W_O
Where W_O ∈ ℝ[d_model, d_model]

---

SHAPES:
Input:   [B, T, d_model]
Q, K, V: [B, H, T, D]   (D = d_model/H)
Scores:  [B, H, T, T]
Attn:    [B, H, T, D]
Concat:  [B, T, d_model]
Output:  [B, T, d_model]

---

PARAMETERS:
- Q, K, V projection matrices: 3 × d_model × d_model
- Output projection: 1 × d_model × d_model
- Total ≈ 4 × d_model² parameters (ignoring biases)

Example:
For d_model = 768 (like GPT-2 small), H=12, D=64:
- Parameters ≈ 4 × (768²) = ~2.36M just for attention.

---

PYTORCH NOTES:
- Use `nn.Linear(d_model, d_model)` for Q, K, V, O projections.
- Reshape with `.view(B, T, H, D).transpose(1,2)` to get [B, H, T, D].
- Use the SDPA function (from Section 4) per head.
- Concatenate heads with `.transpose(1,2).contiguous().view(B, T, d_model)`.

---

GOTCHAS:
- d_model must be divisible by H (otherwise D is fractional).
- Forgetting to scale scores by √D.
- Shape mismatches during reshaping/concatenation.
- Not applying final output projection W_O (important for mixing head info).
- Training vs inference: must handle KV cache across multiple heads.

---

INTUITION:
Think of each head as a "specialist" focusing on different linguistic patterns:
- One may focus on the next word,
- Another on verb tense agreement,
- Another on global topic coherence.
The output projection W_O mixes these specialists back into a unified representation.

---

REAL-WORLD NOTES:
- GPT-2, GPT-3, LLaMA, etc. all use MHA as their main building block.
- Some models experiment with different head sizes or shared projections.
- Efficiency: Multi-Query Attention (MQA) and Grouped Query Attention (GQA) reduce memory cost by sharing keys/values across heads.
"""