# Attention Mechanism (Scaled dot-product + Causal Masking)
"""
WHAT/WHY:
The attention mechanism allows each token to "look back" at other tokens in the
sequence and gather relevant information.
This is the *core* of the Transformer — instead of processing tokens sequentially 
(like RNNs), attention enables parallel processing with flexible dependency modeling.

In a Causal Language Model (CLM), we specifically use **causal attention**
(also called "masked self-attention") so that each token can only attend to itself
and past tokens, not future tokens.  
This enforces the autoregressive property: predicting the next token without peeking ahead.

---

MATH (Scaled Dot-Product Attention):
Given:
- Queries Q ∈ ℝ[B, H, T, D]
- Keys    K ∈ ℝ[B, H, T, D]
- Values  V ∈ ℝ[B, H, T, D]

Where:
- B = batch size
- H = number of attention heads
- T = sequence length
- D = head dimension (d_model / H)

1. Compute raw scores (similarity between queries and keys):
   scores = Q @ Kᵀ
   Shape: [B, H, T, T]

2. Scale the scores:
   scores = scores / sqrt(D)
   Why?
   - Without scaling, dot products grow with D.
   - Large magnitudes push softmax into very peaky distributions.
   - Scaling stabilizes gradients and prevents early saturation.

3. Apply mask:
   - **Causal mask:** Disallow attending to future tokens.
   - Form: a [T, T] upper-triangular matrix with -∞ above the diagonal.
   - scores = scores + mask
   - For padding, an additional mask may be applied to ignore pad tokens.

4. Normalize with softmax:
   attn_weights = softmax(scores, dim=-1)
   Shape: [B, H, T, T]

5. Apply dropout (during training only).

6. Compute output:
   out = attn_weights @ V
   Shape: [B, H, T, D]

---

CAUSAL MASK (example):
Suppose T=4, causal mask is:
[
 [0, -∞, -∞, -∞],
 [0,   0, -∞, -∞],
 [0,   0,   0, -∞],
 [0,   0,   0,   0]
]

Meaning:
- Token 0 attends only to itself.
- Token 1 attends to tokens {0,1}.
- Token 2 attends to tokens {0,1,2}.
- Token 3 attends to tokens {0,1,2,3}.

---

SHAPES:
Q, K, V: [B, H, T, D]
Scores:  [B, H, T, T]
Weights: [B, H, T, T]
Output:  [B, H, T, D]

Then, across heads, outputs are concatenated: [B, T, H*D] = [B, T, d_model]

---

PYTORCH NOTES:
- einsum notation: 
  scores = torch.einsum("bhqd, bhkd -> bhqk", Q, K)
- scaling: scores /= sqrt(D)
- masking: use `masked_fill` or add a large negative number to forbidden positions.
- dropout: `F.dropout(attn_weights, p=dropout, training=self.training)`
- safe softmax: consider `F.softmax(scores, dim=-1, dtype=torch.float32)` 
  even if model is in fp16/bf16 (to prevent overflow).

---

GOTCHAS:
- Forgetting scaling (1/√D).
- Wrong mask dtype (must be additive in logits space, not boolean).
- Mask broadcast errors (ensure [T,T] or [B,1,T,T] to match batch/heads).
- Using softmax after dropout (wrong order — dropout applies after softmax).
- Attending to future tokens if mask is incorrectly constructed.

---

INTUITION:
- Keys and queries measure "relevance".
- Values carry the actual information.
- Attention is "relevant weighted averaging" of values.
- Causal mask ensures autoregression (next-token prediction).
"""
