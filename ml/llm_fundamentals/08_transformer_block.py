# Transformer Block (Pre-Norm with MHA + FFN + Residuals)
"""
WHAT/WHY:
The Transformer block is the fundamental building unit of LLMs.  
It combines:
1. Normalization (stabilization of activations)
2. Multi-Head Attention (token-to-token interactions)
3. Feed-Forward Network (nonlinear transformation per token)
4. Residual connections (ease gradient flow)

Stacking many Transformer blocks allows the model to build increasingly 
abstract representations of the input sequence.

---

STRUCTURE:
The modern standard is **Pre-Norm Transformer Block**:

Input x
│
├─► Norm(x) ─► Multi-Head Attention ─► Dropout ─► + residual (x)
│
└─► Norm(output) ─► Feed-Forward Network ─► Dropout ─► + residual

So, formally:

1. a = MHA(Norm(x)) + x
2. out = FFN(Norm(a)) + a

This pattern repeats for N layers.

---

MATH (per block):
Let x ∈ ℝ[B, T, d_model]

1. Norm:
   x_norm = Norm₁(x)

2. Attention:
   attn_out = MHA(x_norm)   → ℝ[B, T, d_model]

3. Residual add:
   a = x + attn_out

4. Norm:
   a_norm = Norm₂(a)

5. Feed-Forward:
   ff_out = FFN(a_norm)     → ℝ[B, T, d_model]

6. Residual add:
   out = a + ff_out

Final output: out ∈ ℝ[B, T, d_model]

---

SHAPES:
- Input/Output: [B, T, d_model]
- Attention: [B, T, d_model]
- FFN: [B, T, d_model]
- Block preserves dimensionality across layers.

---

WHY PRE-NORM?
- Pre-Norm (Norm before sublayer):
  - Better gradient flow in deep networks.
  - Avoids vanishing/exploding gradients.
  - Standard in modern LLMs (GPT-3, LLaMA, PaLM, etc.)
- Post-Norm (Norm after sublayer):
  - Used in original Transformer (2017).
  - Works for shallow networks but unstable for 50+ layers.

---

PYTORCH NOTES:
- Define block as `nn.Module` with two Norms, one MHA, one FFN.
- Use residual connections via addition.
- Dropout typically applied after MHA and FFN (before residual add).
- In practice, "DropPath" or stochastic depth may be added for regularization.

Example skeleton:
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, norm="layernorm"):
        super().__init__()
        self.norm1 = RMSNorm(d_model) if norm=="rmsnorm" else nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model) if norm=="rmsnorm" else nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

---

PARAMETERS:
- Attention: ~4 x d_model²
- FFN: ~2 x d_model x d_ff
- Norm: 2 x d_model (scale + bias if LN, just scale if RMSNorm)
- Typically, FFN dominates parameter count.

---

GOTCHAS:
- Forgetting residual connections → model won't train deep.
- Wrong placement of Norm (Pre vs Post) → stability issues.
- Mixing RMSNorm/LayerNorm across blocks → inconsistent behavior.
- Dropout must be applied before residual add, not after.

---

INTUITION:
- Attention: “Who should I listen to?”
- FFN: “How do I transform what I heard?”
- Residual: “Remember what I was before, just in case.”
- Norm: “Keep activations in a stable range.”

---

REAL-WORLD NOTES:
- GPT-2 used Post-Norm LayerNorm.
- GPT-3, LLaMA use Pre-Norm + RMSNorm.
- Some models use parallel attention/FFN instead of sequential (PaLM style):
  out = x + MHA(Norm(x)) + FFN(Norm(x))
- This reduces depth but increases width of computation graph.
"""
