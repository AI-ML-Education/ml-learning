# Normalization in Transformers (LayerNorm, RMSNorm, Pre-norm vs Post-norm)
"""
WHAT/WHY:
Deep neural networks often suffer from unstable training due to exploding or
vanishing activations/gradients.
Normalization layers stabilize training by ensuring that activations have
controlled scale.

In Transformers, normalization is critical for:
- Stable gradient flow across many stacked layers.
- Faster convergence.
- Reducing sensitivity to weight initialization.

Two main types in modern LLMs:
1. Layer Normalization (LayerNorm)
2. Root Mean Square Normalization (RMSNorm)

---

1) LAYER NORMALIZATION (LayerNorm)
- Normalizes across the feature dimension for each token independently.
- Formula:
  LN(x) = (x - μ) / √(σ² + ε) * γ + β

  Where:
  - x ∈ ℝ[d_model] (one token's hidden vector)
  - μ = mean(x) over d_model
  - σ² = variance(x) over d_model
  - γ, β ∈ ℝ[d_model] are learned scale and bias parameters

- Effect: ensures zero mean and unit variance across features, then rescales/shifts.

- Used in early Transformers (e.g., original "Attention is All You Need").

---

2) RMS NORMALIZATION (RMSNorm)
- A simplified variant of LayerNorm:
  - Removes the mean subtraction step.
  - Normalizes only by the root mean square (RMS) of activations.

- Formula:
  RMSNorm(x) = (x / RMS(x)) * γ
  Where RMS(x) = √( (1/d) ∑ x_i² + ε )

- Parameters: only γ (scale), no bias β.
- Typically slightly faster and more memory efficient.
- Empirically performs as well or better in large LLMs (e.g., LLaMA, GPT-J).

---

3) PRE-NORM vs POST-NORM TRANSFORMER BLOCKS

- In a Transformer block, we have:
  Input → (Norm → SubLayer → Residual)

  Two variants exist:

  **Post-Norm (original Transformer, 2017):**
  - Block = (SubLayer(x) + x) → Norm
  - Normalization is applied after adding the residual.

  **Pre-Norm (modern LLMs):**
  - Block = SubLayer(Norm(x)) + x
  - Normalization is applied before the sublayer.
  - Residual connection is added after the sublayer.

- WHY PRE-NORM?
  - Improves training stability for deep models (50+ layers).
  - Ensures gradients can flow directly through residual paths without being distorted by normalization.
  - Almost all modern large LMs (GPT-3, LLaMA, etc.) use Pre-Norm.

---

SHAPES:
- Input/output per token: [B, T, d_model]
- LayerNorm: applies independently across last dimension (d_model).
- RMSNorm: same shape handling, but only scaling by RMS.

---

PYTORCH NOTES:
- LayerNorm: `nn.LayerNorm(d_model, eps=1e-5)`
- RMSNorm: not in standard PyTorch, usually custom implemented.

Example RMSNorm implementation:

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: [B, T, d_model]
        norm_x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.scale * norm_x

---

GOTCHAS:
- ε placement: must be inside sqrt for RMS stability.
- Don't apply weight decay to γ, β parameters.
- Switching Pre-Norm ↔ Post-Norm drastically changes training stability.
- RMSNorm does not center the activations (mean ≠ 0), but this is fine in practice.

---

INTUITION:
- LayerNorm: makes every token vector have mean=0 and variance=1.
- RMSNorm: just ensures consistent scale (length) of token vectors.
- Pre-Norm: lets gradients “flow around” the sublayer by passing through
  the residual path almost unchanged, avoiding vanishing/exploding.

---

REAL-WORLD NOTES:
- GPT-2 used LayerNorm with Post-Norm blocks.
- GPT-3 switched to Pre-Norm for stability at scale.
- LLaMA series use RMSNorm for efficiency + Pre-Norm structure.
"""
