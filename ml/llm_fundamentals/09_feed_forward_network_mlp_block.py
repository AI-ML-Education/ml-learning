# Feed-Forward Network (FFN / MLP Block)
"""
WHAT/WHY:
In the Transformer architecture, attention handles *token-to-token interactions*
(global mixing), but we also need a mechanism to transform each token's
representation *individually* in a nonlinear way.

That's where the **Feed-Forward Network (FFN)** comes in.
It applies the same small neural network to every token independently.

Think of it as:
- Attention = "Who should this token listen to?"
- FFN = "How should this token's representation be transformed after listening?"

The FFN provides nonlinear capacity and increases the representational power
of the model.

---

STRUCTURE:
The standard FFN consists of:
1. Linear layer (d_model → d_ff)  — expansion
2. Activation function (nonlinear, e.g., GELU or SiLU)
3. Linear layer (d_ff → d_model) — projection back

So, the token vector goes from a smaller hidden dimension (d_model), 
up to a larger intermediate space (d_ff), then back down again.

This is sometimes referred to as a "bottleneck MLP".

---

MATH:
Let x ∈ ℝ[B, T, d_model]

1. h = x W₁ + b₁,   with W₁ ∈ ℝ[d_model, d_ff]
2. h_act = φ(h),     where φ is activation (e.g., GELU, SiLU, ReLU)
3. out = h_act W₂ + b₂, with W₂ ∈ ℝ[d_ff, d_model]

Final output: out ∈ ℝ[B, T, d_model]

---

SHAPES:
- Input:  [B, T, d_model]
- Hidden: [B, T, d_ff]   (where d_ff is usually 4x d_model)
- Output: [B, T, d_model]

Example:
If d_model=768 and d_ff=3072:
- First Linear: (768 → 3072)
- Activation: elementwise
- Second Linear: (3072 → 768)
- Params ≈ 2 x (768x3072) ≈ 4.7M

---

ACTIVATION FUNCTIONS:
- **GELU** (Gaussian Error Linear Unit): smooth, used in GPT-2/3, BERT.
- **SiLU / Swish**: similar smooth nonlinearity, used in some LLMs.
- **ReLU**: simple, but less effective in deep transformers.
- **SwiGLU (Gated variant)**: improves expressiveness, used in LLaMA.
  - Formula: (xW₁ ⊗ σ(xW₂)) W₃
  - Slightly more parameters, but better performance.

---

PYTORCH NOTES:
- Implementation with `nn.Linear` layers.
- Activation: `F.gelu` or `F.silu`.
- In gated variants (SwiGLU), use two parallel linear layers for gating.

Example (classic FFN):
```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = getattr(F, activation)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

---

PARAMETER COUNT:
- Total ≈ 2 × d_model × d_ff (ignoring biases)
- In practice, the FFN block contains more parameters than the attention block.

Example:
- d_model = 4096, d_ff = 16384 (LLaMA-13B)
- Params in FFN ≈ 2 × 4096 × 16384 ≈ 134M per layer!

---

GOTCHAS:
- Forgetting to apply activation between linear layers.
- Using wrong activation (ReLU can work but is less effective).
- d_ff too small = underfitting; too large = memory blow-up.
- In SwiGLU, must double the projection dimensions to handle gating properly.

---

INTUITION:
- Attention does “who talks to whom”.
- FFN does “how do we transform the message once it arrives”.
- The expansion to d_ff and contraction back to d_model gives the model more
  capacity while keeping the interface size fixed.

---

REAL-WORLD NOTES:
- Original Transformer: ReLU.
- GPT-2/GPT-3: GELU.
- LLaMA, PaLM: SwiGLU (gated activations) for better performance-to-parameter ratio.
- FFN is usually the largest contributor to parameter count in modern LLMs.
"""
