# LoRA & QLoRA (Parameter-Efficient Fine-Tuning)
"""
WHAT/WHY:
Large Language Models (LLMs) often have billions of parameters, making full
fine-tuning (updating all weights) very costly in terms of GPU memory, time, 
and storage.  

**LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)** are methods for
parameter-efficient fine-tuning:
- Freeze most of the base model’s weights.
- Introduce small trainable modules (LoRA adapters).
- Greatly reduce the number of trainable parameters.
- Enable fine-tuning on consumer-grade hardware.

---

LORA (Low-Rank Adaptation):

IDEA:
- Instead of updating full weight matrix W ∈ ℝ[d_out, d_in],
  approximate update ΔW as a low-rank factorization:
  
  ΔW = B A * (α / r)

  Where:
  - A ∈ ℝ[r, d_in]
  - B ∈ ℝ[d_out, r]
  - r = rank (small, e.g., 4, 8, 16)
  - α = scaling factor (controls effective learning rate of adapters)

- Final weight during training:
  W' = W (frozen) + ΔW (trainable)

- At inference, ΔW can be merged into W for efficiency.

WHERE TO APPLY:
- Commonly applied to projection matrices in attention layers:
  W_q, W_k, W_v, W_o
- Sometimes also applied to FFN layers.

BENEFITS:
- Only train ~0.1–2% of parameters.
- Faster, cheaper fine-tuning.
- Can maintain multiple adapters for different tasks/domains.

---

QLO-RA (Quantized LoRA):

IDEA:
- Quantize base model weights (e.g., to 4-bit NF4).
- Keep frozen quantized weights in memory (saves VRAM).
- Train LoRA adapters in higher precision (fp16/bf16).

BENEFITS:
- Enables fine-tuning 65B parameter models on a single 48GB GPU.
- Preserves base model performance (minimal quantization degradation).
- Memory savings from quantization + adapter efficiency.

---

MATH RECAP:
Given input x ∈ ℝ[d_in]:

LoRA forward pass:
  y = W x   +   (B (A x)) * (α / r)

- W is frozen
- A, B are trainable low-rank matrices
- Only ~O(r(d_in + d_out)) trainable parameters (instead of d_in × d_out)

---

PYTORCH NOTES:
- Implement LoRA as a wrapper around nn.Linear.
- Forward pass adds LoRA contribution to frozen Linear output.
- Merge/unmerge adapters at inference for speed.

Example:
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=8, alpha=16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.base = nn.Linear(in_dim, out_dim, bias=False)
        self.base.weight.requires_grad = False
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scaling

---

GOTCHAS:
- Choosing too small rank r → adapters underfit.
- Forgetting to freeze base weights → defeats purpose.
- Scaling factor α must be tuned (too large = instability).
- With QLoRA, must handle quantization/dequantization efficiently.

---

INTUITION:
- LoRA: “Instead of moving the entire house (W), just add a small,
  trainable extension (ΔW) to adjust it for new tasks.”
- QLoRA: “Compress the house into a tiny blueprint (quantization)
  but still let you renovate with small extensions.”

---

REAL-WORLD NOTES:
- LoRA popularized by Hugging Face PEFT library (2022).
- QLoRA (Dettmers et al., 2023) enabled fine-tuning 65B LLaMA on a single GPU.
- Industry standard for adapting large models to domain-specific tasks
  without massive compute budgets.
"""
