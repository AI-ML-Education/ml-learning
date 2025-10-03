# Training Mechanics & Efficiency
"""
WHAT/WHY:
Training large language models is resource-intensive and unstable without
careful optimization choices.
Key components: optimizer, learning rate schedule, gradient management,
precision handling, and memory efficiency.

---

OPTIMIZER:
- Standard: AdamW (Adam with decoupled weight decay).
- Why not plain Adam?
  - Adam mixes L2 penalty with gradient step → scale-sensitive.
  - AdamW decouples weight decay → better regularization.
- Common hyperparameters:
  - β1 = 0.9 (momentum)
  - β2 = 0.95-0.98 (longer-term smoothing)
  - eps = 1e-8 (numerical stability)
  - weight_decay = 0.01 (apply only to weights, not biases/norms)

---

LEARNING RATE SCHEDULE:
- Warmup: linear increase from 0 → peak LR over first N steps.
- Decay: cosine decay or step decay after warmup.
- Why?
  - Warmup prevents instability from large updates at the start.
  - Decay prevents overfitting and reduces oscillations at the end.

Typical formula:
   lr(step) = lr_peak * 0.5 * (1 + cos(step / total_steps * π))
   for step > warmup_steps

---

GRADIENT MANAGEMENT:
- Gradient Clipping:
  - Clip global norm (e.g., 1.0) to prevent exploding gradients.
- Gradient Accumulation:
  - Simulate larger batch sizes by accumulating gradients over N steps.
- Parameter Groups:
  - Exclude bias and normalization weights from weight decay.

---

MIXED PRECISION TRAINING:
- Use FP16 or BF16 for forward/backward pass, FP32 for master weights.
- Benefits: faster training, lower memory use.
- Risks: gradient underflow/overflow.
- Solution: Gradient scaling.
  - PyTorch: torch.cuda.amp + GradScaler

---

MEMORY EFFICIENCY TRICKS:
- Gradient checkpointing:
  - Recompute activations during backward instead of storing → saves VRAM.
- FlashAttention:
  - IO-aware attention kernel → faster, less memory.
- Model sharding / tensor parallelism:
  - Distribute weights across GPUs.

---

PYTORCH NOTES:
- AdamW: torch.optim.AdamW with param_groups for weight decay exclusion.
- AMP:
  scaler = GradScaler()
  with autocast():
      loss = model(x).loss
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()

- Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

---

GOTCHAS:
- Forgetting to exclude LayerNorm/embedding biases from weight decay → hurts training.
- No warmup → unstable early loss spikes.
- Using FP16 without loss scaling → NaNs.
- Too much gradient accumulation → optimizer states may diverge.
- Scheduler step must be consistent (per batch vs per epoch).

---

INTUITION:
- Optimizer defines how weights update.
- Scheduler defines when updates shrink.
- Clipping prevents catastrophic updates.
- Mixed precision trades precision for efficiency but needs safeguards.
- Efficiency tricks (checkpointing, FlashAttention) are essential at scale.

---

REAL-WORLD NOTES:
- GPT-3 trained with Adam (β1=0.9, β2=0.95), weight decay=0.1, cosine LR decay.
- LLaMA uses AdamW + RMSNorm + SwiGLU, with warmup + cosine schedule.
- DeepSpeed/FSDP commonly used for gradient sharding and memory scaling.
"""
