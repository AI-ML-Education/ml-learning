# Sampling & Decoding
"""
WHAT/WHY:
After training, a causal LM produces logits over the vocabulary at each step.
We need a decoding strategy to turn these probability distributions into
concrete tokens. Different strategies trade off between determinism, diversity,
and fluency.

STRATEGIES:
1. Greedy Decoding (argmax)
   - Always pick the highest-probability token.
   - Deterministic, but often repetitive or dull.

2. Temperature Scaling
   - Adjusts "confidence" of the distribution.
   - logits = logits / T
   - T < 1 → sharper, more deterministic
   - T > 1 → flatter, more random
   - T = 1 → unchanged

3. Top-k Sampling
   - Keep only the k most probable tokens, set others to -∞.
   - Renormalize and sample.
   - Controls vocabulary size per step.
   - Example: k=50.

4. Top-p Sampling (Nucleus Sampling)
   - Keep the smallest set of tokens whose cumulative probability ≥ p.
   - More adaptive than top-k.
   - Example: p=0.9 → only tokens covering 90% probability mass are sampled.

5. Repetition Penalty
   - Penalize logits of tokens that appeared recently to reduce loops.
   - Common formula: if token ∈ recent_context:
       if logit > 0: logit /= penalty
       else: logit *= penalty

6. EOS Handling
   - Stop generation if End-of-Sequence token is sampled.
   - Always set a max length as a fallback.

---

MATH:
Given logits z ∈ ℝ[V]:
1. Apply temperature: z' = z / T
2. (Optional) Apply repetition penalty adjustments
3. Convert to probabilities: p = softmax(z')
4. Apply top-k or top-p filtering to mask unlikely tokens
5. Sample from categorical distribution: token ~ Categorical(p)

---

SHAPES:
- Input logits: [B, V]
- After scaling/masking: [B, V]
- Output: next_token_id ∈ [0, V)

---

PYTORCH NOTES:
- Use torch.topk for top-k filtering.
- Use cumulative probability + torch.sort for top-p filtering.
- Sampling: torch.multinomial(p, num_samples=1).
- Greedy: torch.argmax(p, dim=-1).

---

EXAMPLES:
Greedy:
   logits = [2.1, 0.3, -0.5] → probs = softmax → pick index 0

Top-k (k=2):
   keep top-2 logits, mask rest → renormalize → sample

Top-p (p=0.8):
   keep smallest set of tokens whose cumulative prob ≥ 0.8

---

GOTCHAS:
- Temperature=0 is not defined mathematically; in code it's often treated
  as "greedy decoding".
- Top-k and top-p can be combined, but usually one is enough.
- Always renormalize probabilities after filtering.
- For batched decoding, EOS must be tracked per sample (mask out finished ones).

---

INTUITION:
- Temperature controls overall randomness.
- Top-k limits randomness to a fixed shortlist.
- Top-p adapts randomness to the uncertainty of the model.
- Repetition penalty encourages novelty but must be tuned carefully.

---

REAL-WORLD NOTES:
- GPT-2 paper popularized top-k (k=40).
- Nucleus (top-p) sampling (Holtzman et al. 2020) is now standard in LLMs.
- Many production LLMs use: temperature ~0.7, top-p=0.9, repetition_penalty ~1.1.
"""
