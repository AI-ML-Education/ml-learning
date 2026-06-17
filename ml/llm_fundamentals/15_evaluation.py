# Evaluation Basics
"""
WHAT/WHY:
Once a language model is trained or fine-tuned, we need to evaluate how well it 
performs. Evaluation ensures:
- The model actually learned useful patterns.
- We can compare models across tasks.
- We can track overfitting or underfitting.
- We can measure quality beyond just training loss.

Evaluation is not one-size-fits-all. Different tasks require different metrics.

---

1) PERPLEXITY (PPL):
- Standard metric for language modeling.
- Definition: PPL = exp(cross_entropy_loss)
- Intuition: Lower perplexity = model is less "surprised" by the data.
- Example:
  Cross-entropy = 2.0 → PPL = exp(2.0) ≈ 7.39
- Use case: Evaluate on held-out test set (teacher forcing).
- Limitation: Correlates poorly with human judgments of text quality.

---

2) EXACT MATCH (EM):
- Checks if the model output exactly matches the ground-truth answer.
- Used in QA, closed-form tasks.
- Example: Question: "Capital of France?" → Output: "Paris"
  - EM = 1 if exact match, else 0
- Limitation: Very strict, does not allow partial correctness.

---

3) BLEU (Bilingual Evaluation Understudy):
- n-gram overlap between generated and reference text.
- Common in machine translation.
- BLEU-4 uses up to 4-grams.
- Example: Reference = "the cat is on the mat"
           Prediction = "the cat sat on the mat"
  - BLEU captures partial n-gram overlap.
- Limitation: Rigid, does not capture meaning or fluency well.

---

4) ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
- Measures n-gram recall (how much of the reference text is recovered).
- Common in summarization.
- ROUGE-L uses longest common subsequence.
- Limitation: Like BLEU, relies on surface-level overlap.

---

5) EMBEDDING-BASED METRICS (BERTScore, MoverScore):
- Compare similarity in embedding space instead of surface overlap.
- More semantic awareness.
- Example: "the boy is playing football" vs "a kid plays soccer"
  - Low BLEU, high BERTScore (semantically similar).

---

6) TASK-SPECIFIC EVALUATION:
- QA: F1 (overlap of predicted vs ground-truth tokens).
- Dialogue: Human ratings (helpfulness, safety).
- Code: Pass@k (probability of generating correct solution within k samples).
- Safety: Prompt-based evaluators (toxicity, bias, hallucination).

---

PYTORCH / IMPLEMENTATION NOTES:
- Perplexity:
  ```python
  import torch.nn.functional as F
  def perplexity(model, dataloader):
      total_loss, total_tokens = 0, 0
      for x, y in dataloader:
          logits = model(x)
          loss = F.cross_entropy(logits.transpose(1,2), y, ignore_index=-100, reduction="sum")
          total_loss += loss.item()
          total_tokens += (y != -100).sum().item()
      return math.exp(total_loss / total_tokens)

- BLEU/ROUGE: Use NLTK or Hugging Face datasets for standardized implementations.

---

GOTCHAS:
- Always separate training, validation, and test sets.
- Teacher forcing required for perplexity (feed ground-truth tokens).
- BLEU/ROUGE penalize paraphrasing → use semantic metrics if possible.
- Evaluation on in-domain vs out-of-domain data may differ drastically.
- For safety evals, automatic metrics are limited → need human-in-the-loop.

---

INTUITION:
- Perplexity = “How surprised is the model?”
- Exact Match = “Did it get the right answer word-for-word?”
- BLEU/ROUGE = “Does it overlap with the reference surface text?”
- Embedding metrics = “Does it mean the same thing?”
- Human eval = “Is it actually useful, safe, and trustworthy?”

---

REAL-WORLD NOTES:
- LLM leaderboards (HELM, OpenLLM, MLPerf) report multiple metrics
  (reasoning, coding, math, safety).
- Perplexity used for model pretraining analysis.
- BLEU/ROUGE less popular now; replaced by BERTScore, human evals,
  or task-specific benchmarks.
"""
