# Modeling Paradigms in LLMs
"""
WHAT/WHY:
There are multiple ways to pretrain and structure large language models (LLMs).
The choice of objective (causal, masked, seq2seq, etc.) affects what the model learns
and what tasks it is best suited for.

---

1. Causal Language Modeling (CLM, Autoregressive)
- Predict next token x_{t+1} from past tokens x_{≤t}.
- Requires causal (triangular) mask so tokens cannot peek into the future.
- Enables autoregressive text generation.

MATH:
Objective = sum_{t=1}^{T-1} log p(x_{t+1} | x_{≤t})

SHAPES:
Inputs:  [B, T] → Logits: [B, T, V]
Labels:  [B, T] (shifted left, -100 for last/pad)

EXAMPLES:
GPT family (GPT-2, GPT-3, GPT-4), LLaMA, Mistral

GOTCHAS:
- Must apply causal mask correctly.
- Can't directly see right context (one-directional).

INTUITION:
Autoregressive forecasting: each token predicts the next one.
Ideal for text generation and dialogue.

---

2. Masked Language Modeling (MLM)
- Randomly mask tokens in the input and predict them from surrounding context.
- Allows bidirectional context use during training.

MATH:
Objective = sum log p(x_{mask} | x_{<mask}, x_{>mask})

SHAPES:
Inputs:  [B, T] with [MASK] tokens
Labels:  [B, T] (masked positions filled with token IDs, others = -100)

EXAMPLES:
BERT, RoBERTa, DeBERTa

GOTCHAS:
- Not autoregressive: cannot generate text directly.
- Distribution of masks matters (mask too much → too hard, too little → too easy).

INTUITION:
Fill-in-the-blanks training. Great for embeddings and classification tasks.

---

3. Sequence-to-Sequence (Seq2Seq, Encoder-Decoder)
- Encoder reads the full input sequence (bidirectional).
- Decoder generates output sequence autoregressively with causal mask.

MATH:
Objective = sum_{t=1}^{T-1} log p(y_{t+1} | y_{≤t}, x_{1:T})

SHAPES:
Encoder input:   [B, T_enc]
Decoder logits:  [B, T_dec, V]
Decoder labels:  [B, T_dec]

EXAMPLES:
T5, BART, mT5

GOTCHAS:
- Training requires aligned input-output pairs (text-to-text).
- More complex than pure CLM or MLM.

INTUITION:
Think "translation machine": one sequence → another sequence.
Ideal for summarization, translation, text-to-text tasks.

---

4. Span Corruption / Infilling
- Generalization of MLM where contiguous spans (not just single tokens) are masked out.
- Model predicts missing spans given context.

MATH:
Input: [x₁, ..., <mask>, ..., x_T]
Target: the missing span tokens

SHAPES:
Inputs:  [B, T] with special <mask> tokens
Labels:  Missing spans in order

EXAMPLES:
T5 (span corruption), Codex (code infilling), GPT family with fill-in-the-middle (FIM)

GOTCHAS:
- Requires special handling for span IDs and targets.
- More complex batching.

INTUITION:
"Fill in the blanks" but with flexible span length → improves robustness and editing ability.

---

SUMMARY:
- CLM = autoregressive, great for generation.
- MLM = bidirectional context, great for understanding/embeddings.
- Seq2Seq = encoder-decoder, great for translation & summarization.
- Span Corruption = flexible masking, great for editing & robustness.
"""
