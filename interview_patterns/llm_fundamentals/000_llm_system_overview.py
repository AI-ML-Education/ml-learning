# LLM System Overview (End-to-End Components)
"""
WHAT/WHY:
Large Language Models (LLMs) process raw text through a pipeline of components.
Each step transforms the data into richer representations until the model can
predict the next token. Adding analogies helps ground the technical flow.

---

1. INPUT PREPARATION
- Raw text → tokenized into IDs (like turning words into dictionary indexes) using a subword tokenizer
  (e.g., BPE, WordPiece, SentencePiece).
  Example: "I love cats" → [12, 47, 99]

- Sequence length (T) is fixed; sequences longer than T are truncated, shorter ones are padded. 
  This is called the context window — the maximum number of tokens the model can see at once.
  Analogy: like how many words you can keep in short-term memory before forgetting the start.

- Batch size (B) = number of sequences processed in parallel.
  Analogy: like a class of students each reading their own sentence simultaneously.

- Padding is applied to sequences shorter than T so that all have uniform shape [B, T].
  Analogy: filling empty seats in a row with placeholders so all rows have equal length.

- Each token ID is mapped to a dense vector via a learnable embedding matrix of shape [V, d_model],
  where V = vocabulary size and d_model = embedding dimension.
  Analogy: like looking up a word in a dictionary to get its meaning represented as a vector.

---

2. POSITIONAL INFORMATION
- Attention is order-agnostic; it treats tokens as a bag of words unless we add positions.
- A positional encoding is calculated for each position (absolute sinusoidal, learned embeddings, or relative encodings like RoPE/ALiBi).
- Token embeddings and positional encodings are combined (typically added).
- Shape after this step: [B, T, d_model]
  Analogy: like page numbers in a book — the words make sense, but you need the page number to know their order.

---

3. TRANSFORMER BLOCKS (repeated N times)
Each block processes token representations to capture dependencies and context.

Inside each block:
- LayerNorm: normalize activations for stable training.
  Analogy: like adjusting volume so all speakers in a room are balanced.

- Multi-Head Self-Attention:
  - Queries, Keys, Values projected from input.
  - Dot-products compute similarity, scaled by sqrt(d_head).
  - Softmax produces attention weights.
  - Causal mask applied so tokens cannot see the future.
  - Padding mask applied so pad tokens are ignored.
  Analogy: at a party, each person listens to everyone else but only pays more attention 
  to the most relevant voices; causal mask means no one can hear "future speakers".

- Residual connection: add the input back to the output of attention.
  Analogy: like layering new paint but keeping some of the base color visible.

- Feed Forward Network (FFN):
  - Two linear layers with nonlinearity (e.g., ReLU, GELU, SwiGLU).
  - Expands dimension (often 4x d_model) then projects back.
  - Adds nonlinear mixing of features.
  Analogy: like digesting food after a conversation — you process and extract nutrients 
  before sending energy back into the system.

- Another residual connection.

Optional enhancements:
- RMSNorm instead of LayerNorm.
- Rotary Position Embeddings (RoPE), ALiBi for positional info.
- FlashAttention for efficient computation.

---

4. FINAL PROJECTION
- After N blocks, apply final LayerNorm.
- Linear projection maps hidden states [B, T, d_model] → logits [B, T, V].
- Softmax converts logits into probability distributions over vocabulary tokens.
  Analogy: like the brain deciding which word is most likely to come next given all the context.

---

5. TRAINING OBJECTIVE
- Labels are the input shifted left by 1 (each token predicts the next).
- Cross-entropy loss compares logits vs. labels.
- Ignore index (e.g., -100 in PyTorch) is used for padding and the final token.
  Analogy: like a teacher grading answers — correct answers get marks, wrong answers get penalties,
  and blank ones (padding) are skipped.

- Objective: maximize sum_{t=1}^{T-1} log p(x_{t+1} | x_{≤t})

---

6. TRAINING ENHANCEMENTS
- Weight tying: share parameters between embedding matrix and final projection.
- Regularization: dropout, label smoothing.
- Optimizer: AdamW with learning rate schedules (warmup, cosine decay).
- Gradient clipping to prevent exploding updates.
- Mixed precision (fp16 or bf16) for speed and memory efficiency.
  Analogy: like training with a lighter backpack so you can move faster but still reach the same goal.

---

7. FINE-TUNING & ALIGNMENT
- Supervised Fine-Tuning (SFT): train on instruction or QA datasets.
- Preference modeling: train a reward model to score responses.
- RLHF (e.g., PPO) or Direct Preference Optimization (DPO) to align model outputs with human values.
- Additional safety/alignment data ensures responsible outputs.
  Analogy: like taking a trained student and giving them etiquette lessons 
  so they answer politely and responsibly.

---

8. INFERENCE & GENERATION
- Start with a prompt (initial tokens).
- Model predicts distribution for next token.
- Choose next token via sampling strategy:
  - Greedy (always pick most likely).
  - Temperature (controls randomness).
  - Top-k (sample from k most likely).
  - Top-p / nucleus (sample from smallest set of tokens covering p% probability).
- Append token to input, repeat until EOS or max length.
- Use key-value (KV) caching to avoid recomputation during autoregressive generation.
  Analogy: like writing a story word by word, remembering what you've already written 
  so you don't have to re-read the whole story every time.

---

SUMMARY:
LLMs process text through:
Raw text → Tokenization → Batching/sequence shaping → Embeddings + Positional info →
Stacked Transformer blocks (attention + FFN + residuals/norm) → Final projection/softmax → Loss.

After training, models are refined through fine-tuning and alignment, then deployed with
efficient inference strategies for real-world use.
"""
