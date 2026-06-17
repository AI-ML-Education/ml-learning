# Putting It All Together: Tiny GPT from Scratch
"""
WHAT/WHY:
We’ve studied each Transformer component (tokenization, embeddings, 
positional encodings, attention, FFN, normalization, residuals, LoRA, etc.).  
Now we put them together into a **Tiny GPT** model — a small, causal language 
model for next-token prediction.

This is the minimal blueprint for GPT-style LLMs.

---

ARCHITECTURE OVERVIEW:
Input IDs → Token Embeddings → Positional Encoding
→ N × Transformer Blocks (Norm → MHA → FFN + residuals)
→ Final LayerNorm → Linear Projection → Logits over Vocabulary

Training Objective: Causal LM loss (cross-entropy with shifted labels).

---

STEP 1. INPUTS:
- Input tokens: [B, T]
- Embedding layer: nn.Embedding(V, d_model)
- Positional info: absolute learned embeddings OR RoPE applied to Q,K.

Output: [B, T, d_model]

---

STEP 2. TRANSFORMER BLOCK STACK:
For each block (L layers):
1. Norm + Multi-Head Attention + residual
2. Norm + Feed-Forward + residual

Output stays [B, T, d_model].

---

STEP 3. FINAL PROJECTION:
- Apply final normalization (LayerNorm/RMSNorm).
- Project hidden states to vocabulary logits:
  logits = hidden @ W_vocabᵀ
  Shape: [B, T, V]

---

STEP 4. TRAINING:
- Labels: input shifted left by 1, with -100 for padding.
- Loss: cross-entropy(logits, labels).
- Optimizer: AdamW with weight decay exclusions.
- Schedule: warmup + cosine decay.
- Tricks: gradient clipping, AMP, gradient checkpointing.

---

MATH RECAP:
Given token IDs X ∈ [B, T]:
1. h0 = Embedding(X) + PositionalEncoding
2. For each block l:
   hl = TransformerBlock(hl-1)
3. h_final = Norm(hl)
4. logits = h_final W_vocabᵀ + b_vocab
5. Loss = CrossEntropyLoss(logits, labels)

---

SHAPES (example with B=2, T=5, V=10000, d_model=512, H=8):
- Input: [2, 5]
- Embeddings: [2, 5, 512]
- After L blocks: [2, 5, 512]
- Logits: [2, 5, 10000]

---

PYTORCH NOTES:
Minimal GPT skeleton:
```python
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, labels=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   labels.view(-1),
                                   ignore_index=-100)
        return logits, loss

---

GOTCHAS:
- Forgetting to shift labels → misaligned supervision.
- Using positional embeddings beyond max_seq_len → index out of bounds.
- Normalization placement (Pre-Norm vs Post-Norm) affects training stability.
- Without dropout/regularization → overfitting on small corpora.
- In large-scale training, must implement KV cache for efficient inference.

---

INTUITION:
- Embeddings tell the model “what token this is”.
- Positional encodings tell the model “where it is in the sequence”.
- Attention tells the model “which other tokens matter right now”.
- FFN transforms the attended representation into richer features.
- Residuals + Norms stabilize training and let information flow.
- The final linear layer predicts the next token.

---

REAL-WORLD NOTES:
- GPT-2: 12 layers, 768 hidden, 117M params.
- GPT-3: up to 175B params, same basic structure scaled up.
- LLaMA/Mistral: Pre-Norm + RMSNorm + SwiGLU, RoPE embeddings.
- TinyGPT (few layers, small vocab) is often used in tutorials
to illustrate the architecture end-to-end.
"""
