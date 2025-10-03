# Architectures in LLM Modeling
"""
WHAT/WHY:
Different neural architectures have been explored for large language models.
The architecture determines how sequences are processed, how dependencies are modeled,
and what tasks the model excels at.

---

1. RNN-based Models (Recurrent Neural Networks, LSTMs, GRUs)
- Process tokens sequentially, maintaining hidden states.
- Can, in principle, model arbitrarily long dependencies, but in practice suffer from vanishing/exploding gradients.
- Limited parallelization → training is slow at scale.

MATH:
h_t = f(W_x x_t + W_h h_{t-1})
y_t = softmax(W_y h_t)

SHAPES:
Inputs:  [B, T]
Hidden states: [B, T, H]

EXAMPLES:
Early seq2seq models, ELMo

GOTCHAS:
- Hard to scale to billions of tokens.
- Long-term dependencies poorly captured without tricks.

INTUITION:
Like reading one token at a time, carrying a “memory state” forward.

---

2. CNN-based Models (Convolutional Sequence Models)
- Use convolutions over tokens (1D CNNs) to capture local dependencies.
- Deep stacks extend receptive field to cover longer context.
- Highly parallelizable, but less flexible than attention for long-range dependencies.

MATH:
h_t = Conv1D(x_{t-k:t+k})

SHAPES:
Inputs:  [B, T, d_model]
Outputs: [B, T, d_model]

EXAMPLES:
ByteNet, WaveNet

GOTCHAS:
- Requires many layers to model long dependencies.
- Superseded by attention in most LLMs.

INTUITION:
Sliding window over sequence; learns local patterns well, but struggles with global context.

---

3. Transformer Encoder-Only
- Stacks of bidirectional self-attention + feed-forward layers.
- Excellent at capturing contextual representations.
- Cannot autoregressively generate text without modification.

MATH:
h_t = SelfAttention(x_{1:T}) + FeedForward(h_t)

SHAPES:
Inputs:  [B, T, d_model]
Outputs: [B, T, d_model]

EXAMPLES:
BERT, RoBERTa, DeBERTa

GOTCHAS:
- Not generative, but strong for classification and embedding tasks.

INTUITION:
Every token sees every other token (bidirectional). Great for “understanding.”

---

4. Transformer Decoder-Only
- Stacks of causal self-attention + feed-forward layers.
- Autoregressive: predicts next token given past.
- Forms the backbone of modern LLMs.

MATH:
h_t = CausalSelfAttention(x_{≤t}) + FeedForward(h_t)

SHAPES:
Inputs:  [B, T, d_model]
Outputs: [B, T, d_model]
Logits:  [B, T, V]

EXAMPLES:
GPT family (GPT-2, GPT-3, GPT-4), LLaMA, Mistral

GOTCHAS:
- Context length limited by quadratic attention cost.
- Requires causal mask to prevent information leak.

INTUITION:
Forecasting machine: generate text one token at a time.

---

5. Transformer Encoder-Decoder (Seq2Seq Transformers)
- Encoder with bidirectional attention, decoder with causal attention.
- Encoder produces context representations; decoder generates output autoregressively.
- Flexible: can handle text-to-text mapping tasks.

MATH:
h_enc = Encoder(x_{1:T_enc})
y_t = Decoder(y_{≤t}, h_enc)

SHAPES:
Encoder input: [B, T_enc, d_model]
Decoder output: [B, T_dec, d_model]

EXAMPLES:
T5, BART, mT5

GOTCHAS:
- More complex than pure encoder or decoder.
- Training requires parallel input-output data.

INTUITION:
Translator: read one sequence, produce another.

---

6. Sparse / Efficient Transformer Variants
- Replace full attention (O(T²)) with sparse or approximate attention to scale to longer contexts.
- Techniques: Local attention, sliding windows, low-rank factorization, kernelized attention.

EXAMPLES:
Longformer, BigBird, Performer, FlashAttention

GOTCHAS:
- May trade off accuracy for efficiency.
- Implementation details critical for stability.

INTUITION:
Approximate “who to attend to” so we can handle much longer documents efficiently.

---

SUMMARY:
- RNNs: sequential memory, but scale poorly.
- CNNs: local context, parallel, but limited long-range.
- Encoder-only Transformers: bidirectional, strong for understanding.
- Decoder-only Transformers: causal, strong for generation (modern LLM standard).
- Encoder-Decoder Transformers: flexible for translation/summarization.
- Sparse/Efficient Transformers: extend context length with approximation tricks.
"""
