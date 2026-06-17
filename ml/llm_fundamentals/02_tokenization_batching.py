# Tokenization and Batching
"""
WHAT/WHY:
Text cannot be directly fed into a neural network. We must first convert it into
a numerical form. Tokenization maps raw text → discrete integer IDs.
Batching then arranges multiple token sequences together so they can be processed
in parallel on GPU hardware.
(By the way inference servers like Triton/vLLM also create batches of user input sequences for efficiency.)

Tokenization ensures the model has a finite, manageable vocabulary.
Batching ensures efficient training by utilizing parallelism.

TYPES OF TOKENIZATION:
- Word-level: splits by words (large vocab, many OOV (out-of-vocab) issues).
- Subword-level (BPE, WordPiece, UnigramLM): balances vocab size with coverage.
- Character-level: robust, but long sequences.
- Byte-level (e.g., GPT-2 BPE): covers all text without OOV.

MATH / SHAPES:
Suppose we have input text: "Hello world".
Tokenizer converts this into token IDs: [15496, 995].

Let:
- V = vocabulary size (e.g., 50,000).
- B = batch size (# of sequences).
- T = sequence length (# of tokens per sequence).

Final batched input shape: [B, T].
Each element is an integer ID in range [0, V).

EXAMPLE:
Text batch:
[
  "Hello world",
  "Machine learning is fun"
]

Token IDs (after padding to equal length):
[
  [15496,   995,     0,     0],
  [  5295,  3187,   318,  2188]
]

Here:
- 0 is the PAD token.
- Shape: [B=2, T=4].

BATCHING FOR CAUSAL LM:
We need input-target pairs (x, y):
- Input x = sequence up to t
- Label y = sequence shifted left by 1 (the next token)

So:
Input IDs (x):  [15496,   995,     0,     0]
Label IDs (y):  [  995,     0,     0,  -100]

Notice:
- At position 0, model sees "Hello" → predict "world".
- At position 1, model sees "world" → predict PAD.
- At position 2, sees PAD → predict PAD.
- At last position, no "next token", so we set label=-100 (ignore).

WHY SHIFT?
Because causal LM is trained to predict *next* token from context.

PYTORCH NOTES:
- Inputs: `torch.long` tensor of IDs, shape [B, T].
- Labels: same shape [B, T], but shifted.
- Dataloader can yield (x, y) pairs directly.
- For contiguous text corpora (like Wiki), one can concatenate text and slice
  long chunks of size T to avoid padding altogether.

EXAMPLE FUNCTION:
make_causal_batches(tokens, block_size, batch_size)
→ returns (x, y) where:
  - x: [B, T] current tokens
  - y: [B, T] tokens shifted left by 1
  - labels filled with -100 where no target exists (e.g., end of sequence).

GOTCHAS:
- Forgetting to shift labels.
- Incorrectly aligning pad tokens (must set them to -100 in labels).
- Variable-length sequences within batch require attention masks.
- In causal LM, sometimes "packed sequences" are used to avoid waste from padding.
"""
