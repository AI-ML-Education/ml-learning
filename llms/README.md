# llms/

Hands-on **"build a GPT from scratch"** track — I tear the architecture down and
re-implement each piece myself so I actually understand it. The notebooks here are
the practical companion to the narrated theory in [`../ml/llm_fundamentals/`](../ml/llm_fundamentals)
(which explains the same ideas as runnable `.py` lessons).

Loosely follows Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

## Suggested reading order

### Foundations (math & building blocks — read first, or dip in as needed)
- [layers.ipynb](layers.ipynb) — what a "layer" really is: neurons, `nn.Linear`, and how they stack.
- [backprop_gradient_descent.ipynb](backprop_gradient_descent.ipynb) — manual forward/backward through an MLP, with a gradient check against autograd.
- [logistic_sigmoid_cross_entropy_maximization.ipynb](logistic_sigmoid_cross_entropy_maximization.ipynb) — sigmoid → maximum likelihood → cross-entropy, plus the PyTorch loss cheatsheet.
- [tokenization_bpe_and_wordpiece_encoding.ipynb](tokenization_bpe_and_wordpiece_encoding.ipynb) — Byte-Pair Encoding and WordPiece built from scratch.
- [basics_llm.ipynb](basics_llm.ipynb) — the big picture: the components of an LLM and how they map to the code we build below.

### Build a GPT (the main path)
Numbered by where each piece sits in the architecture / training lifecycle:
1. [01_data_prep.ipynb](01_data_prep.ipynb) — text → tokens → vocab/BPE → input–target batches with a `DataLoader`.
2. [02_attention.ipynb](02_attention.ipynb) — self-attention → causal masking → multi-head attention.
3. [03_transformer_block.ipynb](03_transformer_block.ipynb) — LayerNorm, GELU feed-forward, residual connections, the transformer block, the full `GPTModel`, and greedy text generation.
   - [shape_tracing.ipynb](shape_tracing.ipynb) — *companion to step 3*: trace tensor shapes end-to-end through the model.
4. [04_pretraining.ipynb](04_pretraining.ipynb) — the training loop on real text: LM cross-entropy, train/val loss, sampling during training, checkpoints. *(Stub — planned build-out, Raschka ch. 5.)*
5. [05_decoding.ipynb](05_decoding.ipynb) — decoding strategies: greedy, temperature scaling, and top-k sampling.
6. [06_finetuning_classification.ipynb](06_finetuning_classification.ipynb) — fine-tune the GPT into a spam classifier.

## Shared code — [`utils/`](utils)
Reusable pieces extracted from the notebooks so later notebooks just import them:
- `utils/gpt_model.py` — `GELU`, `LayerNorm`, `FeedForward`, `MultiHeadAttention`, `TransformerBlock`, `GPTModel`.
- `utils/text_generator.py` — `generate_text` / `generate_text_simple`, `text_to_token_ids`, `token_ids_to_text`.

> Imports are written as `from utils import ...`, so run the notebooks **with `llms/` as the working directory**.

## Roadmap — [`tbd/`](tbd)
Planned deep-dives (stubs for now): LoRA, ViT, adapters, benchmarks, pruning,
quantization, reasoning, and RL for LLMs. Note that `rl/14`–`rl/18` already cover
much of the RL-for-LLMs material.

## Notes
- Generated/downloaded artifacts (the spam dataset zip, extracted data under
  `../data/`, and plot PDFs) are git-ignored — the notebooks recreate them on run.
