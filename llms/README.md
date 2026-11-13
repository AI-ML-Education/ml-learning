# llms/

Hands-on **"build a GPT from scratch"** track — I tear the architecture down and
re-implement each piece myself so I actually understand it. Narrated theory lessons
that back up the build live alongside, in [`theory/`](theory) (reading) and
[`tbd/`](tbd) (theory seeds queued to become notebooks).

Loosely follows Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

## Suggested reading order

### Foundations (math & building blocks — read first, or dip in as needed)
- [layers.ipynb](layers.ipynb) — what a "layer" really is: neurons, `nn.Linear`, and how they stack.
- [backprop_gradient_descent.ipynb](backprop_gradient_descent.ipynb) — manual forward/backward through an MLP, with a gradient check against autograd.
- [sigmoid_to_cross_entropy.ipynb](sigmoid_to_cross_entropy.ipynb) — sigmoid → maximum likelihood → cross-entropy, plus the PyTorch loss cheatsheet.
- [tokenization_bpe_and_wordpiece_encoding.ipynb](tokenization_bpe_and_wordpiece_encoding.ipynb) — Byte-Pair Encoding and WordPiece built from scratch.
- [basics_llm.ipynb](basics_llm.ipynb) — the big picture: the components of an LLM and how they map to the code we build below.

### Build a GPT (the main path)
Numbered by where each piece sits in the architecture / training lifecycle:
1. [01_data_prep.ipynb](01_data_prep.ipynb) — text → tokens → vocab/BPE → input–target batches with a `DataLoader`.
2. [02_attention.ipynb](02_attention.ipynb) — self-attention → causal masking → multi-head attention.
3. [03_transformer_block.ipynb](03_transformer_block.ipynb) — LayerNorm, GELU feed-forward, residual connections, the transformer block, the full `GPTModel`, and greedy text generation.
   - [03b_shape_tracing.ipynb](03b_shape_tracing.ipynb) — *companion to step 3*: trace tensor shapes end-to-end through the model.
4. [04_pretraining.ipynb](04_pretraining.ipynb) — the training loop on real text: LM cross-entropy, train/val loss, sampling during training, checkpoints. Pretrains the from-scratch GPT on *The Verdict* and shows overfitting on a tiny corpus.
5. [05_decoding.ipynb](05_decoding.ipynb) — decoding strategies: greedy, temperature scaling, and top-k sampling.
6. [06_finetuning_classification.ipynb](06_finetuning_classification.ipynb) — fine-tune the GPT into a spam classifier.

## Shared code — [`utils/`](utils)
Reusable pieces extracted from the notebooks so later notebooks just import them:
- `utils/gpt_model.py` — `GELU`, `LayerNorm`, `FeedForward`, `MultiHeadAttention`, `TransformerBlock`, `GPTModel`.
- `utils/text_generator.py` — `generate_text` / `generate_text_simple`, `text_to_token_ids`, `token_ids_to_text`.

> Imports are written as `from utils import ...`, so run the notebooks **with `llms/` as the working directory**.

## Theory — [`theory/`](theory)
Narrated `.py` lessons for the concepts that sit *around* the build (read as needed):
- `theory/llm_system_overview.py` — the end-to-end pipeline at a glance.
- `theory/llm_architecture.py` — how decoder-only GPTs compare to RNN/CNN/seq2seq.
- `theory/modeling_in_llms.py` — training objectives: causal LM, masked LM, seq2seq, span corruption.
- `theory/training_mechanics_and_efficiency.py` — AdamW, LR schedules, grad clipping, mixed precision, checkpointing (reference for [04_pretraining.ipynb](04_pretraining.ipynb)).
- `theory/determinism_and_non_determinism.py` — reproducibility, seeds, float non-associativity.
- `theory/putting_it_all_together.py` — a unified TinyGPT blueprint tying the pieces together.

## Roadmap — [`tbd/`](tbd)
Theory seeds (`.py`) and stubs queued to grow into hands-on notebooks:
- `tbd/kv_cache.py` — key/value caching for fast inference (no notebook yet).
- `tbd/positional_information.py` — sinusoidal / learned / RoPE encodings.
- `tbd/lora_and_qlora.py` (+ `lora_basics.ipynb`) — parameter-efficient fine-tuning.
- `tbd/evaluation.py` (+ `llm_benchmarks.ipynb`) — perplexity, BLEU/ROUGE/BERTScore, task metrics.
- Other stubs: ViT, adapters, pruning, quantization, reasoning, RL for LLMs.
  Note that `rl/14`–`rl/18` already cover much of the RL-for-LLMs material.

## Notes
- Generated/downloaded artifacts (the spam dataset zip, extracted data under
  `../data/`, and plot PDFs) are git-ignored — the notebooks recreate them on run.
