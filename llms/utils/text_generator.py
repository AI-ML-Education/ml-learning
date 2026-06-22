"""Text generation + tokenization helpers from the from-scratch notebooks.

Collected from `03_transformer_block.ipynb` and `05_decoding.ipynb`:

    - generate_text_simple : greedy (argmax) decoding
    - generate_text        : greedy by default, with optional temperature /
                             top-k sampling and early stopping on an eos token
    - text_to_token_ids    : encode a string into a batched tensor of ids
    - token_ids_to_text    : decode a batched tensor of ids back into text
"""

import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Greedy decoding: at each step pick the most likely next token."""
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate_text(model, idx, max_new_tokens, context_size,
                  temperature=0.0, top_k=None, eos_id=None):
    """Decode text from ``model``.

    Defaults to greedy decoding (``temperature=0.0``), so calling this with
    only ``model, idx, max_new_tokens, context_size`` behaves like
    :func:`generate_text_simple`. Set ``temperature`` and/or ``top_k`` to
    enable sampling.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top-k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        # Apply temperature scaling + sampling
        if temperature > 0.0:
            logits = logits / temperature

            # numerical stability: subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            # Greedy: pick the token with the highest logit
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Stop early if the end-of-sequence token is produced
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    """Encode ``text`` into a (1, n_tokens) tensor of token ids."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Decode a (1, n_tokens) tensor of token ids back into a string."""
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
