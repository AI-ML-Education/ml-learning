"""Reusable building blocks extracted from the from-scratch LLM notebooks.

These mirror the implementations developed step by step in the notebooks
(`02_attention.ipynb`, `03_transformer_block.ipynb`, `05_decoding.ipynb`) so that later
notebooks can simply import them instead of re-defining everything.
"""

from .gpt_model import (
    GELU,
    LayerNorm,
    FeedForward,
    MultiHeadAttention,
    TransformerBlock,
    GPTModel,
)
from .text_generator import (
    generate_text,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)

__all__ = [
    "GELU",
    "LayerNorm",
    "FeedForward",
    "MultiHeadAttention",
    "TransformerBlock",
    "GPTModel",
    "generate_text",
    "generate_text_simple",
    "text_to_token_ids",
    "token_ids_to_text",
]
