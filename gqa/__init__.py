"""
gqa
===
A clean, research-oriented implementation of Grouped-Query Attention (GQA).

Modules
-------
attention : repeat_kv, RotaryEmbedding, GroupedQueryAttention
surgery   : MHA → GQA checkpoint compression utilities
"""

from .attention import repeat_kv, RotaryEmbedding, GroupedQueryAttention
from .surgery   import convert_mha_to_gqa, compress_attention_layer, memory_comparison_table

__all__ = [
    "repeat_kv",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "convert_mha_to_gqa",
    "compress_attention_layer",
    "memory_comparison_table",
]
