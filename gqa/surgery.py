"""
surgery.py
----------
Utilities for converting Multi-Head Attention (MHA) checkpoints into
Grouped-Query Attention (GQA) checkpoints via mean-pooling, as described in:

  Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints" (2023) — Section 2.1 (Uptraining)

Key idea: instead of training GQA from scratch, we can *compress* an existing
MHA model by averaging groups of key/value heads together. Mean-pooling
preserves the most information from the pre-trained model compared to
alternatives like selecting a single head or random re-initialization.
"""

import torch
from typing import Optional


def convert_mha_to_gqa(
    mha_weight: torch.Tensor,
    group_size: int,
    weight_is_pytorch_linear: bool = False,
) -> torch.Tensor:
    """
    Compress an MHA key/value weight matrix into a GQA weight matrix by
    mean-pooling groups of heads. No for-loops — pure tensor operations.

    Mathematical view:
        MHA weight shape : (D_in, H_MHA × D_head)
        GQA weight shape : (D_in, H_GQA × D_head)   where H_GQA = H_MHA / group_size

    Args:
        mha_weight               : Weight tensor of shape (D_in, D_out).
        group_size               : How many MHA heads to pool into one GQA head.
        weight_is_pytorch_linear : If True, the tensor is in PyTorch's native
                                   (D_out, D_in) layout and will be transposed
                                   automatically before and after conversion.

    Returns:
        Compressed weight of shape (D_in, D_out // group_size).

    Raises:
        ValueError: if D_out is not divisible by group_size.
    """
    if weight_is_pytorch_linear:
        mha_weight = mha_weight.t()   # (D_out, D_in) → (D_in, D_out)

    d_in, d_out = mha_weight.shape

    if d_out % group_size != 0:
        raise ValueError(
            f"Output dimension {d_out} is not divisible by group_size {group_size}."
        )

    new_heads = d_out // group_size

    # Reshape: (D_in, new_heads, group_size)  →  mean over group_size
    compressed = mha_weight.view(d_in, new_heads, group_size).mean(dim=2)
    # Shape: (D_in, new_heads)

    if weight_is_pytorch_linear:
        compressed = compressed.t()   # back to (D_out_new, D_in)

    return compressed


def compress_attention_layer(
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    group_size: int,
    weight_is_pytorch_linear: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper: compress both K and V weight matrices of an attention
    layer in one call.

    Args:
        k_weight, v_weight       : Key and Value projection weights.
        group_size               : Heads to pool per GQA group.
        weight_is_pytorch_linear : See convert_mha_to_gqa.

    Returns:
        (compressed_k, compressed_v)
    """
    return (
        convert_mha_to_gqa(k_weight, group_size, weight_is_pytorch_linear),
        convert_mha_to_gqa(v_weight, group_size, weight_is_pytorch_linear),
    )


def kv_cache_bytes(
    num_kv_heads: int,
    seq_len: int = 4096,
    batch_size: int = 1,
    head_dim: int = 128,
    bytes_per_element: int = 2,   # float16
) -> int:
    """
    Calculate the total KV cache size in bytes for a single transformer layer.

    Formula:
        2 (K and V) × B × T × H_KV × D_head × bytes_per_element
    """
    return 2 * batch_size * seq_len * num_kv_heads * head_dim * bytes_per_element


def memory_comparison_table(
    num_query_heads: int = 32,
    seq_len: int = 4096,
    batch_size: int = 1,
    d_model: int = 4096,
) -> list[dict]:
    """
    Build a comparison table of KV cache sizes for MHA, GQA, and MQA.

    Returns a list of dicts, one row per configuration.
    """
    head_dim = d_model // num_query_heads
    configs  = [
        ("MHA",  num_query_heads),
        ("GQA",  num_query_heads // 4),   # Llama 3 style
        ("MQA",  1),
    ]
    mha_bytes = kv_cache_bytes(num_query_heads, seq_len, batch_size, head_dim)
    rows = []
    for name, h_kv in configs:
        total = kv_cache_bytes(h_kv, seq_len, batch_size, head_dim)
        rows.append({
            "Architecture"   : name,
            "H_Q"            : num_query_heads,
            "H_KV"           : h_kv,
            "KV Cache (MB)"  : round(total / (1024 ** 2), 2),
            "Reduction"      : f"{mha_bytes // total}x",
        })
    return rows
