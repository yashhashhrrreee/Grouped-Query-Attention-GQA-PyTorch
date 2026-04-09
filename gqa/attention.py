"""
attention.py
------------
Core attention implementations:
  - repeat_kv      : broadcasting utility for GQA
  - RotaryEmbedding: RoPE positional encoding (Su et al., 2021)
  - GroupedQueryAttention: full GQA module with optional RoPE + causal mask
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 1.  repeat_kv
# ---------------------------------------------------------------------------

def repeat_kv(x: torch.Tensor, num_repeats: int) -> torch.Tensor:
    """
    Expand KV heads to match the number of Query heads without copying memory.

    Args:
        x           : (B, n_kv_heads, T, head_dim)
        num_repeats : H_Q // H_KV

    Returns:
        Tensor of shape (B, n_kv_heads * num_repeats, T, head_dim)
    """
    if num_repeats == 1:
        return x

    B, n_kv, T, D = x.shape
    # Insert a broadcast dimension, expand, then reshape
    return (
        x[:, :, None, :, :]                                    # (B, n_kv, 1, T, D)
         .expand(B, n_kv, num_repeats, T, D)                   # (B, n_kv, R, T, D)
         .reshape(B, n_kv * num_repeats, T, D)                 # (B, H_Q, T, D)
    )


# ---------------------------------------------------------------------------
# 2.  Rotary Positional Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as used in Llama / GPT-NeoX.

    Reference: Su et al. "RoFormer: Enhanced Transformer with Rotary Position
    Embedding" (2021) — https://arxiv.org/abs/2104.09864

    The key insight: instead of adding positional info, we *rotate* the Q and K
    vectors. Relative position is then captured automatically in the dot product.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10_000):
        super().__init__()
        # Compute inverse frequencies — one per pair of head dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute cos/sin cache for speed
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)           # (T, D/2)
        emb   = torch.cat([freqs, freqs], dim=-1)       # (T, D)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])  # (1,1,T,D)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension by -90°."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ---------------------------------------------------------------------------
# 3.  Grouped-Query Attention
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) as described in:
      Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models
      from Multi-Head Checkpoints" (2023) — https://arxiv.org/abs/2305.13245

    Supports:
      - Standard GQA (H_KV > 1, H_KV < H_Q)
      - MHA equivalent (num_kv_groups == num_heads)
      - MQA equivalent (num_kv_groups == 1)
      - Optional RoPE positional encoding
      - Optional causal (autoregressive) masking

    Args:
        d_in          : Input embedding dimension.
        d_out         : Output embedding dimension.
        num_heads     : Number of query heads (H_Q).
        num_kv_groups : Number of key/value heads (H_KV). Must divide num_heads.
        use_rope      : If True, apply Rotary Positional Embedding to Q and K.
        causal        : If True, apply a causal mask (for autoregressive decoding).
        dropout       : Attention dropout probability (0 = disabled).
        max_seq_len   : Maximum sequence length (used for RoPE cache and causal mask).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        num_kv_groups: int,
        use_rope: bool = False,
        causal: bool = False,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
    ):
        super().__init__()

        if num_heads % num_kv_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_groups ({num_kv_groups})."
            )

        self.num_heads     = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim      = d_out // num_heads
        self.repeat_factor = num_heads // num_kv_groups
        self.causal        = causal
        self.scale         = 1.0 / math.sqrt(self.head_dim)

        kv_dim = num_kv_groups * self.head_dim

        self.q_proj = nn.Linear(d_in, d_out,    bias=False)
        self.k_proj = nn.Linear(d_in, kv_dim,   bias=False)
        self.v_proj = nn.Linear(d_in, kv_dim,   bias=False)
        self.o_proj = nn.Linear(d_out, d_out,   bias=False)

        self.attn_drop = nn.Dropout(dropout)

        # Optional RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len) if use_rope else None

        # Causal mask buffer (lower-triangular)
        if causal:
            mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
            self.register_buffer("causal_mask", mask)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x         : (B, T, d_in)
            attn_mask : Optional boolean mask of shape (T, T) or (B, 1, T, T).
                        True = keep, False = mask out.

        Returns:
            (B, T, d_out)
        """
        B, T, _ = x.shape

        # 1. Linear projections
        q = self.q_proj(x)   # (B, T, d_out)
        k = self.k_proj(x)   # (B, T, kv_dim)
        v = self.v_proj(x)   # (B, T, kv_dim)

        # 2. Reshape into heads
        q = q.view(B, T, self.num_heads,     self.head_dim).transpose(1, 2)  # (B, H_Q,  T, D_h)
        k = k.view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)  # (B, H_KV, T, D_h)
        v = v.view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)  # (B, H_KV, T, D_h)

        # 3. Optional RoPE
        if self.rope is not None:
            q, k = self.rope(q, k, T)

        # 4. GQA broadcast: repeat K and V to match Q head count
        k = repeat_kv(k, self.repeat_factor)   # (B, H_Q, T, D_h)
        v = repeat_kv(v, self.repeat_factor)   # (B, H_Q, T, D_h)

        # 5. Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) * self.scale   # (B, H_Q, T, T)

        # 6. Apply masks
        if self.causal:
            causal = self.causal_mask[:T, :T]              # (T, T)
            scores = scores.masked_fill(~causal, float("-inf"))

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # 7. Aggregate values
        out = attn_weights @ v                             # (B, H_Q, T, D_h)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_out)

        return self.o_proj(out)

    # ------------------------------------------------------------------
    @property
    def config(self) -> dict:
        """Human-readable summary of the layer configuration."""
        return {
            "num_heads"    : self.num_heads,
            "num_kv_groups": self.num_kv_groups,
            "head_dim"     : self.head_dim,
            "repeat_factor": self.repeat_factor,
            "rope"         : self.rope is not None,
            "causal"       : self.causal,
        }
