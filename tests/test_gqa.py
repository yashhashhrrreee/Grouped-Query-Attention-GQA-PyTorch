"""
test_gqa.py
-----------
Unit tests covering:
  - repeat_kv shape and data correctness
  - RotaryEmbedding output shape and dtype preservation
  - GroupedQueryAttention: MHA / GQA / MQA configurations
  - Causal mask behaviour
  - RoPE integration
  - Model surgery (convert_mha_to_gqa)
  - memory_comparison_table correctness

Run with:
    pytest tests/test_gqa.py -v
"""

import math
import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gqa.attention import repeat_kv, RotaryEmbedding, GroupedQueryAttention
from gqa.surgery   import convert_mha_to_gqa, memory_comparison_table


# ===========================================================================
# repeat_kv
# ===========================================================================

class TestRepeatKV:

    def test_output_shape(self):
        x = torch.randn(1, 2, 4, 8)
        out = repeat_kv(x, num_repeats=2)
        assert out.shape == (1, 4, 4, 8), f"Expected (1,4,4,8) got {out.shape}"

    def test_no_repeat_returns_same(self):
        x = torch.randn(1, 4, 6, 16)
        out = repeat_kv(x, num_repeats=1)
        assert out is x, "num_repeats=1 should return the same object"

    def test_data_correctness(self):
        """Each KV head should be repeated contiguously."""
        x = torch.randn(1, 2, 4, 8)
        out = repeat_kv(x, num_repeats=3)
        # head 0 of input → heads 0,1,2 of output
        for i in range(3):
            assert torch.allclose(out[:, i, :, :], x[:, 0, :, :])
        # head 1 of input → heads 3,4,5 of output
        for i in range(3, 6):
            assert torch.allclose(out[:, i, :, :], x[:, 1, :, :])

    def test_large_repeat(self):
        x = torch.randn(2, 1, 128, 64)
        out = repeat_kv(x, num_repeats=32)
        assert out.shape == (2, 32, 128, 64)


# ===========================================================================
# RotaryEmbedding
# ===========================================================================

class TestRotaryEmbedding:

    def test_output_shape(self):
        rope = RotaryEmbedding(head_dim=64)
        q = torch.randn(1, 8, 16, 64)
        k = torch.randn(1, 8, 16, 64)
        q_r, k_r = rope(q, k, seq_len=16)
        assert q_r.shape == q.shape
        assert k_r.shape == k.shape

    def test_dtype_preserved(self):
        rope = RotaryEmbedding(head_dim=64)
        q = torch.randn(1, 4, 10, 64).half()
        k = torch.randn(1, 4, 10, 64).half()
        q_r, k_r = rope(q, k, seq_len=10)
        assert q_r.dtype == torch.float16
        assert k_r.dtype == torch.float16

    def test_rotation_changes_values(self):
        rope = RotaryEmbedding(head_dim=64)
        q = torch.randn(1, 4, 8, 64)
        k = torch.randn(1, 4, 8, 64)
        q_r, k_r = rope(q, k, seq_len=8)
        assert not torch.allclose(q, q_r), "RoPE should change Q values"


# ===========================================================================
# GroupedQueryAttention
# ===========================================================================

class TestGQAModule:

    @pytest.fixture
    def default_cfg(self):
        return dict(d_in=512, d_out=512, num_heads=8, num_kv_groups=2)

    def test_output_shape(self, default_cfg):
        layer = GroupedQueryAttention(**default_cfg)
        x     = torch.randn(1, 10, 512)
        out   = layer(x)
        assert out.shape == (1, 10, 512)

    def test_mha_equivalent(self):
        """num_kv_groups == num_heads → standard MHA."""
        layer = GroupedQueryAttention(d_in=256, d_out=256, num_heads=4, num_kv_groups=4)
        x = torch.randn(2, 20, 256)
        out = layer(x)
        assert out.shape == (2, 20, 256)

    def test_mqa_equivalent(self):
        """num_kv_groups == 1 → MQA."""
        layer = GroupedQueryAttention(d_in=256, d_out=256, num_heads=4, num_kv_groups=1)
        x = torch.randn(2, 20, 256)
        out = layer(x)
        assert out.shape == (2, 20, 256)

    def test_invalid_groups_raises(self):
        with pytest.raises(ValueError):
            GroupedQueryAttention(d_in=256, d_out=256, num_heads=8, num_kv_groups=3)

    def test_causal_mask_shape(self):
        layer = GroupedQueryAttention(
            d_in=128, d_out=128, num_heads=4, num_kv_groups=2,
            causal=True, max_seq_len=64
        )
        x   = torch.randn(1, 32, 128)
        out = layer(x)
        assert out.shape == (1, 32, 128)

    def test_causal_mask_is_causal(self):
        """
        The output at position t should not depend on positions > t.
        We verify by zeroing out tokens t+1..T and checking token t is unchanged.
        """
        torch.manual_seed(0)
        layer = GroupedQueryAttention(
            d_in=64, d_out=64, num_heads=2, num_kv_groups=1,
            causal=True, max_seq_len=16
        ).eval()

        x = torch.randn(1, 8, 64)

        with torch.no_grad():
            out_full = layer(x)

            x_masked        = x.clone()
            x_masked[:, 5:] = 0.0
            out_masked = layer(x_masked)

        # Positions 0–4 must be identical
        assert torch.allclose(out_full[:, :5], out_masked[:, :5], atol=1e-5), \
            "Causal masking broken: future tokens leaked into past positions."

    def test_rope_integration(self):
        layer = GroupedQueryAttention(
            d_in=128, d_out=128, num_heads=4, num_kv_groups=2,
            use_rope=True, max_seq_len=64
        )
        x   = torch.randn(1, 32, 128)
        out = layer(x)
        assert out.shape == (1, 32, 128)

    def test_config_property(self, default_cfg):
        layer = GroupedQueryAttention(**default_cfg)
        cfg   = layer.config
        assert cfg["num_heads"]     == 8
        assert cfg["num_kv_groups"] == 2
        assert cfg["repeat_factor"] == 4

    def test_batch_invariance(self, default_cfg):
        """Output for item 0 in a batch must match single-item forward pass."""
        layer = GroupedQueryAttention(**default_cfg).eval()
        torch.manual_seed(7)
        x     = torch.randn(4, 10, 512)
        with torch.no_grad():
            out_batch  = layer(x)
            out_single = layer(x[:1])
        assert torch.allclose(out_batch[:1], out_single, atol=1e-5)


# ===========================================================================
# Model Surgery
# ===========================================================================

class TestSurgery:

    def test_manual_mean_pooling(self):
        """Heads [1, 3] grouped by 2 → averaged to [2]."""
        w       = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
        result  = convert_mha_to_gqa(w, group_size=2)
        expected = torch.tensor([[2.0, 6.0]])
        assert torch.allclose(result, expected), f"Got {result}"

    def test_output_shape(self):
        layer   = torch.nn.Linear(10, 32, bias=False)
        w_in    = layer.weight.t()          # (10, 32)
        compressed = convert_mha_to_gqa(w_in, group_size=4)
        assert compressed.shape == (10, 8), f"Expected (10,8) got {compressed.shape}"

    def test_pytorch_layout_flag(self):
        """weight_is_pytorch_linear=True should handle (D_out, D_in) layout."""
        layer      = torch.nn.Linear(16, 32, bias=False)
        compressed = convert_mha_to_gqa(
            layer.weight, group_size=4, weight_is_pytorch_linear=True
        )
        # Original: (32, 16)  →  after compression: (8, 16)
        assert compressed.shape == (8, 16)

    def test_indivisible_raises(self):
        w = torch.randn(4, 10)
        with pytest.raises(ValueError):
            convert_mha_to_gqa(w, group_size=3)

    def test_group1_is_identity(self):
        """group_size=1 should not change the weight."""
        w = torch.randn(8, 16)
        assert torch.allclose(convert_mha_to_gqa(w, group_size=1), w)


# ===========================================================================
# Memory Table
# ===========================================================================

class TestMemoryTable:

    def test_table_has_three_rows(self):
        rows = memory_comparison_table()
        assert len(rows) == 3

    def test_mha_is_baseline(self):
        rows = memory_comparison_table()
        mha  = next(r for r in rows if r["Architecture"] == "MHA")
        assert mha["Reduction"] == "1x"

    def test_gqa_is_4x(self):
        rows = memory_comparison_table(num_query_heads=32)
        gqa  = next(r for r in rows if r["Architecture"] == "GQA")
        assert gqa["Reduction"] == "4x"

    def test_mqa_is_32x(self):
        rows = memory_comparison_table(num_query_heads=32)
        mqa  = next(r for r in rows if r["Architecture"] == "MQA")
        assert mqa["Reduction"] == "32x"
