# Grouped-Query Attention (GQA)

A clean, research-oriented PyTorch implementation of **Grouped-Query Attention** — the attention mechanism used in Llama 2/3, Mistral, and other modern LLMs to solve the KV-cache memory bottleneck.

> **Paper:** [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., Google Research (2023)

---

## What is GQA?

When an LLM generates text token-by-token, it caches the Key (K) and Value (V) vectors for every past token in GPU memory (the **KV Cache**). For long sequences, this cache dominates VRAM and makes generation *memory-bandwidth-bound*, not compute-bound.

GQA solves this by reducing the number of KV heads relative to Query heads:

| Architecture | H_Q | H_KV | KV Cache (1 layer, T=4096) | Quality |
|---|---|---|---|---|
| **MHA** (GPT-2, BERT) | 32 | 32 | 64 MB | ✅ Best |
| **GQA** (Llama 3) | 32 | 8 | 16 MB | ✅ Near-MHA |
| **MQA** (extreme) | 32 | 1 | 2 MB | ⚠️ Degraded |

GQA achieves a **4× memory reduction** with quality nearly identical to full MHA.

---

## Features

- **`repeat_kv`** — broadcast KV heads to match Q heads using memory-efficient views (no copy)
- **`RotaryEmbedding`** — RoPE positional encoding (Su et al., 2021) as used in Llama / Mistral
- **`GroupedQueryAttention`** — full GQA module supporting:
  - All three regimes: MHA / GQA / MQA (controlled by `num_kv_groups`)
  - Optional RoPE positional encoding
  - Optional causal (autoregressive) masking
  - Attention dropout
- **`convert_mha_to_gqa`** — model surgery via mean-pooling (no for-loops)
- **Benchmark script** — latency and memory comparison across MHA / GQA / MQA
- **25 unit tests** — shape, data correctness, causal mask leak test, and more

---

## Project Structure

```
gqa_project/
├── gqa/
│   ├── __init__.py
│   ├── attention.py      # repeat_kv, RotaryEmbedding, GroupedQueryAttention
│   └── surgery.py        # MHA → GQA checkpoint compression
├── benchmarks/
│   └── benchmark.py      # Latency + memory benchmark (CLI)
├── notebooks/
│   └── demo.ipynb        # End-to-end interactive walkthrough
├── tests/
│   └── test_gqa.py       # 25 unit tests (pytest)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/grouped-query-attention
cd grouped-query-attention
pip install -r requirements.txt
```

### Run the demo notebook
```bash
jupyter notebook notebooks/demo.ipynb
```

### Use the module directly

```python
import torch
from gqa.attention import GroupedQueryAttention

# GQA with RoPE and causal masking (Llama-style)
layer = GroupedQueryAttention(
    d_in=512,
    d_out=512,
    num_heads=8,
    num_kv_groups=2,   # 4x fewer KV heads than Q heads
    use_rope=True,
    causal=True,
)

x   = torch.randn(1, 128, 512)   # (batch, seq_len, d_model)
out = layer(x)
print(out.shape)   # torch.Size([1, 128, 512])
```

### Convert an MHA checkpoint to GQA

```python
import torch.nn as nn
from gqa.surgery import convert_mha_to_gqa

# Existing MHA key-projection layer (32 heads)
mha_k = nn.Linear(512, 512, bias=False)

# Compress to 8 KV heads (group_size = 32/8 = 4)
gqa_k_weight = convert_mha_to_gqa(
    mha_k.weight,
    group_size=4,
    weight_is_pytorch_linear=True,  # handles (D_out, D_in) layout automatically
)
print(gqa_k_weight.shape)   # torch.Size([128, 512])
```

### Run the benchmark

```bash
python benchmarks/benchmark.py
python benchmarks/benchmark.py --seq_len 1024 --num_heads 16 --device cuda
```

Example output (CPU, seq_len=256):
```
Arch    H_KV   KV Params   Latency (ms)   Speedup
-------------------------------------------------
MHA        8     524,288      14.1 ± 21.0     1.00x
GQA        2     131,072      10.5 ± 18.2     1.35x
MQA        1      65,536      11.3 ± 19.5     1.25x
```

### Run tests

```bash
pytest tests/test_gqa.py -v
```

---

## Implementation Notes

### Why `.expand()` instead of `.repeat()`?

`repeat_kv` uses `.expand()` (a *view* — no memory allocation) before `.reshape()`.
While `.reshape()` may trigger a copy when the tensor is not contiguous, the
expand step avoids allocating `num_repeats` full copies of the KV tensor in the
intermediate representation, which matters at scale.

### RoPE design

The `RotaryEmbedding` module pre-computes a cos/sin cache at construction time and
slices it at forward-pass time, avoiding redundant computation per step. The
`_rotate_half` operation splits the head dimension into two halves and applies the
rotation as:

```
q' = q * cos + rotate_half(q) * sin
```

### Causal masking

The causal mask is registered as a persistent buffer (not a parameter), so it moves
with `.to(device)` automatically and is not included in `model.parameters()`.

---

## References

- Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* https://arxiv.org/abs/2305.13245
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* https://arxiv.org/abs/2104.09864
- Shazeer (2019). *Fast Transformer Decoding: One Write-Head is All You Need.* https://arxiv.org/abs/1911.02150

---

## License

MIT
