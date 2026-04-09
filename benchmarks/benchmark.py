"""
benchmark.py
------------
Benchmarks latency and (theoretical) KV-cache memory for:
    MHA  (num_kv_groups == num_heads)
    GQA  (num_kv_groups == num_heads // 4)
    MQA  (num_kv_groups == 1)

Run with:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --device cuda   # if GPU available
"""

import argparse
import time
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gqa.attention import GroupedQueryAttention
from gqa.surgery   import memory_comparison_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layer(num_kv_groups: int, d_model: int, num_heads: int,
                use_rope: bool, causal: bool, device: str) -> GroupedQueryAttention:
    return GroupedQueryAttention(
        d_in=d_model, d_out=d_model,
        num_heads=num_heads, num_kv_groups=num_kv_groups,
        use_rope=use_rope, causal=causal,
    ).to(device).eval()


def measure_latency(
    layer: GroupedQueryAttention,
    x: torch.Tensor,
    warmup: int = 10,
    repeats: int = 100,
) -> dict:
    """Return mean and std latency in milliseconds."""
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(x)

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if x.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = layer(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = layer(x)
                times.append((time.perf_counter() - t0) * 1000)

    t = torch.tensor(times)
    return {"mean_ms": t.mean().item(), "std_ms": t.std().item()}


def count_kv_params(layer: GroupedQueryAttention) -> int:
    return sum(p.numel() for p in [layer.k_proj.weight, layer.v_proj.weight])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    d_model: int = 512,
    num_heads: int = 8,
    seq_len: int = 256,
    batch_size: int = 1,
    use_rope: bool = True,
    causal: bool = True,
    device: str = "cpu",
    warmup: int = 5,
    repeats: int = 50,
):
    print("=" * 62)
    print(" GQA Benchmark")
    print(f" d_model={d_model}  num_heads={num_heads}  seq_len={seq_len}")
    print(f" batch={batch_size}  rope={use_rope}  causal={causal}  device={device}")
    print("=" * 62)

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    configs = [
        ("MHA",  num_heads),
        ("GQA",  max(1, num_heads // 4)),
        ("MQA",  1),
    ]

    results = []
    for name, kv_groups in configs:
        layer   = _make_layer(kv_groups, d_model, num_heads, use_rope, causal, device)
        latency = measure_latency(layer, x, warmup=warmup, repeats=repeats)
        kv_params = count_kv_params(layer)
        results.append({
            "name"    : name,
            "H_KV"    : kv_groups,
            "kv_params": kv_params,
            **latency,
        })

    # --- Print latency table ---
    header = f"{'Arch':<6} {'H_KV':>5} {'KV Params':>11} {'Latency (ms)':>14} {'Speedup':>9}"
    print(header)
    print("-" * len(header))

    base_ms = results[0]["mean_ms"]
    for r in results:
        speedup = base_ms / r["mean_ms"]
        print(
            f"{r['name']:<6} {r['H_KV']:>5} {r['kv_params']:>11,} "
            f"{r['mean_ms']:>10.3f} ± {r['std_ms']:.3f}   {speedup:>6.2f}x"
        )

    # --- Print memory table ---
    print()
    head_dim = d_model // num_heads
    mem_rows = memory_comparison_table(
        num_query_heads=num_heads,
        seq_len=seq_len,
        batch_size=batch_size,
        d_model=d_model,
    )
    print(f"\n{'Arch':<6} {'H_KV':>5} {'KV Cache (MB)':>14} {'Reduction':>11}")
    print("-" * 40)
    for row in mem_rows:
        print(
            f"{row['Architecture']:<6} {row['H_KV']:>5} "
            f"{row['KV Cache (MB)']:>14.4f} {row['Reduction']:>11}"
        )

    print()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GQA Benchmark")
    parser.add_argument("--d_model",    type=int,   default=512)
    parser.add_argument("--num_heads",  type=int,   default=8)
    parser.add_argument("--seq_len",    type=int,   default=256)
    parser.add_argument("--batch_size", type=int,   default=1)
    parser.add_argument("--no_rope",    action="store_true")
    parser.add_argument("--no_causal",  action="store_true")
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--warmup",     type=int,   default=5)
    parser.add_argument("--repeats",    type=int,   default=50)
    args = parser.parse_args()

    run_benchmark(
        d_model=args.d_model,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        use_rope=not args.no_rope,
        causal=not args.no_causal,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
