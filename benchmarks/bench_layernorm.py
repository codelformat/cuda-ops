"""LayerNorm 前向基准测试：对比 cuda_ops、PyTorch，以及纯拷贝（实际可达带宽上限）。

示例：
  python benchmarks/bench_layernorm.py
  python benchmarks/bench_layernorm.py --shapes 8192x4096 --dtype fp32 bf16

  # 配合 ncu：--profile 时每个配置只执行一次 cuda_ops 实现，不跑对照组和计时循环
  ncu -k regex:layer_norm_fwd python benchmarks/bench_layernorm.py --profile --shapes 8192x4096 --dtype fp32
"""

import argparse

import torch
import torch.nn.functional as F

from bench_utils import benchmark, gbps, peak_dram_bandwidth_gbps
from cuda_ops import layer_norm

EPS = 1e-5
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# 小 N 大 M、常见 hidden size、大 N 小 M；fp32 下数据量均远大于 L2
DEFAULT_SHAPES = [
    "32768x256",
    "8192x768",
    "8192x1024",
    "8192x4096",
    "4096x8192",
    "2048x12288",
    "256x65536",
]


def parse_shape(text: str) -> tuple[int, int]:
    m, n = text.lower().split("x")
    return int(m), int(n)


def min_dram_bytes(m: int, n: int, dtype: torch.dtype) -> int:
    """理论最小 DRAM 访存量：读 x/gamma/beta，写 y/mean/rstd。"""
    elem = torch.empty((), dtype=dtype).element_size()
    return 2 * m * n * elem + 2 * n * elem + 2 * m * 4


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES, help="MxN，例如 8192x4096")
    parser.add_argument("--dtype", nargs="+", choices=list(DTYPES), default=list(DTYPES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--profile", action="store_true", help="每个配置只执行一次 cuda_ops 实现，供 ncu/nsys 采集"
    )
    args = parser.parse_args()

    peak = peak_dram_bandwidth_gbps()
    if not args.profile:
        print(f"{torch.cuda.get_device_name()}，理论峰值带宽 {peak:.0f} GB/s\n")
        print(
            "| shape | dtype | 理论访存 MB | cuda_ops us | GB/s | 峰值% "
            "| torch us | GB/s | copy us | GB/s | 对 torch 加速比 |"
        )
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for shape in args.shapes:
        m, n = parse_shape(shape)
        for name in args.dtype:
            dtype = DTYPES[name]
            x = torch.randn(m, n, device="cuda", dtype=dtype)
            w = torch.randn(n, device="cuda", dtype=dtype)
            b = torch.randn(n, device="cuda", dtype=dtype)

            if args.profile:
                layer_norm(x, w, b, EPS)
                continue

            out = torch.empty_like(x)
            t_ours = benchmark(lambda: layer_norm(x, w, b, EPS), args.warmup, args.iters)
            t_torch = benchmark(lambda: F.layer_norm(x, (n,), w, b, EPS), args.warmup, args.iters)
            t_copy = benchmark(lambda: out.copy_(x), args.warmup, args.iters)

            nbytes = min_dram_bytes(m, n, dtype)
            copy_bytes = 2 * x.numel() * x.element_size()
            bw_ours = gbps(nbytes, t_ours)
            print(
                f"| {m}x{n} | {name} | {nbytes / 1e6:.1f} "
                f"| {t_ours:.1f} | {bw_ours:.0f} | {bw_ours / peak:.1%} "
                f"| {t_torch:.1f} | {gbps(nbytes, t_torch):.0f} "
                f"| {t_copy:.1f} | {gbps(copy_bytes, t_copy):.0f} "
                f"| {t_torch / t_ours:.2f}x |"
            )

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
