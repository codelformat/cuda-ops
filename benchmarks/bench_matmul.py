"""Matmul 基准测试：对比 cuda_ops 与 torch.matmul（cuBLAS），按实测 SM 频率计算算力利用率。

示例：
  python benchmarks/bench_matmul.py
  python benchmarks/bench_matmul.py --shapes 4096x4096x4096 --dtype fp32 fp16

  # 配合 ncu：--profile 时每个配置只执行一次 cuda_ops 实现，不跑对照组和计时循环
  ncu -k regex:matmul python benchmarks/bench_matmul.py --profile --shapes 4096x4096x4096 --dtype fp32

峰值算力：fp32 按 CUDA core 计；fp16/bf16 按 Tensor Core（fp32 累加）计。
笔记本 GPU 会因功耗墙降频，频率取计时期间 nvidia-smi 采样的平均值，不同行之间的峰值因此不同。
权威的算力利用率看 ncu Roofline 一节：FFMA 实际每周期执行数 / 理论值（按周期归一化，与频率无关）。
不要看 Speed Of Light 的 Compute (SM) Throughput：它取 SM 内最忙的单元，访存指令（LSU）打满时也接近 100%。
"""

import argparse
import math

import torch

from bench_utils import (
    CUDA_CORE_FP32_FLOPS_PER_CLK,
    TENSOR_CORE_FP32_ACC_FLOPS_PER_CLK,
    SmClockSampler,
    benchmark,
    peak_tflops,
    tflops,
)
from cuda_ops import matmul

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# 方阵；K 较小；扁平（M=16，算术强度低，接近访存瓶颈）
DEFAULT_SHAPES = [
    "1024x1024x1024",
    "2048x2048x2048",
    "4096x4096x4096",
    "4096x4096x1024",
    "16x4096x4096",
]
MIN_TIMED_SECONDS = 1.0  # 计时循环至少持续这么久，让频率进入稳态，nvidia-smi 也能采到足够样本


def parse_shape(text: str) -> tuple[int, int, int]:
    m, n, k = text.lower().split("x")
    return int(m), int(n), int(k)


def timed(fn, warmup: int, iters: int) -> tuple[float, float | None]:
    """返回 (耗时中位数 us, 平均 SM 频率 MHz)。迭代次数按单次耗时放大到至少 MIN_TIMED_SECONDS。"""
    estimate_us = benchmark(fn, warmup=1, iters=3)
    iters = max(iters, math.ceil(MIN_TIMED_SECONDS * 1e6 / estimate_us))
    with SmClockSampler() as clock:
        time_us = benchmark(fn, warmup, iters)
    return time_us, clock.mean_mhz


def fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES, help="MxNxK，例如 4096x4096x4096")
    parser.add_argument("--dtype", nargs="+", choices=list(DTYPES), default=["fp32"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20, help="最少迭代次数")
    parser.add_argument(
        "--profile", action="store_true", help="每个配置只执行一次 cuda_ops 实现，供 ncu/nsys 采集"
    )
    args = parser.parse_args()

    # 对照组用真正的 fp32 SGEMM，不允许 cuBLAS 走 TF32
    torch.backends.cuda.matmul.allow_tf32 = False

    if not args.profile:
        props = torch.cuda.get_device_properties()
        print(
            f"{props.name}，{props.multi_processor_count} SMs；峰值/SM/周期："
            f"CUDA core fp32 {CUDA_CORE_FP32_FLOPS_PER_CLK} FLOP，"
            f"Tensor Core fp32 累加 {TENSOR_CORE_FP32_ACC_FLOPS_PER_CLK} FLOP\n"
        )
        print(
            "| MxNxK | dtype | 算术强度 FLOP/B | cuda_ops us | TFLOPS | MHz | 峰值% "
            "| torch us | TFLOPS | MHz | 峰值% | 对 torch 加速比 |"
        )
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for shape in args.shapes:
        m, n, k = parse_shape(shape)
        for name in args.dtype:
            dtype = DTYPES[name]
            a = torch.randn(m, k, device="cuda", dtype=dtype)
            b = torch.randn(k, n, device="cuda", dtype=dtype)

            if args.profile:
                matmul(a, b)
                continue

            # 结果错误时计时没有意义（例如 kernel 还没写，耗时只有几微秒），直接跳过。
            # 必须先算 cuda_ops 再算参考值：输出是 at::empty，若参考结果的显存刚被释放，
            # 缓存分配器会把同一块显存分给它，没写任何值也会"恰好正确"
            out = matmul(a, b).float()
            ref = torch.matmul(a, b).float()
            rel_err = ((out - ref).norm() / ref.norm()).item()
            if not rel_err < 1e-2:
                print(f"| {m}x{n}x{k} | {name} | 结果错误（相对误差 {rel_err:.2g}），跳过计时 |")
                continue

            flop = 2 * m * n * k
            min_bytes = (m * k + k * n + m * n) * a.element_size()
            per_clk = (
                CUDA_CORE_FP32_FLOPS_PER_CLK
                if dtype == torch.float32
                else TENSOR_CORE_FP32_ACC_FLOPS_PER_CLK
            )

            t_ours, mhz_ours = timed(lambda: matmul(a, b), args.warmup, args.iters)
            t_torch, mhz_torch = timed(lambda: torch.matmul(a, b), args.warmup, args.iters)

            row = [f"{m}x{n}x{k}", name, f"{flop / min_bytes:.0f}"]
            for t, mhz in ((t_ours, mhz_ours), (t_torch, mhz_torch)):
                achieved = tflops(flop, t)
                pct = achieved / peak_tflops(per_clk, mhz) if mhz else None
                row += [f"{t:.1f}", f"{achieved:.2f}", fmt(mhz, ".0f"), fmt(pct, ".1%")]
            row.append(f"{t_torch / t_ours:.2f}x")
            print("| " + " | ".join(row) + " |")

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
