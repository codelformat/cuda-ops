"""基准测试公共工具：GPU 计时（每次迭代前清空 L2）、理论峰值带宽与算力、SM 频率采样。"""

import statistics
import subprocess
from collections.abc import Callable

import torch


def peak_dram_bandwidth_gbps(device: int | None = None) -> float:
    """由显存频率和位宽计算理论峰值带宽（GB/s），按双倍数据率计算。"""
    props = torch.cuda.get_device_properties(device)
    return 2 * props.memory_clock_rate * 1e3 * (props.memory_bus_width / 8) / 1e9


def gbps(nbytes: int, time_us: float) -> float:
    return nbytes / (time_us * 1e-6) / 1e9


# Ampere GA10x 每个 SM 每周期的峰值 FLOP（一次 FMA 计 2 次）
CUDA_CORE_FP32_FLOPS_PER_CLK = 256  # 128 个 FP32 单元
TENSOR_CORE_FP32_ACC_FLOPS_PER_CLK = 512  # fp16/bf16 输入、fp32 累加、稠密；GeForce 上是 fp16 累加的一半


def peak_tflops(flops_per_clk_per_sm: int, sm_clock_mhz: float, device: int | None = None) -> float:
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count * flops_per_clk_per_sm * sm_clock_mhz * 1e6 / 1e12


def tflops(flop: int, time_us: float) -> float:
    return flop / (time_us * 1e-6) / 1e12


class SmClockSampler:
    """用 nvidia-smi 在后台采样 SM 频率（MHz），with 块结束后通过 mean_mhz 取平均值。

    笔记本 GPU 持续满载时会撞功耗墙降频（实测 cuBLAS SGEMM 时约 790 MHz，而基础频率为 1395 MHz），
    计算瓶颈的 kernel 必须按运行时的实际频率计算峰值，否则利用率会被严重低估。
    """

    def __init__(self, interval_ms: int = 20, device: int | None = None):
        uuid = torch.cuda.get_device_properties(device).uuid
        self._cmd = [
            "nvidia-smi",
            f"--id=GPU-{uuid}",
            "--query-gpu=clocks.sm",
            "--format=csv,noheader,nounits",
            f"--loop-ms={interval_ms}",
        ]
        self.samples: list[int] = []

    def __enter__(self) -> "SmClockSampler":
        self._proc = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, text=True)
        return self

    def __exit__(self, *exc) -> None:
        self._proc.terminate()
        out, _ = self._proc.communicate()
        self.samples = [int(v) for v in out.split() if v.isdigit()]

    @property
    def mean_mhz(self) -> float | None:
        return statistics.mean(self.samples) if self.samples else None


def benchmark(fn: Callable[[], object], warmup: int = 10, iters: int = 100) -> float:
    """返回 fn 单次执行的 GPU 耗时中位数（微秒）。

    每次迭代前写一块远大于 L2 的缓冲区把 L2 冲掉，避免小数据常驻缓存导致带宽虚高；
    清空操作同时让 GPU 保持忙碌，使 Python 端的调用开销不计入 kernel 时间。
    """
    l2_bytes = torch.cuda.get_device_properties().L2_cache_size
    flush_buf = torch.empty(max(16 * l2_bytes, 64 << 20), dtype=torch.uint8, device="cuda")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        flush_buf.zero_()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()

    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends)) * 1e3
