"""基准测试公共工具：GPU 计时（每次迭代前清空 L2）与理论峰值带宽。"""

import statistics
from collections.abc import Callable

import torch


def peak_dram_bandwidth_gbps(device: int | None = None) -> float:
    """由显存频率和位宽计算理论峰值带宽（GB/s），按双倍数据率计算。"""
    props = torch.cuda.get_device_properties(device)
    return 2 * props.memory_clock_rate * 1e3 * (props.memory_bus_width / 8) / 1e9


def gbps(nbytes: int, time_us: float) -> float:
    return nbytes / (time_us * 1e-6) / 1e9


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
