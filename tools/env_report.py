"""输出与性能测量相关的环境信息，方便复现和对比数字。

用法（在容器内）：
  python tools/env_report.py

回报测量结果时，请把这段输出一起贴上。任何一项拿不到时显示“未知”，不会报错退出。
"""

import ctypes
import glob
import os
import platform
import shutil
import subprocess
import unicodedata

UNKNOWN = "未知"


def run(cmd):
    """运行命令并返回去掉首尾空白的输出；命令不存在或失败时返回 None。"""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def nvidia_smi(query):
    out = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else UNKNOWN


def nvcc_version():
    out = run(["nvcc", "--version"])
    if not out:
        return UNKNOWN
    for line in out.splitlines():
        if "release" in line:
            return line.split("release", 1)[1].strip()
    return UNKNOWN


def ncu_version():
    out = run(["ncu", "--version"])
    return out.splitlines()[-1].strip() if out else "未安装或不在 PATH 中"


def profiling_permission():
    """Linux 驱动默认只允许管理员读取性能计数器，此时 ncu 会报 ERR_NVGPUCTRPERM。"""
    try:
        with open("/proc/driver/nvidia/params") as f:
            for line in f:
                if line.startswith("RmProfilingAdminOnly"):
                    value = line.split(":", 1)[1].strip()
                    return "所有用户可用" if value == "0" else "仅管理员（ncu 可能报 ERR_NVGPUCTRPERM）"
    except OSError:
        pass
    return UNKNOWN


def cublas_version(torch):
    """通过 cublasGetProperty 读取 cuBLAS 版本，优先使用 PyTorch 自带的库。"""
    candidates = []
    if torch is not None:
        site = os.path.dirname(os.path.dirname(torch.__file__))
        candidates += sorted(glob.glob(os.path.join(site, "nvidia", "*", "lib", "libcublas.so*")))
        candidates += sorted(glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcublas.so*")))
    candidates += ["libcublas.so.13", "libcublas.so.12", "libcublas.so"]
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            parts = []
            for prop in range(3):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
                value = ctypes.c_int()
                if lib.cublasGetProperty(prop, ctypes.byref(value)) != 0:
                    raise OSError
                parts.append(str(value.value))
            return ".".join(parts)
        except (OSError, AttributeError):
            continue
    return UNKNOWN


def display_width(text):
    """终端显示宽度：中文等全角字符占两格。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def main():
    try:
        import torch
    except ImportError:
        torch = None

    rows = [("系统", f"{platform.system()} {platform.release()}"), ("Python", platform.python_version())]

    if torch is None:
        rows.append(("PyTorch", "未安装"))
    else:
        rows.append(("PyTorch", torch.__version__))
        rows.append(("PyTorch 的 CUDA 版本", torch.version.cuda or "无（CPU 版本）"))

    cuda_ok = torch is not None and torch.cuda.is_available()
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "未设置")
    if cuda_ok:
        p = torch.cuda.get_device_properties(0)
        cap = f"{p.major}.{p.minor}"
        rows += [
            ("GPU", p.name),
            ("计算能力", f"{cap}（sm_{p.major}{p.minor}）"),
            ("SM 数", str(p.multi_processor_count)),
            ("显存", f"{p.total_memory / 2**30:.1f} GiB"),
        ]
    else:
        cap = None
        rows.append(("GPU", "PyTorch 检测不到可用的 CUDA 设备"))

    rows += [
        ("驱动", nvidia_smi("driver_version")),
        ("SM 频率（当前 / 最高）", f"{nvidia_smi('clocks.sm')} / {nvidia_smi('clocks.max.sm')}"),
        ("显存频率（当前 / 最高）", f"{nvidia_smi('clocks.mem')} / {nvidia_smi('clocks.max.mem')}"),
        ("功耗上限", nvidia_smi("power.limit")),
        ("nvcc", nvcc_version()),
        ("cuBLAS", cublas_version(torch)),
        ("TORCH_CUDA_ARCH_LIST", arch_list),
        ("ncu", ncu_version()),
        ("性能计数器权限", profiling_permission()),
    ]

    width = max(display_width(k) for k, _ in rows)
    for key, value in rows:
        print(f"{key}{' ' * (width - display_width(key))}  {value}")

    if cap is not None and arch_list != "未设置":
        archs = [a.strip().removesuffix("+PTX") for a in arch_list.replace(",", ";").replace(" ", ";").split(";") if a.strip()]
        if cap not in archs:
            print(
                f"\n注意：当前 GPU 是 {cap}，但 TORCH_CUDA_ARCH_LIST 是 {arch_list}，"
                f"扩展运行时会报 no kernel image。重新编译：\n"
                f'  rm -rf build && TORCH_CUDA_ARCH_LIST="{cap}" pip install -e . --no-build-isolation'
            )


if __name__ == "__main__":
    main()
