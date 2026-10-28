import torch

from cuda_ops import _C


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = a @ b。a: [M, K]，b: [K, N]，均为连续的 CUDA 张量且 dtype 相同；内部 fp32 累加。"""
    return _C.matmul(a, b)
