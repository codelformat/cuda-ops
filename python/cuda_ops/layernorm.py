import torch

from cuda_ops import _C


def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm 前向，返回 (y, mean, rstd)；mean/rstd 形状为 [M]，dtype 为 fp32。"""
    # C++ 侧返回 std::vector，到 Python 是 list，这里转成 tuple 与类型标注一致
    y, mean, rstd = _C.layer_norm_fwd(x, weight, bias, eps)
    return y, mean, rstd


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """语义等同于 F.layer_norm(x, (x.shape[-1],), weight, bias, eps)。"""
    return layer_norm_fwd(x, weight, bias, eps)[0]
