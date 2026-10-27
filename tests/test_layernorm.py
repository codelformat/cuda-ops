import pytest
import torch
import torch.nn.functional as F

from cuda_ops import layer_norm, layer_norm_fwd

EPS = 1e-5

# (rtol, atol)，取 torch.testing.assert_close 对各 dtype 的默认值
TOL = {
    torch.float32: (1.3e-6, 1e-5),
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
}

# 覆盖：单元素、奇数长度、非向量宽度整数倍、常见 hidden size、跨分档边界、寄存器缓存上限、单行超长
SHAPES = [
    (1, 1),
    (3, 7),
    (5, 127),
    (128, 768),
    (64, 1024),
    (31, 1032),
    (33, 4096),
    (16, 5121),
    (4, 12288),
    (3, 16384),
    (2, 65536),
]


def make_inputs(M, N, dtype):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    b = torch.randn(N, device="cuda", dtype=dtype)
    return x, w, b


def reference(x, w, b):
    """在 fp32 下计算参考值，再转回输入 dtype。"""
    xf = x.float()
    y = F.layer_norm(xf, (x.shape[-1],), w.float(), b.float(), EPS).to(x.dtype)
    mean = xf.mean(dim=-1).reshape(-1)
    rstd = torch.rsqrt(xf.var(dim=-1, unbiased=False) + EPS).reshape(-1)
    return y, mean, rstd


@pytest.mark.parametrize("dtype", list(TOL), ids=str)
@pytest.mark.parametrize("M,N", SHAPES)
def test_forward(M, N, dtype):
    x, w, b = make_inputs(M, N, dtype)
    y, mean, rstd = layer_norm_fwd(x, w, b, EPS)
    y_ref, mean_ref, rstd_ref = reference(x, w, b)

    rtol, atol = TOL[dtype]
    torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(mean, mean_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rstd, rstd_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", list(TOL), ids=str)
def test_misaligned_pointers(dtype):
    # 首地址偏移一个元素：仍然连续，但不满足 16 字节对齐，应退回逐元素读写
    M, N = 16, 1024
    x = torch.randn(M * N + 1, device="cuda", dtype=dtype)[1:].view(M, N)
    w = torch.randn(N + 1, device="cuda", dtype=dtype)[1:]
    b = torch.randn(N + 1, device="cuda", dtype=dtype)[1:]
    y, mean, rstd = layer_norm_fwd(x, w, b, EPS)
    y_ref, mean_ref, rstd_ref = reference(x, w, b)

    rtol, atol = TOL[dtype]
    torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(mean, mean_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rstd, rstd_ref, rtol=1e-4, atol=1e-5)


def test_3d_input():
    x = torch.randn(4, 16, 768, device="cuda")
    w = torch.randn(768, device="cuda")
    b = torch.randn(768, device="cuda")
    torch.testing.assert_close(layer_norm(x, w, b, EPS), F.layer_norm(x, (768,), w, b, EPS))


def test_large_mean_is_stable():
    # 均值远大于标准差：E[x^2] - E[x]^2 写法会出现灾难性抵消
    x = torch.randn(64, 4096, device="cuda", dtype=torch.float64).add_(1000.0)
    w = torch.ones(4096, device="cuda", dtype=torch.float64)
    b = torch.zeros(4096, device="cuda", dtype=torch.float64)
    y_ref = F.layer_norm(x, (4096,), w, b, EPS).float()
    y = layer_norm(x.float(), w.float(), b.float(), EPS)
    torch.testing.assert_close(y, y_ref, rtol=0, atol=1e-3)


@pytest.mark.parametrize(
    "case",
    ["cpu", "non_contiguous", "dtype_mismatch", "bad_gamma_shape", "unsupported_dtype"],
)
def test_invalid_inputs(case):
    x, w, b = make_inputs(64, 64, torch.float32)
    if case == "cpu":
        x = x.cpu()
    elif case == "non_contiguous":
        x = x.t()
    elif case == "dtype_mismatch":
        w = w.half()
    elif case == "bad_gamma_shape":
        w = w[:32]
    elif case == "unsupported_dtype":
        x, w, b = x.double(), w.double(), b.double()
    with pytest.raises(RuntimeError):
        layer_norm(x, w, b, EPS)
