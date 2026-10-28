import pytest
import torch

from cuda_ops import matmul

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# 相对误差：fp32 来自累加，fp16/bf16 主要来自输出舍入（半个 ulp：2^-11、2^-8），各留约 2 倍余量
RTOL = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 8e-3}

# (M, N, K)
SHAPES = [
    (1, 1, 1),
    (3, 5, 7),
    (127, 129, 65),  # 都不是分块大小的整数倍
    (256, 256, 256),
    (513, 257, 1023),
    (1024, 1024, 1024),
    (64, 64, 8192),  # 长 K：累加误差最大
    (16, 4096, 1024),  # 扁平：推理时的小 batch
    (70000, 16, 16),  # M 方向若每个 block 只算一行并映射到 grid.y，会超过 65535 的上限
    (16, 70000, 16),
]


def atol(k: int) -> float:
    # fp32 累加的绝对误差随 K 增长；结果接近 0 的元素无法只靠 rtol 覆盖
    return 1e-6 * k


def reference(a, b):
    """在 CPU 上以 fp64 计算参考值。

    不用 fp32 的 torch.matmul：它可能走 TF32，本身只有约 1e-3 的精度；
    不在 GPU 上算：cuBLAS 会让 compute-sanitizer 报出与本算子无关的 API 错误。
    """
    return a.cpu().double() @ b.cpu().double()


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_matmul(M, N, K, dtype):
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c = matmul(a, b)

    assert c.shape == (M, N) and c.dtype == dtype
    torch.testing.assert_close(c.cpu().double(), reference(a, b), rtol=RTOL[dtype], atol=atol(K))


def test_tolerance_rejects_tf32():
    # 确认容差足够严格：TF32（10 位尾数）的结果应当判为不通过
    a = torch.randn(256, 1024, device="cuda")
    b = torch.randn(1024, 256, device="cuda")
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        c_tf32 = a @ b
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            c_tf32.cpu().double(), reference(a, b), rtol=RTOL[torch.float32], atol=atol(1024)
        )


@pytest.mark.parametrize("M,N,K", [(0, 8, 4), (8, 0, 4), (8, 4, 0)])
def test_empty(M, N, K):
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    c = matmul(a, b)
    assert c.shape == (M, N)
    torch.testing.assert_close(c, torch.zeros(M, N, device="cuda"))  # K=0 时是空求和，结果为 0


@pytest.mark.parametrize(
    "case",
    ["cpu", "non_contiguous", "not_2d", "shape_mismatch", "dtype_mismatch", "unsupported_dtype"],
)
def test_invalid_inputs(case):
    a = torch.randn(64, 32, device="cuda")
    b = torch.randn(32, 64, device="cuda")
    if case == "cpu":
        a = a.cpu()
    elif case == "non_contiguous":
        b = torch.randn(64, 32, device="cuda").t()
    elif case == "not_2d":
        a = a.unsqueeze(0)
    elif case == "shape_mismatch":
        b = torch.randn(33, 64, device="cuda")
    elif case == "dtype_mismatch":
        b = b.half()
    elif case == "unsupported_dtype":
        a, b = a.double(), b.double()
    with pytest.raises(RuntimeError):
        matmul(a, b)
