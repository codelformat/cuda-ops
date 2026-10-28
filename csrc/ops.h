#pragma once

#include <ATen/core/Tensor.h>

#include <vector>

namespace cuda_ops {
// LayerNorm 前向，对最后一维归一化。返回 {y, mean, rstd}, mean/rstd 形状为 [M], dtype 为 fp32
std::vector<at::Tensor> layer_norm_fwd(const at::Tensor& x, const at::Tensor& gamma,
                                       const at::Tensor& beta, double eps);
// C[M, N] = A[M, K] @ B[K, N]，输入为二维、连续存储且 dtype 相同，输出 dtype 与输入一致
at::Tensor matmul(const at::Tensor& a, const at::Tensor& b);
}  // namespace cuda_ops
