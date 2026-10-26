#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cuda_ops {
// 对 x[M, N]的每一行： y = (x - mean) * rstd * gamma + beta
// mean[M], rstd[M] 以 fp32 输出，供反向使用。 T 为 float / c10::Half / c10::BFloat16
template <typename T>
void launch_layer_norm_fwd(const T* x, const T* gamma, const T* beta, T* y, float* mean,
                           float* rstd, int64_t M, int64_t N, float eps, cudaStream_t stream) {
  // TODO: 实现 kernel 并在这里 launch
}
}  // namespace cuda_ops
