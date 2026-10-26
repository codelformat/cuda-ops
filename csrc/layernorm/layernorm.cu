#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "common/dispatch.h"
#include "layernorm/layernorm.cuh"
#include "ops.h"

namespace cuda_ops {
std::vector<at::Tensor> layer_norm_fwd(const at::Tensor& x, const at::Tensor& gamma,
                                       const at::Tensor& beta, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);

  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension.");
  const int64_t N = x.size(-1);
  TORCH_CHECK(N > 0, "Normalized dimension must be non-empty");
  TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == N, "gamma must have shape [", N, "]");
  TORCH_CHECK(beta.dim() == 1 && beta.size(0) == N, "beta must have shape [", N, "]");
  TORCH_CHECK(gamma.scalar_type() == x.scalar_type() && beta.scalar_type() == x.scalar_type(),
              "gamma/beta dtype must match x");
  TORCH_CHECK(gamma.device() == x.device() && beta.device() == x.device(),
              "x/gamma/beta must be on the same device");
  const int64_t M = x.numel() / N;
  const c10::cuda::CUDAGuard guard(x.device());
  const auto stat_ops = x.options().dtype(at::kFloat);
  auto mean = at::empty({M}, stat_ops);
  auto rstd = at::empty({M}, stat_ops);
  auto y = at::empty_like(x);
  if (M == 0) {
    return {y, mean, rstd};
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CUDA_OPS_DISPATCH_FLOAT_HALF_BF16(x.scalar_type(), "layer_norm_fwd", [&] {
    launch_layer_norm_fwd<scalar_t>(x.data_ptr<scalar_t>(), gamma.data_ptr<scalar_t>(),
                                    beta.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),
                                    mean.data_ptr<float>(), rstd.data_ptr<float>(), M, N,
                                    static_cast<float>(eps), stream);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, mean, rstd};
}
}  // namespace cuda_ops
