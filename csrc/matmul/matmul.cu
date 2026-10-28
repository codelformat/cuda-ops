#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "common/dispatch.h"
#include "matmul/matmul.cuh"
#include "ops.h"

namespace cuda_ops {
at::Tensor matmul(const at::Tensor& a, const at::Tensor& b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);

  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "matmul expects 2D inputs, got ", a.sizes(), " and ",
              b.sizes());
  TORCH_CHECK(a.size(1) == b.size(0), "shape mismatch: ", a.sizes(), " @ ", b.sizes());
  // dispatch 只看 a 的 dtype，b 的类型不同时会被按 a 的类型解释，结果错误且不报错
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a/b dtype must match, got ", a.scalar_type(),
              " and ", b.scalar_type());
  TORCH_CHECK(a.device() == b.device(), "a/b must be on the same device");

  const int64_t M = a.size(0);
  const int64_t K = a.size(1);
  const int64_t N = b.size(1);
  const c10::cuda::CUDAGuard guard(a.device());
  // kernel 会写满 C 的每个元素，不需要 zeros
  auto c = at::empty({M, N}, a.options());
  if (M == 0 || N == 0) {
    return c;
  }
  if (K == 0) {  // 空求和，结果为 0
    return c.zero_();
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CUDA_OPS_DISPATCH_FLOAT_HALF_BF16(a.scalar_type(), "matmul", [&] {
    launch_matmul<scalar_t>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(),
                            M, N, K, stream);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
}  // namespace cuda_ops
