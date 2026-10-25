#include <algorithm>

#include "ops/vector_add.cuh"

namespace ops {
namespace {

constexpr int kBlockSize = 256;
constexpr std::size_t kMaxGridSize = 65535;

__global__ void vector_add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ c, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    c[i] = a[i] + b[i];
  }
}

}  // namespace

cudaError_t vector_add(const float* a, const float* b, float* c, std::size_t n,
                       cudaStream_t stream) {
  if (n == 0) {
    return cudaSuccess;
  }
  const std::size_t blocks = std::min((n + kBlockSize - 1) / kBlockSize, kMaxGridSize);
  vector_add_kernel<<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(a, b, c, n);
  return cudaGetLastError();
}

}  // namespace ops
