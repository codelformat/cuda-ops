#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ops {

// c[i] = a[i] + b[i]，a/b/c 均为长度 n 的设备指针。在 stream 上异步执行。
cudaError_t vector_add(const float* a, const float* b, float* c, std::size_t n,
                       cudaStream_t stream = nullptr);

}  // namespace ops
