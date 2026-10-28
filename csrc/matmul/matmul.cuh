#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cuda_ops {

    template<typename T, int kThreads>
    __global__ void __launch_bounds__(kThreads) matmul_kernel(const T* a, const T* b, T* c, 
                                            int64_t M, int64_t N, int64_t K){ // 一个线程计算C的一个元素
      float sum = 0.0f;

      // 先写直觉上的 x 对应行
      // const int row = blockDim.x * blockIdx.x + threadIdx.x;
      // const int col = blockDim.y * blockIdx.y + threadIdx.y;

      // 改成 x 对应列
      const int row = blockDim.y * blockIdx.y + threadIdx.y;
      const int col = blockDim.x * blockIdx.x + threadIdx.x;

      if (row >= M || col >= N) return;
      
      for (int k = 0; k < K; ++ k) {
        // 如果 x 对应列，每轮只有b的col变，相邻32个线程可以访问连续地址的float，8个float4就能合并
        sum += static_cast<float>(a[row * K + k]) * static_cast<float>(b[k * N + col]); // 如果 x 对应行，相邻的线程在A上跨行读取，b不变，没法做合并访存
      }

      // 写回c
      c[row * N + col] = static_cast<T>(sum);
    }

    // C[M, N] = A[M, K] @ B[K, N]，均为行主序、连续存储，fp32 累加。T 为 float / c10::Half /
    // c10::BFloat16 注意：gridDim.y/z 上限 65535，下标乘积（row * K 等）用 int64_t
    template <typename T>
    void launch_matmul(const T* a, const T* b, T* c, int64_t M, int64_t N, int64_t K,
                      cudaStream_t stream) {
      // TODO: 实现 kernel 并在这里 launch
      constexpr int kThreads = 256;
      constexpr int BX = 32; constexpr int BY = 8;
      dim3 block(BX, BY);
      // const dim3 grid(
      //   static_cast<unsigned int>((M + BX - 1) / BX),
      //   static_cast<unsigned int>((N + BY - 1) / BY)
      // ); 

      // 改成 x 对应列
      const dim3 grid(
        static_cast<unsigned int>((N + BX - 1) / BX),
        static_cast<unsigned int>((M + BY - 1) / BY)
      );

      matmul_kernel<T, kThreads><<<grid, block, 0, stream>>>(a, b, c, M, N, K);
    }

}  // namespace cuda_ops
