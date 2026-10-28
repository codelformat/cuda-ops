#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cuda_ops {

    template<typename T, int kThreads, int BM, int BN, int BK, int TM, int TN>
    __global__ void __launch_bounds__(kThreads) matmul_kernel(const T* a, const T* b, T* c, 
                                            int64_t M, int64_t N, int64_t K){ // 一个线程计算C的一个元素
      // float sum = 0.0f;

      // 先写直觉上的 x 对应行
      // const int row = blockDim.x * blockIdx.x + threadIdx.x;
      // const int col = blockDim.y * blockIdx.y + threadIdx.y;

      // 改成 x 对应列
      // 

      const int tid = threadIdx.y * blockDim.x + threadIdx.x; // 该thread在block中的编码

      // 先移动到a, b, c在该block的对应起始位置
      a += BM * blockIdx.y * K;
      b += BN * blockIdx.x;
      c += BM * blockIdx.y * N + BN * blockIdx.x;

      // if (row >= M || col >= N) return; // TODO: 不能直接return，会让__sync_threads()卡死，需要填0

      __shared__ float A_TILE[BM][BK], B_TILE[BK][BN];
      float acc[TM][TN] = {0.0f}; // 每个线程只负责 TM * TN 个输出

      for (int k = 0; k < K; k += BK) {
        // v1: 先搬运
        // 每个线程搬 BM * BK / KThreads 个 float 到 A_TILE
        for(int i = 0; i < (BM * BK / kThreads); ++ i) {
          int idx = tid + i * kThreads;
          int r = idx / BK; // 一个线程只搬1个float，相邻线程自动合并访存
          int c = idx % BK;
          if (blockIdx.y * BM + r >= M || k + c >= K) A_TILE[r][c] = 0.0f; // 越界填0
          else A_TILE[r][c] = static_cast<float>(a[r * K + c + k]); // 只需要指定在当前block里的相对位置
        }

        for (int i = 0; i < (BK * BN / kThreads); ++ i) {
          int idx = tid + i * kThreads;
          int r = idx / BN;
          int c = idx % BN;
          if (blockIdx.x * BN + c >= N || k + r >= K) B_TILE[r][c] = 0.0f; // 越界填0
          else B_TILE[r][c] = static_cast<float>(b[(r + k) * N + c]); 
        }

        // 每个线程搬运的位置不一样，所以搬运时不需要同步
        // 搬运后需要同步，防止后续计算访问到还没搬运进来的值
        __syncthreads();

        // 开始计算
        #pragma unroll
        for (int kk = 0; kk < BK; ++ kk) {
          #pragma unroll
          for (int i = 0; i < TM; ++ i) { // 该线程只负责 [ty*TM .. ty*TM+TM-1]
            #pragma unroll
            for (int j = 0; j < TN; ++ j) {
              acc[i][j] += A_TILE[threadIdx.y * TM + i][kk] * B_TILE[kk][threadIdx.x * TN + j];
            }
          }
        }

        // 防止有的线程算完了已经开始写下一轮TILE，结果有的warp还在计算，导致race
        __syncthreads();

        // 如果 x 对应列，每轮只有b的col变，相邻32个线程可以访问连续地址的float，8个float4就能合并
        // sum += static_cast<float>(a[row * K + k]) * static_cast<float>(b[k * N + col]); // 如果 x 对应行，相邻的线程在A上跨行读取，b不变，没法做合并访存
      }

      // 写回c
      // c[row * N + col] = static_cast<T>(sum);

      // 搬运回C
      // 先移动到线程起始位置
      c += threadIdx.y * TM * N + threadIdx.x * TN;
      // 循环上界是模版常量，需要加#pragma unroll
      #pragma unroll
      for (int i = 0; i < TM; ++ i) {
        #pragma unroll
        for (int j = 0; j < TN; ++ j) {
          if (blockIdx.y * BM + threadIdx.y * TM + i < M && blockIdx.x * BN + threadIdx.x * TN + j < N)
            c[i * N + j] = static_cast<T>(acc[i][j]);
        }
      }
    }

    // C[M, N] = A[M, K] @ B[K, N]，均为行主序、连续存储，fp32 累加。T 为 float / c10::Half /
    // c10::BFloat16 注意：gridDim.y/z 上限 65535，下标乘积（row * K 等）用 int64_t
    template <typename T>
    void launch_matmul(const T* a, const T* b, T* c, int64_t M, int64_t N, int64_t K,
                      cudaStream_t stream) {
      // TODO: 实现 kernel 并在这里 launch
      // constexpr int kThreads = 256;
      // v0: naive
      // constexpr int BX = 32; constexpr int BY = 8;
      // dim3 block(BX, BY);
      // const dim3 grid(
      //   static_cast<unsigned int>((M + BX - 1) / BX),
      //   static_cast<unsigned int>((N + BY - 1) / BY)
      // ); 

      // v1: shared memory 分块
      constexpr int BM = 128; constexpr int BN = 128;
      constexpr int TM = 8; constexpr int TN = 8;
      constexpr int BK = 8; // TODO: 根据情况具体定BK的值
      constexpr int kThreads = (BM * BN) / (TM * TN);

      dim3 block(BN/ TN, BM / TM);

      // 改成 x 对应列
      const dim3 grid(
        static_cast<unsigned int>((N + BN - 1) / BN),
        static_cast<unsigned int>((M + BM - 1) / BM)
      );

      matmul_kernel<T, kThreads, BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(a, b, c, M, N, K);
    }

}  // namespace cuda_ops
