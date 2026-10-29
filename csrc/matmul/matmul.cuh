#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cuda_ops {

    template<typename T, int kThreads, int BM, int BN, int BK, int TM, int TN>
    __global__ void __launch_bounds__(kThreads) matmul_kernel(const T* a, const T* b, T* c, 
                                            int64_t M, int64_t N, int64_t K){ // 一个线程计算C的一个元素
      // v1: matmul tiling 共享内存分块重写
      // 计算 tid
      const int tid = threadIdx.y * blockDim.x + threadIdx.x;

      // 将a, b, c先移动到该 block 负责小块的起始位置
      a += blockIdx.y * BM * K;
      b += blockIdx.x * BN;
      c += blockIdx.y * BM * N + blockIdx.x * BN;

      // 定义共享内存（ __shared__ 变量无法初始化），累加器，每次累加时的寄存器
      __shared__ float ATILE[BM][BK], BTILE[BK][BN];
      // 每个线程负责 TM * TN 个输出
      float acc[TM][TN] = {0.0f};
      float regM[TM] = {0.0f}; float regN[TN] = {0.0f};

      // 开始循环。先搬运，再计算。在 K 维度上计算，以 BK 为步长
      // 循环次数是编译期常量的，用 #pragma unroll 展开
      for (int t = 0; t < K; t += BK) {
        // 先搬 ATILE, 共 BM * BK 个元素。目的是让相邻线程尽可能地址相邻，合并访存
        #pragma unroll
        for (int i = 0; i < (BM * BK) / kThreads; ++ i) {
          int idx = tid + i * kThreads;
          int r = idx / BK;
          int c = idx % BK;
          // 需要加判断，如果越界访问则填 0
          if (blockIdx.y * BM + r >= M || t + c >= K) ATILE[r][c] = 0;
          else ATILE[r][c] = static_cast<float>(a[r * K + c + t]);
        }

        #pragma unroll
        for (int i = 0; i < (BK * BN) / kThreads; ++ i) {
          int idx = tid + i * kThreads;
          int r = idx / BN;
          int c = idx % BN;
          // 需要加判断，如果越界访问则填 0
          if (t + r >= K || blockIdx.x * BN + c >= N) BTILE[r][c] = 0;
          else BTILE[r][c] = static_cast<float>(b[(r + t) * N + c]);
        }

        // 搬运完先同步，防止还没搬完其他线程就开始计算
        __syncthreads();

        // 开始计算。沿共同的 BK 维度累加
        #pragma unroll
        for (int kk = 0; kk < BK; ++ kk) {
          // 先把a小块对应的搬运到regM
          // 每个线程负责的是 BM * BN 小块中的 [ty * TM ... ty * TM + TM - 1, tx * TN ... tx * TN + TN - 1]
          #pragma unroll
          for (int i = 0; i < TM; ++ i) {
            regM[i] = ATILE[threadIdx.y * TM + i][kk];
          }

          #pragma unroll
          for (int j = 0; j < TN; ++ j) {
            regN[j] = BTILE[kk][threadIdx.x * TN + j];
          }

          #pragma unroll
          for (int i = 0; i < TM; ++ i) {
            #pragma unroll
            for (int j = 0; j < TN; ++ j) {
              acc[i][j] += regM[i] * regN[j];
            }
          }
        }

        // 这里防止慢线程还在搬寄存器，快线程已经写入新值到共享内存了。
        __syncthreads();
      }

      // 写回C
      // 先计算 C 在当前线程的偏移
      c += threadIdx.y * TM * N + threadIdx.x * TN;
      // 每个线程只负责 c 的 TM * TN 个输出
      #pragma unroll
      for (int i = 0; i < TM; ++ i) {
        #pragma unroll
        for (int j = 0; j < TN; ++ j) {
          // 写回这里也要加越界判断
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
