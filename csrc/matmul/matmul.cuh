#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include "common/vec.h"

namespace cuda_ops {

    template<typename T, int kThreads, int BM, int BN, int BK, int TM, int TN, int PAD, int kVec>
    __global__ void __launch_bounds__(kThreads) matmul_kernel(const T* a, const T* b, T* c, 
                                            int64_t M, int64_t N, int64_t K){ // 一个线程计算C的一个元素
      using Vec = AlignedVec<T, kVec>;
      static_assert((BM * BK) % (kThreads * kVec) == 0);

      // 每个线程负责的 TN 列被分成 TN/4 组，组内 4 列连续，组与组之间间隔 BN/(TN/4) 列
      // 同一条 LDS.128 中，半个 warp 的16个线程可以读连续的 64 float = 256 字节，不产生 bank conflict
      static_assert(TN % 4 == 0);
      constexpr int kGroupStride = BN / (TN / 4);
      auto col_of = [](int j) { return (j / 4) * kGroupStride + 4 * static_cast<int>(threadIdx.x) + j % 4; };
      // v1: matmul tiling 共享内存分块重写
      // 计算 tid
      const int tid = threadIdx.y * blockDim.x + threadIdx.x;

      // 将a, b, c先移动到该 block 负责小块的起始位置
      a += blockIdx.y * BM * K;
      b += blockIdx.x * BN;
      c += blockIdx.y * BM * N + blockIdx.x * BN;

      // 定义共享内存（ __shared__ 变量无法初始化），累加器，每次累加时的寄存器
      // v3: 转置A，减少每线程寄存器数，能让共享内存读取连续地址；（如果不转置，kk循环读ATILE时会一整行读进寄存器来后续复用）
      __shared__ alignas(16) float ATILE[BK][BM + PAD], BTILE[BK][BN]; // 显式声明首地址 16 字节对齐
      // 每个线程负责 TM * TN 个输出
      float acc[TM][TN] = {0.0f};
      float regM[TM] = {0.0f}; float regN[TN] = {0.0f};

      // 开始循环。先搬运，再计算。在 K 维度上计算，以 BK 为步长
      // 循环次数是编译期常量的，用 #pragma unroll 展开
      for (int t = 0; t < K; t += BK) {
        // 先搬 ATILE, 共 BM * BK 个元素。目的是让相邻线程尽可能地址相邻，合并访存
        /// v3: float4 读取，减少指令发射次数
        #pragma unroll
        for (int i = 0; i < (BM * BK) / (kThreads * kVec); ++ i) {
          int idx = tid + i * kThreads;
          int r = idx * kVec / BK;
          int c = idx * kVec % BK;
          // 需要加判断，如果越界访问则填 0
          if (blockIdx.y * BM + r >= M || t + c >= K) { // **这里依赖 K 是 kVec 倍数情况下**，只判断 t + c >= K 因为第一个元素不越界，后面几个元素均不会越界
            #pragma unroll
            for (int j = 0; j < kVec; ++ j) {
              ATILE[c + j][r] = 0;
            }
          }
          else {
            // ATILE[c][r] = static_cast<float>(a[r * K + c + t]);
            auto in = *reinterpret_cast<const Vec*>(a + r * K + c + t);
            #pragma unroll
            for (int j = 0; j < kVec; ++ j) {
              ATILE[c + j][r] = static_cast<float>(in.v[j]);
            }
          }
        }

        #pragma unroll
        for (int i = 0; i < (BK * BN) / (kThreads * kVec); ++ i) {
          int idx = tid + i * kThreads;
          int r = idx * kVec / BN;
          int c = idx * kVec % BN;
          // 需要加判断，如果越界访问则填 0
          // 先在寄存器里拼好 kVec 个 float，再整体写入：kVec=4 时为一条 STS.128，kVec=1 时为一条 STS.32
          AlignedVec<float, kVec> tmpB;
          if (t + r >= K || blockIdx.x * BN + c >= N) { // **这里也依赖 N 是 kVec倍数** 因为按行读 kVec
            #pragma unroll
            for (int j = 0; j < kVec; ++ j) {
              tmpB.v[j] = 0.0f;
            }
          }
          else {
            // BTILE[r][c] = static_cast<float>(b[(r + t) * N + c]);
            auto in = *reinterpret_cast<const Vec*>(b + (r + t) * N + c);
            #pragma unroll
            for (int j = 0; j < kVec; ++ j) {
              tmpB.v[j] = static_cast<float>(in.v[j]);
            }
          }
          *reinterpret_cast<AlignedVec<float, kVec>*>(&BTILE[r][c]) = tmpB;
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
            regM[i] = ATILE[kk][threadIdx.y * TM + i];
          }

          #pragma unroll
          for (int j = 0; j < TN; ++ j) {
            // 这里 32 Threads / Warp 被分成了两个 半warp, 因为ty=0..1, tx=0..15, 所以每个 半warp 的ty相同，只变tx
            // 导致这里的相邻线程相隔了TN=8个float，这样的话每4个线程会产生 bank conflict(0, 4)
            // 每个线程要读连续的8列float，共2条LDS.128指令。先从tx*8读4个，起始bank在(tx*8)%32，占连续4个bank。
            // 所以理想情况每 半warp 应该有 32 / 4 = 8 线程/wavefront, 所以至少2次wavefront

            // 但这里相邻线程相隔TN=8个float，每4个线程会产生冲突；所以变成需要4次wavefront
            regN[j] = BTILE[kk][col_of(j)]; 
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
      // 先计算 C 在当前线程的行偏移
      c += threadIdx.y * TM * N;
      // 每个线程只负责 c 的 TM * TN 个输出
      #pragma unroll
      for (int i = 0; i < TM; ++ i) {
        #pragma unroll
        for (int j = 0; j < TN; ++ j) {
          // 写回这里也要加越界判断
          if (blockIdx.y * BM + threadIdx.y * TM + i < M && blockIdx.x * BN + col_of(j) < N)
            c[i * N + col_of(j)] = static_cast<T>(acc[i][j]);
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

      constexpr int PAD = 4;

      dim3 block(BN/ TN, BM / TM);

      // 改成 x 对应列
      const dim3 grid(
        static_cast<unsigned int>((N + BN - 1) / BN),
        static_cast<unsigned int>((M + BM - 1) / BM)
      );

      if ( K % 4 != 0 || N % 4 != 0 
        || reinterpret_cast<uintptr_t>(a) % sizeof(AlignedVec<T, 4>) != 0 
        || reinterpret_cast<uintptr_t>(b) % sizeof(AlignedVec<T, 4>) != 0) 
        matmul_kernel<T, kThreads, BM, BN, BK, TM, TN, PAD, 1><<<grid, block, 0, stream>>>(a, b, c, M, N, K);
      else 
        matmul_kernel<T, kThreads, BM, BN, BK, TM, TN, PAD, 4><<<grid, block, 0, stream>>>(a, b, c, M, N, K);
      
    }

}  // namespace cuda_ops
