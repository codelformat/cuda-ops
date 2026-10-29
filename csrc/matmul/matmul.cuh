#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include "common/vec.h"

namespace cuda_ops {

    // 每个 block 计算 C 的 BM * BN 小块，每个线程计算其中 TM * TN 个元素，沿 K 每次处理 BK
    // __launch_bounds__ 第二个参数要求每个 SM 至少放 2 个 block，编译器会把寄存器限制在 128 以内
    // （v4 不加时 kVec=1 的版本为 130 个，每个 SM 只能放 1 个 block）
    template<typename T, int kThreads, int BM, int BN, int BK, int TM, int TN, int PAD, int kVec>
    __global__ void __launch_bounds__(kThreads, 2) matmul_kernel(const T* a, const T* b, T* c,
                                            int64_t M, int64_t N, int64_t K){
      using Vec = AlignedVec<T, kVec>;
      using FloatVec = AlignedVec<float, kVec>;
      // 每个线程每个 tile 从 global 读 A、B 各几次，每次 kVec 个元素
      constexpr int kLoadsA = (BM * BK) / (kThreads * kVec);
      constexpr int kLoadsB = (BK * BN) / (kThreads * kVec);
      static_assert(kLoadsA >= 1 && (BM * BK) % (kThreads * kVec) == 0);
      static_assert(kLoadsB >= 1 && (BK * BN) % (kThreads * kVec) == 0);

      // 每个线程负责的 TN 列被分成 TN/4 组，组内 4 列连续，组与组之间间隔 BN/(TN/4) 列
      // 同一条 LDS.128 中，半个 warp 的16个线程可以读连续的 64 float = 256 字节，不产生 bank conflict
      static_assert(TN % 4 == 0);
      constexpr int kGroupStride = BN / (TN / 4);
      auto col_of = [](int j) { return (j / 4) * kGroupStride + 4 * static_cast<int>(threadIdx.x) + j % 4; };

      const int tid = threadIdx.y * blockDim.x + threadIdx.x;

      // 将a, b, c先移动到该 block 负责小块的起始位置
      a += blockIdx.y * BM * K;
      b += blockIdx.x * BN;
      c += blockIdx.y * BM * N + blockIdx.x * BN;

      // v3: 转置A，减少每线程寄存器数，能让共享内存读取连续地址；（如果不转置，kk循环读ATILE时会一整行读进寄存器来后续复用）
      // v4: 双缓冲。计算 [cur] 的同时，下一个 tile 写入 [cur ^ 1]
      __shared__ alignas(16) float ATILE[2][BK][BM + PAD], BTILE[2][BK][BN]; // 显式声明首地址 16 字节对齐
      // 每个线程负责 TM * TN 个输出
      float acc[TM][TN] = {0.0f};
      float regM[TM]; float regN[TN];
      // v4: 下一个 tile 的 LDG 结果先放在寄存器里，等当前 tile 算完再写入共享内存
      Vec a_pre[kLoadsA], b_pre[kLoadsB];

      // 第一步：发出 LDG，把第 t 列起的 tile 读进 a_pre / b_pre。越界的位置在这里直接填 0
      // LDG 发出后线程不会等待，直到后面的指令用到 a_pre / b_pre 时才会等数据（Long Scoreboard）
      auto load_tile_to_regs = [&](int t) {
        #pragma unroll
        for (int i = 0; i < kLoadsA; ++ i) {
          const int idx = tid + i * kThreads;
          const int r = idx * kVec / BK;
          const int col = idx * kVec % BK;
          // **这里依赖 K 是 kVec 倍数**：只判断第一个元素，第一个不越界则后面几个元素均不会越界
          if (blockIdx.y * BM + r >= M || t + col >= K) {
            #pragma unroll
            for (int j = 0; j < kVec; ++ j) a_pre[i].v[j] = static_cast<T>(0.0f);
          } else {
            a_pre[i] = *reinterpret_cast<const Vec*>(a + r * K + t + col);
          }
        }

        #pragma unroll
        for (int i = 0; i < kLoadsB; ++ i) {
          const int idx = tid + i * kThreads;
          const int r = idx * kVec / BN;
          const int col = idx * kVec % BN;
          // **这里依赖 N 是 kVec 倍数**，理由同上
          if (t + r >= K || blockIdx.x * BN + col >= N) {
            #pragma unroll
            for (int j = 0; j < kVec; ++ j) b_pre[i].v[j] = static_cast<T>(0.0f);
          } else {
            b_pre[i] = *reinterpret_cast<const Vec*>(b + (r + t) * N + col);
          }
        }
      };

      // 第二步：把 a_pre / b_pre 写入 ATILE[buf] / BTILE[buf]（A 转置存放），越界已在第一步处理
      auto store_regs_to_smem = [&](int buf) {
        #pragma unroll
        for (int i = 0; i < kLoadsA; ++ i) {
          const int idx = tid + i * kThreads;
          const int r = idx * kVec / BK;
          const int col = idx * kVec % BK;
          #pragma unroll
          for (int j = 0; j < kVec; ++ j) ATILE[buf][col + j][r] = static_cast<float>(a_pre[i].v[j]);
        }

        #pragma unroll
        for (int i = 0; i < kLoadsB; ++ i) {
          const int idx = tid + i * kThreads;
          const int r = idx * kVec / BN;
          const int col = idx * kVec % BN;
          // 先在寄存器里拼好 kVec 个 float，再整体写入：kVec=4 时为一条 STS.128，kVec=1 时为一条 STS.32
          FloatVec tmpB;
          #pragma unroll
          for (int j = 0; j < kVec; ++ j) tmpB.v[j] = static_cast<float>(b_pre[i].v[j]);
          *reinterpret_cast<FloatVec*>(&BTILE[buf][r][col]) = tmpB;
        }
      };

      // 用 ATILE[buf] / BTILE[buf] 计算一个 tile，沿共同的 BK 维度累加
      auto compute_tile = [&](int buf) {
        #pragma unroll
        for (int kk = 0; kk < BK; ++ kk) {
          // 每个线程负责 BM * BN 小块中的第 ty * TM ... ty * TM + TM - 1 行，列见 col_of
          #pragma unroll
          for (int i = 0; i < TM; ++ i) regM[i] = ATILE[buf][kk][threadIdx.y * TM + i];
          // LDS.128 按半个 warp（ty 相同、tx=0..15）处理，按 col_of 分组后 16 个线程读连续的 256 字节，
          // 每条指令 4 个 wavefront，达到理想值
          #pragma unroll
          for (int j = 0; j < TN; ++ j) regN[j] = BTILE[buf][kk][col_of(j)];

          #pragma unroll
          for (int i = 0; i < TM; ++ i) {
            #pragma unroll
            for (int j = 0; j < TN; ++ j) {
              acc[i][j] += regM[i] * regN[j];
            }
          }
        }
      };

      // 第 0 个 tile 没有可以重叠的计算，直接读入并同步
      load_tile_to_regs(0);
      store_regs_to_smem(0);
      __syncthreads();

      // v4: 每轮先发出下一个 tile 的 LDG，然后计算当前 tile，算完再把下一个 tile 写入另一块缓冲区。
      // LDG 的延迟被当前 tile 的 FFMA 掩盖。每轮只需一次同步：第 t 轮写的 [cur ^ 1] 最后一次被读是在
      // 第 t-1 轮的计算中，所有线程都已越过第 t-1 轮末尾的同步，不会覆盖别人正在读的数据
      int cur = 0;
      for (int t = 0; t < K; t += BK) {
        const bool has_next = t + BK < K; // 整个 block 取值相同，不会造成线程分歧
        if (has_next) load_tile_to_regs(t + BK);
        compute_tile(cur);
        if (has_next) store_regs_to_smem(cur ^ 1);
        // 让 [cur ^ 1] 的写入对所有线程可见，下一轮再计算它
        __syncthreads();
        cur ^= 1;
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
