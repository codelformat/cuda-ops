#pragma once

#include <mma.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include "common/vec.h"

namespace cuda_ops {

    // c10::Half / c10::BFloat16 与 __half / __nv_bfloat16 二进制相同，WMMA 只认后者
    // 只为这两种类型特化，其他类型（如 float）误用时直接编译失败
    template <typename T> struct WmmaType;
    template <> struct WmmaType<c10::Half> { using type = __half; };
    template <> struct WmmaType<c10::BFloat16> { using type = __nv_bfloat16; };

    // V5: Tensor Core
    template<typename T, int kThreads, int BM, int BN, int BK, int WM, int WN, int FM, int FN, int PAD>
    __global__ void __launch_bounds__(kThreads) matmul_kernel_tc(const T* A, const T* B, T* C,
                                    int64_t M, int64_t N, int64_t K) {
      using namespace nvcuda;
      using FragT = typename WmmaType<T>::type;
      // fragment<用途， M, N, K, 元素类型， 布局>
      wmma::fragment<wmma::matrix_a, 16, 16, 16, FragT, wmma::row_major> a_frag[FM]; // A: 16 * 16
      wmma::fragment<wmma::matrix_b, 16, 16, 16, FragT, wmma::row_major> b_frag[FN]; // B: 16 * 16
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[FM][FN]; // C: 16 * 16, fp32, 不写布局
      
      // v5.2：先按线程并行搬运到共享内存，然后再按warp并行加载到fragment和计算
      // __shared__ alignas(16) T ATILE[BK][BM + PAD], BTILE[BK][BN]; // 这里需要16字节对齐，同时为了防止STS.128带来的bank 冲突，ATILE转置并PAD
      // 首先，load_matrix_sync要求32位对齐
      // 其次，之前 v4 的ATILE转置是为了读到寄存器的 LDS.128能读到连续地址，而现在 load_matrix_sync使用的是 ldmatrix，能自己处理布局
      __shared__ alignas(32) T ATILE[BM][BK + PAD], BTILE[BK][BN + PAD];

      // 写回C的float缓冲
      // 因为是整个 warp = 32线程协作，所以 每个只负责 256 / 32 = 8个float
      // store_matrix_sync 要求目标地址 32 字节对齐；每个 warp 一块 16*16*4 = 1 KB，块首地址也都是 32 的倍数
      __shared__ alignas(32) float tmpC[WM * WN][16][16];

      // 先移动到要计算的位置
      int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32; // 0, 1, 2, 3
      int warpM = warpId / WN; // 0, 1
      int warpN = warpId % WN; // 0, 1
      // 线程负责写回的位置
      int tid = threadIdx.y * blockDim.x + threadIdx.x;
      // 负责搬运：按线程在block里分工
      constexpr int kACols = BK / 8; // 进入该算子的已经是 fp16 变量，一次float4读可以读8个数
      const int aRow = tid / kACols; // 每个tid对应的在A小块中的初始行号
      const int aCol = tid % kACols; // 每个tid对应的在A小块中的列号
      constexpr int kARows = kThreads / kACols; // 每次循环所有线程可以搬ATILE多少行
      constexpr int kALoads = BM / kARows; // 总共需要多少次循环可以搬运完A小块

      constexpr int kBCols = BN / 8;
      const int bRow = tid / kBCols;
      const int bCol = tid % kBCols;
      constexpr int kBRows = kThreads / kBCols;
      constexpr int kBLoads = BK / kBRows;

      int lane = tid % 32; // 线程在 warp 中的位置
      // 负责写回：在warp里分工
      // 256个float，32个线程，每个线程写8个，每次4个，共 16 * 16 / (32 * 4) = 2 次
      // 一行 16 列由 4 个相邻线程负责，32 个线程一次覆盖 [8][16]，第 i 次写第 i * 8 + laneY 行
      constexpr int kVec = 4;
      constexpr int kLaneCols = 16 / kVec;         // 4
      constexpr int kLaneRows = 32 / kLaneCols;    // 8
      constexpr int kStores = 16 / kLaneRows;      // 2
      int laneY = lane / kLaneCols;
      int laneX = lane % kLaneCols;


      // host 只保证 M、N 是 16 的倍数，不一定是 BM、BN 的倍数（如 M = 16 时第二行 warp 整个越界）
      // 整个 warp 一起退出是安全的：32 个线程都不调用 _sync 函数，kernel 里也没有 __syncthreads
      // if (blockIdx.y * BM + warpM * 16 >= M || blockIdx.x * BN + warpN * 16 >= N) {
      //   // return; // v5.1: 使用了线程做搬运，越界的warp也有搬运任务，因此越界改为填 0
      // }

      // A += (blockIdx.y * BM + warpM * 16) * K; // 一个block里2个warpM负责一次t - t+BK的所有A中的[BM][BK]小块计算， BM是WM 2倍
      // B += blockIdx.x * BN + warpN * 16; // 一个block里2个warpM负责一次t - t+BK的所有B中的[BK][BN]小块计算， BN是WN 2倍
      // C += (blockIdx.y * BM + FM * warpM * 16) * N + blockIdx.x * BN + FN * warpN * 16; // v5.1 基址偏移重算

      // v5.1 线程负责搬运
      A += (blockIdx.y * BM) * K;
      B += blockIdx.x * BN;
      C += (blockIdx.y * BM + FM * warpM * 16) * N + blockIdx.x * BN + FN * warpN * 16; // v5.1 C按照每warp写入，转移到warp起始地址
      #pragma unroll
      for (int i = 0; i < FM; ++ i) 
        #pragma unroll
        for (int j = 0; j < FN; ++ j) 
          wmma::fill_fragment(c_frag[i][j], 0.0f);
 
      for(int t = 0; t < K; t += BK) { // 边界情况已经排除给v4 kernel处理
        // v5.2 先按线程搬运到ATILE，BTILE
        #pragma unroll
        for (int i = 0; i < kALoads; ++ i) {
          int r = aRow + i * kARows;
          if (blockIdx.y * BM + r >= M || aCol * 8 + t >= K) {
            AlignedVec<T, 8> emptyA; 
            #pragma unroll
            for (int j = 0; j < 8; ++ j) emptyA.v[j] = static_cast<T>(0.0f);
            *reinterpret_cast<AlignedVec<T, 8>*>(&ATILE[r][aCol * 8]) = emptyA;
          } else {
            *reinterpret_cast<AlignedVec<T, 8>*>(&ATILE[r][aCol * 8]) = *reinterpret_cast<const AlignedVec<T, 8>*>(A + r * K + aCol * 8 + t);
          }
        }

        #pragma unroll
        for (int i = 0; i < kBLoads; ++ i) {
          int r = bRow + i * kBRows;
          if (blockIdx.x * BN + bCol * 8 >= N || t + r >= K) {
            AlignedVec<T, 8> emptyB; 
            #pragma unroll 
            for (int j = 0; j < 8; ++ j) emptyB.v[j] = static_cast<T>(0.0f);
            *reinterpret_cast<AlignedVec<T, 8>*>(&BTILE[r][bCol * 8]) = emptyB;
          } else {
            *reinterpret_cast<AlignedVec<T, 8>*>(&BTILE[r][bCol * 8]) = *reinterpret_cast<const AlignedVec<T, 8>*>(B + (r + t) * N + bCol * 8);
          }
        }
        // 搬运完做同步
        __syncthreads();

        // 针对ATILE, BTILE再做 Tensor Core计算
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
          // wmma::load_matrix_sync(a_frag, reinterpret_cast<const FragT*>(A + t), K); // 从A指向的16*16小块左上角； lda: 一行有多少元素
          // wmma::load_matrix_sync(b_frag, reinterpret_cast<const FragT*>(B + t * N), N);
          // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); // c = a·b + c

          // 这里和 v4_kernel 一样
          #pragma unroll
          for (int i = 0; i < FM; ++ i) wmma::load_matrix_sync(a_frag[i], reinterpret_cast<const FragT*>(&ATILE[warpM * FM * 16 + i * 16][kk]), BK + PAD);
          #pragma unroll
          for (int i = 0; i < FN; ++ i) wmma::load_matrix_sync(b_frag[i], reinterpret_cast<const FragT*>(&BTILE[kk][warpN * FN * 16 + i * 16]), BN + PAD);

          #pragma unroll
          for (int i = 0; i < FM; ++ i){
            #pragma unroll
            for (int j = 0; j < FN; ++ j){
              wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
          }
        }

        __syncthreads();
      }

      // 写回
      // 一个 warp 负责 64(FM*16) * 64(FN*16), 一个fragment是其中的16*16, 现在要一个个写回fragment(共循环FM*FN次)，要定位到block里fragment对应位置
      #pragma unroll
      for (int t = 0; t < (FM * FN); ++ t) {
        // 判断是否在界内，因为M, N整除16，所以要么整个fragment全在界内要么全不在
        if (blockIdx.y * BM + warpM * FM * 16 + (t / FN) * 16 >= M || blockIdx.x * BN + warpN * FN * 16 + (t % FN) * 16 >= N) continue;

        // 必须先存进float缓冲，再转换成半精度写回C
        // tmpC[warpId] 是 float[16][16]，退化后是 float(*)[16]，要用 &[0][0] 拿到 float*
        wmma::store_matrix_sync(&tmpC[warpId][0][0], c_frag[t / FN][t % FN], 16, wmma::mem_row_major);
        // warp 内同步：本 warp 32 个线程都写完 tmpC[warpId]，才能读别的线程写的位置
        // 每个 warp 只读写自己那块，不需要 __syncthreads
        __syncwarp();
        // 每个线程读 4 个 float，逐个转换成 T，再整体写入 C（half 时为一条 8 字节的 STG.64）
        #pragma unroll
        for (int i = 0; i < kStores; ++ i) {
          const int row = i * kLaneRows + laneY;
          const int col = laneX * kVec;
          const AlignedVec<float, kVec> src = *reinterpret_cast<const AlignedVec<float, kVec>*>(&tmpC[warpId][row][col]);
          AlignedVec<T, kVec> dst;
          #pragma unroll
          for (int j = 0; j < kVec; ++ j) dst.v[j] = static_cast<T>(src.v[j]);
          *reinterpret_cast<AlignedVec<T, kVec>*>(C + (row + (t / FN) * 16) * N + col + (t % FN) * 16) = dst;
        }
      }
    }

    // 每个 block 计算 C 的 BM * BN 小块，每个线程计算其中 TM * TN 个元素，沿 K 每次处理 BK
    // __launch_bounds__ 第二个参数要求每个 SM 至少放 2 个 block，编译器会把寄存器限制在 128 以内
    // （v4 不加时 kVec=1 的版本为 130 个，每个 SM 只能放 1 个 block）
    template<typename T, int kThreads, int BM, int BN, int BK, int TM, int TN, int PAD, int kVec>
    __global__ void __launch_bounds__(kThreads, 2) matmul_kernel(const T* a, const T* b, T* c,
                                            int64_t M, int64_t N, int64_t K) {
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

      auto v4_kernel = [&] {
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
      };

      if constexpr (std::is_same_v<T, float>) { // fp32
        v4_kernel();
      } else { //fp16 / bf16; M, N, K 都是 16 倍数时走 Tensor Core
        // load_matrix_sync 要求指针 32 字节对齐：K、N 是 16 的倍数时每行起点都是 32 字节的倍数，只需检查基地址
        // c 来自 at::empty，至少 256 字节对齐，8 字节写入不需要检查
        if ((M % 16 != 0) || (N % 16 != 0) || (K % 16 != 0)
        || reinterpret_cast<uintptr_t>(a) % 32 != 0
        || reinterpret_cast<uintptr_t>(b) % 32 != 0)
          v4_kernel();
        else {
          // v5.2: BM=BN=128, BK=32
          constexpr int BM = 128; constexpr int BN = 128; constexpr int BK = 32;
          constexpr int WM = 2; constexpr int WN = 2;
          constexpr int kThreads = 128;
          constexpr int PAD = 8; // v5.1 预留
          constexpr int FM = BM / WM / 16, FN = BN / WN / 16;


          dim3 block(WN * 32, WM);
          const dim3 grid (
            static_cast<unsigned int>((N + BN - 1) / (BN)),
            static_cast<unsigned int>((M + BM - 1) / (BM))
          );

          matmul_kernel_tc<T, kThreads, BM, BN, BK, WM, WN, FM, FN, PAD><<<grid, block, 0, stream>>>(a, b, c, M, N, K);
        }
      }
    }
}  // namespace cuda_ops
