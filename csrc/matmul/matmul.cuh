#pragma once

#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>
#include "common/vec.h"

namespace cuda_ops {

    // c10::Half / c10::BFloat16 与 __half / __nv_bfloat16 二进制相同，WMMA 只认后者
    // 只为这两种类型特化，其他类型（如 float）误用时直接编译失败
    template <typename T> struct WmmaType;
    template <> struct WmmaType<c10::Half> { using type = __half; };
    template <> struct WmmaType<c10::BFloat16> { using type = __nv_bfloat16; };

    // v5.4 两个内联 PTX Helper 函数
    template <bool kTrans>
    __device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], const void* p) {
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
      if constexpr (kTrans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
      else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
    }

    template <typename T>
    __device__ __forceinline__ void mma_k16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
      float d0, d1, d2, d3;
      if constexpr (std::is_same_v<T, c10::Half>)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
      else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
      c[0] = d0; c[1] = d1; c[2] = d2; c[3] = d3;
    }

    // V5: Tensor Core
    template<typename T, int kThreads, int BM, int BN, int BK, int WM, int WN, int FM, int FN, int PAD>
    __global__ void __launch_bounds__(kThreads) matmul_kernel_tc(const T* A, const T* B, T* C,
                                    int64_t M, int64_t N, int64_t K) {
      using namespace nvcuda;
      using FragT = typename WmmaType<T>::type;
      // fragment<用途， M, N, K, 元素类型， 布局>
      // wmma::fragment<wmma::matrix_a, 16, 16, 16, FragT, wmma::row_major> a_frag[FM]; // A: 16 * 16
      // wmma::fragment<wmma::matrix_b, 16, 16, 16, FragT, wmma::row_major> b_frag[FN]; // B: 16 * 16
      // wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[FM][FN]; // C: 16 * 16, fp32, 不写布局
      
      // v5.2：先按线程并行搬运到共享内存，然后再按warp并行加载到fragment和计算
      // __shared__ alignas(16) T ATILE[BK][BM + PAD], BTILE[BK][BN]; // 这里需要16字节对齐，同时为了防止STS.128带来的bank 冲突，ATILE转置并PAD
      // 首先，load_matrix_sync要求32位对齐
      // 其次，之前 v4 的ATILE转置是为了读到寄存器的 LDS.128能读到连续地址，而现在 load_matrix_sync使用的是 ldmatrix，能自己处理布局(每次8行，每行16字节，还是需要PAD处理冲突)
      // v5.2: 双缓冲 cp.async
      __shared__ alignas(32) T ATILE[2][BM][BK + PAD], BTILE[2][BK][BN + PAD];

      // v5.4 写回C的中转缓冲：每个 warp 一块 16*16 的 half，共 2 KB
      // 寄存器里每个线程只有相邻 2 列，直接写 C 时一条指令每行只能写 16 B，一个 32 B sector 要发两次 L2 写请求
      // 先拼到 tmpH，再每线程写 8 B，每行 4 个线程正好写满一个 sector
      __shared__ alignas(32) T tmpH[WM * WN][16][16];

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

      // v5.4 计算lane所在行与列
      int ldm_r = lane % 16;
      int ldm_c = (lane / 16) * 8;

      // v5.3 M、N、K 都是 16 的倍数，内部 block 不可能越界。这个条件在 block 内取值相同，不会造成线程分歧
      const bool block_inside = blockIdx.y * BM + BM <= M && blockIdx.x * BN + BN <= N;
      // v5.3 预先算好每次搬运相对 A、B 小块起点的偏移，与 t 无关，每轮不用重算
      int64_t aOff[kALoads], bOff[kBLoads];
      #pragma unroll
      for (int i = 0; i < kALoads; ++ i) aOff[i] = (aRow + i * kARows) * K + aCol * 8;
      #pragma unroll
      for (int i = 0; i < kBLoads; ++ i) bOff[i] = (bRow + i * kBRows) * N + bCol * 8;

      // v5.4 用PTX内联函数手写wmma，加载和计算统一用PTX内联函数替代
      // 定义所需的寄存器，每线程对A 4reg, B 4reg，C c0[4] + c1[4];
      // 共随着warp处理FM * FN个 16*16 区域的计算
      uint32_t a_reg[FM][4], b_reg[FN][4];
      float c_reg[FM][FN * 2][4] = {};

      // v5.2 预先定义好搬运函数
      auto copy_tile = [&] (int buf, int t) {
        // v5.3 快路径：整个 tile 都在界内时，不做逐线程越界判断，减少非 HMMA 指令和分支
        if (block_inside && t + BK <= K) {
          const T* aTile = A + t;
          const T* bTile = B + t * N; // v5.3 这里的 t * N是64位乘法，每轮都要计算一遍
          #pragma unroll
          for (int i = 0; i < kALoads; ++ i)
            __pipeline_memcpy_async(&ATILE[buf][aRow + i * kARows][aCol * 8], aTile + aOff[i], 16); // 每轮地址重算，且buf改变基址也要 IMAD 重算
          #pragma unroll
          for (int i = 0; i < kBLoads; ++ i)
            __pipeline_memcpy_async(&BTILE[buf][bRow + i * kBRows][bCol * 8], bTile + bOff[i], 16);
          __pipeline_commit();
          return;
        }

        // v5.1 先按线程搬运到ATILE，BTILE
        // v5.2: 全面改用cp.async搬运并在最后提交，绕过寄存器中介
        #pragma unroll
        for (int i = 0; i < kALoads; ++ i) {
          int r = aRow + i * kARows;
          if (blockIdx.y * BM + r >= M || aCol * 8 + t >= K) { // v5.2 越界时不能用cp.async，只能普通写入 0，不变
            AlignedVec<T, 8> emptyA; 
            #pragma unroll
            for (int j = 0; j < 8; ++ j) emptyA.v[j] = static_cast<T>(0.0f);
            *reinterpret_cast<AlignedVec<T, 8>*>(&ATILE[buf][r][aCol * 8]) = emptyA;
          } else {
            // *reinterpret_cast<AlignedVec<T, 8>*>(&ATILE[buf][r][aCol * 8]) = *reinterpret_cast<const AlignedVec<T, 8>*>(A + r * K + aCol * 8 + t);
            // v5.2 改搬运逻辑
            __pipeline_memcpy_async(&ATILE[buf][r][aCol * 8], A + r * K + aCol * 8 + t, 16); // 一次传16字节 
          }
        }

        #pragma unroll
        for (int i = 0; i < kBLoads; ++ i) {
          int r = bRow + i * kBRows;
          if (blockIdx.x * BN + bCol * 8 >= N || t + r >= K) {
            AlignedVec<T, 8> emptyB; 
            #pragma unroll 
            for (int j = 0; j < 8; ++ j) emptyB.v[j] = static_cast<T>(0.0f);
            *reinterpret_cast<AlignedVec<T, 8>*>(&BTILE[buf][r][bCol * 8]) = emptyB;
          } else {
            // *reinterpret_cast<AlignedVec<T, 8>*>(&BTILE[buf][r][bCol * 8]) = *reinterpret_cast<const AlignedVec<T, 8>*>(B + (r + t) * N + bCol * 8);
            __pipeline_memcpy_async(&BTILE[buf][r][bCol * 8], B + (r + t) * N + bCol * 8, 16); // 一次传16字节
          }
        }

        // 最后需要把之前发出的拷贝打包成一批 commit
        __pipeline_commit();
      };

      auto compute_tile = [&] (int buf, bool has_next, int t) {
        // 针对ATILE, BTILE再做 Tensor Core计算
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
          // v5.5 先计算，再搬运，防止没有HMMA指令使 Tensor Core空转
          // kk=0时Tensor Core还没运行，先直接进行计算;
          // 接下来第二轮循环（kk == 16）有快的线程已经算完了，就先搬运下一块
          // 这样快的线程搬运时慢的线程在计算上一个kk，慢线程算完后快线程已经搬运完自己的部分开始下一个kk的计算，Tensor Core不空转
          // 省下了 kk = 0时, 所有线程一起拷贝下一块让tensor core空转的开销
          if (kk == 16 & has_next) copy_tile(buf ^ 1, t + BK); 

          // v5.4 不再用wmma::load_matrix_sync, 直接使用 PTX 内联
          // 加载 FN 组 B(全部加载进来)
          #pragma unroll
          for (int i = 0; i < FN; ++ i)
            ldm_x4<true>(b_reg[i], &BTILE[buf][kk + ldm_r][i * 16 + warpN * FN * 16 + ldm_c]);

          #pragma unroll
          for (int i = 0; i < FM; ++ i) {
            // 先加载一组 A
            ldm_x4<false>(a_reg[i], &ATILE[buf][i * 16 + warpM * FM * 16 + ldm_r][kk + ldm_c]); // 下面进行hmma计算的时候可以异步搬运接下来的a_reg[i]
            #pragma unroll
            for (int j = 0; j < FN; ++ j) { // 直接开始一组组A进行计算
              mma_k16<T>(c_reg[i][j * 2], a_reg[i], b_reg[j][0], b_reg[j][1]); // 分两次计算，第一次计算前8列
              mma_k16<T>(c_reg[i][j * 2 + 1], a_reg[i], b_reg[j][2], b_reg[j][3]); // 第二次计算后8列
            }
          }
        }
      };
      
      // v5.1 线程负责搬运
      A += (blockIdx.y * BM) * K;
      B += blockIdx.x * BN;
      C += (blockIdx.y * BM + FM * warpM * 16) * N + blockIdx.x * BN + FN * warpN * 16; // v5.1 C按照每warp写入，转移到warp起始地址

      int cur = 0;
      // 先发出第0份tile的拷贝并commit
      copy_tile(cur, 0);
 
      for(int t = 0; t < K; t += BK) { // 边界情况已经排除给v4 kernel处理
        // 等待上一轮（本线程）的 tile t搬运到齐了
        __pipeline_wait_prior(0);
        __syncthreads(); // 所有线程搬运到齐且上一轮计算全部结束
        const bool has_next = t + BK < K;
        // v5.5 合并到compute_tile中
        compute_tile(cur, has_next, t);

        // 搬运完做同步
        cur ^= 1;
      }

      // 线程结果在寄存器里的位置：c[0..1] 在 rowL 行，c[2..3] 在 rowL + 8 行，起始列都是 colL
      const int rowL = lane / 4;
      const int colL = (lane % 4) * 2;
      // 线程写回 C 时负责的位置：每行 16 列由 4 个线程分担，每个线程 4 列
      const int colW = (lane % 4) * 4;

      // 写回
      // 一个 warp 负责 64(FM*16) * 64(FN*16), 一个fragment是其中的16*16, 现在要一个个写回fragment(共循环FM*FN次)，要定位到block里fragment对应位置
      #pragma unroll
      for (int i = 0; i < FM; ++ i){
        #pragma unroll
        for (int j = 0; j < FN; ++ j){
          // 先判断是否整块越界
          if (blockIdx.y * BM + warpM * FM * 16 + i * 16 >= M || blockIdx.x * BN + warpN * FN * 16 + j * 16 >= N) continue;

          T* frag = C + i * 16 * N + j * 16;
          T* tmp = &tmpH[warpId][0][0];

          AlignedVec<T, 2> d00, d01, d10, d11;
          d00.v[0] = static_cast<T>(c_reg[i][j * 2][0]); // c0[0], [R, C]
          d00.v[1] = static_cast<T>(c_reg[i][j * 2][1]); // c0[1], [R, C+1]

          d01.v[0] = static_cast<T>(c_reg[i][j * 2][2]); // c0[2], [R+8, C]
          d01.v[1] = static_cast<T>(c_reg[i][j * 2][3]); // c0[3], [R+8, C+1]

          d10.v[0] = static_cast<T>(c_reg[i][j * 2 + 1][0]); // c1[0], [R, C+8]
          d10.v[1] = static_cast<T>(c_reg[i][j * 2 + 1][1]); // c1[1], [R, C+9]

          d11.v[0] = static_cast<T>(c_reg[i][j * 2 + 1][2]); // c1[2], [R+8, C+8]
          d11.v[1] = static_cast<T>(c_reg[i][j * 2 + 1][3]); // c1[3], [R+8, C+9]

          // 位置规则和直接写 C 相同，只是行距从 N 换成 16
          *reinterpret_cast<AlignedVec<T, 2>*>(tmp + rowL * 16 + colL) = d00;
          *reinterpret_cast<AlignedVec<T, 2>*>(tmp + (rowL + 8) * 16 + colL) = d01;
          *reinterpret_cast<AlignedVec<T, 2>*>(tmp + rowL * 16 + colL + 8) = d10;
          *reinterpret_cast<AlignedVec<T, 2>*>(tmp + (rowL + 8) * 16 + colL + 8) = d11;
          __syncwarp(); // 下面要读其他线程写入 tmpH 的数据，必须等它们写完

          *reinterpret_cast<AlignedVec<T, 4>*>(frag + rowL * N + colW) = *reinterpret_cast<const AlignedVec<T, 4>*>(tmp + rowL * 16 + colW);
          *reinterpret_cast<AlignedVec<T, 4>*>(frag + (rowL + 8) * N + colW) = *reinterpret_cast<const AlignedVec<T, 4>*>(tmp + (rowL + 8) * 16 + colW);
          __syncwarp(); // 下一个小块会复用 tmpH，必须等所有线程读完
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
