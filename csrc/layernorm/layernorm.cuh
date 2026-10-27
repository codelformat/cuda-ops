#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

namespace cuda_ops {

    // __launch_bounds__ 告诉编译器 block 最多 kThreads 个线程，寄存器分配不会超过
    // 65536 / kThreads 个/线程，否则 kThreads=1024 时可能 "too many resources requested for launch"
    template <typename T, int kThreads, int kSize>
    __global__ void __launch_bounds__(kThreads) layer_norm_fwd_kernel(const T* __restrict__ x, const T* __restrict__ gamma,
                                        const T* __restrict__ beta, T* __restrict__ y,
                                        float* __restrict__ mean, float* __restrict__ rstd,
                                        int64_t N, float eps) {
        const int64_t row = blockIdx.x;
        const int64_t tid = threadIdx.x;
        // KThreads = 256个线程处理N个数
        // 为了合并访存，每个线程负责处理tid, 共循环 N / KThreads 次
        __shared__ float warp_sum[kThreads / 32];

        float x_reg[kSize] = {0}; // 每个线程存下负责的（最多）kSize个x对应元素

        // 所有线程先搬运
        #pragma unroll
        for (int k = 0; k < kSize; ++k) {
            const int i = k * kThreads + threadIdx.x;
            if (i < N) x_reg[k] = static_cast<float>(x[row * N + i]);
        }

        // TODO: 越界情况处理
        float sum = 0.0f;
        // 计算每个线程负责元素的sum
        #pragma unroll
        for (int k = 0; k < kSize; ++k) {
            const int i = k * kThreads + threadIdx.x;
            if (i < N) sum += x_reg[k]; // 这里存在race，sum = sum + ... 都有先读sum的动作
        }
        // 计算线程所属warp内的sum
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        const int warpId = threadIdx.x / 32; const int lane = threadIdx.x % 32;
        if (lane == 0) warp_sum[warpId] = sum;
        // 等所有线程写完warp_sum
        __syncthreads();
        float total = 0.0f;
        for (int w = 0; w < kThreads / 32; ++w) total += warp_sum[w];

        float row_mean = total / N;// 先不做搬运，暂存到寄存器 // 后来想想错了，搬运不影响线程往下走，接下来会继续计算，所以直接搬到内存不影响。 // 后来看到后续计算还要用到这个值，防止重复搬运，还是存在寄存器里先
        if (tid == 0) mean[row] = total / N; 
        
        // 计算每个线程负责的方差分子和
        float square_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < kSize; ++k) {
            const int i = k * kThreads + threadIdx.x;
            if (i < N) square_sum += (x_reg[k] - row_mean) * (x_reg[k] - row_mean);
        }
        // 计算每个warp的方差分子和
        for (int offset = 16; offset > 0; offset >>= 1) {
            square_sum += __shfl_xor_sync(0xffffffff, square_sum, offset);
        }
        __syncthreads();
        // 复用 warp_sum 和 total
        total = 0.0f;
        if (lane == 0) warp_sum[warpId] = square_sum;
        __syncthreads();
        for (int w = 0; w < kThreads / 32; ++w) total += warp_sum[w];

        float row_rstd = rsqrtf(total / N + eps);
        if (tid == 0) rstd[row] = row_rstd;
        
        // 写回
        #pragma unroll
        for (int k = 0; k < kSize; ++k) {
            const int i = k * kThreads + threadIdx.x;
            // 这里x是fp16类型，最好显示转换类型
            if (i < N) y[row * N + i] = static_cast<T>((x_reg[k] - row_mean) * row_rstd * gamma[i] + beta[i]);
        }
        __syncthreads();

    }

    // 兜底 kernel（v1）：不缓存，x 从全局内存读三遍。寄存器放不下整行时（N > 16384）使用
    template <typename T, int kThreads>
    __global__ void __launch_bounds__(kThreads) layer_norm_fwd_uncached_kernel(
        const T* __restrict__ x, const T* __restrict__ gamma, const T* __restrict__ beta,
        T* __restrict__ y, float* __restrict__ mean, float* __restrict__ rstd, int64_t N,
        float eps) {
        const int64_t row = blockIdx.x;
        const int64_t tid = threadIdx.x;
        __shared__ float warp_sum[kThreads / 32];

        float sum = 0.0f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            sum += static_cast<float>(x[row * N + i]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        const int warpId = threadIdx.x / 32; const int lane = threadIdx.x % 32;
        if (lane == 0) warp_sum[warpId] = sum;
        __syncthreads();
        float total = 0.0f;
        for (int w = 0; w < kThreads / 32; ++w) total += warp_sum[w];

        float row_mean = total / N;
        if (tid == 0) mean[row] = row_mean;

        float square_sum = 0.0f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            const float d = static_cast<float>(x[row * N + i]) - row_mean;
            square_sum += d * d;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            square_sum += __shfl_xor_sync(0xffffffff, square_sum, offset);
        }
        __syncthreads();
        total = 0.0f;
        if (lane == 0) warp_sum[warpId] = square_sum;
        __syncthreads();
        for (int w = 0; w < kThreads / 32; ++w) total += warp_sum[w];

        float row_rstd = rsqrtf(total / N + eps);
        if (tid == 0) rstd[row] = row_rstd;

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            y[row * N + i] = static_cast<T>((static_cast<float>(x[row * N + i]) - row_mean) * row_rstd *
                                            static_cast<float>(gamma[i]) + static_cast<float>(beta[i]));
        }
    }

    // 对 x[M, N]的每一行： y = (x - mean) * rstd * gamma + beta
    // mean[M], rstd[M] 以 fp32 输出，供反向使用。 T 为 float / c10::Half / c10::BFloat16
    template <typename T>
    void launch_layer_norm_fwd(const T* x, const T* gamma, const T* beta, T* y, float* mean,
                            float* rstd, int64_t M, int64_t N, float eps, cudaStream_t stream) {
        const dim3 grid(static_cast<unsigned int>(M)); // 每行一个block处理
        auto launch = [&](auto kernel, int threads) {
            kernel<<<grid, threads, 0, stream>>>(x, gamma, beta, y, mean, rstd, N, eps);
        };
        // 每线程缓存 kSize = ceil(N / kThreads) 个元素，向上取到 2 的幂以控制模板实例数量。
        // kSize 封顶 16，行再长就加线程而不是加寄存器：寄存器多了会降 occupancy，甚至溢出到 local memory
        if (N <= 256) launch(layer_norm_fwd_kernel<T, 256, 1>, 256);
        else if (N <= 512) launch(layer_norm_fwd_kernel<T, 256, 2>, 256);
        else if (N <= 1024) launch(layer_norm_fwd_kernel<T, 256, 4>, 256);
        else if (N <= 2048) launch(layer_norm_fwd_kernel<T, 256, 8>, 256);
        else if (N <= 4096) launch(layer_norm_fwd_kernel<T, 256, 16>, 256);
        else if (N <= 8192) launch(layer_norm_fwd_kernel<T, 512, 16>, 512);
        else if (N <= 16384) launch(layer_norm_fwd_kernel<T, 1024, 16>, 1024);
        else launch(layer_norm_fwd_uncached_kernel<T, 256>, 256);
    }
}  // namespace cuda_ops
