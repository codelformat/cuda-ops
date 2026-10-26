#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>

namespace cuda_ops {

    template <typename T, int kThreads>
    __global__ void layer_norm_fwd_kernel(const T* __restrict__ x, const T* __restrict__ gamma,
                                        const T* __restrict__ beta, T* __restrict__ y,
                                        float* __restrict__ mean, float* __restrict__ rstd,
                                        int64_t N, float eps) {
        const int64_t row = blockIdx.x;
        const int64_t tid = threadIdx.x;
        // KThreads = 256个线程处理N个数
        // 为了合并访存，每个线程负责处理tid % KTthreads, 共循环 N / KThreads 次
        // **越界情况**：N不能整除Kthreads
        const int64_t BN = N / blockDim.x;
        // const int64_t bias = tid % blockDim.x;
        __shared__ float warp_sum[kThreads / 32];
        // __shared__ float 

        // if (tid == 0) {
        //     // sum = 0.0f;
        //     square_sum = 0.0f;
        // }
        // __syncthreads();

        // TODO: 越界情况处理
        float sum = 0.0f;
        // 计算每个线程负责元素的sum
        for (int i = threadIdx.x ; i < N; i += blockDim.x) {
            sum += x[row * N + i]; // 这里存在race，sum = sum + ... 都有先读sum的动作
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
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            square_sum += (x[row * N + i] - row_mean) * (x[row * N + i] - row_mean);
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
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            // 这里x是fp16类型，最好显示转换类型
            y[row * N + i] = static_cast<T>((static_cast<float>(x[row * N + i]) - row_mean) * row_rstd * gamma[i] + beta[i]);
        }
        __syncthreads();

    }

    // 对 x[M, N]的每一行： y = (x - mean) * rstd * gamma + beta
    // mean[M], rstd[M] 以 fp32 输出，供反向使用。 T 为 float / c10::Half / c10::BFloat16
    template <typename T>
    void launch_layer_norm_fwd(const T* x, const T* gamma, const T* beta, T* y, float* mean,
                            float* rstd, int64_t M, int64_t N, float eps, cudaStream_t stream) {
        // TODO: 实现 kernel 并在这里 launch
        constexpr int kThreads = 256;
        const dim3 grid(static_cast<unsigned int>(M)); // 每行一个block处理
        const dim3 block(kThreads); // KThreads=256个线程（8个warp）同时处理同一行
        layer_norm_fwd_kernel<T, kThreads><<<grid, block, 0, stream>>>(x, gamma, beta, y, mean, rstd, N, eps); // shared memory 开 16 int 存sum，y

    }
}  // namespace cuda_ops
