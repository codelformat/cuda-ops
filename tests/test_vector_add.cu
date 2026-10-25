#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ops/vector_add.cuh"

#define CHECK_CUDA(expr)                                                              \
  do {                                                                                \
    cudaError_t err_ = (expr);                                                        \
    if (err_ != cudaSuccess) {                                                        \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,            \
                   cudaGetErrorString(err_));                                         \
      std::exit(EXIT_FAILURE);                                                        \
    }                                                                                 \
  } while (0)

// 与 CPU 参考实现逐元素比对；浮点加法单步运算结果确定，因此要求完全相等
static bool run_case(std::size_t n) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> a(n), b(n), c(n), ref(n);
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = dist(rng);
    b[i] = dist(rng);
    ref[i] = a[i] + b[i];
  }

  const std::size_t bytes = n * sizeof(float);
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

  CHECK_CUDA(ops::vector_add(d_a, d_b, d_c, n));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));

  for (std::size_t i = 0; i < n; ++i) {
    if (c[i] != ref[i]) {
      std::fprintf(stderr, "n=%zu: mismatch at %zu: got %f, expected %f\n", n, i, c[i], ref[i]);
      return false;
    }
  }
  return true;
}

int main() {
  // 覆盖空输入、不满一个 block、block 边界、非 2 的幂，以及超过最大 grid 需要循环的规模
  const std::size_t sizes[] = {0, 1, 255, 256, 257, (1u << 20) + 3, 65535u * 256 * 2 + 7};
  int failures = 0;
  for (std::size_t n : sizes) {
    const bool ok = run_case(n);
    std::printf("[%s] n=%zu\n", ok ? "PASS" : "FAIL", n);
    failures += ok ? 0 : 1;
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
