#pragma once

namespace cuda_ops {

// kVec 个 T 打包成一个按总大小对齐的结构体，总大小为 16 字节时编译为一条 128 位读写指令
template <typename T, int kVec>
struct alignas(sizeof(T) * kVec) AlignedVec {
  T v[kVec];
};

}  // namespace cuda_ops
