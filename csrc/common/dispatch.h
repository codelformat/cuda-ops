#pragma once

#include <ATen/Dispatch.h>
#include <c10/util/Exception.h>

// 只为 fp32 / fp16 / bf16 实例化模版；其他 dtype 会在运行时报 "not implemented for"
#define CUDA_OPS_DISPATCH_FLOAT_HALF_BF16(TYPE, NAME, ...)                       \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                                 \
                     AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__) AT_DISPATCH_CASE( \
                         at::kHalf, __VA_ARGS__) AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__))

#define CHECK_CUDA(t) TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(t) TORCH_CHECK((t).is_contiguous(), #t " must be contiguous")
#define CHECK_INPUT(t) \
  CHECK_CUDA(t);       \
  CHECK_CONTIGUOUS(t)
