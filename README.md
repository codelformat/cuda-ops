# cuda-ops

CUDA 算子开发仓库：算子以 PyTorch C++/CUDA 扩展的形式实现，通过 pybind11 暴露为 Python 包 `cuda_ops`，并以 PyTorch 实现作为精度基准。

开发环境完全由 Docker 定义（`docker/Dockerfile` + `compose.yaml`），宿主机只需要 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。

## 环境

| 组件 | 版本 |
|---|---|
| 基础镜像 | `nvidia/cuda:13.2.1-devel-ubuntu24.04` |
| PyTorch | `2.14.1+cu132` |
| 目标架构 | `sm_86`（由镜像中的 `TORCH_CUDA_ARCH_LIST` 指定） |

## 快速开始

VS Code：Remote-SSH 连到开发机，打开本目录，执行 `Dev Containers: Reopen in Container`。容器创建后会自动执行一次 `pip install -e .`。

命令行：

```bash
docker compose up -d
docker compose exec dev bash

# 以下在容器内执行
pip install -e . --no-build-isolation   # 修改任何 .cu/.cuh/.cpp 后都需要重新执行
pytest tests/
compute-sanitizer python -m pytest tests/ -q
```

修改 `docker/Dockerfile` 或 `requirements-dev.txt` 后需要重新构建镜像（VS Code 中执行 `Dev Containers: Rebuild Container`）。

## 目录结构

```
csrc/
  common/             各算子共用的工具（dtype 分发、参数检查、归约、向量化访存）
  <op>/<op>.cu        host 代码：参数检查、输出分配、dtype 分发、调用 launcher
  <op>/<op>.cuh       kernel 模板与 launcher，不引入 torch
  ops.h               所有算子 host 函数的声明
  bindings.cpp        唯一的 PYBIND11_MODULE
python/cuda_ops/      Python 封装层（对外接口）
tests/                正确性测试（以 PyTorch 为参考实现）
benchmarks/           性能测试
docs/                 每个算子的手算与 Nsight Compute 分析记录
```

## 新增一个算子

1. 新建 `csrc/<op>/<op>.cuh`、`csrc/<op>/<op>.cu`；在 `csrc/ops.h` 声明 host 函数，在 `csrc/bindings.cpp` 注册（`setup.py` 会自动收集源文件）。
2. 新建 `python/cuda_ops/<op>.py` 并在 `__init__.py` 导出。
3. 新建 `tests/test_<op>.py`：覆盖 fp32/fp16/bf16、边界 shape 与非法输入。
4. 新建 `benchmarks/bench_<op>.py` 与 `docs/<op>.md`：记录理论访存/计算量、ncu 数据和每个优化版本的结果。

## 约定

- 环境变更只能通过修改 `docker/Dockerfile` 或 `requirements-dev.txt` 并提交，不要在运行中的容器里临时安装依赖。
- 只有 `csrc/bindings.cpp` 引入 `torch/extension.h`；`.cu` 只引入 ATen/c10 头文件，kernel 头文件不引入 torch。
- 统计量（mean、rstd 等）一律以 fp32 输出；fp16/bf16 的 kernel 内部以 fp32 累加。
- 提交前在容器内运行一次 `pre-commit install`，由 hook 执行 clang-format。
- 每个新算子都要附带正确性测试，并通过 `compute-sanitizer` 检查。
