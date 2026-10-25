# cuda-ops

CUDA 算子开发仓库。开发环境完全由 Docker 定义（`docker/Dockerfile` + `compose.yaml`），宿主机只需要 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。

## 环境

| 组件 | 版本 |
|---|---|
| 基础镜像 | `nvidia/cuda:13.2.1-devel-ubuntu24.04` |
| 目标架构 | `sm_86`（可通过 `-DCMAKE_CUDA_ARCHITECTURES` 覆盖） |

## 快速开始

VS Code：Remote-SSH 连到开发机，打开本目录，执行 `Dev Containers: Reopen in Container`。

命令行：

```bash
docker compose build          # 修改 Dockerfile / requirements-dev.txt 后需要重新构建
docker compose up -d
docker compose exec dev bash

# 以下在容器内执行
cmake --preset release
cmake --build --preset release
ctest --preset release
compute-sanitizer ./build/release/test_vector_add
```

国内网络可以在 `.env` 中配置镜像源后再构建：

```
APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
```

## 目录结构

```
include/ops/   对外头文件
src/           算子实现
tests/         正确性测试（与 CPU 参考实现比对）
benchmarks/    性能测试
docker/        开发镜像定义
```

## 约定

- 环境变更只能通过修改 `docker/Dockerfile` 或 `requirements-dev.txt` 并提交，不要在运行中的容器里临时安装依赖。
- 提交前在容器内运行 `pre-commit install`（只需一次），由 hook 执行 clang-format。
- 每个新算子都要附带正确性测试，并通过 `compute-sanitizer` 检查。
