# cuda-ops

手写 CUDA 算子：LayerNorm、SGEMM 和 Tensor Core GEMM，以 PyTorch C++/CUDA 扩展的形式实现，通过 pybind11 暴露为 Python 包 `cuda_ops`，并以 PyTorch 作为精度基准。每一版都先手算访存量和指令数，再用 Nsight Compute 对账，记录在 [`docs/`](docs/) 中。

**English:** Hand-written CUDA kernels (LayerNorm, SGEMM and a Tensor Core GEMM) built as a PyTorch C++/CUDA extension. Every version is estimated by hand first and then checked against Nsight Compute. At 4096³ on an RTX 3080 Ti Laptop with the SM clock locked at 802 MHz, the fp16 GEMM runs at 96.9% of cuBLAS and the fp32 SGEMM at 85.6%. The docs are written in Chinese.

## 性能

矩阵乘法，4096³，相对同精度的 cuBLAS：

| dtype | 实现 | 相对 cuBLAS |
|---|---|---:|
| fp16 | Tensor Core（`ldmatrix` + `mma.sync`） | 96.9% |
| bf16 | Tensor Core（`ldmatrix` + `mma.sync`） | 96.9% |
| fp32 | CUDA core（分块 + 双缓冲） | 85.6% |

LayerNorm，8192×4096：fp32 为 477 GB/s，达到显存带宽峰值的 93%，与同样数据量的显存拷贝相当，是 PyTorch 的 1.5 倍。

**测试条件：** RTX 3080 Ti Laptop（sm_86），矩阵乘法在宿主机用 `tools/bench_locked.sh` 把 SM 频率锁在 802 MHz 后计时，关闭 TF32；LayerNorm 未锁频。详细数据、按周期的对比和测量方法见 [`docs/matmul.md`](docs/matmul.md) 的“当前结果”一节和 [`docs/layernorm.md`](docs/layernorm.md)。

**还有差距的地方：**
- 上面的矩阵乘法数字只针对 4096³。其他形状还有差距，例如 2048³ 为 0.76×，64×4096×4096 为 0.36×，原因和计划见 `docs/matmul.md`。
- fp16 / bf16 只在 M、N、K 都是 16 的倍数，且输入首地址按 32 字节对齐时走 Tensor Core 版本，否则退回 CUDA core 版本。
- LayerNorm 在很宽的行上较慢，例如 256×65536 时是 PyTorch 的 0.76×。

## 视频课程与版本索引

这个仓库的开发过程正在制作成中文视频课程。每节对应一个 tag，指向那一节结束时的代码：

| 节 | 主题 | tag | 提交 |
|---|---|---|---|
| 1 | CUDA 基础与开发全链路 | `ep01-vector-add`（第一个 kernel）、`ep01` | `7ee9745`、`c250b86` |
| 2 | LayerNorm：访存密集型算子 | 待发布 | `5bdbed0` |
| 3 | 朴素矩阵乘法与性能分析 | 待发布 | `38ffae9` |
| 4 | Tiling：分块与数据复用 | 待发布 | `daab96c` |
| 5 | Shared memory 与 bank conflict | 待发布 | `d4eb993` |
| 6 | 延迟隐藏：双缓冲 | 待发布 | `a19e30d` |
| 7 | Tensor Core 入门 | 待发布 | `745fb41` |
| 8 | 异步拷贝与指令开销 | 待发布 | `5a7405e` |
| 9 | 手写 PTX 与写回优化 | 待发布 | `cc63ebb` |
| 10 | 最后一公里：同步点与调度 | 待发布 | 待定 |

```bash
git checkout ep01        # 切到某一节的代码
git fetch --tags         # 之前 clone 过的仓库，先拉取新发布的 tag
git checkout main        # 回到最新代码
```

tag 是当时的代码快照，里面的 README 和文档可能还是旧版本，说明以 `main` 为准。部分提交的说明与实际内容不符，对照表见 `docs/matmul.md` 的“版本与提交对照”一节。

## 硬件与系统要求

- **GPU：** NVIDIA GPU，计算能力 8.0（Ampere）及以上。Tensor Core 版本用到 `cp.async`、`ldmatrix` 和 `mma.sync.m16n8k16`，更早的 GPU 目前无法编译。只在 RTX 3080 Ti Laptop（sm_86）上测试过。
- **系统：** Linux，装好 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。macOS 不能运行 CUDA，只能编辑代码；Windows 上的 WSL2 没有测试过。
- **Nsight Compute 权限：** Linux 驱动默认只允许管理员读取性能计数器，此时 `ncu` 会报 `ERR_NVGPUCTRPERM`。可以在宿主机上新建 `/etc/modprobe.d/nvidia-profiling.conf`，写入 `options nvidia NVreg_RestrictProfilingToAdminUsers=0` 后重启（Ubuntu 等发行版重启前还要运行 `sudo update-initramfs -u`）；也可以用 root 运行 `ncu`。当前状态可以用 `python tools/env_report.py` 查看。

## 环境

开发环境完全由 Docker 定义（`docker/Dockerfile` + `compose.yaml`）：

| 组件 | 版本 |
|---|---|
| 基础镜像 | `nvidia/cuda:13.2.1-devel-ubuntu24.04` |
| PyTorch | `2.14.1+cu132` |
| 目标架构 | 默认 `sm_86`，由 `TORCH_CUDA_ARCH_LIST` 指定，见下文 |

## 快速开始

VS Code：Remote-SSH 连到开发机，打开本目录，执行 `Dev Containers: Reopen in Container`。容器创建后会自动执行一次 `pip install -e .`。

命令行：

```bash
docker compose up -d
docker compose exec dev bash

# 以下在容器内执行
pip install -e . --no-build-isolation   # 修改任何 .cu/.cuh/.cpp 后都需要重新执行
pytest tests/
python tools/env_report.py              # 查看 GPU、驱动、CUDA、cuBLAS 等环境信息
```

国内网络可以在构建镜像时使用镜像源：

```bash
APT_MIRROR=mirrors.tuna.tsinghua.edu.cn PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple docker compose build
```

修改 `docker/Dockerfile` 或 `requirements-dev.txt` 后需要重新构建镜像（VS Code 中执行 `Dev Containers: Rebuild Container`）。

### 换成自己的 GPU

扩展默认只为 sm_86 编译。在其他 GPU 上直接运行会报 `no kernel image is available`。先查计算能力，`python tools/env_report.py` 或 `nvidia-smi --query-gpu=compute_cap --format=csv` 都可以，例如 RTX 4090 为 8.9。然后任选一种方式：

```bash
# 方式一：构建镜像时指定，之后每次编译都生效（在宿主机执行）
TORCH_CUDA_ARCH_LIST=8.9 docker compose build

# 方式二：不重建镜像，编译扩展时临时指定（在容器内执行）
rm -rf build && TORCH_CUDA_ARCH_LIST=8.9 pip install -e . --no-build-isolation
```

值的格式是 `8.9`，不是 `sm_89`；多个架构用分号分隔，例如 `"8.6;8.9"`。改完要确认重新编译了：删掉 `build/` 目录就会全部重编。

### 测试

```bash
pytest tests/
compute-sanitizer python -m pytest tests/ -q                     # 越界与非法访问
compute-sanitizer --tool racecheck python -m pytest tests/ -q    # shared memory 竞争
```

`test_tolerance_rejects_tf32` 会在 GPU 上调用 cuBLAS。如果 compute-sanitizer 报出 cuBLAS 内部的错误，可以加 `-k "not tolerance_rejects_tf32"` 排除。

### 跑基准

```bash
python benchmarks/bench_matmul.py --shapes 4096x4096x4096 --dtype fp32 fp16 bf16
python benchmarks/bench_layernorm.py --shapes 8192x4096 --dtype fp32

# 用 Nsight Compute 分析某个 kernel
ncu --set full -k regex:matmul_kernel_tc -c 1 python benchmarks/bench_matmul.py --profile --shapes 4096x4096x4096 --dtype fp16
```

笔记本 GPU 满载时会因功耗墙降频，计时结果不稳定。可以在宿主机上锁频后再测（需要 sudo，结束时自动恢复）：

```bash
tools/bench_locked.sh 802 python benchmarks/bench_matmul.py --shapes 4096x4096x4096 --dtype fp16
```

不同 GPU 的结果会不一样。分享测量结果时，请附上 `python tools/env_report.py` 的输出。

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
tools/
  env_report.py       输出与测量相关的环境信息
  bench_locked.sh     锁定 SM 频率后运行命令（在宿主机执行）
  ncu_hotspots.py     按源码行汇总 ncu 的停顿采样
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

## 许可证

[MIT](LICENSE)，适用于本仓库的全部代码，包括历史提交和各个 tag。
