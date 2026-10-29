# Matmul 优化记录

`C[M, N] = A[M, K] @ B[K, N]`，行主序、连续存储，输入输出同为 fp32 / fp16 / bf16，内部 fp32 累加。
代码：`csrc/matmul/matmul.cuh`；测试：`tests/test_matmul.py`；基准：`benchmarks/bench_matmul.py`。

## 硬件与测量方法

RTX 3080 Ti Laptop（GA104，sm_86）：58 SM，每 SM 每周期 FP32 FFMA 峰值 256 FLOP（128 条线程级 FFMA），
Tensor Core（fp16/bf16 输入、fp32 累加）512 FLOP；DRAM 峰值 512 GB/s，L2 4 MB。

- **按周期比较，不按毫秒**：笔记本 GPU 满载时功耗墙会把频率压到 800 MHz 左右，且每次运行不同。
  ncu 的周期数、百分比与频率无关；bench 需要锁频：`tools/bench_locked.sh 800 python benchmarks/bench_matmul.py`（宿主机运行）。
- **算力利用率看 Roofline 一节的 FFMA 每周期执行数 / 7424**（58 SM × 128），不要看 Speed Of Light 的
  Compute (SM) Throughput：它取 SM 内最忙的单元，访存指令打满时也接近 100%（见 v0）。
- **每一版先手算预期，再与 ncu 对账**：请求数、sector 数、wavefront 数、指令数对不上，说明对 kernel 行为的理解有误。

采集与查看（容器内）：

```bash
pip install -e . --no-build-isolation        # 改了 .cuh 必须重新编译
ncu -f -o reports/matmul_vX --set full -k regex:matmul -c 1 \
    python benchmarks/bench_matmul.py --profile --shapes 4096x4096x4096
R=reports/matmul_vX.ncu-rep
ncu -i $R --page details --section SpeedOfLight_RooflineChart --print-details all   # FFMA 每周期执行数
ncu -i $R --page details --section WarpStateStats --print-details all              # 停顿原因
ncu -i $R --page details --section LaunchStats --section Occupancy                 # 寄存器、occupancy
ncu -i $R --page raw --metrics <指标>                                              # 原始指标，不知道名字时 --page raw | grep
ncu -i $R --page source --print-source sass --csv                                  # 逐条 SASS 的执行次数、wavefront、停顿
python tools/ncu_hotspots.py $R                                                     # 按源码行汇总停顿采样
```

常用原始指标：

| 指标 | 含义 |
|---|---|
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` / `..._sectors_...` | global load 的 warp 级请求数 / 32 B sector 数 |
| `smsp__sass_inst_executed_op_shared_ld.sum` / `_st.sum` | LDS / STS 指令数（warp 级） |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` / `_st.sum` | shared memory 的 wavefront 数（约等于占用 shared memory 的周期数） |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` / `_st.sum` | bank conflict（实际 wavefront − 理想 wavefront） |
| `smsp__issue_active.avg.pct_of_peak_sustained_active` | 调度器发射指令的周期占比 |
| `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FMA 单元利用率（含 IMAD） |

## 版本总览（4096³ fp32，ncu）

| 版本 | 主要改动 | 周期数 | FFMA 利用率 | 相对 v0 | 当时的瓶颈 |
|---|---|---:|---:|---:|---|
| v0 | 每线程算 C 的 1 个元素，block 32×8，x 对应列 | 148,479,768 | 6.2% | 1.0× | LSU 发射（每次 FFMA 配 2 条 global load） |
| v1 | shared memory 分块 + 8×8 寄存器分块（BM=BN=128，BK=8，TM=TN=8） | 27,458,046 | 33.7% | 5.4× | occupancy 16.7%，延迟遮不住 |
| v2.1 | A 在 shared memory 中转置存放 | 20,526,784 | 45.1% | 7.2× | 写 A 转置时 8 路 bank conflict |
| v2.2 | 转置后的 A 每行加 PAD=4 | 19,663,501 | 47.1% | 7.6× | 指令开销（整数指令约 20%） |
| v3 | global 用 128 位向量读，多 dtype，不对齐时退回标量 | 18,758,722 | 49.3% | 7.9× | 写 B 拆成 4 条标量 STS，4 路冲突 |
| v3 | 写 B 改为一条 STS.128 | 17,879,523 | 51.8% | 8.3× | 读 B 的 LDS 冲突 |
| v3 | 每个线程负责的 8 列拆成两组（`col_of`） | 16,264,114 | 56.9% | 9.1× | global 延迟（Long Scoreboard）、同步 |
| **v4** | global→shared memory 双缓冲，每轮一次同步 | **14,883,579** | **62.2%** | **10.0×** | 发射率 76.8%（Barrier、Short Scoreboard） |
| cuBLAS | `cutlass_80_simt_sgemm_128x64_8x5`（5 级 cp.async 流水） | 12,727,814 | 72.7% | 11.7× | — |

FFMA 利用率 = 4096³ / 周期数 / 7424，与 Roofline 一节的读数一致。

## v0：朴素实现

每个线程算 C 的一个元素：`for k: sum += a[row*K+k] * b[k*N+col]`。block 32×8（256 线程，每 SM 6 个 block，occupancy 100%）。

**线程映射**：`threadIdx.x` 对应列。一个 warp 是同一行的 32 个连续列：读 A 是同一地址（广播，1 个 sector），
读 B 是连续 128 B（4 个 sector）。反过来映射时读 A 的每个请求要 32 个 sector。

**手算对账**（warp 数 = 4096² / 32 = 524,288）：

| 量 | 手算 | ncu |
|---|---|---|
| global load 请求 | warp 数 × K × 2 = 4,294,967,296 | 4,294,967,296 |
| load sector | warp 数 × K × (1 + 4) = 10,737,418,240 | 10,737,418,240 |
| FFMA / 周期 | 2³⁶ / 148,479,768 = 463.5 | 463.48 |
| DRAM 读 | L1 未命中 1.36e9 sector × 32 B × (1 − L2 命中 19.8%) ≈ 34.9 GB | 34.93 GB（理想 134 MB 的 260 倍） |

**瓶颈**：Speed Of Light 的 Compute 显示 99.9%，但那是 LSU（99.92%）；FMA 只有 7.84%。停顿以 `lg_throttle`
为主（34 个周期里占 21.2 个）：每次 FFMA 要 2 条 load 指令，LSU 队列被打满。DRAM 只有 53%，不是瓶颈。

## v1：shared memory 分块 + 寄存器分块

block 负责 C 的 128×128 小块，沿 K 每次把 A 的 128×8、B 的 8×128 读进 shared memory；每个线程负责 8×8 个输出，
每个 kk 读 8 个 A、8 个 B 到寄存器，做 64 次 FFMA。

**复用分析**：global 读取量 ∝ MNK(1/BM + 1/BN)，shared memory 读取量 ∝ MNK(1/TM + 1/TN)。
累加器必须在寄存器里（下标是编译期常量、循环完全展开）。

**结果**：158 寄存器/线程 → 每 SM 1 个 block，occupancy 16.7%。FMA 36.6%，LSU 26.4%，都没打满；
停顿以 Long Scoreboard（等 global）为主。

**SASS**：kk 循环展开后，编译器把线程的整个 8×8 A 子块（ATILE 中连续的 256 B）用 16 条 LDS.128 一次读进寄存器
（这也是 158 个寄存器的来源之一），B 每个 kk 2 条 LDS.128，每个 tile 共 32 条 LDS、512 条 FFMA。

**bank conflict**（逐条 SASS 的 Wavefronts / Ideal）：LDS.128 按半个 warp（16 线程）处理。
读 A 半个 warp 同一地址，每条 2 个 wavefront，无冲突；读 B 时相邻 tx 相隔 8 个 float，起始 bank `(8·tx) % 32`
只有 4 种，每组 bank 4 个线程，每条 8 个 wavefront（理想 4）。平均每条 5 个 wavefront、2 个冲突，
与 `wavefronts_mem_shared_op_ld`（671,088,640）和 `bank_conflicts`（268,435,456）一致。

## v2.1：A 转置存放

**问题**：读 A 时固定 kk 需要的 8 个数在 ATILE 中相隔一行（不连续），编译器只好整块读进寄存器。
**改动**：`ATILE[BK][BM]`，固定 kk 时 8 个 A 连续，与 B 一样每个 kk 2 条 LDS.128。

**结果**：寄存器 158 → 120，每 SM 2 个 block，occupancy 33.3%，周期 −25%。
新问题：写转置的 A 时 `ATILE[c][r]` 中 r 相同的线程落在同一 bank（每行 128 个 float，128 % 32 = 0），
8 路冲突，STS 每条 4.5 个 wavefront（A 8 个、B 1 个），MIO Throttle 从 0.16 升到 0.68。

## v2.2：PAD

每行改为 `BM + PAD`。bank = `(c·(128+PAD) + r) % 32`，PAD 需同时满足：标量写不冲突、行首 16 B 对齐（读 A 仍能用 LDS.128）、
以后的 float4 映射也不冲突 → **PAD = 4**（PAD = 1 只能降到 4 路，且破坏 LDS.128 对齐）。

**结果**：STS 冲突 117,440,512 → 0，但周期只 −4.4%：shared memory 当时只有 69% 繁忙，消掉的不是瓶颈。
此时 FMA 利用率 = 发射率 67% × FFMA 占比 71.4%（整数指令约 20%，多来自搬运循环的地址计算和越界判断）。

## v3：向量化 global 读取、多 dtype

- `AlignedVec<T, kVec>`（`csrc/common/vec.h`）一次读 kVec 个元素，shared memory 统一存 float，计算部分与 dtype 无关。
  kVec 固定为 4（fp32 为 LDG.128，fp16/bf16 为 LDG.64），各 dtype 共用一套 tile 参数。
- host 端在 K、N 是 4 的倍数且 a、b 首地址按 `sizeof(AlignedVec<T, 4>)` 对齐时用 kVec=4，否则 kVec=1。
  越界判断只检查向量的第一个元素，依赖 K、N 是 kVec 的倍数。
- 结果：LDG 每 tile 8 → 2 条，指令 717 → 607 条/(warp·tile)，FFMA 占比 71.4% → 84.4%，周期 −4.6%；
  FMA 利用率不变：指令少了但停顿没少，发射率从 67% 降到 59.7%。

**写 B 的 4 路冲突**：编译器把 4 个 float 拆成 4 条标量 STS（shared 数组只保证 4 B 对齐），相邻线程相隔 4 个 float，
32 线程落在 8 个 bank。修正：`alignas(16)` + 先拼 `AlignedVec<float, kVec>` 再整体写入 → 一条 STS.128，
STS wavefront 8,390 万 → 3,355 万（每 warp·tile 8 个 = 理想值），周期 −4.7%。

**读 B 的冲突**：PAD 只改变行与行的距离，对同一行内的冲突无效。改为每个线程负责的 TN 列分成 TN/4 组、组内 4 列连续、
组间相隔 `BN / (TN/4)` 列（`col_of`），半个 warp 读连续 256 B。LDS 冲突 268,435,456 → 0，
wavefront 每 warp·tile 160 → 96，shared memory 需求 / FMA 周期从 1.31（FMA 上限约 76%）降到 0.81。周期 −9.0%。

## v4：双缓冲

**问题**：Long Scoreboard 停顿 100% 落在 STS 上（LDG 后马上 STS，等几百周期）；Barrier 次之。
**改动**：搬运拆成 `load_tile_to_regs`（LDG 到寄存器）和 `store_regs_to_smem`（STS 到另一块缓冲区），中间夹当前 tile 的计算；
shared memory 双份（16.6 KB），每轮一次 `__syncthreads`。LDG 发出后线程不等待，到用到寄存器时才等。
`__launch_bounds__(kThreads, 2)` 保证 kVec=1 版本也压到 128 寄存器以内（否则 130 个，每 SM 只有 1 个 block）。

**结果**：Long Scoreboard 0.83 → 0.00，发射率 67.6% → 76.8%，周期 −8.5%。Barrier 总停顿几乎没变（13.6 亿 → 13.2 亿），
推测由 warp 之间的进度差决定而不是同步次数，未验证。compute-sanitizer racecheck / synccheck / memcheck 均为 0。

## 与 cuBLAS 对比

ncu（4096³ fp32）：

| 指标 | v4 | cuBLAS |
|---|---|---|
| 周期数 / 相对速度 | 14,883,345 / 85.5% | 12,727,814 / 100% |
| FFMA 利用率 | 62.2% | 72.7% |
| 发射率 | 76.8% | 90.5% |
| 总指令数 | 25.95 亿 | 26.22 亿 |
| bank conflict | 0 | 1.8 亿 |
| 每 block 线程数 / 寄存器 / shared memory | 256 / 127 / 16.6 KB | 128 / 126 / 30.7 KB（5 级流水） |

指令数相近，差距在发射率（停顿）。

锁频 802 MHz（`tools/bench_locked.sh 800 python benchmarks/bench_matmul.py --dtype fp32 fp16`），占峰值比例：

| shape | fp32 v4 / cuBLAS | 加速比 | fp16 v4 / cuBLAS（Tensor Core 峰值） | 加速比 |
|---|---|---:|---|---:|
| 4096³ | 62.2% / 72.7% | 0.86× | 30.5% / 97.3% | 0.31× |
| 4096×4096×1024 | 59.8% / 74.2% | 0.81× | 29.8% / 95.4% | 0.31× |
| 2048³ | 54.3% / 61.8% | 0.88× | 26.4% / 92.0% | 0.29× |
| 1024³ | 33.4% / 54.2% | 0.62× | 16.5% / 64.7% | 0.25× |
| 16×4096×4096 | 4.3% / 19.5% | 0.22× | 1.7% / 11.5% | 0.15× |

- 1024³ 只有 64 个 block（每 SM 可放 2 个，共 116 个位置）；M=16 只有 32 个 block，填不满 58 个 SM（需要 split-K）。
- fp16 cuBLAS 用 Tensor Core（`ampere_fp16_s1688gemm`），v4 仍是 CUDA core FFMA。
  注意 GeForce 上 ncu 的 `sm__pipe_tensor_cycles_active` 按 fp16 累加的速率计峰值，fp32 累加时 49.65% 已接近满载。

## 经验

1. 先判断瓶颈在哪一类，再细分到具体单元：Speed Of Light → Pipe Utilization → Warp State → 逐条 SASS。
2. 改不是瓶颈的地方收益很小（v2.2），但要算清它什么时候会变成瓶颈（shared memory 需求 / FMA 周期 > 1 时限制上限）。
3. 停顿比值是"每条指令"的，指令数变化后要乘以指令数再比较。
4. 编译器会重排和合并访存（v1 整块读 A、v3 拆分 STS），以 SASS 为准。
5. PAD 解决不同行之间的冲突，同一行内的冲突要改线程与数据的映射。
6. LDG 不阻塞，用到结果时才等：预取的关键是把"用"推迟到计算之后。

## 下一步

- v5：Tensor Core（WMMA → `mma.sync` + `ldmatrix` + cp.async 多级流水），目标 fp16/bf16 达到 cuBLAS 的 80% 以上。
- v6：小 M（split-K）。
- fp32 剩余空间：BK=16、寄存器双缓冲、float4 写回 C。
