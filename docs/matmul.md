# Matmul 优化记录

`C[M, N] = A[M, K] @ B[K, N]`，行主序、连续存储，输入输出同为 fp32 / fp16 / bf16，内部 fp32 累加。
代码：`csrc/matmul/matmul.cuh`；测试：`tests/test_matmul.py`；基准：`benchmarks/bench_matmul.py`。
fp32 的优化见 v0 到 v4，fp16 / bf16 的 Tensor Core 版本见 v5，最新数据见“当前结果”。

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

2026-11-05 复测：v4 为 14,947,225 周期，cuBLAS 为 12,753,371 周期，按周期为 85.3%，锁频计时为 85.6%（见“当前结果”）。

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

## v5：Tensor Core（fp16 / bf16）

fp16、bf16 输入在 M、N、K 都是 16 的倍数，且 `a`、`b` 首地址按 32 字节对齐时走 Tensor Core 版本 `matmul_kernel_tc`，
否则退回 v4（`csrc/matmul/matmul.cuh` 的 host 分发）。fp32 始终走 v4。v5.2 起用到 `cp.async`，v5.4 起用到
`ldmatrix` 和 `mma.sync.m16n8k16`，需要 Ampere（sm_80）及以后的 GPU。

### 版本总览（4096³ fp16，ncu）

各行取自当时保存的 ncu 报告（单次运行），相对速度以同期 cuBLAS 报告的 4,780,927 周期为基准。

| 版本 | 主要改动 | 周期数 | 相对 cuBLAS | 寄存器 | shared memory | 指令数 |
|---|---|---:|---:|---:|---:|---:|
| v4 | fp16 输入，仍是 CUDA core FFMA | 15,217,344 | 31.4% | 123 | 16.6 KB | 26.5 亿 |
| v5.0 | WMMA，fragment 直接从 global 读取 | 65,412,510 | 7.3% | 40 | 4.1 KB | 6.45 亿 |
| v5.1 | A、B 先搬进 shared memory；PAD=8 | 5,714,743 | 83.7% | 230 | 23.0 KB | 1.17 亿 |
| v5.2 | `cp.async` 双缓冲 | 5,677,438 | 84.2% | 210 | 42.0 KB | 1.52 亿 |
| v5.3 | 界内 tile 走快路径，搬运偏移预先算好 | 5,369,694 | 89.0% | 242 | 42.0 KB | 9,456 万 |
| v5.4（直接写回） | 手写 `ldmatrix` + `mma.sync`，寄存器直接写回 | 5,211,680 | 91.7% | 240 | 37.9 KB | 8,626 万 |
| v5.4 | 写回先在 shared memory 拼成整 sector | 5,096,853 | 93.8% | 240 | 39.9 KB | 8,681 万 |
| v5.5 | 下一块的拷贝移到 tile 中段 | 4,917,838 | 97.2% | 188 | 39.9 KB | 9,867 万 |
| cuBLAS | `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn` | 4,780,927 | 100% | 234 | 32.8 KB | 9,884 万 |

“v5.4（直接写回）”是开发过程中的中间状态，没有单独提交。v5.5 的三次平均见“当前结果”一节。

主要停顿（每条发射指令平均等待的 warp 数，`smsp__average_warps_issue_stalled_*_per_issue_active.ratio`）：

| 版本 | MIO Throttle | Long Scoreboard | LG Throttle | Math Pipe Throttle |
|---|---:|---:|---:|---:|
| v5.0 | 117.9 | 86.7 | 48.9 | 0.4 |
| v5.1 | 0.48 | 4.97 | 0.00 | 10.4 |
| v5.2 | 0.71 | 0.38 | 0.00 | 8.8 |
| v5.3 | 1.38 | 0.60 | 0.71 | 14.4 |
| v5.4 | 0.81 | 0.66 | 0.54 | 17.3 |
| v5.5 | 0.16 | 0.27 | 0.13 | 13.6 |
| cuBLAS | 0.07 | 0.05 | 0.00 | 13.7 |

v5.1 之后的第一停顿都是 Math Pipe Throttle，即 Tensor Core 已经是瓶颈。warp 在等 Tensor Core 不等于 Tensor Core 空转：
只有 SM 里所有 warp 都拿不出 HMMA 时，Tensor Core 才会闲下来。所以后面几版看的是 Tensor Core 利用率和周期数，
而不是停顿比值。ncu 在 GeForce 上按 fp16 累加的速率计 `sm__pipe_tensor_cycles_active` 的峰值，fp32 累加时读数要乘以 2：
v5.5 为 48.0%（约 96%），cuBLAS 为 49.6%（约 99%）。

### v5.0：WMMA 直接读 global

每个 warp 负责 C 的一个 16×16 小块，每步从 global 加载 A、B 各一个 16×16 fragment，warp 之间不共享数据。

**手算对账**：每个 warp 每 16 个 k 读 2 × 256 × 2 B = 1 KB，共 (M/16)(N/16)(K/16) 步，合计 MNK/4 B ≈ 17.2 GB，
即 5.37 亿个 sector；ncu 的 L2 读取为 5.28 亿个 sector（`lts__t_sectors_srcunit_tex_op_read.sum`）。DRAM 读 8.11 GB。

**瓶颈**：MIO Throttle、Long Scoreboard、LG Throttle 三项都很高，Tensor Core 利用率只有 3.5%。
换上更快的计算单元后，访存没有任何复用，比 v4 慢了 4.3 倍。

### v5.1：shared memory 分块

和 v1 的思路相同：block 负责 128×128，BK=32，4 个 warp 按 2×2 排列，每个 warp 负责 64×64（4×4 个 16×16 fragment）。
A、B 先由全部线程搬进 shared memory，再由各 warp 用 `load_matrix_sync` 读取。

**手算对账**：L2 读取量 = MNK × (1/BM + 1/BN) × 2 B = 2,147,483,648 B，即 6,711 万个 sector；ncu 为 6,672 万。

**PAD**：WMMA 从 shared memory 读取时编译为 `LDSM`，每条按 8 行 × 16 B 为一组，8 行要落在不同的 bank，
即行距必须是 16 B 的奇数倍。BK=32 时行距 64 B、BN=128 时行距 256 B，都是偶数倍；PAD=8 后变为 80 B 和 272 B。
读方向的 bank conflict 降到 0，周期从 5.92M 降到 5.71M。

### v5.2：cp.async 双缓冲

`__pipeline_memcpy_async` 把数据从 global 直接拷进 shared memory，不经过寄存器（SASS 为 `LDGSTS`）。
顺序为：等待本线程的拷贝完成 → `__syncthreads()` → 发出下一块的拷贝 → 计算当前块，每块一次同步。

**结果**：寄存器 230 → 210，Long Scoreboard 4.97 → 0.38，但周期只少 0.6%：瓶颈已经是 Tensor Core，访存延迟本来就被遮住了。
L2 读取量变为手算值的 1.5 倍（每条指令 24 个 sector，理想为 16），原因没有深究；改用 `.ca` 变体能恢复读取量，但反而略慢，所以保留默认写法。

### v5.3：快路径

整个 tile 都在矩阵范围内时（4096³ 时所有 tile 都是），跳过逐线程的越界判断；每次搬运的偏移在主循环外算好。
每个 warp 每个 tile 的非 HMMA 指令从 226 条降到 116 条，周期减少 5.4%。

### v5.4：手写 ldmatrix 与 mma.sync

用内联 PTX 替换 WMMA：`ldmatrix.sync.aligned.m8n8.x4`（B 加 `.trans`）和 `mma.sync.aligned.m16n8k16.row.col.f32`。

**线程映射**：加载一个 16×16 小块时，第 ℓ 号线程传入第 `ℓ % 16` 行、第 `(ℓ / 16) × 8` 列的地址；
拿到的数据则是第 `ℓ / 4` 行（和第 `ℓ / 4 + 8` 行）、第 `(ℓ % 4) × 2` 列起的两个元素。线程传入的地址和它拿到的数据不是一回事。
一条 m16n8k16 只算 8 列，16×16 的结果要两条指令。

**结果**：
- 不写回时周期减少 4.6%（不执行写回的对照版本：5.345M → 5.100M）。
- A 的加载和计算交错与否，周期相同（5.215M 对 5.216M）；改用 k=8 的 `mma` 慢 4.6%。

**写回**：寄存器里每个线程只有相邻 2 列，直接写 C 时一条指令每行只写 16 B，一个 32 B sector 要两次写请求：
L2 写请求 230.4 万，理想值为 4096² × 2 B / 32 B = 104.9 万（`lts__t_requests_aperture_device_op_write.sum`），
写回耗时约 11.5 万周期。先在 shared memory（每 warp 一块 16×16，共 2 KB）拼好，再每线程写 8 B，
写请求降到 105.1 万，写回耗时约 1 万周期，周期 5.21M → 5.10M。

### v5.5：拷贝移到 tile 中段

v5.4 中 `__syncthreads()` 之后所有 warp 同时执行拷贝指令。把下一块的拷贝移到 `kk == 16` 之前：

| 拷贝位置 | 周期 | Long Scoreboard |
|---|---:|---:|
| 同步之后、计算之前（v5.4） | 5.11M | 0.66 |
| tile 中段（v5.5） | 4.93M | 0.27 |
| tile 末尾 | 5.86M | 4.32 |

放在末尾时拷贝没有时间被遮住，最慢；放在中段比放在开头快 3.5%。寄存器从 240 降到 188，指令数增加 13.7%，
DRAM 读取从 624 MB 降到 354 MB，这三项变化的原因还没有查清。

### 未解决：block 在 SM 间分配不均

v5.5 在部分形状上比 v5.4 慢，例如 2048³（锁频计时 0.86× → 0.76×）。2048³ 只有 16 × 16 = 256 个 block，
每个 SM 最多同时放 2 个，平均每个 SM 分到 256 / 58 ≈ 4.4 个，有的 SM 要多跑一到两个：

| 2048³ | SM 活跃周期最小 | 最大 | 比值 |
|---|---:|---:|---:|
| v5.4 | 59.1 万 | 73.5 万 | 1.24（约 4 对 5 个 block） |
| v5.5 | 55.1 万 | 82.6 万 | 1.50（约 4 对 6 个 block） |

（`sm__cycles_active.min`、`sm__cycles_active.max`）单个 block 在 v5.5 更快，但最忙的 SM 多分到了 block。
换成 block 数正好是 58 的整数倍的形状（3712×1024×2048，232 个 block），v5.5 比 v5.4 快 7.1%。
每个 SM 实际分到几个 block 是由比值推算的，还需要在 kernel 里读取 `%smid` 直接统计。

## 当前结果（2026-11-05 复测）

**环境**：RTX 3080 Ti Laptop（sm_86，58 个 SM），驱动 595.99.02，CUDA 13.2（nvcc V13.2.78），cuBLAS 13.4.1，
PyTorch 2.14.1+cu132，Nsight Compute 2026.1.1；`allow_tf32 = False`。环境信息可以用 `python tools/env_report.py` 输出。

**4096³**：

| dtype | 本仓库（周期） | cuBLAS（周期） | 按周期 | 锁频 802 MHz 计时 |
|---|---:|---:|---:|---:|
| fp32 | 14,947,225 | 12,753,371 | 85.3% | 85.6% |
| fp16 | 4,932,261 | 4,782,600 | 97.0% | 96.9% |
| bf16 | 4,927,563 | 4,771,878 | 96.8% | 96.9% |

- 周期：ncu `sm__cycles_elapsed.max`，三次平均，与频率无关。cuBLAS 的 kernel：fp32 为 `cutlass_80_simt_sgemm_128x64_8x5_nn_align1`
  （CUDA core，确认没有走 TF32），fp16 / bf16 为 `ampere_{fp16,bf16}_s1688gemm_*_128x128_ldg8_f2f_stages_32x1_nn`。
- 计时：宿主机运行 `tools/bench_locked.sh 802 ...`，预热 3 次后取 9 次的中位数。两种口径相差不超过 0.3 个百分点。

**其他形状（fp16，锁频计时，相对 cuBLAS）**：

| 形状 M×N×K | 相对 cuBLAS | 形状 M×N×K | 相对 cuBLAS |
|---|---:|---|---:|
| 256³ | 0.50× | 4096×4096×256 | 1.11× |
| 512³ | 0.56× | 1024×4096×1024 | 0.90× |
| 1024³ | 0.79× | 4096×1024×1024 | 0.92× |
| 2048³ | 0.76× | 1024×4096×4096 | 0.86× |
| 4096³ | 0.97× | 4096×1024×4096 | 0.98× |
| 4096×4096×1024 | 0.97× | 64×4096×4096 | 0.36× |
| 256×4096×4096 | 0.66× | 4096×64×4096 | 0.35× |
| 4096×256×4096 | 0.65× | | |

小矩阵和 M 或 N 很小的形状 block 太少，填不满 SM；当前只有一种 tile 配置，下一步按形状选择。

## 版本与提交对照

| 版本 | 提交 | 提交说明 | 说明 |
|---|---|---|---|
| v0 | `38ffae9` | matmul v0 | 一致 |
| v1（第一步） | `2b10068` | matmul v2: add tiling shared memory | 提交说明写作 v2，实际是 v1 的 shared memory 分块 |
| v1（第二步） | `e55478a` | matmul v1.1 | 先把 shared memory 的数据读进寄存器，减少重复读取 |
| v1 | `daab96c` | hand rewrite matmul v1 | 本文 v1 一节的数据 |
| v2.1 到 v3 | `d4eb993` | matmul v3 | 包含转置、PAD、向量化和 `col_of` |
| v4 | `a19e30d` | matmul v4 | 一致 |
| v5.0 | 无 | | 没有单独提交 |
| v5.1 | `745fb41` | v5.2: add shared memory | 提交说明写作 v5.2，实际是 v5.1 |
| v5.2 | `0df982a` | v5.2: add cp.async | 一致 |
| v5.3 | `5a7405e` | v5.3 | 一致 |
| v5.4 | `cc63ebb` | v5.4 | 包含 shared memory 拼装写回 |
| v5.5 | `10b2fdf` | v5.5 | 一致 |

视频课程各节对应的 tag（`ep01` 起）见 README。

## 经验

1. 先判断瓶颈在哪一类，再细分到具体单元：Speed Of Light → Pipe Utilization → Warp State → 逐条 SASS。
2. 改不是瓶颈的地方收益很小（v2.2），但要算清它什么时候会变成瓶颈（shared memory 需求 / FMA 周期 > 1 时限制上限）。
3. 停顿比值是"每条指令"的，指令数变化后要乘以指令数再比较。
4. 编译器会重排和合并访存（v1 整块读 A、v3 拆分 STS），以 SASS 为准。
5. PAD 解决不同行之间的冲突，同一行内的冲突要改线程与数据的映射。
6. LDG 不阻塞，用到结果时才等：预取的关键是把"用"推迟到计算之后。
7. 瓶颈变成 Tensor Core 之后，warp 在等 Tensor Core 不代表它空转；判断收益看利用率和周期，不看停顿比值（v5.2）。
8. 指令少了不一定更快，指令多了也不一定更慢（v5.5 多了 13.7% 的指令，周期少了 3.5%），以周期为准。
9. 写回按 sector 计：一次写不满 32 B，L2 写请求就会翻倍（v5.4）。
10. 单个 block 变快，总时间仍可能变长：block 数不是 SM 数的整数倍时，看最忙的 SM（v5.5）。

## 下一步

- 按形状选择 tile，让 block 在各 SM 上分得均匀，并用 `%smid` 验证。
- 小 M 或小 N 的形状：split-K。
- M、N、K 不是 16 的倍数时，补零到 16 的倍数后走 Tensor Core 版本，而不是退回 v4。
- fp32 剩余空间：BK=16、寄存器双缓冲、float4 写回 C。
