#!/usr/bin/env bash
# 锁定 GPU SM 频率后在 dev 容器里执行命令，结束（包括 Ctrl+C、出错）时自动恢复默认频率。
#
# 用法（在宿主机的仓库目录下执行，不要在容器里：容器内没有改频率的权限）：
#   tools/bench_locked.sh <MHz> <容器内命令...>
#   tools/bench_locked.sh 800 python benchmarks/bench_matmul.py --shapes 4096x4096x4096
#
# 为什么锁频：笔记本 GPU 满载时会因功耗墙降频（SGEMM 实测约 800 MHz），不同版本、不同次运行的
# 频率不同，TFLOPS 无法直接对比。锁在功耗墙内能稳定维持的频率后，结果可复现。
# 注意：锁低 SM 频率也会拖慢访存瓶颈的算子（如 LayerNorm），带宽测试一般不要锁。
set -euo pipefail

if [[ $# -lt 2 || ! $1 =~ ^[0-9]+$ ]]; then
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
if [[ -f /.dockerenv ]]; then
  echo "请在宿主机上运行：容器内没有修改 GPU 频率的权限" >&2
  exit 1
fi

mhz=$1
shift
root=$(cd "$(dirname "$0")/.." && pwd)

restore() {
  sudo nvidia-smi -rgc >/dev/null && echo "[bench_locked] 已恢复默认 SM 频率"
}
trap restore EXIT

sudo nvidia-smi -lgc "$mhz,$mhz" >/dev/null
echo "[bench_locked] SM 频率已锁定为 $mhz MHz"

# 有终端时分配 TTY，便于看到实时输出；被管道或脚本调用时不分配
tty_flag=()
[[ -t 0 && -t 1 ]] || tty_flag=(-T)
docker compose -f "$root/compose.yaml" exec "${tty_flag[@]}" dev "$@"
