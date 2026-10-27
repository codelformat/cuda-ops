"""按 CUDA 源码行统计 ncu 报告中的 warp 停顿采样，找出最耗时的代码行（相当于 ncu-ui 的 Source 页）。

用法：
  python tools/ncu_hotspots.py reports/v1_big.ncu-rep
  python tools/ncu_hotspots.py reports/v1_big.ncu-rep --top 20

报告需包含 SourceCounters 和 WarpStateStats（--set full 已包含），kernel 需带 -lineinfo 编译。
报告中有多个 kernel 时，结果是它们的合计；采集时用 -c 1 只抓一个 kernel。
"""

import argparse
import csv
import io
import subprocess

STALL_SAMPLES = "Warp Stall Sampling (All Samples)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("report", help=".ncu-rep 报告路径")
    parser.add_argument("--top", type=int, default=10, help="显示前几行")
    args = parser.parse_args()

    text = subprocess.run(
        ["ncu", "-i", args.report, "--page", "source", "--print-source", "cuda,sass", "--csv"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    file = header = None
    lines = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2:
            continue
        if row[0] == "File Path":
            file = row[1]
            continue
        if row[0] == "Line No":
            header = row
            continue
        # cuda,sass 视图中，源码行的 Address 列为 "-"，其指标是该行对应全部 SASS 指令的汇总
        if header and file and len(row) == len(header) and row[2] == "-":
            metrics = dict(zip(header[4:], row[4:]))
            samples = int(metrics.get(STALL_SAMPLES) or 0)
            lines.append((samples, f"{file.rsplit('/', 1)[-1]}:{row[0]}", row[1].strip()))

    total = sum(samples for samples, _, _ in lines)
    if total == 0:
        raise SystemExit("报告中没有 warp 停顿采样数据：采集时需包含 SourceCounters 和 WarpStateStats")
    print(f"warp 停顿采样共 {total} 个，按源码行排序：")
    for samples, location, source in sorted(lines, reverse=True)[: args.top]:
        print(f"{samples / total:6.1%}  {location:<28} {source[:90]}")


if __name__ == "__main__":
    main()
