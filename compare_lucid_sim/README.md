# LUCiD 与 FSMP/JUNOsw 三方比较工具

`Makefile` 是全部图表的复现入口，可从任意目录运行：

```bash
make -C compare_lucid_sim help          # target 与变量一览
make -C compare_lucid_sim compare-all   # 例: 四个 run 的三方对比
```

## Target

| Target | 产物 | 说明 |
|---|---|---|
| `compare-<run>` / `compare-all` | time/summary/amplitude PDF + CSV | 三方对比主图（run: 15040/15030/15022/15012，对应 z=0/15/16.1/17.1 m） |
| `compare-z Z=16.1` | 同上 | run 由 z 查 run_532.csv 得到；FSMP 用默认根 /junofs/C/PE |
| `tts-irf-<run>` / `tts-irf-all` | 五曲线时间谱 PDF | TTS vs 实测 IRF 交叉检查（125 特征化通道，自动推导） |
| `pe-scale-<run>` / `pe-scale-all` | PE/event scale PDF | 全通道（17612）探测器级对比 |
| `theta-intensity-<run>` / `-all` | 三面板 PDF + CSV | 每 PMT PE-θ 分布；复用 pe-scale 缓存 |
| `scale-vs-z` | scale-z 汇总 PDF | 跨源位置汇总 |
| `submit-<target>` | 同上 | 以 Slurm 作业运行（仅主仓库） |
| `all` | 以上全部 | |

## 变量

路径变量（**默认是作者目录，运行前务必覆盖 `DATA_ROOT`**，输出跟随它）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `DATA_ROOT` | `/junofs/C/liukainan/LUCiD/simulation_updated_diffuserHardware` | LUCiD 模拟根（含 z0/z1500/... 子目录） |
| `FSMP_ROOT` | `/junofs/C/PE` | FSMP 数据根 |
| `JUNOSW_ROOT` | `/mnt/stage/haoch/junores/laser/532` | JUNOsw 模拟根 |
| `PMT_MAP` / `IRF_H5` | `/junofs/C/junoid_J25.7.3.pq` 等 | PMT 布局 / 实测 IRF 核 |

参数变量：`CHANNEL_COUNT`(100) `SEED`(0) `BINS`(600)
`FSMP_PE_SCALE`(10) `QE`(0.097) `PAGE_CHANNEL_COUNT`(空=全画)；
`args="..."` 把额外脚本参数原样透传给任意 target。

## 输出与缓存

输出根：`DIFFUSER_OUT` / `IRF125_OUT` / `PE_SCALE_OUT` /
`SCALE_Z_OUT`，默认都在 `$(DATA_ROOT)/compare_sim-*` 下；
每个 run 的产物在 `<输出根>/<run>/`，缓存（`*_all.npz` 等）与
PDF 同目录。缓存按内容校验：通道集合、分箱数、处理口径
（`metrics=8,qe=...,fsmp_target=...` 串）或输入数据变化时自动
重建，否则命中复用。`--plot-only`（可经 `args` 透传）只重画。

## 作业提交

`submit-<target>` 把同一 target 提交为 Slurm 作业（作业内递归调
本 Makefile，命令模板单点维护）。`SBATCH` / `SBATCH_OPTS` /
`SLURM_LOG_DIR` 可覆盖；日志默认 `<DATA_ROOT>/slurm-compare/`。

## 模块

`compare_sim_vs_real.py` 主流水线；
`plot_tts_vs_irf.py`、`plot_pe_scale.py`、`plot_theta_intensity.py`、`plot_source_scale_vs_z.py` 检查/汇总图；`compare_sim.py`、`laser_compare_*.py`、`pe_scale_io.py` 为内部模块。
