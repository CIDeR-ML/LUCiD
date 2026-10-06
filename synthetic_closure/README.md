# JUNO synthetic event-wise closure workflow

本目录集中管理全几何 LUCiD synthetic closure 的运行、提交和专项验证。
前向模型包含当前基线中的三介质传播、PMT surface、support rods、
chimneys、acrylic bowls 和 diffuser ball。观测端使用 event-wise、
unbinned Poisson point-process likelihood。

## 组账号提交入口

`SBATCH=/path/to/group/sbatch-proxy` 是此前说明中用来表示“组内统一提交
入口”的占位文本，并不指向仓库中的真实文件，因此不应原样执行。本项目
已有的 `compare_lucid_sim/Makefile` 使用组账号命令，本工作流与它保持
一致，默认直接执行：

```text
sudo -u#35905 sbatch
```

也就是说，`make submit-*` 会以课题组统一账号提交 Slurm 作业。Makefile
使用 `sbatch --wrap` 把配置写入作业命令，再执行
`run/run_juno_full_geometry_closure.sh`，避免依赖 `sudo` 是否保留调用者的
环境变量。若只需在当前账号测试，可显式覆盖 `SBATCH=sbatch`。

## 固定的资格测试配置

`qualification-one` 和 `qualification-five` 共用以下配置：

| Item | Value |
|---|---:|
| Optimization plan | `joint-only` |
| Events per source | 2 |
| Data proposal photons | 400,000 |
| Fit proposal photons | 100,000 |
| Transport depth `K` | 32 |
| Data/Fit/MC-validation ensembles | 2 / 1 / 2 |
| Joint optimizer steps | 128 |
| Time window | [0, 1000] ns |
| Numerical time basis | 2 ns |
| Signal intensity | 100,000 |
| Background rate | 30,000 Hz |
| Fit seed offset | 1,000,003 |
| MC-validation seed offset | 2,000,003 |
| Validation cadence | every 2 steps |
| External source chunk size | 0 (single dynamic source loop) |
| Multi-source objective execution | `per-source` |
| Checkpoint cadence | every 4 steps |

以上 `QUAL_*` 配置用于复用已有 2-event observations 的 fit-only 资格
验证。正式 `submit-one` 和 `submit-five` 默认改为：

| Formal item | Value |
|---|---:|
| Events per source | 100 |
| Joint optimizer steps | 300 |
| Objective execution | `per-source` |
| AD--FD relative step | `1e-3` |

其余 transport、ensemble、时间窗和背景配置沿用资格基线。正式五源作业
必须等待正式单源作业验证显存、运行时间和输出契约后再提交。

单源资格测试默认使用 `z=17.1 m`，因为它比中心源更敏感于界面和工程
结构。五源测试使用以下嵌套集合：

```text
(0, 0, 0.0) m
(0, 0, 9.0) m
(0, 0, 15.0) m
(0, 0, 16.1) m
(0, 0, 17.1) m
```

## 提交前检查

先查看最终 Slurm 命令，不会提交作业：

```bash
make -n -C synthetic_closure submit-one \
    OUTDIR=/junofs/C/liukainan/LUCiD/closure_qualification/one_source

make -n -C synthetic_closure submit-five \
    OUTDIR=/junofs/C/liukainan/LUCiD/closure_qualification/five_sources
```

确认输出目录、源位置和资源后正式提交：

```bash
make -C synthetic_closure submit-one \
    OUTDIR=/junofs/C/liukainan/LUCiD/closure_qualification/one_source

make -C synthetic_closure submit-five \
    OUTDIR=/junofs/C/liukainan/LUCiD/closure_qualification/five_sources
```

若 generation 已成功、fit 失败，不要复用同一个 `OUTDIR` 运行完整 worker，
因为它会拒绝覆盖已有片段。使用新的 fit 输出目录，并显式复用原片段：

```bash
make -C synthetic_closure submit-fit-only \
    OUTDIR=/junofs/C/liukainan/LUCiD/closure_twophase/fit_ensemble1 \
    OBSERVATION_ROOT=/junofs/C/liukainan/LUCiD/closure_twophase/\
one_source/generation
```

`submit-fit-only` 仍会核对 data photons、data ensemble、K、seed、时间窗、
intensity、background 和 IRF，但允许改变 fit photons、fit ensemble、
fit seed offset 和起始参数，适合做单变量 GPU 故障定位。
它继续使用与既有 observations 一致的 2 events、128 steps；它不是新的
100-event 正式数据生产入口。

五源 generation 已存在时，使用逐源目标直接复用五份数据：

```bash
make -C synthetic_closure submit-five-fit-only \
    OUTDIR=/junofs/C/liukainan/LUCiD/closure_twophase/\
five_sources_split_k32 \
    OBSERVATION_ROOT=/junofs/C/liukainan/LUCiD/closure_twophase/\
five_sources/generation
```

资格检查阶段不要同时提交单源和五源。先等待单源完成并生成
`closure_result.json`，再提交五源，避免同一节点上的两个大型 XLA 编译
同时争用主机内存或 ROCm 编译资源。

Makefile 拒绝覆盖已经含有 `closure_result.json` 的目录。Slurm 标准输出和
错误日志默认写入
`/junofs/C/liukainan/LUCiD/slurm/calibration/`。

当前计算节点的 JAX/ROCm 在同一进程连续建立 data、fit 和
MC-validation 图时曾报 `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT`。
`jax.clear_caches()` 不能销毁 ROCm context 和 queue，因此正式 worker
采用两阶段、多进程执行：

1. 每个 source 启动一个独立 Python 进程，以 400,000 photons 生成
   event-wise synthetic PE 和 truth bin mass，写入独立 HDF5；
2. 每个生成进程正常退出，GPU context 随进程释放；
3. 启动全新的 fit 进程，校验并读取所有 source HDF5，只构建
   100,000-photon fit/MC-validation forward 和 VJP。

这不改变 photon、K、ensemble、随机 key 或 PPP likelihood，只改变计算图
的生命周期。每个片段记录绝对 source index，拟合进程同时核对源位置、
时间窗、K、data photons、seed、background 和 IRF，拒绝重复、遗漏或过期
片段。此外 closure worker 仍会
通过 `run/env.sh` 显式加入：

```text
XLA_FLAGS=--xla_gpu_enable_command_buffer=
```

该规避只对 closure worker 生效，不改变普通 forward simulation 作业。
作业启动日志会打印最终 `XLA_FLAGS`，便于复现和核查。
每个 Python phase 正常退出时，ROCm 可能打印 `Queue ... is idle`；在该行
之前 phase 已写出完成 JSON 时，这是 context 回收信息，不是作业卡死。

Closure worker 显式使用
`/opt/gentoo/usr/bin/python3.13`，不再依赖 generic `python` 在不同节点上
解析到哪个版本。启动日志同时记录 Python 完整版本、`sys.executable`、
JAX/JAXLIB 版本和 GPU devices。`CLOSURE_PYTHON` 只保留为诊断覆盖入口。

ROCm 7 实测表明，多源融合 executable 在相同形状第二次重放时可能产生
畸形 AQL packet。资格作业因此使用 `OBJECTIVE_EXECUTION=per-source`：每个
source 独立编译和执行 forward/VJP，显式同步后把未归一化 NLL 与梯度复制
到 host 求和，最后只除一次全局 PE 数。五个 source 在同一个参数点全部
完成后才执行一次 Adam update。因此它与联合 PPP 的数学定义严格相同，
只避免生成不稳定的多源 GPU executable。`fused` 模式继续用于单元测试和
数值等价对照。

## 任意源集合与本地测试

底层 `run` 接口仍接收任意数量的源位置；位置之间用空格分隔：

```bash
make -C synthetic_closure run \
    OUTDIR=/tmp/lucid_custom_closure \
    SOURCES="0,0,0 0,0,16.1" \
    PLATFORM=cpu \
    N_EVENTS=1 DATA_N_PHOTONS=128 FIT_N_PHOTONS=128 \
    K=4 JOINT_STEPS=1 TIME_MAX_NS=300 TIME_BIN_NS=50
```

需要用资格测试配置提交其他源集合时，使用：

```bash
make -C synthetic_closure submit-custom \
    OUTDIR=/junofs/C/liukainan/LUCiD/closure_qualification/custom \
    SOURCES="0,0,0 0,0,12.0 0,0,17.1"
```

CPU 单元测试：

```bash
make -C synthetic_closure test
```

其余目标 `grid-convergence`、`key-comparison`、`importance-audit`、
`credibility-scan`、`fisher` 和 `heldout-smoke` 用于专项检查，不属于本次
1/5-source 正式资格作业的必要步骤。

## 主要输出

- `generation/source_XX/closure_observations.h5`：每源独立生成片段；
- `generation/source_XX/generation_result.json`：每源生成状态与计时；
- `closure_observations.h5`：fit 进程校验后合并的 event-wise 输入；
- `closure_result.json`：配置、优化轨迹、梯度和验证量；
- `closure_diagnostics.pdf`：train/MC loss、梯度、计时和参数轨迹；
- `closure_diagnostics.json`：best-step 等机器可读摘要；
- `checkpoint_STAGE.npz`：周期恢复点；
- Slurm 日志：编译时间、逐步运行时间、错误与资源诊断。

Jacobian/Fisher 使用独立作业，避免诊断失败破坏 closure 主结果：

```bash
make -C synthetic_closure submit-one-fisher \
    FISHER_OUTDIR=/junofs/C/liukainan/LUCiD/closure_formal/one_source_fisher

make -C synthetic_closure submit-five-fisher \
    FISHER_OUTDIR=/junofs/C/liukainan/LUCiD/closure_formal/five_sources_fisher
```

输出 `fisher_result.json` 和 `fisher_diagnostics.pdf`。这里计算的是
IRF-convolved numerical-time-basis expected Poisson Fisher，可用于初步
identifiability 判断，但还不是 continuous-time unbinned PPP Fisher。

提交前应保存当前 Git commit 和 `make -n` 展开的完整命令；分析结果时应
同时核查有限梯度、MC-validation、checkpoint 状态和逐 source 贡献。
