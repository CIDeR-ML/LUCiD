# JUNO 支撑杆有效几何

## 目标与范围

本组件在全探测器 LUCiD 前向模型中回答两个问题：光线是否先于
PMT/介质边界击中支撑杆；击中后是被有效吸收，还是以镜面或漫反射
方向继续传播。它不追求复刻全部 CAD 细节。

第一版仅保留 JUNOSW J26.3.2 `geometry_acrylic.gdml` 中两类
`StrutSteel` 主体管：

| family | count | radius | full length |
|---|---:|---:|---:|
| `main_85mm` | 370 | 42.5 mm | 1548.054 mm |
| `secondary_60mm` | 220 | 30.0 mm | 1548.077 mm |

共 590 根。中心坐标直接读取官方 GDML；两类管的局部轴线均严格沿
径向指向探测器中心，因此运行时由中心坐标推导单位轴。这既保留方向，
又避免把 GDML Euler 角约定带入运行接口。

未包含：球头、连接件、螺栓、支撑节点的 Boolean/CAD 细节，以及
亚克力碗和烟囱。这些是独立组件或有证据后才考虑的 v2 候选。

## 有限圆柱交点

对中心为 $\mathbf c$、单位轴为 $\mathbf a$、半长为 $h$、半径为
$r$ 的杆，以及光线

$$
\mathbf x(t)=\mathbf o+t\mathbf d,\qquad t>0,
$$

把 $\mathbf d$ 和 $\mathbf o-\mathbf c$ 分解为平行、垂直于
$\mathbf a$ 的分量。圆柱侧壁由

$$
\left\|\mathbf o_\perp+t\mathbf d_\perp\right\|^2=r^2,
\qquad
\left|o_\parallel+t d_\parallel\right|\le h
$$

给出；两个端盖由

$$
o_\parallel+t d_\parallel=\pm h,
\qquad \|\mathbf x_\perp(t)\|\le r
$$

给出。取所有合法正根中最小者，再与当前的 LS--Acrylic、
Acrylic--Water 或 PMT/外边界交点比较。只有更近的支撑杆交点会覆盖
原交点，因此该组件是传播几何的一部分，不是对最终 PE 的事后修正。

为避免构造 `n_photon x 590` 的大数组，球面角网格为每个方向保存
12 根邻近杆。每条光线同时查询平均杆半径交点与当前候选边界交点的
网格，共检查至多 24 个圆柱候选。固定种子生成的 4000 条覆盖全径向
范围的随机光线与全量 590 杆求交对照中，40 条真实支撑杆命中均被候选
方法找回，最近杆编号和距离一致。

## 表面过程与诊断

J26.3.2 的 `StrutSteel` 材料/表面引用在 532 nm 覆盖范围内给出
固定反射率 0.535，因而采用：

- `support_reflection_rate = 0.535`：总反射概率；
- `support_fspec = 0.5`：发生反射后选择镜面方向的有效份额；
- `1 - support_reflection_rate`：有效吸收概率。

`support_fspec` 不是声称 JUNOSW 的精确表面微结构，而是配置驱动、
后续可微刻度的低维有效参数。逐事件 `results_summary.json` 记录：
`support_hits`、`support_blocked`、`support_absorbed`、
`support_reflected`、`support_specular` 和 `support_diffuse`。
其中 `blocked` 表示原传播路径被杆件几何截断，数值等于 `hits`；
`absorbed + reflected = hits`，且
`specular + diffuse = reflected`。

## 消融与验收

命令行参数 `--support-geometry config|on|off`（批处理环境变量
`SUPPORT_GEOMETRY`）独立控制该组件。相同 seed、光子数、K 和其他
配置下比较：

$$
\Delta G_{\rm support}(t)=
G_{\rm PMT+support}(t)-G_{\rm PMT}(t).
$$

首轮使用 $z=0$ 和 $z=16.1\,\mathrm m$，再以
$z=17.1\,\mathrm m$ 验证。验收目标是定位支撑杆改变的 prompt、late、
遮挡和反射分量，而不是要求 LUCiD 数值必然更接近 JUNOSW 或数据。

## 是否需要更新

当前已满足当前 `diffsim.tex` 与 `AGENTS.md` 对“已知结构逐项部署”的
要求：位置、方向、主要尺寸有官方来源；形状可解释；组件可独立消融；
交点与反射过程可微；诊断可追溯。

只有出现以下证据之一，才建议定义下一阶段更新：

1. 消融图显示残差在支撑节点位置稳定聚集，且无法由主体杆解释；
2. 逐光子诊断显示大量光线从杆端附近穿漏，球头/连接件会改变分量归属；
3. 候选网格与全量 590 杆交点的验证出现漏检；
4. 后续可微刻度对杆半径或表面参数产生边界解，提示有效形状不足。

v2 也应只增加被上述证据支持的球头或连接节点，不应直接复制完整 CAD。
