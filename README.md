# hotspot_al

`hotspot_al` 是一个面向大规模反应型 MLIP-MD 的局域热点主动学习原型框架。当前版本只支持 `Allegro` 作为 MLIP 后端，并围绕 `Allegro + LAMMPS + CP2K` 工作流组织代码。

方法名暂定为 `PHAL`：

`Physics-aware Hotspot Active Learning for Allegro-LAMMPS-CP2K`.

它不是 DP-GEN 的简单包装，而是一个 atom-resolved, physics-aware, event-triggered, hotspot-localized active learning 框架。

## 方法定位

标准 DP-GEN 更偏向：

- committee-based force deviation；
- frame-level candidate selection；
- 整帧进入标注流程；
- 与 DeepMD 生态强耦合。

本项目当前聚焦于：

- `Allegro` 单后端主动学习闭环；
- `LAMMPS` 统一 MD 执行层；
- `CP2K` 统一 DFT 标注层；
- atom-wise OOD monitoring；
- LJ projection residual 等 physics-aware 指标；
- event-triggered pre/trigger/post frame extraction；
- hotspot-localized cluster/slab/graph extraction；
- H capping / frozen boundary / embedding hook；
- core-region masked force retraining；
- 避免对大规模 MD 整帧做 DFT 标注。

如果 DP-GEN 回答的是“哪一帧不确定”，PHAL 进一步回答“哪一个原子异常、异常为什么发生、应该截取哪一块区域、哪些原子真正进入监督损失”。

## 工作流

```text
Large-scale Allegro-LAMMPS MLIP-MD
        ↓
Atom-wise real-time / offline OOD monitoring
        ↓
Event-triggered frame extraction
        ↓
Hotspot atom clustering
        ↓
Local cluster / slab / graph extraction
        ↓
Boundary treatment / H capping / embedding hook
        ↓
CP2K DFT labeling
        ↓
Masked training data generation
        ↓
Allegro retraining
        ↓
Updated Allegro-LAMMPS simulation
```

## 当前实现范围

已经实现的最小闭环：

- LAMMPS custom dump 读取，统一输出 `FrameData`；
- 逐原子监测：`force`、`delta_force`、`displacement`、`r_min`、`coordination`、`delta_q`；
- staged OOD scoring：`light` / `physics` / `full`；
- rolling buffer 与事件元数据；
- hotspot 检测与空间聚类；
- cluster / slab / graph 三种局域截取 baseline；
- 保守式 H capping；
- CP2K H-only optimization 与 single-point input 生成；
- CP2K force parser；
- 通用 `extxyz + npz + metadata` 数据输出；
- Allegro extxyz + per-atom mask 导出；
- 轻量 candidate pool 与几何去重接口。

当前仍保留为接口或 TODO 的部分：

- 在线 Allegro 模型推理与 committee evaluator；
- LAMMPS / CP2K 真实任务调度与失败恢复；
- point-charge embedding 的具体实现；
- Allegro 训练代码中的 mask-aware loss 深度集成；
- 生产级候选池排序、版本管理和 retraining orchestration。

## 目录结构

```text
project/
  README.md
  pyproject.toml
  requirements.txt
  config/
    default.yaml
    allegro.yaml
    cp2k.yaml
    lammps.yaml
  src/hotspot_al/
    io/
    lammps/
    monitor/
    buffer/
    hotspot/
    extraction/
    cp2k/
    training/
    active_learning/
    utils/
  tests/
  examples/
```

## 核心数据结构

`FrameData`

- `atoms: ase.Atoms`
- `step: int`
- `time: float | None`
- `forces: np.ndarray | None`
- `velocities: np.ndarray | None`
- `energy: float | None`
- `metadata: dict`

`ExtractedRegion`

- `atoms`
- `original_indices`
- `core_indices`
- `inner_buffer_indices`
- `outer_buffer_indices`
- `boundary_indices`
- `h_cap_indices`
- `hotspot_indices`
- `region_labels`
- `mask_weights`
- `metadata`

## 安装

```bash
pip install -e .
```

开发测试：

```bash
pip install -e .[dev]
pytest
```

如果只想安装基础依赖：

```bash
pip install -r requirements.txt
```

## 配置

主配置在 [config/default.yaml](/home/guozy/workspace/lammps_monitor/config/default.yaml)。

其中包括：

- `backend`：Allegro / LAMMPS / CP2K 总体后端；
- `lammps`：dump 字段、type map、timestep；
- `allegro`：模型路径、deployed model 路径、committee 模式；
- `monitor` / `ood_score`：三阶段触发参数；
- `buffer` / `hotspot`：事件缓存和聚类半径；
- `extraction`：cluster/slab/graph 截取参数；
- `h_capping`：保守式补氢策略；
- `cp2k`：functional、basis、cutoff、SCF、H-only optimization；
- `training_mask`：core/buffer/boundary/H-cap 权重；
- `candidate_pool`：去重与每轮上限。

## 最小示例

- `examples/01_monitor_lammps_dump.py`
- `examples/02_extract_hotspots.py`
- `examples/03_generate_cp2k_inputs.py`
- `examples/04_parse_cp2k_forces.py`
- `examples/05_write_allegro_dataset.py`

这些示例展示的是模块连接方式，不是完整生产流水线。

## 测试覆盖

当前测试包括：

- `test_lammps_reader.py`
- `test_hotspot_detection.py`
- `test_cluster_extraction.py`
- `test_h_capping.py`
- `test_cp2k_parser.py`
- `test_mask_generator.py`
- `test_allegro_adapter.py`

## Allegro 数据输出说明

当前实现导出 `extxyz`，包含：

- `forces`
- `mask_weights`
- `region_code`

如果 Allegro 训练脚本支持 per-atom weights，可直接接入；否则需要在 dataloader / loss 中显式使用 `mask_weights`。项目默认不允许把 masked atoms 的力简单设成 0 再走普通 loss。

## 局限性

- 当前版本是研究原型，不是完整生产平台。
- Allegro backend 目前主要提供输入生成、数据导出和接口占位，还没有绑定真实训练与推理运行时。
- LAMMPS 和 CP2K runner 目前主要是命令构建与输入准备层，还没有完整 HPC job management。
- OOD 还缺训练集统计校准、在线 callback 与真实 committee 外部评估。
- 局域截取和边界化学处理仍是 baseline，实现上偏保守。
- 数据血缘、模型版本管理、失败恢复和 CLI 操作层还没有系统化。
- 当前测试主要是单元测试，还缺真实 LAMMPS / CP2K / Allegro integration tests。

## 目标应用

本项目面向以下类型体系：

- tribochemistry
- 摩擦界面反应
- 氧化物表面反应
- 大规模界面与局域热点反应型 MLIP-MD

核心目标是在这些体系中实现更低成本、更可解释、更贴近真实反应事件的 Allegro 主动学习闭环。
