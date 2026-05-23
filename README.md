# hotspot_al

`hotspot_al` 是一个面向大规模反应型 MLIP-MD 的局域热点主动学习研究原型。当前版本只支持 `Allegro` 作为 MLIP 后端，并围绕 `Allegro + LAMMPS + CP2K` 工作流组织代码。

当前项目处于 research prototype / validation-in-progress 阶段：核心 Python 模块可测试、可导入，但真实 Allegro 在线推理、LAMMPS/CP2K 任务调度、生产级 retraining 闭环仍需要外部 runner、evaluator 或 HPC 适配层接入。

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

- `Allegro` 单后端主动学习协议；
- `LAMMPS` MD 输入与 dump 读取接口；
- `CP2K` DFT 输入生成与 force parser；
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

已经实现的最小可验证模块：

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
- Allegro 薄 runner 骨架：可注入单模型 force evaluator，以及 train/export command template；
- 轻量 candidate pool 与几何去重接口。

当前仍保留为接口、骨架或外部适配部分：

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

当前仓库实际只提供 `config/default.yaml`。如果后续需要按后端拆分配置，可以扩展出 `config/allegro.yaml`、`config/cp2k.yaml`、`config/lammps.yaml`，但它们不是当前版本已包含的文件。

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

推荐用 editable install，确保普通 Python 进程无需设置 `PYTHONPATH` 就能导入包：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python -m pytest -q
```

基础运行安装：

```bash
pip install -e .
```

可视化可选依赖：

```bash
pip install -e ".[viz]"
```

如果只想安装基础运行依赖而不安装本地包：

```bash
pip install -r requirements.txt
```

正式使用本仓库源码时仍建议执行 `pip install -e .`，而不是依赖 `PYTHONPATH=src`。

## 配置

主配置在 [config/default.yaml](config/default.yaml)。

其中包括：

- `backend`：Allegro / LAMMPS / CP2K 总体后端；
- `lammps`：dump 字段、type map、timestep；
- `allegro`：模型路径、deployed model 路径、committee 模式；
- `allegro.dataset_dir` / `train_output_dir` / `checkpoint_path`：runner skeleton 使用的外部训练与导出路径；
- `allegro.train_command_template` / `export_command_template`：外部 Allegro runtime 命令模板；
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

这些示例展示的是模块连接方式，不是完整生产流水线。示例中的 `dump.lammpstrj`、`trajectory.extxyz` 等输入文件需要用户替换为自己的数据路径，例如 `./data/trajectory.extxyz`。真实 Allegro、LAMMPS、CP2K 外部程序不会由这些示例自动配置。

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

## Allegro 接口骨架

当前提供了一个薄封装 [src/hotspot_al/training/allegro_runner.py](src/hotspot_al/training/allegro_runner.py)，目标是把仓库内部协议和外部 Allegro 运行时解耦：

- `AllegroBackend.evaluate_forces(...)`：通过 `AllegroRunner(force_evaluator=...)` 注入真实单模型推理回调；
- `AllegroBackend.evaluate_committee(...)`：对多个 model path 逐个调用 evaluator，返回形状 `(n_models, n_atoms, 3)`；
- `AllegroBackend.train(...)`：读取 `allegro.dataset_dir`、`allegro.train_output_dir` 和 `allegro.train_command_template`，默认以 dry-run 方式返回外部训练命令；
- `AllegroBackend.export_model(...)`：读取 `allegro.checkpoint_path` 和 `allegro.export_command_template`，默认以 dry-run 方式返回外部导出命令。

如果没有注入 `force_evaluator`，调用 `AllegroRunner.evaluate_forces(...)` 或依赖它的 committee 评估会抛出 `NotImplementedError`。这是预期的接口边界，不是安装错误。

命令模板使用 Python `str.format(...)` 占位符：

- `train_command_template` 可用 `{dataset_dir}`、`{output_dir}`、`{train_config_path}`；
- `export_command_template` 可用 `{checkpoint_path}`、`{output_dir}`。

最小 mock evaluator 接法示意：

```python
import numpy as np

from hotspot_al.lammps.allegro_lammps import AllegroBackend
from hotspot_al.training.allegro_runner import AllegroRunner


def my_force_evaluator(atoms, model_path, config):
    # Replace this with a real Allegro inference call.
    return np.zeros((len(atoms), 3))


backend = AllegroBackend(
    config=config,
    runner=AllegroRunner(force_evaluator=my_force_evaluator),
)
```

需要用户提供或适配的外部组件：

- Allegro 单模型推理函数，签名兼容 `force_evaluator(atoms, model_path, config)`；
- Allegro train/export 命令模板，写入 `allegro.train_command_template` 和 `allegro.export_command_template`；
- LAMMPS / CP2K 可执行文件、输入模板、作业提交、重试和失败恢复逻辑；
- 如果要做真实 masked retraining，需要在外部 Allegro dataloader / loss 中消费 `mask_weights`。

## 局限性

- 当前版本是 research prototype / validation-in-progress，不是完整生产平台。
- Allegro backend 现在提供了薄 runner 骨架，但默认仍未绑定真实训练与推理运行时。
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
