# grid-cells-torch

[🌐 English](README.md)

本仓库最初是对 [`google-deepmind/grid-cells`](https://github.com/google-deepmind/grid-cells) 的忠实 PyTorch 迁移，之后扩展成了更实用的数据生成、评估、可视化、日志与分析工作流。

## ✨ 结果

参考运行：`results/20260416-040934`，展示快照为 `epoch 12`。
适合 README 发布的资源已镜像到 `docs/assets/readme/`。

| 指标 | 数值 |
|---|---:|
| `pos_mse` | `0.027598` |
| `grid_score_60 max` | `1.3284` |
| `grid_score_90 max` | `1.5351` |

[![Epoch 12 evaluation animation](docs/assets/readme/eval_animation_epoch_0012_thumb.png)](docs/assets/readme/eval_animation_epoch_0012.mp4)

视频：`docs/assets/readme/eval_animation_epoch_0012.mp4`

[PDF 1](docs/assets/readme/rates_and_sac_epoch_0012.pdf) | [PDF 2](docs/assets/readme/hdc_tuning_epoch_0012.pdf)

![PDF 1 第 1 页](docs/assets/readme/rates_and_sac_epoch_0012-1.png)
![PDF 1 第 2 页](docs/assets/readme/rates_and_sac_epoch_0012-2.png)

## 🚀 相比官方仓库的扩展

- 预生成 `train/eval` 数据集，并保留在线生成回退模式。
- 精简 `train.log` 和 TensorBoard 日志。
- 在训练与评估中加入解码位置指标 `pos_mse`。
- 支持分页 rate-map PDF、HDC tuning PDF，以及训练评估和数据生成共用的 3-panel MP4。
- 提供以 CLI 为中心的数据生成、可视化和实验管理流程。

## 📚 参考

| 参考项 | 作用 |
|---|---|
| Banino et al. (2018), [Vector-based navigation using grid-like representations in artificial agents](https://doi.org/10.1038/s41586-018-0102-6) | 原始 Nature 论文 |
| DeepMind 官方实现, [google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells) | 本仓库最初对齐的原始代码库 |

## 🧭 概览

本仓库一开始严格对齐 DeepMind 官方 `grid-cells` 实现，并将核心训练流程迁移到 PyTorch。之后又加入了固定数据集生成、评估 PDF、HDC tuning 图、共用的 3-panel MP4 动画、TensorBoard、`pos_mse` 解码指标以及更完整的 CLI 工作流，更适合做可复现实验和后续分析。

代码采用以 `grid_cells/` 为根的分层 Python 包结构。根目录提供 `train.py` 和 `generate_data.py` 两个 CLI 入口；库代码导入统一使用 `grid_cells.*` 路径。

## ⚡ 快速开始

```bash
pip install torch numpy scipy matplotlib pyyaml tqdm tensorboard

# 如果需要导出 MP4，请先安装 ffmpeg
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# 默认会生成一个数据集目录到 data/datasets/<dataset-id>/，并导出预览产物
python generate_data.py --visualize --animate

# 调大动画采样步长，缩短预览视频
python generate_data.py --animate --visualization.anim_step 2

# 从 config.yaml 默认值出发做一次性覆盖
python generate_data.py --data_generation.num_samples 5000 --task.seq_len 800

# 显式指定输出路径时，仍可使用 legacy 单文件模式
python generate_data.py --output data/train_small.npz --eval_output data/eval_small.npz

# 训练默认读取 data/latest/train.npz 和 data/latest/eval.npz
python train.py

# 也可以覆盖评估视频共用的动画参数
python train.py --visualization.anim_num_traj 4 --visualization.anim_step 2

# 或查看常用命令清单
bash run_scripts.sh

# 查看指标
tensorboard --logdir results
```

默认约定：

- 默认生成目录：`data/datasets/<dataset-id>/`
- 默认训练入口（最新 train）：`data/latest/train.npz`
- 默认训练入口（最新 eval）：`data/latest/eval.npz`
- 结果目录：`results/<timestamp>/`

目录模式成功生成后，会把最新的 `train.npz` / `eval.npz` 暴露到 `data/latest/`，供默认训练流程复用。

如果 `data/latest/train.npz` 不存在，`train.py` 会回退到在线生成轨迹模式。

## 📦 输出内容

- `train.log`：精简训练日志。
- `tensorboard/`：标量指标和配置快照。
- `rates_and_sac_epoch_XXXX.pdf`：rate map 和空间自相关图。
- `hdc_tuning_epoch_XXXX.pdf`：HDC tuning 曲线。
- `eval_animation_epoch_XXXX.mp4`：eval 风格的 3-panel 轨迹动画。

## 🗂️ 仓库结构

```text
grid-cells-torch/
├── config.yaml
├── generate_data.py
├── train.py
├── grid_cells/
│   ├── common/
│   ├── cells/
│   ├── data/
│   ├── training/
│   ├── analysis/
│   └── viz/
├── tests/
├── docs/assets/readme/
└── results/
```

<details>
<summary>🔍 更多说明</summary>

- `config.yaml` 是默认实验入口，并支持命令行覆盖，例如 `python train.py --training.epochs 100 --training.lr 1e-3`。
- `train.py` 和 `generate_data.py` 现在统一使用显式的 `--section.key value` 覆盖风格。
- `generate_data.py` 默认会在 `data/datasets/<dataset-id>/` 下生成一个完整数据集目录，写入元数据，并同步 `data/latest/*` 作为训练默认入口。
- 如果显式传入 `--output` / `--eval_output`，仍可走 legacy 单文件模式。
- 长期复用的数据生成默认值放在 `config.yaml` 的 `data_generation.*` 下；`--visualize`、`--animate`、`--train_only`、`--visualize_progress` 这类一次性运行开关继续保留在 CLI。
- 分层包边界如下：
  `grid_cells/common` 管理共享配置辅助。
  `grid_cells/cells` 管理 ensembles、encoding 和 model。
  `grid_cells/data` 管理 dataset IO、数据生成、预览和轨迹合成。
  `grid_cells/training` 管理 CLI 解析、runtime 装配、evaluation 和训练 session。
  `grid_cells/analysis` 管理评分与绘图辅助。
  `grid_cells/viz` 管理动画渲染。
- 共享动画默认参数统一放在 `config.yaml` 的 `visualization.anim_*` 下，`train.py` 和 `generate_data.py` 都可以通过 CLI 覆盖。
- 常见覆盖示例：
  `python train.py --task.env_size 2.4 --training.batch_size 32`
  `python generate_data.py --visualization.anim_fps 30 --data_generation.num_workers 4`
- `run_scripts.sh` 会打印一份精简的常用训练、数据生成和 TensorBoard 命令清单。
- 当前默认配置更偏向扩展后的工程化实验流程，而不是对原始超参数做逐行逐值锁定。
- README 中使用的媒体资源会从选定运行结果镜像到 `docs/assets/readme/`，避免首页依赖被忽略的 `results/` 文件。

</details>

## 🙏 致谢

本项目的主要开发工作在很大程度上依赖 Claude Code（Claude Sonnet 4.6）和 OpenCode（GPT-5.4）的协助。整体开发大约用时一天，累计使用四个 Claude Code Pro session 和两个 GPT Plus session，迭代节奏非常快。
