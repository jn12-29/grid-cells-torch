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
- 支持分页 rate-map PDF、HDC tuning PDF 和评估 MP4。
- 提供以 CLI 为中心的数据生成、可视化和实验管理流程。

## 📚 参考

| 参考项 | 作用 |
|---|---|
| Banino et al. (2018), [Vector-based navigation using grid-like representations in artificial agents](https://doi.org/10.1038/s41586-018-0102-6) | 原始 Nature 论文 |
| DeepMind 官方实现, [google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells) | 本仓库最初对齐的原始代码库 |

## 🧭 概览

本仓库一开始严格对齐 DeepMind 官方 `grid-cells` 实现，并将核心训练流程迁移到 PyTorch。之后又加入了固定数据集生成、评估 PDF、HDC tuning 图、MP4 动画、TensorBoard、`pos_mse` 解码指标以及更完整的 CLI 工作流，更适合做可复现实验和后续分析。

## ⚡ 快速开始

```bash
pip install torch numpy scipy matplotlib pyyaml tqdm tensorboard

# 如果需要导出 MP4，请先安装 ffmpeg
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# generate train/eval splits plus preview artifacts
python generate_data.py --visualize --animate

# train with the generated dataset
python train.py

# or print the common command list
bash run_scripts.sh

# inspect metrics
tensorboard --logdir results
```

默认约定：

- 训练集：`data/train.npz`
- 评估集：`data/eval.npz`
- 结果目录：`results/<timestamp>/`

如果 `data/train.npz` 不存在，`train.py` 会回退到在线生成轨迹模式。

## 📦 输出内容

- `train.log`：精简训练日志。
- `tensorboard/`：标量指标和配置快照。
- `rates_and_sac_epoch_XXXX.pdf`：rate map 和空间自相关图。
- `hdc_tuning_epoch_XXXX.pdf`：HDC tuning 曲线。
- `eval_animation_epoch_XXXX.mp4`：评估轨迹动画。

## 🗂️ 仓库结构

```text
grid-cells-torch/
├── docs/assets/readme/
├── config.yaml
├── generate_data.py
├── train.py
├── model.py
├── dataset.py
├── ensembles.py
├── scores.py
├── utils.py
└── results/
```

<details>
<summary>🔍 更多说明</summary>

- `config.yaml` 是默认实验入口，并支持命令行覆盖，例如 `python train.py --training.epochs 100 --training.lr 1e-3`。
- `generate_data.py` 可以在同一工作流中导出 `.npz`、PDF 汇总和 MP4 动画。
- `run_scripts.sh` 会打印一份精简的常用训练、数据生成和 TensorBoard 命令清单。
- 当前默认配置更偏向扩展后的工程化实验流程，而不是对原始超参数做逐行逐值锁定。
- README 中使用的媒体资源会从选定运行结果镜像到 `docs/assets/readme/`，避免首页依赖被忽略的 `results/` 文件。

</details>

## 🙏 致谢

本项目的主要开发工作在很大程度上依赖 Claude Code（Claude Sonnet 4.6）和 OpenCode（GPT-5.4）的协助。整体开发大约用时一天，累计使用四个 Claude Code Pro session 和两个 GPT Plus session，迭代节奏非常快。
