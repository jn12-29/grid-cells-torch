# grid-cells-torch

PyTorch 重构版本，复现 DeepMind 2018 年 Nature 论文：

> Banino et al., *"Vector-based navigation using grid-like representations in artificial agents"*, Nature 557, 429–433 (2018)

原版代码（TensorFlow 1.12 + Sonnet）位于 `../grid-cells/`，本目录不依赖 TensorFlow，所有数据均在运行时用 numpy 生成，无需下载任何数据集。

---

## 环境依赖

```bash
pip install torch numpy scipy matplotlib pyyaml
```

Python ≥ 3.8，PyTorch ≥ 2.0。

---

## 文件结构

```
grid-cells-torch/
├── config.yaml      # 所有超参数的默认值
├── train.py         # 训练入口 + argparse 命令行覆盖
├── model.py         # GridCellsRNN (nn.Module)
├── ensembles.py     # PlaceCellEnsemble / HeadDirectionCellEnsemble
├── dataset.py       # TrajectoryDataset + get_dataloader
├── scores.py        # GridScorer（网格评分 + 可视化）
└── utils.py         # 编码函数 + 绘图辅助
```

---

## 快速开始

### 推荐工作流：先生成数据，再训练

```bash
cd grid-cells-torch

# 第一步：生成并保存轨迹数据（只需运行一次）
python generate_data.py --output data/train.npz --visualize

# 第二步：用保存的数据训练（每 epoch 直接从文件加载，无需重复生成）
python train.py --data_path data/train.npz
```

使用 `--data_path` 时，训练路径会启用当前实现里的几项性能优化：
1. `model.py` 使用 `nn.LSTM(batch_first=True)`，不再手写 `LSTMCell` 时间循环。
2. `dataset.py` 会预计算 `init_cond`，并在 DataLoader worker 中按样本编码 `pc_targets_i` / `hdc_targets_i`。
3. DataLoader 启用 `persistent_workers=True`，避免每个 epoch 重建 worker。
4. CUDA 训练自动启用 AMP（`torch.autocast` + `GradScaler`）。

### 不保存数据（原始模式）

```bash
python train.py   # 每 epoch 实时生成新轨迹，适合快速实验
```

### 命令行覆盖参数

```bash
# 修改训练轮数和学习率
python train.py --training.epochs 500 --training.lr 5e-6

# 换更大的位置细胞数
python train.py --task.n_pc 512

# 指定保存目录
python train.py --training.save_dir /data/results/run1

# 使用自定义配置文件
python train.py --config my_config.yaml
```

### 3. 在代码里使用

```python
from train import load_config, train

cfg = load_config("config.yaml")
cfg.training.epochs = 200          # 直接修改配置字段
train(cfg)
```

---

## 轨迹数据生成

本项目不依赖任何外部数据文件。`get_dataloader` 可以直接在内存中生成随机游走轨迹，也可以从 `generate_data.py` 导出的 `.npz` 文件加载。训练使用预生成数据时，会额外复用预编码的初始条件并把目标编码下放到 DataLoader worker 中，以减少主进程 CPU 开销。

### 运动模型

轨迹基于 Sargolini et al. (2006) 的啮齿动物运动模型生成，核心参数：

| 参数 | 值 | 含义 |
|------|-----|------|
| `dt` | 0.02 s | 时间步长（20 ms） |
| `B` | 0.26 | 角速度 OU 衰减系数 |
| `MV` | 0.1 m/s | 平移速度均值（10 cm/s） |
| `SV` | 0.13 m/s | 平移速度标准差 |
| `SW` | 0.52π rad/s | 角速度标准差（约 94°/s） |

每步生成过程：
1. **角速度**：`ω_t = (1−B)·ω_{t−1} + B·N(0, SW²)`（Ornstein-Uhlenbeck 过程，向 0 均值回归）
2. **平移速度**：从 `N(MV, SV²)` 截断采样（只取正值，动物只向前运动）
3. **朝向更新**：`θ_t = θ_{t−1} + ω_t · dt`，wrap 到 `[−π, π]`
4. **位置更新**：`(x, y) += v · [cos(θ), sin(θ)] · dt`
5. **边界反射**：超出环境范围时对应轴速度取反（镜面反射）
6. **速度记录**：在边界反射之前记录 `ego_vel = [v·cos(θ), v·sin(θ), ω]`，确保速度与位移因果一致

### 生成并保存数据（推荐）

```bash
# 默认：生成 steps_per_epoch × batch_size = 10000 条轨迹
python generate_data.py --output data/train.npz

# 生成更多数据 + 同时输出可视化 PDF
python generate_data.py --output data/train.npz --num_samples 100000 --visualize

# 生成单独的评估集
python generate_data.py --output data/eval.npz --num_samples 4000 --seed 9999

# 指定自定义参数
python generate_data.py \
    --output data/large_env.npz \
    --env_size 3.0 \
    --num_samples 50000 \
    --visualize
```

生成完成后 `data/train.npz` 包含所有轨迹，`data/train_vis.pdf` 是可视化报告（见下节）。

### 手动生成并查看轨迹

```python
import matplotlib.pyplot as plt
from dataset import TrajectoryDataset

ds = TrajectoryDataset(
    num_samples=10,
    seq_len=100,
    env_size=2.2,
    seed=42
)

traj = ds[0]
print("init_pos :", traj["init_pos"])        # (2,)   初始 x, y 坐标（米）
print("init_hd  :", traj["init_hd"])         # (1,)   初始朝向（弧度）
print("ego_vel  :", traj["ego_vel"].shape)   # (100, 3) [vx, vy, ω] 每步
print("target_pos:", traj["target_pos"].shape)  # (100, 2)
print("target_hd:", traj["target_hd"].shape)    # (100, 1)

# 可视化轨迹
plt.figure(figsize=(5, 5))
plt.plot(traj["target_pos"][:, 0], traj["target_pos"][:, 1])
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title("Sample trajectory")
plt.savefig("trajectory.png")
```

### 可视化报告说明

`generate_data.py --visualize` 生成一张包含 6 个面板的 PDF 报告：

| 面板 | 内容 |
|------|------|
| Sample trajectories | 最多 16 条轨迹叠加在环境中 |
| Position coverage | 所有时间步的位置热力图（验证覆盖均匀性） |
| Translational speed | 速度分布直方图（均值应约 0.1 m/s） |
| Angular velocity | 角速度分布（应对称，均值约 0） |
| Head direction rose | 朝向极坐标直方图（应均匀分布） |
| Velocity–displacement | 速度×dt 与实际位移的散点图（验证数据一致性） |

### 控制数据生成行为

通过 `config.yaml` 中的 `task` 节控制：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `env_size` | 2.2 | 方形环境边长（米），位置范围 `[−1.1, 1.1]²` |
| `seq_len` | 100 | 每条轨迹的时间步数（100步 × 20ms = 2秒） |
| `velocity_noise` | [0, 0, 0] | 叠加到 ego_vel 三个分量上的高斯噪声标准差 `[σvx, σvy, σω]` |
| `neurons_seed` | 8341 | 数据生成随机种子（同时控制细胞位置的采样） |

每个 epoch 生成的轨迹数 = `steps_per_epoch × batch_size`（默认 1000 × 10 = 10000 条）。

---

## 配置参数详解

所有参数在 `config.yaml` 中定义，分为三个节。

### `task` — 任务与环境

```yaml
task:
  env_size: 2.2          # 方形环境边长（米）。
                         # 影响位置细胞均匀采样的范围（-env_size/2 ~ env_size/2）。

  n_pc: [256]            # 位置细胞数列表。列表长度决定有几组 PlaceCellEnsemble。
                         # 默认 [256] 表示 1 组 256 个细胞。
                         # 可设为 [128, 128] 以创建 2 组，各有独立输出头。

  pc_scale: [0.01]       # 位置细胞高斯调谐曲线的标准差（米）。
                         # 原论文用 0.01 m（1 cm），比环境尺寸小很多，使响应高度局部化。
                         # 增大此值会让每个细胞的响应区域更大，覆盖更宽的位置范围。

  n_hdc: [12]            # 方向细胞数列表。默认 12 个细胞均匀覆盖 360°。

  hdc_concentration: [20.0]  # Von Mises 分布的浓度参数 κ。
                              # κ 越大，方向细胞的调谐曲线越尖锐（响应角度范围越窄）。
                              # κ=20 对应约 ±25° 的有效响应宽度。

  seq_len: 100           # 每条轨迹的时间步数。
                         # 100 步 × dt(0.02s) = 2 秒轨迹。

  neurons_seed: 8341     # 控制细胞优选方向/位置的随机采样，以及轨迹生成的随机性。
                         # 固定此值可复现相同的细胞配置。与原版论文保持一致。

  targets_type: "softmax"  # 训练目标的编码方式，决定监督信号的形式：
                            # "softmax"    — 对 log_posterior 取 softmax，软概率目标（默认，推荐）
                            # "voronoi"    — argmax 的 one-hot，硬目标（Voronoi 分区）
                            # "sample"     — 从 softmax 分布采样的 one-hot
                            # "normalized" — exp(unnormalized log-pdf)，每个细胞独立二分类

  lstm_init_type: "softmax"  # LSTM 初始隐状态的编码方式，选项与 targets_type 相同。
                              # 额外支持 "zeros"：用全零初始化（不依赖初始位置）。

  velocity_noise: [0.0, 0.0, 0.0]  # 速度噪声 [σvx, σvy, σω]。
                                    # 在训练时给 ego_vel 的三个分量加独立高斯噪声。
                                    # 0.0 表示不加噪声。
```

### `model` — 模型架构

```yaml
model:
  nh_lstm: 128         # LSTM 隐层维度。
                       # 原论文 128，增大可提升表达能力，但训练更慢。

  nh_bottleneck: 256   # 瓶颈线性层输出维度（LSTM → 瓶颈 → 输出头）。
                       # 这一层的激活就是我们分析"网格细胞"涌现的地方。
                       # 原论文 256，对应可视化中 256 个独立的 rate map。

  dropout_rate: 0.5    # 瓶颈层之后的 Dropout 比率（训练时生效）。
                       # 较大的 dropout 有助于表示去相关，促进网格细胞涌现。

  bottleneck_has_bias: false  # 瓶颈线性层是否包含偏置项。
                               # 原版为 false，不加偏置。

  init_weight_disp: 0.0  # 输出头权重初始化的均值偏移量。
                          # 权重从 U[−1/√256 + disp, 1/√256 + disp] 初始化。
                          # 默认 0.0（对称初始化）。
```

### `training` — 训练超参数

```yaml
training:
  epochs: 1000           # 总训练轮数。原论文 1000 epoch。

  steps_per_epoch: 1000  # 每个 epoch 的梯度更新次数。
                         # 每步处理一个 batch，共 1000 × 10 = 10000 条轨迹/epoch。

  batch_size: 10         # 每个梯度更新步使用的轨迹数。
                         # 原论文 10，较小 batch 与 RMSprop 配合更稳定。

  eval_batch_size: 4000  # 评估时使用的轨迹总数（4000 条）。
                         # 用于计算 rate map 和网格评分，数量越多越准确。

  lr: 1.0e-5             # RMSprop 学习率。原论文 1e-5，偏小但训练稳定。

  momentum: 0.9          # RMSprop 动量系数。

  weight_decay: 1.0e-5   # L2 正则化系数，通过 optimizer 的 weight_decay 实现。

  grad_clip: 1.0e-5      # 梯度裁剪阈值（clip_grad_value_），限制每个梯度分量的绝对值。
                         # 原论文使用极小的裁剪值（1e-5）以确保训练稳定性。

  eval_every: 2          # 每隔几个 epoch 评估一次（计算网格评分 + 生成 PDF）。
                         # 评估比较耗时，建议设为 5~10 以加速实验。

  save_dir: "./results"  # 评估 PDF 的保存目录（自动创建）。
                         # 文件命名格式：rates_and_sac_epoch_XXXX.pdf
```

---

## 模型架构详解

```
初始位置 (B, 2)  ──→  PlaceCellEnsemble.get_init  ──→  (B, 256)  ──┐
初始朝向 (B, 1)  ──→  HDCEnsemble.get_init         ──→  (B, 12)   ──┤
                                                                      │ concat
                                                              (B, 268)┘
                                                                      │
                                              ┌── Linear(268, 128) ──→ h₀
                                              └── Linear(268, 128) ──→ c₀
                                                         ↓
速度序列 (B, 100, 3) ──→ LSTMCell(3 → 128) [手动展开 100步] ──→ (B, 100, 128)
                                                         ↓
                                           Linear(128 → 256, no bias)
                                                         ↓
                                              Dropout(0.5, train only)
                                                  [bottleneck]
                                            (B, 100, 256)
                                              /            \
                        Linear(256 → 256)           Linear(256 → 12)
                          [pc_head]                   [hdc_head]
                        pc_logits                   hdc_logits
                      (B, 100, 256)              (B, 100, 12)
                              \                    /
                        交叉熵损失 (softmax cross-entropy)
                               total_loss
```

**关键设计**：
- 瓶颈层（bottleneck）**没有激活函数**，直接线性投影，这是论文的关键设计
- 分析网格细胞时，我们关注的是 bottleneck 激活（256 维），不是 LSTM 隐状态
- 训练目标是有监督的：让模型通过积分速度来预测当前的位置/方向细胞激活

---

## 评估与可视化

每隔 `eval_every` 个 epoch，训练脚本会：

1. 用 4000 条轨迹跑前向传播，收集 bottleneck 激活
2. 对每个 bottleneck 神经元计算 **rate map**（2D 位置直方图，统计各位置的平均激活）
3. 对每个 rate map 计算 **空间自相关图（SAC）**
4. 计算 **grid score**：在 SAC 上衡量 60° 旋转对称性（`grid_score_60`）和 90° 旋转对称性（`grid_score_90`）
5. 将所有 rate map 和 SAC 按 `grid_score_60` 降序排列，保存为 PDF

输出示例：
```
Epoch    2 | loss mean=4.8246 std=0.0071
  grid_score_60 max=0.35  grid_score_90 max=0.84
```

`grid_score_60 > 0.3` 表明对应神经元已涌现出类似六边形网格的响应模式。

### 手动计算网格评分

```python
import numpy as np
from scores import GridScorer

starts = [0.2] * 10
ends   = list(np.linspace(0.4, 1.0, num=10))
scorer = GridScorer(
    nbins=20,
    coords_range=[[-1.1, 1.1], [-1.1, 1.1]],
    mask_parameters=list(zip(starts, ends))
)

# rate_map: (20, 20) numpy array，某个神经元的空间响应图
score_60, score_90, mask_60, mask_90, sac = scorer.get_scores(rate_map)
print(f"Grid score 60°: {score_60:.4f}")
print(f"Grid score 90°: {score_90:.4f}")
```

---

## generate_data.py 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--output` | 必填 | 保存路径，如 `data/train.npz` |
| `--config` | `config.yaml` | 读取默认参数的配置文件 |
| `--num_samples` | `steps_per_epoch × batch_size` | 生成的轨迹总数 |
| `--seq_len` | config 中的值 | 每条轨迹的时间步数 |
| `--env_size` | config 中的值 | 方形环境边长（米） |
| `--seed` | config 中的 `neurons_seed` | 随机种子 |
| `--visualize` | 关闭 | 是否生成可视化 PDF |
| `--vis_output` | `<output>_vis.pdf` | 可视化 PDF 的保存路径 |

训练时使用保存的数据：

```bash
python train.py --data_path data/train.npz
```

数据只在 `DataLoader` 初始化时加载一次，每个 epoch 通过 shuffle 重新打乱，**不会重复读取文件**。

---

## 快速实验

### 用小模型验证流程（约 5 分钟）

```bash
python train.py \
    --task.n_pc 64 \
    --task.n_hdc 8 \
    --model.nh_bottleneck 128 \
    --training.epochs 50 \
    --training.steps_per_epoch 200 \
    --training.eval_every 10 \
    --training.save_dir ./results_quick
```

### 复现论文结果（约数小时，需要 GPU）

```bash
python train.py  # 使用 config.yaml 全部默认值即可
```

论文中 1000 epoch 后 bottleneck 层会有相当比例的神经元涌现出六边形网格响应（`grid_score_60 > 0.3`）。

---

## 与原版 TF 实现的主要差异

| 方面 | 原版（TF 1.12 + Sonnet） | 本版（PyTorch） |
|------|--------------------------|-----------------|
| 数据来源 | TFRecord 文件（需下载） | numpy 实时生成 |
| 数据加载 | TF Queue（多线程） | DataLoader（num_workers=4） |
| LSTM | `snt.LSTM` + `dynamic_rnn` | `nn.LSTMCell`（手动展开） |
| 优化器 | `tf.train.RMSPropOptimizer` | `torch.optim.RMSprop` |
| 概率计算 | `tensorflow_probability` | `scipy.stats` / numpy |
| 配置管理 | `tf.flags` | `config.yaml` + `argparse` |
| 评分/可视化 | 已是 numpy（直接移植） | 同左 |

所有超参数与原论文完全一致。
