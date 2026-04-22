# 当前数据集生成逻辑与 MuJoCo 迁移参考

日期：2026-04-23

## 背景

当前仓库的数据集生成并不是“在线与环境交互并即时喂给训练”，而是先离线生成一批固定长度的轨迹序列，再将这些序列保存成训练可直接读取的 `.npz` 数据集文件。

这个设计对复现实验很有价值：

- 数据分布固定，便于比较不同模型或超参数。
- 训练与数据生成解耦，排查问题时更容易区分是“模型问题”还是“数据问题”。
- 可以在生成阶段加入预览、统计与完整性检查，而不是等训练异常后再回头看数据。

本文档的目的不是重复 CLI 用法，而是提炼当前实现中的通用技术逻辑，并说明在迁移到 MuJoCo 一类仿真引擎时，哪些设计应当保留，哪些实现可以替换。

## 目标

- 解释当前数据集生成的整体流水线。
- 说明关键参数的来源、默认值和作用边界。
- 说明每条轨迹样本和整份数据集包含哪些字段。
- 说明 train / eval 生成策略与附属可视化产物。
- 给出迁移到 MuJoCo 时的保留项、替换项和建议接口。

## 非目标

- 不设计新的 MuJoCo 训练管线。
- 不在本文档中规定 MuJoCo 环境的具体动力学或奖励函数。
- 不要求未来实现必须逐字段逐文件完全复刻当前仓库。
- 不覆盖当前项目中与训练、评估 PDF、模型结构无关的细节。

## 1. 数据生成总流程

当前实现可以抽象为一条与具体仿真引擎无关的流水线：

```text
配置输入
  -> 轨迹采样器
  -> 序列状态生成
  -> train / eval 数据集构造
  -> 落盘存储
  -> 可选预览与检查产物
```

这个抽象非常重要。迁移到 MuJoCo 时，最自然的做法不是推翻整套数据集机制，而是仅替换“轨迹采样器”这一层，将当前的手写二维随机游走采样器替换为 MuJoCo 环境中的 rollout 采样器。

### 当前仓库的两层结构

当前仓库的数据集生成逻辑分成两层：

- CLI / 配置层：决定生成多少条样本、输出到哪里、是否导出 PDF / MP4 / 进度图。
- 生成内核层：真正采样轨迹、组织字段、保存 `.npz` 文件。

对应实现位置：

- [generate_data.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/generate_data.py)
- [grid_cells/data/generation.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/grid_cells/data/generation.py)
- [grid_cells/data/dataset.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/grid_cells/data/dataset.py)
- [grid_cells/data/trajectory_generation.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/grid_cells/data/trajectory_generation.py)

### 当前实现的执行顺序

一次标准数据生成的大致顺序如下：

1. 读取 `config.yaml` 与命令行覆盖项。
2. 确定训练集输出路径，以及是否还要生成 eval 集。
3. 解析轨迹相关参数，例如 `seq_len`、`env_size`、`velocity_noise`、`seed`。
4. 解析样本数量、并行 worker 数、进度预览刷新频率。
5. 构造 place-cell / head-direction-cell ensembles，仅供动画或后续编码逻辑使用。
6. 调用统一的 `generate_dataset_file(...)` 生成训练集文件。
7. 如果没有开启 `--train_only`，再生成一份 eval 数据集文件。
8. 按需导出 PDF、MP4、进度 PNG。

### 应迁移的核心思想

迁移到 MuJoCo 时，建议保留以下总体思想：

- 继续采用“先生成离线数据集，再训练”的工作流。
- 让 train / eval 仍然是显式生成的工件，而不是依赖训练脚本隐式在线采样。
- 将“轨迹采样器”设计成可以单独替换的模块，使其既可以来自当前随机模型，也可以来自 MuJoCo rollout。

## 2. 参数来源与默认值决策

当前实现中的参数并不是都直接来自命令行，而是经过“配置文件默认值 + CLI 覆盖 + 代码回填默认值”三层决策。

对迁移设计来说，最重要的是区分两类参数：

- 决定数据分布的参数。
- 仅影响生成效率或可视化的参数。

前者必须在不同实验之间谨慎控制，后者则主要影响运行速度和调试体验。

### 2.1 环境与轨迹参数

这类参数决定“一条轨迹长什么样”：

- `task.seq_len`
  - 每条轨迹的时间步数。
- `task.env_size`
  - 当前二维环境边长，位置范围限制为 `[-env_size/2, env_size/2]`。
- `task.velocity_noise`
  - 对最终 `ego_vel` 观测量加的噪声，格式为 3 维列表。
- `task.neurons_seed`
  - 当前数据生成使用的基础随机种子。

其中：

- `seq_len` 决定了每条样本的时间长度。
- `env_size` 决定空间边界与覆盖尺度。
- `velocity_noise` 决定训练输入是否带观测噪声。
- `seed` 决定数据集是否可复现。

### 2.2 数据集规模参数

这类参数决定要生成多少条轨迹：

- `data_generation.num_samples`
- `data_generation.eval_num_samples`

当前实现的默认策略不是完全独立指定，而是带有训练流程导向：

- 如果 `data_generation.num_samples` 未指定，则默认使用
  `training.steps_per_epoch * training.batch_size`
- 如果 `data_generation.eval_num_samples` 未指定，则默认使用
  `training.eval_batch_size`

这意味着当前项目默认把“训练集大小”绑定到了“一轮训练需要消耗多少条样本”，这是工程约定，不是理论要求。

迁移到 MuJoCo 时可以保留这个默认行为，但更建议将其视为“方便的默认值”，而不是强约束。对于规模更大或生成更贵的仿真环境，通常会希望显式控制离线 train / eval 数据量。

### 2.3 生成过程参数

这类参数主要影响性能和调试：

- `data_generation.num_workers`
  - 用于并行生成轨迹的 worker 数。
- `data_generation.progress_every`
  - 每完成多少个 chunk 刷新一次进度图。

这些参数不会改变当前样本定义，但可能影响生成耗时、刷新频率和任务切块方式。

### 2.4 可视化与动画参数

这类参数不影响训练样本本身，只影响附属产物：

- `visualization.spatial_bins`
- `visualization.directional_bins`
- `visualization.anim_num_traj`
- `visualization.anim_fps`
- `visualization.anim_step`
- `visualization.anim_workers`

其中：

- `spatial_bins` / `directional_bins` 控制统计图或预览图的分箱分辨率。
- `anim_num_traj` 控制动画中展示多少条轨迹。
- `anim_fps`、`anim_step`、`anim_workers` 控制动画的播放帧率、采样步幅和并行渲染速度。

### 2.5 参数优先级

当前实现采用明确的优先级：

1. CLI 显式覆盖值
2. `config.yaml` 中的配置值
3. 代码中的保底默认值

例如动画参数的解析就会优先使用 CLI，其次使用 `visualization.*`，最后回退到代码中的固定默认值。

迁移到 MuJoCo 时，建议保留这种优先级模型。数据集生成通常需要大量 sweep，而“稳定默认值 + 方便覆盖”的配置模式比硬编码更适合长期实验。

## 3. 轨迹状态、随机性与存储格式

这一节是迁移时最需要保留的抽象层：一份样本到底包含什么信息，训练到底拿到什么监督。

### 3.1 当前样本字段

当前实现中，每条轨迹样本包含以下字段：

- `init_pos`
- `init_hd`
- `ego_vel`
- `target_pos`
- `target_hd`

单条样本的形状如下：

- `init_pos`: `(2,)`
- `init_hd`: `(1,)`
- `ego_vel`: `(seq_len, 3)`
- `target_pos`: `(seq_len, 2)`
- `target_hd`: `(seq_len, 1)`

整份数据集保存时，会按样本维堆叠为：

- `init_pos`: `(num_samples, 2)`
- `init_hd`: `(num_samples, 1)`
- `ego_vel`: `(num_samples, seq_len, 3)`
- `target_pos`: `(num_samples, seq_len, 2)`
- `target_hd`: `(num_samples, seq_len, 1)`

### 3.2 字段语义

这些字段背后的监督结构是：

- 初始状态：`init_pos`、`init_hd`
- 运动输入：`ego_vel`
- 每一步的状态真值：`target_pos`、`target_hd`

也就是说，当前数据集提供的是一种“路径积分型监督”：

给定轨迹起点状态和每步自运动输入，模型需要恢复后续位置与朝向表征。

这个抽象在 MuJoCo 中依然成立。真正应该迁移的是这种字段关系，而不是二维随机游走本身。

### 3.3 当前轨迹采样逻辑

当前仓库使用的是一个二维正方形环境下的手写随机运动模型：

- 初始位置在边界内均匀采样。
- 初始朝向在 `[-pi, pi]` 均匀采样。
- 角速度使用带衰减的随机过程更新。
- 平移速度来自高斯采样，并强制为正。
- 每一步依据当前朝向将速度积分为位置变化。
- 如果运动即将越出边界，则通过反射方式修正朝向。
- 最后再对 `ego_vel` 叠加观测噪声 `velocity_noise`。

有两个关键技术点值得迁移：

- 状态真值的演化与输入噪声是分开的。
- 位置 / 朝向是由连续时间步积分得到的，而不是独立逐步采样。

前者让训练输入可以带噪，而监督真值仍保持干净。后者保证样本具有真实的时间连续性。

### 3.4 随机性与可复现性

当前实现不是简单地把一个全局 RNG 扔给多个 worker 并行使用，而是显式构建“每条样本的独立 seed”：

- 先由 `base_seed` 构造 `SeedSequence`
- 再为每条样本派生单独的 sample seed
- 并行 worker 只负责消费这些预先分配好的 seed

这样做的直接收益是：

- 修改 `num_workers` 不会改变样本内容
- 并行度变化只影响生成顺序和性能，不影响数据分布

这一点对 MuJoCo 迁移尤其关键。MuJoCo rollout 常见的问题是 worker 数变化会导致环境重置顺序、随机数消费顺序和最终数据集内容一起变化。建议显式保留“样本级 seed 列表”的设计，而不是依赖隐式多进程随机性。

### 3.5 存储格式

当前实现使用压缩 `.npz` 落盘，并将最小元数据存到 `meta` 字段中。元数据包括：

- `env_size`
- `seq_len`
- `velocity_noise`
- `num_samples`

因此，一份 `.npz` 文件本身就足以恢复当前数据集的结构和必要上下文。

迁移到 MuJoCo 时，不一定必须继续使用 `.npz`，但建议至少保留“数组主体 + 轻量元数据”的自描述设计。最小元数据建议包括：

- 环境或场景标识
- 序列长度
- 样本数
- 控制或观测噪声配置
- 关键字段定义
- 随机种子策略

## 4. train / eval 生成差异与可视化附属产物

### 4.1 train / eval 的生成关系

当前实现不是从一份大数据集里后切分 train / eval，而是分别生成两份独立数据：

- 训练集写到 `training.data_path`
- 评估集写到 `training.eval_data_path`

主要差异如下：

- train 样本数默认取 `steps_per_epoch * batch_size`
- eval 样本数默认取 `eval_batch_size`
- eval 使用 `seed + 1` 作为基础种子

这种设计很简单，但足够实用：

- train / eval 分布相近
- 它们不会共享完全相同的样本
- 不需要额外切分逻辑

迁移到 MuJoCo 时，建议仍然保留“显式生成 train / eval”而不是“先生成一大坨再随机切分”的方式。原因是仿真环境下的评估分布通常更值得单独控制，例如：

- 固定地图
- 固定障碍物
- 固定重置策略
- 固定动作采样器
- 固定 rollout policy

### 4.2 可选附属产物

当前数据集生成可附带以下工件：

- PDF 预览
- MP4 动画
- 生成过程进度 PNG

这些文件不参与训练，但对数据质量检查很重要。

默认命名规则由主输出路径派生：

- `train.npz` -> `train_vis.pdf`
- `train.npz` -> `train_traj.mp4`
- `train.npz` -> `train_progress.png`
- `eval.npz` -> `eval_progress.png`

### 4.3 进度预览的作用

当前实现中的进度图不是简单的进度条截图，而是一个轻量的数据质量快照，包含：

- 部分采样轨迹
- 空间覆盖热图
- 速度分布直方图
- 角速度分布直方图
- 朝向分布图
- 当前完成度、吞吐和 ETA

这代表了一种很值得保留的设计观念：数据生成过程本身就应当可观测，而不是仅在结束后检查文件是否存在。

迁移到 MuJoCo 时，建议至少保留两类检查：

- 最终数据检查：空间覆盖、速度分布、状态边界、异常值。
- 生成过程检查：定期导出摘要，尽早发现 rollout 卡死、分布塌缩、碰撞异常等问题。

### 4.4 在 MuJoCo 中可扩展的检查项

如果 MuJoCo 环境更复杂，可以在当前检查项基础上扩展：

- 状态变量范围统计
- 控制输入分布
- 碰撞率或接触事件统计
- 回合提前终止比例
- 不同场景或重置模式下的覆盖均衡性

这些都不要求改变训练接口，但会显著提高数据生成的可控性和可解释性。

## 5. MuJoCo 迁移时保留项、替换项、建议接口

### 5.1 建议保留的设计

迁移到 MuJoCo 时，建议优先保留以下结构性设计：

- 离线固定数据集工作流
- train / eval 分开生成
- 样本级 seed 派生机制
- 初始状态、运动输入、目标状态分离
- 自描述的数据文件格式
- 可视化和生成过程检查机制

这些设计不依赖当前二维环境，属于更通用的数据工程逻辑。

### 5.2 建议替换的部分

以下内容属于当前仓库的具体实现，不应视为必须继承的约束：

- 二维正方形边界
- 反射式撞墙规则
- 当前固定的速度与角速度统计模型
- 仅用 `(x, y, heading)` 描述状态的简化动力学

在 MuJoCo 中，轨迹更可能来自：

- 环境 reset 后的状态初始化
- 控制器或策略驱动的动作序列
- 真实动力学积分
- 传感器或估计器输出的观测

### 5.3 建议的抽象接口

迁移后，建议把“轨迹采样器”抽象成如下接口。

输入：

- 环境配置
- `seq_len`
- 样本 seed
- 控制策略或动作采样器
- 噪声配置

输出：

- 初始状态
- 每步控制或自运动输入
- 每步状态真值
- 可选附加观测
- rollout 元信息

一个更通用的字段命名可以是：

- `init_state`
- `ego_motion` 或 `control`
- `target_state`
- `observation`
- `meta`

### 5.4 与当前训练接口的兼容策略

是否继续使用当前 5 个字段名，取决于迁移目标。

如果目标是“尽量少改当前训练代码”，则应在 MuJoCo 侧将 rollout 结果映射回当前语义：

- `init_pos`
- `init_hd`
- `ego_vel`
- `target_pos`
- `target_hd`

如果目标是“建立更通用的仿真数据接口”，则可以升级为更抽象的命名，再在训练侧增加一层适配。

更稳妥的方案是折中处理：

- 底层 rollout 采样接口抽象化
- 落盘时导出一份与当前训练兼容的视图

这样可以同时获得：

- MuJoCo 状态空间更丰富的表达能力
- 与现有训练代码的低改动兼容

### 5.5 迁移判断标准

可以用下面的标准判断迁移应当做到什么程度：

如果任务本质仍然是二维或低维位姿积分预测，则应尽量保留当前字段语义，只替换轨迹采样器。

如果任务已经变成通用机器人动力学建模、复杂控制或多模态观测预测，则应升级字段抽象，再实现一层到当前训练输入格式的映射。

## 实现映射速查

为了方便后续对照代码，这里给出当前实现的最小映射：

- CLI 入口与参数装配：
  [generate_data.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/generate_data.py)
- 单文件数据集生成与附属工件输出：
  [grid_cells/data/generation.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/grid_cells/data/generation.py)
- 数据集对象、保存、加载、字段堆叠：
  [grid_cells/data/dataset.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/grid_cells/data/dataset.py)
- 轨迹采样、并行 worker、样本级 seed：
  [grid_cells/data/trajectory_generation.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/grid_cells/data/trajectory_generation.py)
- 生成过程监控与进度预览：
  [grid_cells/data/monitor.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/grid_cells/data/monitor.py)

## 结论

当前仓库最值得迁移到 MuJoCo 的，不是二维随机游走的具体公式，而是下面这套数据工程骨架：

- 固定长度序列的离线数据集生成
- 初始状态 / 运动输入 / 状态真值三段式监督结构
- 样本级可复现随机性控制
- train / eval 显式分离
- 生成期即提供分布检查与调试可视化

换句话说，MuJoCo 迁移时真正该替换的是“轨迹如何产生”，而不是“数据集如何组织”。
