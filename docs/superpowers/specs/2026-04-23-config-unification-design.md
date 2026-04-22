# 配置统一化设计：generate_data.py 与 train.py 参数收敛

## 背景

当前项目已经部分采用“`config.yaml` 提供默认值，CLI 只做覆盖”的模式，但两个入口脚本并不一致：

- `train.py` 已将大部分 CLI 解析下沉到公共模块，并支持 `--section.key value` 风格覆盖部分配置项。
- `generate_data.py` 仍保留大量独立的 CLI 参数定义，参数来源分散，和 `train.py` 的行为边界不一致。

这导致三个问题：

1. 参数定义重复，维护成本高。
2. `train.py` 与 `generate_data.py` 的覆盖方式不统一，使用体验不一致。
3. 一部分稳定默认值没有沉淀到配置文件中，难以复用。

## 目标

本设计的目标如下：

1. 让 `generate_data.py` 与 `train.py` 共享一套配置覆盖机制。
2. 将稳定默认值统一沉淀到 `config.yaml`。
3. 保留显式 CLI 覆盖风格，支持 `task.*`、`model.*`、`training.*`、`visualization.*` 等 section 下的参数随时覆盖。
4. 保留一次性运行开关在 CLI 中，例如 `--visualize`、`--animate`、`--train_only`。
5. 减少重复 `argparse` 代码，新增参数时只需要维护一处定义。

## 非目标

本次设计不包含以下内容：

1. 不引入自动从 YAML 结构推导全部 CLI 参数的动态方案。
2. 不改变训练或数据生成的核心业务逻辑。
3. 不在本阶段移除所有历史兼容行为，兼容策略按平滑迁移处理。

## 设计原则

### 单一默认值来源

稳定默认值应优先定义在 `config.yaml` 中，而不是散落在多个入口脚本内。

### CLI 仅做覆盖

CLI 的职责是：

1. 指定配置文件入口，例如 `--config`。
2. 对配置项做显式覆盖，例如 `--training.batch_size 32`。
3. 提供只影响本次执行的一次性运行开关。

### 配置与运行开关分层

配置文件负责“实验默认值”和“稳定路径”；CLI 负责“一次执行行为”。这样边界明确，便于长期维护。

## 参数边界

### 放入 `config.yaml` 的内容

以下内容应作为长期默认值，进入配置文件：

- `task.*`
- `model.*`
- `training.*`
- `visualization.*`
- 只属于数据生成但应长期复用的默认值，例如默认样本数、默认 worker 数、默认可视化输出路径等

### 保留在 CLI 的内容

以下内容继续保留为 CLI 运行开关，不作为配置语义的一部分：

- `--config`
- `--visualize`
- `--animate`
- `--train_only`

### 路径参数策略

路径参数按“稳定路径优先配置化”的原则处理：

- 训练数据路径、评估数据路径、结果目录等稳定路径放入配置。
- `generate_data.py` 所需的默认输出路径也允许配置化。
- 运行时仍允许通过 CLI 显式覆盖这些路径。

## 推荐配置结构

在现有 `task`、`model`、`training`、`visualization` 基础上，新增独立 section：

```yaml
data_generation:
  num_samples: null
  eval_num_samples: null
  num_workers: 8
  progress_every: 4
  vis_output: null
  anim_output: null
  progress_output: null
  eval_progress_output: null
```

说明：

- `task` 继续承载环境、轨迹和神经元相关语义，例如 `env_size`、`seq_len`、`neurons_seed`。
- `model` 继续承载模型结构参数。
- `training` 继续承载训练行为和训练相关路径，例如 `data_path`、`eval_data_path`、`save_dir`。
- `visualization` 继续承载动画和可视化默认值，例如 bins、FPS、渲染步长。
- `data_generation` 专门承载 `generate_data.py` 的长期默认值，避免把数据生成语义混入 `training` 或 `visualization`。

## CLI 统一方案

### 用户可见行为

两个入口脚本统一支持以下覆盖风格：

```bash
python train.py --task.env_size 2.4 --training.batch_size 32
python generate_data.py --task.seq_len 800 --visualization.anim_fps 30
```

配置优先级统一为：

1. 代码内最小保底默认值
2. `config.yaml`
3. CLI 显式覆盖
4. CLI 一次性运行开关

### 公共注册表

引入一份公共“可覆盖配置字段注册表”，用于描述：

- 参数路径，例如 `task.env_size`
- 目标类型，例如 `int`、`float`、`bool`
- 是否多值，例如 `nargs='+'`
- 对应的 `argparse` 元信息

该注册表由 `train.py` 和 `generate_data.py` 共用，避免分别维护同一组 `add_argument()`。

### 公共注册函数

提供统一的 CLI 注册函数，负责根据注册表向 `argparse.ArgumentParser` 挂载参数。两个入口脚本的模式统一为：

1. 创建 parser
2. 注册公共覆盖参数
3. 注册当前脚本专属参数
4. 解析参数
5. 加载 `config.yaml`
6. 应用 CLI 覆盖
7. 执行业务逻辑

## 模块职责建议

### `grid_cells/common/config.py`

继续承担基础能力：

- `load_config`
- `dict_to_namespace`
- `namespace_to_dict`
- `str2bool`
- `apply_namespace_overrides`

### 新增公共 CLI 配置模块

建议新增类似 `grid_cells/common/config_cli.py` 的模块，专门负责：

- 维护可覆盖字段注册表
- 提供注册公共覆盖参数的函数
- 封装类型和列表参数解析细节
- 对未知 section 或字段提供一致的报错

这样可以避免把 CLI 注册细节继续堆入 `train.py` 或 `generate_data.py`。

## `generate_data.py` 的收敛方式

`generate_data.py` 调整后的参数来源应如下：

### 从配置读取默认值

- `task.seq_len`
- `task.env_size`
- `task.neurons_seed`
- `task.velocity_noise`
- `training.data_path`
- `training.eval_data_path`
- `visualization.spatial_bins`
- `visualization.directional_bins`
- `visualization.anim_num_traj`
- `visualization.anim_fps`
- `visualization.anim_step`
- `visualization.anim_workers`
- `data_generation.num_samples`
- `data_generation.eval_num_samples`
- `data_generation.num_workers`
- `data_generation.progress_every`
- `data_generation.vis_output`
- `data_generation.anim_output`
- `data_generation.progress_output`
- `data_generation.eval_progress_output`

### 继续保留为专属 CLI 开关

- `--visualize`
- `--animate`
- `--train_only`

### 路径解析规则

若 CLI 未显式传入路径，则优先使用配置中的默认值；若配置也为空，再按现有逻辑推导派生路径，例如基于输出文件名拼接 `_vis.pdf`、`_traj.mp4`、`_progress.png`。

## `train.py` 的收敛方式

`train.py` 继续保留当前总体模式，但不再在自身重复维护大段配置覆盖参数定义，而是改为复用公共注册逻辑。

训练入口仍可保留专属参数：

- `--config`
- `--data_path`
- `--eval_data_path`

其中：

- `training.data_path` 和 `training.eval_data_path` 依然是配置中的长期默认值。
- `--data_path` 与 `--eval_data_path` 作为更高优先级的入口级覆盖保留，避免破坏已有用法。

## 错误处理

系统需要统一处理以下错误：

1. CLI 覆盖的 section 不存在。
2. CLI 覆盖的字段不存在。
3. 类型不匹配，例如期望整数却传入非法字符串。
4. 正整数约束失败，例如 bins、workers、fps 小于等于 0。
5. 训练与评估输出路径冲突。

报错信息应明确指出具体参数名，避免静默失败。

## 测试策略

至少新增或调整以下测试：

1. 公共注册表能正确向 parser 注册覆盖参数。
2. `apply_namespace_overrides` 能正确处理 `task.*`、`model.*`、`training.*`、`visualization.*`、`data_generation.*`。
3. `train.py` 能通过 CLI 正确覆盖配置值。
4. `generate_data.py` 能通过 CLI 正确覆盖配置值。
5. `generate_data.py` 中运行开关 `--visualize`、`--animate`、`--train_only` 仍保持原语义。
6. 默认值优先级符合“配置 < CLI 覆盖 < 入口级运行开关”的预期。

## 迁移策略

采用平滑迁移，不做激进切换：

1. 先引入公共注册表与公共注册函数。
2. 让 `train.py` 迁移到公共注册逻辑，但尽量保持外部用法不变。
3. 让 `generate_data.py` 从手写独立参数迁移到统一覆盖逻辑。
4. 若存在旧参数名或历史兼容逻辑，短期内保留兼容。
5. 待测试稳定后，再评估是否清理冗余兼容分支。

## 方案对比与结论

讨论过的三个方案如下：

1. 只给 `generate_data.py` 补手写覆盖参数。
2. 用统一注册表和公共注册函数收敛参数定义。
3. 全自动从 YAML 结构动态生成 CLI。

最终采用方案 2，原因如下：

- 最符合“减少传参代码”和“复用参数定义”的目标。
- 兼顾可维护性与可控性，不会引入过度动态化带来的类型和帮助文本问题。
- 与项目现有结构兼容，改造成本可控。

## 成功标准

本设计完成后，应满足以下标准：

1. `train.py` 与 `generate_data.py` 都支持统一的 `--section.key value` 覆盖风格。
2. `task`、`model`、`training`、`visualization` 参数可被随时覆盖。
3. `generate_data.py` 的稳定默认值主要来自配置，而不是大量散落在入口脚本中。
4. 一次性运行开关仍保持 CLI 语义，不被混入配置文件职责。
5. 新增一个可覆盖配置字段时，只需维护一处公共定义。
