# `generate_data.py` 数据集目录化设计

日期：2026-04-23

## 背景

当前 `generate_data.py` 默认将训练集与评估集直接写到固定路径：

- `data/train.npz`
- `data/eval.npz`

这会带来几个问题：

- 多次生成的数据容易互相覆盖，难以留档。
- 预览 PDF、动画 MP4、进度 PNG 通过文件名派生，产物分散在目录中，不利于管理。
- 数据集关键信息只能从命令或文件内容中追溯，目录列表本身辨识度不足。

本设计的目标是将一次生成结果收敛为一个可识别、可追溯、可复用的数据集目录，同时保留旧的单文件输出兼容能力。

## 目标

- `generate_data.py` 默认输出改为“每次运行一个数据集目录”。
- 目录名包含足够的识别信息：时间戳、train/eval 样本数、`seq_len`、`seed`、`env_size`、可选 `tag`。
- 一次运行产生的所有文件统一写入同一目录。
- 目录内新增机器可读的 `meta.json` 和人类可读的 `README.txt`。
- 默认训练入口继续可用，不要求用户每次手动寻找最新目录。
- 显式传入旧式 `--output` / `--eval_output` 时仍支持单文件模式，避免破坏已有工作流。

## 非目标

- 不引入数据集索引数据库或集中注册表。
- 不实现旧数据集自动清理。
- 不支持复杂标签体系或多级命名策略。
- 不在本次改动中引入额外的训练逻辑重构。

## 设计概览

默认情况下，`generate_data.py` 不再直接写 `data/train.npz` 与 `data/eval.npz`，而是在 `data/datasets/` 下创建一个独立目录，并将本次运行的所有产物写入该目录。

同时，为了让现有训练默认配置仍可工作，每次成功生成目录化数据集后，额外维护一个固定入口：

- `data/latest/train.npz`
- `data/latest/eval.npz`
- `data/latest/meta.json`

`train.py` 的默认数据路径改为读取 `data/latest/*.npz`，从而兼顾“保留历史数据集”和“默认训练入口稳定”。

对于显式传入 `--output` / `--eval_output` 的调用，继续沿用当前单文件模式，不强制切换为目录模式。

## 输出目录结构

默认目录结构如下：

```text
data/datasets/<dataset-id>/
├── train.npz
├── eval.npz
├── meta.json
├── README.txt
├── train_vis.pdf
├── train_traj.mp4
├── train_progress.png
└── eval_progress.png
```

说明：

- `eval.npz` 仅在非 `--train_only` 时生成。
- `train_vis.pdf` 仅在 `--visualize` 时生成。
- `train_traj.mp4` 仅在 `--animate` 时生成。
- `train_progress.png` 仅在 `--visualize_progress` 时生成。
- `eval_progress.png` 仅在非 `--train_only` 且启用 `--visualize_progress` 时生成。

目录中允许缺少未启用功能对应的可选产物，但 `train.npz`、`meta.json`、`README.txt` 必须存在。

## 数据集目录命名

目录名格式固定为：

```text
YYYY-MM-DD_HHMMSS_train{train_num}_eval{eval_num}_seq{seq_len}_seed{seed}_env{env_size}_{tag}
```

规则如下：

- `tag` 可选；未提供时不追加尾部段。
- `train_only` 不进入目录名。
- 当 `--train_only` 生效时，目录名中的 `eval_num` 固定为 `0`，即 `eval0`。
- `env_size` 需要做文件名安全化，例如 `2.2 -> 2p2`。
- `tag` 只保留适合路径的字符，建议保留字母、数字、`-`、`_`，其余字符替换为 `_`。

示例：

```text
data/datasets/2026-04-23_143512_train10000_eval4000_seq1000_seed8341_env2p2_baseline/
```

## CLI 设计

新增参数：

- `--output_dir`
  - 指定目录模式下的数据集输出目录。
  - 如果提供，则直接向该目录写入 `train.npz`、`eval.npz`、`meta.json` 等文件。
- `--tag`
  - 指定用于目录名和元数据的可读标签，例如 `baseline`、`ablation-a`。

保留现有参数：

- `--output`
- `--eval_output`
- `--train_only`
- `--visualize`
- `--animate`
- `--visualize_progress`

模式选择规则：

1. 如果显式提供 `--output` 或 `--eval_output`，进入旧式单文件模式。
2. 否则，如果显式提供 `--output_dir`，进入目录模式并使用该目录。
3. 否则，进入默认目录模式，在 `data/datasets/` 下自动创建一个新目录。

这样可以保证：

- 旧调用方式仍可用。
- 新调用方式默认获得更完整的目录化结构。
- 新旧模式之间的行为边界明确。

## 目录模式下的路径派生

目录模式下，所有文件路径统一由目录派生，不再允许可选产物散落到其他位置。

固定派生规则：

- `train.npz` -> `<output_dir>/train.npz`
- `eval.npz` -> `<output_dir>/eval.npz`
- `meta.json` -> `<output_dir>/meta.json`
- `README.txt` -> `<output_dir>/README.txt`
- 训练集预览 PDF -> `<output_dir>/train_vis.pdf`
- 训练集动画 MP4 -> `<output_dir>/train_traj.mp4`
- 训练集进度 PNG -> `<output_dir>/train_progress.png`
- 评估集进度 PNG -> `<output_dir>/eval_progress.png`

`data_generation.vis_output`、`anim_output`、`progress_output`、`eval_progress_output` 在目录模式下不再单独生效，统一使用目录派生路径。它们在旧式单文件模式下继续按当前逻辑工作。

这个约束的目的很明确：目录模式的价值就在于收拢产物，如果目录模式下仍允许附属文件输出到别处，则会重新引入管理混乱。

## 元数据设计

目录内新增 `meta.json`，记录本次运行的实际生效参数与产物关系。它是机器可读主记录。

建议结构如下：

```json
{
  "dataset_id": "2026-04-23_143512_train10000_eval4000_seq1000_seed8341_env2p2_baseline",
  "created_at": "2026-04-23T14:35:12+08:00",
  "tag": "baseline",
  "train_only": false,
  "paths": {
    "train": "train.npz",
    "eval": "eval.npz",
    "train_vis": "train_vis.pdf",
    "train_anim": "train_traj.mp4",
    "train_progress": "train_progress.png",
    "eval_progress": "eval_progress.png"
  },
  "task": {
    "seq_len": 1000,
    "env_size": 2.2,
    "velocity_noise": [0.0, 0.0, 0.0],
    "neurons_seed": 8341
  },
  "generation": {
    "num_samples": 10000,
    "eval_num_samples": 4000,
    "num_workers": 8,
    "progress_every": 4
  },
  "visualization": {
    "enabled": true,
    "spatial_bins": 32,
    "directional_bins": 20
  },
  "animation": {
    "enabled": true,
    "anim_num_traj": 4,
    "anim_fps": 20,
    "anim_step": 4,
    "anim_workers": 4
  },
  "config_source": "config.yaml"
}
```

约束：

- 记录的是“实际生效值”，而不是未展开的默认值。
- 路径字段使用相对路径，便于整体搬迁目录。
- 时间使用 ISO 8601 且带时区。
- `train_only=true` 时，`paths.eval` 与 `eval_progress` 可以省略或置为 `null`，但写法必须在实现中统一。

## `README.txt` 设计

目录内新增一个简短的 `README.txt` 作为人类快速查看摘要，内容聚焦：

- 数据集 ID
- 创建时间
- train/eval 样本数
- `seq_len`
- `seed`
- `env_size`
- `tag`
- 目录内产物清单

`README.txt` 只承担“快速浏览”职责，不是机器解析来源。任何程序化读取都应依赖 `meta.json`。

## 与训练默认流程的兼容

当前训练默认从以下路径读取：

- `training.data_path: data/train.npz`
- `training.eval_data_path: data/eval.npz`

如果 `generate_data.py` 默认切换到时间戳目录，而训练默认路径仍保持不变，则“先生成再训练”的默认流程会断裂。因此需要同时引入稳定入口。

推荐方案：

- 每次目录模式生成成功后，同步更新 `data/latest/`。
- `config.yaml` 中默认训练路径改为：
  - `training.data_path: "data/latest/train.npz"`
  - `training.eval_data_path: "data/latest/eval.npz"`

同步策略优先选择软链接；若考虑平台兼容性或权限问题，也可以使用复制，但实现时必须二选一并保持一致。

设计倾向：

- 若项目运行环境以 Linux 为主，优先软链接。
- 如果软链接失败，主数据集目录不回滚；只报告 `latest` 更新失败，并明确提示默认训练入口可能未同步。

## 错误处理

需要明确以下行为：

- `--train_only` 与 `--eval_output` 同时使用时，继续报错。
- 单文件模式下，`--output` 与 `--eval_output` 解析到同一路径时，继续报错。
- 目录模式下，如果目标目录已存在且非空，默认报错，避免覆盖已有数据集。
- 显式传入 `--output_dir` 且目录为空时，允许写入。
- 自动生成目录名时，如果生成出的目标目录已存在，则应报错而不是自动附加随机后缀，保持命名可预测。
- `tag` 非法字符需要标准化处理，避免创建不可预期路径。

## 测试策略

### 保留旧模式回归

需要保留并继续通过以下测试场景：

- 显式传入 `--output` 时仍能写出单个训练集文件。
- 显式传入 `--output` 与 `--eval_output` 时仍能写出 train/eval 两个文件。
- 单文件模式下的冲突校验仍有效。

### 新增目录模式测试

需要新增覆盖：

- 未传 `--output` / `--eval_output` / `--output_dir` 时，自动创建 `data/datasets/<dataset-id>/`。
- 目录名包含：
  - 时间戳
  - `train{n}`
  - `eval{n}` 或 `eval0`
  - `seq{seq_len}`
  - `seed{seed}`
  - `env{env_size}`
  - 可选 `tag`
- 目录内必须生成：
  - `train.npz`
  - `meta.json`
  - `README.txt`
- 非 `train_only` 时额外生成 `eval.npz`。
- 启用 `--visualize` / `--animate` / `--visualize_progress` 时，产物位于同一目录中。
- `meta.json` 内容记录的是实际生效参数。
- 目录模式成功后，`data/latest/*` 被正确更新。
- 目录模式遇到已存在非空目录时会报错。

### 文档更新

需要同步更新：

- `README.md`
- `README.zh.md`
- 如有必要，`run_scripts.sh`

文档中应明确：

- 默认输出已变为数据集目录。
- 历史数据保存在 `data/datasets/`。
- 默认训练数据入口为 `data/latest/*.npz`。
- 如需兼容旧工作流，仍可显式使用 `--output` / `--eval_output`。

## 推荐实现边界

为了控制范围，本次实现限定为：

- 默认目录模式
- 新增 `--output_dir`
- 新增 `--tag`
- 生成 `meta.json`
- 生成 `README.txt`
- 更新 `data/latest/*`
- 保留旧单文件模式兼容
- 补充测试与文档

本次实现不包含：

- 数据集中心索引
- 历史数据自动清理
- 更复杂的数据集版本管理

## 规格自检

已检查以下项目：

- 无 `TODO`、`TBD`、占位章节。
- 目录模式与旧单文件模式边界明确，没有冲突。
- 训练默认入口与数据集目录化之间的兼容方案已明确。
- 范围保持在单一实现计划可覆盖的层级，没有扩展成多子项目。
- 对 `train_only`、目录已存在、可选产物路径归属等易歧义点均已明确。

## 验收标准

以下条件全部满足时，视为设计被正确实现：

1. `generate_data.py` 默认生成一个带可读 ID 的数据集目录。
2. 目录中统一包含本次运行的所有已启用产物。
3. 目录内存在 `meta.json`，且记录实际生效参数。
4. 目录内存在简要 `README.txt`。
5. 目录模式成功后，默认训练入口 `data/latest/*.npz` 可被训练流程直接使用。
6. 显式传入 `--output` / `--eval_output` 时，旧模式仍可用。
7. 相关测试与文档已更新。
