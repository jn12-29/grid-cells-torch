# 配置统一化实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 统一 `generate_data.py` 与 `train.py` 的配置覆盖机制，让稳定默认值进入 `config.yaml`，并保留显式 `--section.key` 覆盖与少量 CLI 运行开关。

**架构：** 在公共配置层新增一份“可覆盖字段注册表”和 parser 注册函数，由训练与数据生成两个入口共享。`generate_data.py` 补齐基于配置的默认值解析，`train.py` 去掉重复的手写覆盖参数定义，测试围绕公共注册、覆盖优先级和入口行为展开。

**技术栈：** Python, argparse, PyYAML, pytest

---

### 任务 1：建立公共 CLI 覆盖注册层

**文件：**
- 创建：[grid_cells/common/config_cli.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/grid_cells/common/config_cli.py)
- 修改：[grid_cells/common/config.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/grid_cells/common/config.py)
- 测试：[tests/test_train_step.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/tests/test_train_step.py)

- [ ] **步骤 1：补一个失败测试，锁定公共注册层的目标行为**

```python
def test_register_config_overrides_supports_shared_sections():
    parser = argparse.ArgumentParser()
    register_config_overrides(
        parser,
        sections=("task", "model", "training", "visualization", "data_generation"),
    )

    args = parser.parse_args(
        [
            "--task.env_size", "3.5",
            "--model.nh_lstm", "64",
            "--training.batch_size", "8",
            "--visualization.anim_fps", "30",
            "--data_generation.num_workers", "2",
        ]
    )

    assert args.task__env_size == 3.5
    assert args.model__nh_lstm == 64
    assert args.training__batch_size == 8
    assert args.visualization__anim_fps == 30
    assert args.data_generation__num_workers == 2
```

- [ ] **步骤 2：运行单测并确认失败**

运行：`pytest tests/test_train_step.py -k register_config_overrides_supports_shared_sections -v`
预期：FAIL，报错 `register_config_overrides` 不存在或未导入。

- [ ] **步骤 3：实现公共注册表和注册函数**

```python
CONFIG_OVERRIDE_SPECS = {
    "task": {
        "env_size": {"type": float},
        "n_pc": {"type": int, "nargs": "+"},
        "pc_scale": {"type": float, "nargs": "+"},
        "n_hdc": {"type": int, "nargs": "+"},
        "hdc_concentration": {"type": float, "nargs": "+"},
        "seq_len": {"type": int},
        "neurons_seed": {"type": int},
        "velocity_noise": {"type": float, "nargs": "+"},
    },
    "model": {
        "nh_lstm": {"type": int},
        "nh_bottleneck": {"type": int},
        "dropout_rate": {"type": float},
        "bottleneck_has_bias": {"type": str2bool},
        "init_weight_disp": {"type": float},
    },
}

def register_config_overrides(parser, sections=None):
    for section, fields in CONFIG_OVERRIDE_SPECS.items():
        if sections is not None and section not in sections:
            continue
        for field, spec in fields.items():
            parser.add_argument(
                f"--{section}.{field}",
                dest=f"{section}__{field}",
                **spec,
            )
    return parser
```

- [ ] **步骤 4：扩展覆盖应用逻辑，支持新增 section 并校验 section 格式**

```python
def apply_namespace_overrides(
    cfg: SimpleNamespace,
    args: argparse.Namespace,
    skip_keys=("config", "data_path", "eval_data_path"),
) -> SimpleNamespace:
    for key, value in vars(args).items():
        if key in skip_keys or value is None:
            continue
        if "__" not in key:
            continue
        section, attr = key.split("__", 1)
        target = getattr(cfg, section, None)
        if target is None:
            raise AttributeError(f"Unknown config section: {section}")
        if not hasattr(target, attr):
            raise AttributeError(f"Unknown config field: {section}.{attr}")
        setattr(target, attr, value)
    return cfg
```

- [ ] **步骤 5：运行测试确认通过**

运行：`pytest tests/test_train_step.py -k "register_config_overrides_supports_shared_sections or apply_overrides_updates_nested_sections" -v`
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add tests/test_train_step.py grid_cells/common/config.py grid_cells/common/config_cli.py
git commit -m "refactor: share config override registration"
```

### 任务 2：将训练入口迁移到公共注册层

**文件：**
- 修改：[grid_cells/training/cli.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/grid_cells/training/cli.py)
- 修改：[train.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/train.py)
- 测试：[tests/test_train_step.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/tests/test_train_step.py)

- [ ] **步骤 1：补一个失败测试，锁定训练 parser 的统一覆盖行为**

```python
def test_parse_train_args_supports_task_model_training_and_visualization_overrides(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--task.env_size", "2.8",
            "--model.dropout_rate", "0.25",
            "--training.batch_size", "16",
            "--visualization.anim_step", "2",
        ],
    )

    args = parse_train_args()

    assert args.task__env_size == 2.8
    assert args.model__dropout_rate == 0.25
    assert args.training__batch_size == 16
    assert args.visualization__anim_step == 2
```

- [ ] **步骤 2：运行单测并确认失败**

运行：`pytest tests/test_train_step.py -k parse_train_args_supports_task_model_training_and_visualization_overrides -v`
预期：FAIL，现有 parser 不一定包含计划中的全部字段或测试未导入 `parse_train_args`。

- [ ] **步骤 3：用公共注册函数替换训练入口中的重复 `add_argument`**

```python
def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train grid cells RNN (Banino et al., Nature 2018)"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--data_path", default=None, help="Path to a pre-generated .npz trajectory file")
    parser.add_argument("--eval_data_path", default=None, help="Path to a fixed .npz evaluation dataset")
    register_config_overrides(
        parser,
        sections=("task", "model", "training", "visualization"),
    )
    return parser.parse_args()
```

- [ ] **步骤 4：更新测试导入和断言**

```python
from grid_cells.training.cli import parse_train_args
```

- [ ] **步骤 5：运行测试确认通过**

运行：`pytest tests/test_train_step.py -k "parse_train_args_supports_task_model_training_and_visualization_overrides or apply_overrides_updates_nested_sections" -v`
预期：PASS

- [ ] **步骤 6：Commit**

```bash
git add tests/test_train_step.py grid_cells/training/cli.py train.py
git commit -m "refactor: route train cli through shared config overrides"
```

### 任务 3：将数据生成入口切到配置默认值 + 统一覆盖

**文件：**
- 修改：[config.yaml](/home/xh/ai4neuron/gridCells/grid-cells-torch/config.yaml)
- 修改：[generate_data.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/generate_data.py)
- 测试：[tests/test_generate_data.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/tests/test_generate_data.py)

- [ ] **步骤 1：先补失败测试，锁定 `generate_data.py` 对 `data_generation` 和共享覆盖的支持**

```python
def test_main_uses_data_generation_defaults_and_cli_section_overrides(tmp_path, monkeypatch):
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = _make_cfg(str(train_path), str(eval_path))
    cfg.data_generation = SimpleNamespace(
        num_samples=11,
        eval_num_samples=7,
        num_workers=2,
        progress_every=1,
        vis_output=None,
        anim_output=None,
        progress_output=None,
        eval_progress_output=None,
    )

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: SimpleNamespace(
            config="config.yaml",
            output=None,
            eval_output=None,
            data_generation__num_samples=13,
            task__seq_len=5,
            visualize=False,
            animate=False,
            train_only=False,
        ),
    )

    generate_data.main()

    train_meta = np.load(train_path, allow_pickle=False)["meta"].item()
    eval_meta = np.load(eval_path, allow_pickle=False)["meta"].item()
    assert '"num_samples": 13' in train_meta
    assert '"num_samples": 7' in eval_meta
```

- [ ] **步骤 2：运行单测并确认失败**

运行：`pytest tests/test_generate_data.py -k data_generation_defaults_and_cli_section_overrides -v`
预期：FAIL，`generate_data.parse_args()` 尚未暴露 `data_generation__*` 与 `task__*` 覆盖结果，或 `main()` 尚未应用公共覆盖逻辑。

- [ ] **步骤 3：在配置文件中加入 `data_generation` 默认 section**

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

- [ ] **步骤 4：重写 `generate_data.parse_args()`，只保留入口参数、运行开关和公共覆盖注册**

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectory dataset for grid cell training"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--output", default=None, help="Override training.data_path")
    parser.add_argument("--eval_output", default=None, help="Override training.eval_data_path")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    register_config_overrides(
        parser,
        sections=("task", "training", "visualization", "data_generation"),
    )
    return parser.parse_args()
```

- [ ] **步骤 5：在 `main()` 中统一应用覆盖，并从配置读取 `data_generation` 默认值**

```python
cfg = load_config(args.config)
apply_namespace_overrides(cfg, args, skip_keys=("config", "output", "eval_output"))

data_gen_cfg = getattr(cfg, "data_generation", SimpleNamespace())
output_path = args.output or cfg.training.data_path
eval_output_path = None if args.train_only else (args.eval_output or cfg.training.eval_data_path)
num_samples = data_gen_cfg.num_samples or (cfg.training.steps_per_epoch * cfg.training.batch_size)
eval_num_samples = data_gen_cfg.eval_num_samples or cfg.training.eval_batch_size
num_workers = data_gen_cfg.num_workers
progress_every = data_gen_cfg.progress_every
```

- [ ] **步骤 6：补齐针对路径派生和运行开关的回归测试**

```python
def test_main_visualize_flag_uses_configured_vis_output(tmp_path, monkeypatch):
    ...
    assert captured["visualize_output"] == str(tmp_path / "preview.pdf")
```

- [ ] **步骤 7：运行测试确认通过**

运行：`pytest tests/test_generate_data.py -v`
预期：PASS

- [ ] **步骤 8：Commit**

```bash
git add config.yaml generate_data.py tests/test_generate_data.py
git commit -m "refactor: unify generate-data config overrides"
```

### 任务 4：全量验证与收尾

**文件：**
- 修改：[docs/superpowers/plans/2026-04-23-config-unification.md](/home/xh/ai4neuron/gridCells/grid-cells-torch/docs/superpowers/plans/2026-04-23-config-unification.md)
- 测试：[tests/test_generate_data.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/tests/test_generate_data.py)
- 测试：[tests/test_train_step.py](/home/xh/ai4neuron/gridCells/grid-cells-torch/tests/test_train_step.py)

- [ ] **步骤 1：运行聚焦测试**

运行：`pytest tests/test_train_step.py tests/test_generate_data.py -v`
预期：PASS

- [ ] **步骤 2：如测试失败，按失败用例回到对应任务做最小修正并重跑**

运行：`pytest tests/test_train_step.py tests/test_generate_data.py -v`
预期：PASS，无新增回归。

- [ ] **步骤 3：记录计划执行状态**

```markdown
- [x] 任务 1：建立公共 CLI 覆盖注册层
- [x] 任务 2：将训练入口迁移到公共注册层
- [x] 任务 3：将数据生成入口切到配置默认值 + 统一覆盖
- [x] 任务 4：全量验证与收尾
```

- [ ] **步骤 4：Commit**

```bash
git add docs/superpowers/plans/2026-04-23-config-unification.md
git commit -m "docs: record config unification execution"
```
