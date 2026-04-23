# `generate_data.py` 数据集目录化 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `generate_data.py` 的默认输出从固定 `.npz` 文件切换为带可读 ID 的数据集目录，统一收拢 train/eval/元数据/预览产物，并维护 `data/latest/*` 作为默认训练入口，同时保留显式 `--output` / `--eval_output` 的旧单文件模式兼容。

**架构：** 在 `generate_data.py` 中引入“目录模式 vs 单文件模式”的显式分支，并把目录命名、元数据写入、`latest` 入口同步等职责下沉到专门的帮助函数中，避免入口脚本继续堆叠路径逻辑。测试以 TDD 驱动，先锁定目录命名、元数据内容、路径派生、兼容行为，再最小实现；文档最后更新为新的默认工作流。

**技术栈：** Python 3、pytest、NumPy、现有 `grid_cells.data` 数据生成模块、CLI 配置覆盖系统

---

## 文件结构

### 修改文件

- `generate_data.py`
  - 新增目录模式参数、模式解析、目录命名、路径派生、元数据/README 生成、`data/latest` 同步逻辑。
- `config.yaml`
  - 将训练默认数据路径切到 `data/latest/train.npz` 与 `data/latest/eval.npz`。
- `tests/test_generate_data.py`
  - 为目录模式、元数据、`latest` 同步、旧模式兼容补充回归测试。
- `tests/test_train_step.py`
  - 更新默认路径假设，确保训练流程继续从稳定入口读取数据。
- `README.md`
  - 更新默认数据生成与训练工作流说明。
- `README.zh.md`
  - 同步中文文档。
- `run_scripts.sh`
  - 更新命令示例，使其反映目录模式与 `data/latest` 默认入口。

### 可能新增文件

- 无新的 Python 模块是必须的。优先把目录模式相关的帮助函数放在 `generate_data.py` 内部，只有当入口脚本已明显过大时再考虑抽出 `grid_cells/data/output_layout.py`。本计划默认不引入新模块，遵循 YAGNI。

## 任务 1：锁定目录模式 CLI 与路径派生行为

**文件：**
- 修改：`generate_data.py`
- 测试：`tests/test_generate_data.py`

- [ ] **步骤 1：编写失败的测试，覆盖目录模式默认入口与文件布局**

```python
def test_main_defaults_to_timestamped_dataset_directory(tmp_path, monkeypatch):
    cfg = _make_cfg("data/latest/train.npz", "data/latest/eval.npz")
    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(generate_data, "parse_args", lambda: _make_args())
    monkeypatch.chdir(tmp_path)

    generate_data.main()

    datasets_root = tmp_path / "data" / "datasets"
    created_dirs = [p for p in datasets_root.iterdir() if p.is_dir()]
    assert len(created_dirs) == 1

    dataset_dir = created_dirs[0]
    assert "_train6_" in dataset_dir.name
    assert "_eval4_" in dataset_dir.name
    assert "_seq6_" in dataset_dir.name
    assert "_seed5_" in dataset_dir.name
    assert "_env2p2" in dataset_dir.name

    assert (dataset_dir / "train.npz").exists()
    assert (dataset_dir / "eval.npz").exists()
    assert (dataset_dir / "meta.json").exists()
    assert (dataset_dir / "README.txt").exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_generate_data.py::test_main_defaults_to_timestamped_dataset_directory -v`
预期：FAIL，原因是当前 `main()` 仍直接写配置中的固定 `.npz` 路径，不会创建 `data/datasets/<dataset-id>/`。

- [ ] **步骤 3：在 `tests/test_generate_data.py` 中补充目录模式参数与显式目录覆盖测试**

```python
def test_main_uses_explicit_output_dir_when_provided(tmp_path, monkeypatch):
    cfg = _make_cfg("data/latest/train.npz", "data/latest/eval.npz")
    output_dir = tmp_path / "custom-dataset"
    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output_dir=str(output_dir), tag="baseline"),
    )

    generate_data.main()

    assert (output_dir / "train.npz").exists()
    assert (output_dir / "eval.npz").exists()
    assert (output_dir / "meta.json").exists()
```

- [ ] **步骤 4：运行新增测试验证失败**

运行：`python -m pytest tests/test_generate_data.py::test_main_uses_explicit_output_dir_when_provided -v`
预期：FAIL，报 `AttributeError` 或参数缺失，因为当前 `_make_args()` 与 CLI 还没有 `output_dir` / `tag`。

- [ ] **步骤 5：为测试辅助构造器补齐新参数默认值**

在 `tests/test_generate_data.py` 的 `_make_args()` 默认字典中加入：

```python
        output_dir=None,
        tag=None,
```

这样后续目录模式测试可以稳定传参与 monkeypatch，不会因假参数缺失干扰失败信号。

- [ ] **步骤 6：在 `generate_data.py` 的 `parse_args()` 中加入新 CLI 参数**

```python
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to store one generated dataset run and its artifacts.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional human-readable dataset label appended to directory names.",
    )
```

- [ ] **步骤 7：在 `generate_data.py` 中新增目录命名与路径派生帮助函数**

把以下帮助函数加在 `_default_artifact_path()` 之前，避免 `main()` 继续堆积字符串处理：

```python
from datetime import datetime
from pathlib import Path
import json
import re
import shutil


def _sanitize_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", tag).strip("_")
    return cleaned or None


def _format_env_size_for_name(env_size: float) -> str:
    text = str(env_size)
    return text.replace("-", "m").replace(".", "p")


def _build_dataset_id(
    created_at: datetime,
    train_num: int,
    eval_num: int,
    seq_len: int,
    seed: int,
    env_size: float,
    tag: str | None = None,
) -> str:
    parts = [
        created_at.strftime("%Y-%m-%d_%H%M%S"),
        f"train{train_num}",
        f"eval{eval_num}",
        f"seq{seq_len}",
        f"seed{seed}",
        f"env{_format_env_size_for_name(env_size)}",
    ]
    safe_tag = _sanitize_tag(tag)
    if safe_tag:
        parts.append(safe_tag)
    return "_".join(parts)


def _resolve_directory_mode_paths(output_dir: Path) -> dict:
    return {
        "train": output_dir / "train.npz",
        "eval": output_dir / "eval.npz",
        "meta": output_dir / "meta.json",
        "readme": output_dir / "README.txt",
        "train_vis": output_dir / "train_vis.pdf",
        "train_anim": output_dir / "train_traj.mp4",
        "train_progress": output_dir / "train_progress.png",
        "eval_progress": output_dir / "eval_progress.png",
    }
```

- [ ] **步骤 8：在 `generate_data.py` 中实现模式分流与默认目录模式**

将 `main()` 中早期输出路径解析替换为显式模式逻辑：

```python
    single_file_mode = args.output is not None or args.eval_output is not None
    created_at = datetime.now().astimezone()

    eval_num_samples = getattr(data_generation_cfg, "eval_num_samples", None)
    if eval_num_samples is None:
        eval_num_samples = cfg.training.eval_batch_size

    if single_file_mode:
        output_path = args.output or getattr(cfg.training, "data_path", None)
        eval_output_path = None
        if not args.train_only:
            eval_output_path = args.eval_output or getattr(cfg.training, "eval_data_path", None)
    else:
        datasets_root = Path("data") / "datasets"
        dataset_id = _build_dataset_id(
            created_at=created_at,
            train_num=num_samples,
            eval_num=0 if args.train_only else eval_num_samples,
            seq_len=seq_len,
            seed=seed,
            env_size=env_size,
            tag=args.tag,
        )
        output_dir = Path(args.output_dir) if args.output_dir else (datasets_root / dataset_id)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"Dataset output directory already exists and is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        derived_paths = _resolve_directory_mode_paths(output_dir)
        output_path = str(derived_paths["train"])
        eval_output_path = None if args.train_only else str(derived_paths["eval"])
```

- [ ] **步骤 9：运行目录模式相关测试验证通过**

运行：`python -m pytest tests/test_generate_data.py -k "timestamped_dataset_directory or explicit_output_dir" -v`
预期：PASS，两个测试都能看到目录模式生效且文件布局正确。

- [ ] **步骤 10：Commit**

```bash
git add generate_data.py tests/test_generate_data.py
git commit -m "feat: add dataset directory output mode"
```

## 任务 2：写入 `meta.json`、`README.txt` 并固定目录内产物命名

**文件：**
- 修改：`generate_data.py`
- 测试：`tests/test_generate_data.py`

- [ ] **步骤 1：编写失败的测试，锁定 `meta.json` 的关键字段**

```python
def test_directory_mode_writes_meta_json_with_effective_values(tmp_path, monkeypatch):
    cfg = _make_cfg("data/latest/train.npz", "data/latest/eval.npz")
    cfg.data_generation.num_samples = 11
    cfg.data_generation.eval_num_samples = 7
    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output_dir=str(tmp_path / "dataset"), tag="baseline"),
    )

    generate_data.main()

    meta = json.loads((tmp_path / "dataset" / "meta.json").read_text())
    assert meta["tag"] == "baseline"
    assert meta["task"]["seq_len"] == 6
    assert meta["task"]["env_size"] == 2.2
    assert meta["generation"]["num_samples"] == 11
    assert meta["generation"]["eval_num_samples"] == 7
    assert meta["paths"]["train"] == "train.npz"
    assert meta["paths"]["eval"] == "eval.npz"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_generate_data.py::test_directory_mode_writes_meta_json_with_effective_values -v`
预期：FAIL，因为目录模式还没有写 `meta.json`。

- [ ] **步骤 3：编写失败的测试，锁定 `README.txt` 与可选产物名称**

```python
def test_directory_mode_places_optional_artifacts_inside_dataset_dir(tmp_path, monkeypatch):
    cfg = _make_cfg("data/latest/train.npz", "data/latest/eval.npz")
    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(
            output_dir=str(tmp_path / "dataset"),
            visualize=True,
            animate=True,
            visualize_progress=True,
        ),
    )

    generate_data.main()

    dataset_dir = tmp_path / "dataset"
    assert (dataset_dir / "train_vis.pdf").exists()
    assert (dataset_dir / "train_traj.mp4").exists()
    assert (dataset_dir / "train_progress.png").exists()
    assert (dataset_dir / "eval_progress.png").exists()
    readme = (dataset_dir / "README.txt").read_text()
    assert "train.npz" in readme
    assert "eval.npz" in readme
```

- [ ] **步骤 4：运行测试验证失败**

运行：`python -m pytest tests/test_generate_data.py::test_directory_mode_places_optional_artifacts_inside_dataset_dir -v`
预期：FAIL，因为目录模式下还没有统一把可选产物落到同一目录，也没有生成 `README.txt`。

- [ ] **步骤 5：在 `generate_data.py` 中加入元数据构造函数**

在目录帮助函数区域追加：

```python
def _build_dataset_metadata(
    *,
    dataset_id: str,
    created_at: datetime,
    tag: str | None,
    train_only: bool,
    seq_len: int,
    env_size: float,
    velocity_noise,
    seed: int,
    num_samples: int,
    eval_num_samples: int,
    num_workers: int,
    progress_every: int,
    visualize_enabled: bool,
    animate_enabled: bool,
    spatial_bins: int,
    directional_bins: int,
    animation_settings: dict,
    paths: dict,
    config_source: str,
) -> dict:
    rel = lambda key: paths[key].name if paths.get(key) is not None else None
    return {
        "dataset_id": dataset_id,
        "created_at": created_at.isoformat(),
        "tag": _sanitize_tag(tag),
        "train_only": train_only,
        "paths": {
            "train": rel("train"),
            "eval": None if train_only else rel("eval"),
            "train_vis": rel("train_vis") if visualize_enabled else None,
            "train_anim": rel("train_anim") if animate_enabled else None,
            "train_progress": rel("train_progress"),
            "eval_progress": None if train_only else rel("eval_progress"),
        },
        "task": {
            "seq_len": seq_len,
            "env_size": env_size,
            "velocity_noise": list(velocity_noise),
            "neurons_seed": seed,
        },
        "generation": {
            "num_samples": num_samples,
            "eval_num_samples": 0 if train_only else eval_num_samples,
            "num_workers": num_workers,
            "progress_every": progress_every,
        },
        "visualization": {
            "enabled": visualize_enabled,
            "spatial_bins": spatial_bins,
            "directional_bins": directional_bins,
        },
        "animation": {
            "enabled": animate_enabled,
            **animation_settings,
        },
        "config_source": config_source,
    }
```

- [ ] **步骤 6：在 `generate_data.py` 中加入 `meta.json` 与 `README.txt` 写入函数**

```python
def _write_dataset_metadata(meta_path: Path, metadata: dict) -> None:
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def _write_dataset_readme(readme_path: Path, metadata: dict) -> None:
    lines = [
        f"dataset_id: {metadata['dataset_id']}",
        f"created_at: {metadata['created_at']}",
        f"train_only: {metadata['train_only']}",
        f"train_samples: {metadata['generation']['num_samples']}",
        f"eval_samples: {metadata['generation']['eval_num_samples']}",
        f"seq_len: {metadata['task']['seq_len']}",
        f"seed: {metadata['task']['neurons_seed']}",
        f"env_size: {metadata['task']['env_size']}",
        f"tag: {metadata['tag']}",
        "",
        "artifacts:",
    ]
    for key, rel_path in metadata["paths"].items():
        if rel_path is not None:
            lines.append(f"- {key}: {rel_path}")
    readme_path.write_text("\n".join(lines) + "\n")
```

- [ ] **步骤 7：在 `main()` 中切换目录模式下的可选产物派生路径并写元数据**

在目录模式分支中把 `vis_path`、`anim_path`、`progress_path`、`eval_progress_path` 改成目录派生：

```python
    if single_file_mode:
        vis_path = getattr(data_generation_cfg, "vis_output", None) if args.visualize else None
        if args.visualize and vis_path is None:
            vis_path = _default_artifact_path(output_path, "_vis.pdf")
        ...
    else:
        vis_path = str(derived_paths["train_vis"]) if args.visualize else None
        anim_path = str(derived_paths["train_anim"]) if args.animate else None
        progress_path = str(derived_paths["train_progress"]) if args.visualize_progress else None
        eval_progress_path = (
            None if args.train_only or not args.visualize_progress
            else str(derived_paths["eval_progress"])
        )
```

生成 train/eval 完成后写入：

```python
    if not single_file_mode:
        metadata = _build_dataset_metadata(
            dataset_id=dataset_id,
            created_at=created_at,
            tag=args.tag,
            train_only=args.train_only,
            seq_len=seq_len,
            env_size=env_size,
            velocity_noise=cfg.task.velocity_noise,
            seed=seed,
            num_samples=num_samples,
            eval_num_samples=eval_num_samples,
            num_workers=getattr(data_generation_cfg, "num_workers", 8),
            progress_every=getattr(data_generation_cfg, "progress_every", 4),
            visualize_enabled=args.visualize,
            animate_enabled=args.animate,
            spatial_bins=visualization_bins["spatial_bins"],
            directional_bins=visualization_bins["directional_bins"],
            animation_settings=animation_settings,
            paths=derived_paths,
            config_source=args.config,
        )
        _write_dataset_metadata(derived_paths["meta"], metadata)
        _write_dataset_readme(derived_paths["readme"], metadata)
```

- [ ] **步骤 8：运行元数据与目录内产物测试验证通过**

运行：`python -m pytest tests/test_generate_data.py -k "meta_json or optional_artifacts_inside_dataset_dir" -v`
预期：PASS，`meta.json` 字段完整且可选产物都在目录内。

- [ ] **步骤 9：Commit**

```bash
git add generate_data.py tests/test_generate_data.py
git commit -m "feat: write dataset metadata and grouped artifacts"
```

## 任务 3：同步 `data/latest/*` 并更新训练默认入口

**文件：**
- 修改：`generate_data.py`
- 修改：`config.yaml`
- 测试：`tests/test_generate_data.py`
- 测试：`tests/test_train_step.py`

- [ ] **步骤 1：编写失败的测试，锁定目录模式成功后会更新 `data/latest/*`**

在 `tests/test_generate_data.py` 新增：

```python
def test_directory_mode_updates_latest_symlinks(tmp_path, monkeypatch):
    cfg = _make_cfg("data/latest/train.npz", "data/latest/eval.npz")
    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(generate_data, "parse_args", lambda: _make_args())
    monkeypatch.chdir(tmp_path)

    generate_data.main()

    latest_dir = tmp_path / "data" / "latest"
    assert (latest_dir / "train.npz").exists()
    assert (latest_dir / "eval.npz").exists()
    assert (latest_dir / "meta.json").exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_generate_data.py::test_directory_mode_updates_latest_symlinks -v`
预期：FAIL，因为当前还没有 `latest` 同步逻辑。

- [ ] **步骤 3：在 `generate_data.py` 中加入 `latest` 入口同步函数**

```python
def _sync_latest_entry(output_dir: Path, paths: dict, train_only: bool) -> None:
    latest_dir = Path("data") / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "train.npz": paths["train"],
        "meta.json": paths["meta"],
    }
    if not train_only:
        mapping["eval.npz"] = paths["eval"]

    for name, source in mapping.items():
        target = latest_dir / name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.resolve())
```

如果项目在某些环境下不允许 symlink，可在实现时包一层 `try/except OSError`，回退到 `shutil.copy2(source, target)`：

```python
        try:
            target.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)
```

- [ ] **步骤 4：在目录模式的 `main()` 尾部调用 `latest` 同步**

```python
    if not single_file_mode:
        _sync_latest_entry(output_dir, derived_paths, train_only=args.train_only)
```

- [ ] **步骤 5：更新 `config.yaml` 默认数据路径**

把：

```yaml
  data_path: "data/train.npz"
  eval_data_path: "data/eval.npz"
```

改为：

```yaml
  data_path: "data/latest/train.npz"
  eval_data_path: "data/latest/eval.npz"
```

- [ ] **步骤 6：补训练默认入口回归测试**

在 `tests/test_train_step.py` 中把依赖旧默认路径的构造器改为：

```python
        data_path="data/latest/train.npz",
        eval_data_path="data/latest/eval.npz",
```

并新增一个小测试，锁定缺省配置使用 `data/latest/*`：

```python
def test_training_cli_defaults_point_to_latest_dataset():
    cfg = SimpleNamespace(
        training=SimpleNamespace(
            data_path="data/latest/train.npz",
            eval_data_path="data/latest/eval.npz",
        )
    )
    assert cfg.training.data_path.endswith("data/latest/train.npz")
    assert cfg.training.eval_data_path.endswith("data/latest/eval.npz")
```

- [ ] **步骤 7：运行相关测试验证通过**

运行：`python -m pytest tests/test_generate_data.py::test_directory_mode_updates_latest_symlinks tests/test_train_step.py -v`
预期：PASS，目录模式生成后 `data/latest/*` 可用，训练相关测试继续通过。

- [ ] **步骤 8：Commit**

```bash
git add generate_data.py config.yaml tests/test_generate_data.py tests/test_train_step.py
git commit -m "feat: sync latest dataset entry for training defaults"
```

## 任务 4：保留旧单文件模式兼容并补齐冲突/边界测试

**文件：**
- 修改：`generate_data.py`
- 测试：`tests/test_generate_data.py`

- [ ] **步骤 1：编写失败的测试，锁定显式 `--output` 仍走旧模式**

```python
def test_main_explicit_output_preserves_legacy_single_file_mode(tmp_path, monkeypatch):
    train_path = tmp_path / "legacy-train.npz"
    eval_path = tmp_path / "legacy-eval.npz"
    cfg = _make_cfg("data/latest/train.npz", "data/latest/eval.npz")
    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output=str(train_path), eval_output=str(eval_path)),
    )

    generate_data.main()

    assert train_path.exists()
    assert eval_path.exists()
    assert not (tmp_path / "data" / "datasets").exists()
```

- [ ] **步骤 2：运行测试验证失败或不稳定**

运行：`python -m pytest tests/test_generate_data.py::test_main_explicit_output_preserves_legacy_single_file_mode -v`
预期：如果前面目录模式改造不完整，可能 FAIL 或引入 `data/datasets` 副作用；本测试用来锁住旧模式边界。

- [ ] **步骤 3：编写失败的测试，覆盖已有非空目录时报错**

```python
def test_directory_mode_rejects_existing_non_empty_output_dir(tmp_path, monkeypatch):
    cfg = _make_cfg("data/latest/train.npz", "data/latest/eval.npz")
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("old")
    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output_dir=str(output_dir)),
    )

    with pytest.raises(ValueError, match="already exists and is not empty"):
        generate_data.main()
```

- [ ] **步骤 4：运行测试验证失败**

运行：`python -m pytest tests/test_generate_data.py::test_directory_mode_rejects_existing_non_empty_output_dir -v`
预期：FAIL，如果目录非空校验还没落地或异常消息不一致。

- [ ] **步骤 5：在 `generate_data.py` 中整理单文件模式与目录模式的冲突校验**

`main()` 中保留并梳理以下约束：

```python
    if args.train_only and args.eval_output is not None:
        raise ValueError("Cannot combine --train_only with --eval_output.")

    if single_file_mode and eval_output_path is not None:
        if os.path.abspath(eval_output_path) == os.path.abspath(output_path):
            raise ValueError(
                "Train and eval output paths must be different. Use --train_only to generate a single split."
            )
```

并确保只有单文件模式才读取 `cfg.training.data_path` / `eval_data_path` 作为直接输出路径。

- [ ] **步骤 6：运行旧模式与边界测试验证通过**

运行：`python -m pytest tests/test_generate_data.py -k "legacy_single_file_mode or existing_non_empty_output_dir or rejects_same_train_and_eval_output_paths or rejects_train_only_with_eval_output" -v`
预期：PASS，说明旧兼容行为和新边界都被固定。

- [ ] **步骤 7：Commit**

```bash
git add generate_data.py tests/test_generate_data.py
git commit -m "fix: preserve legacy file outputs alongside dataset directories"
```

## 任务 5：更新文档与命令示例

**文件：**
- 修改：`README.md`
- 修改：`README.zh.md`
- 修改：`run_scripts.sh`

- [ ] **步骤 1：编写失败的文档检查清单**

用人工检查替代自动测试，确认以下旧描述都需要更新：

```text
- README 中 “Train split: data/train.npz / Eval split: data/eval.npz”
- README.zh 中对应中文说明
- run_scripts.sh 中默认生成数据后的路径示例
```

预期：当前文档仍以固定文件为默认输出，需要改。

- [ ] **步骤 2：更新 `README.md` 的默认输出说明**

将默认数据说明改成类似：

```md
- Default generation now creates one dataset directory under `data/datasets/<dataset-id>/`.
- The latest generated dataset is exposed via `data/latest/train.npz` and `data/latest/eval.npz`.
- Pass `--output` / `--eval_output` if you explicitly want the legacy single-file mode.
```

并把示例命令改成：

```bash
python generate_data.py
python generate_data.py --tag baseline --visualize --animate
python train.py
```

- [ ] **步骤 3：同步更新 `README.zh.md`**

对应中文文案：

```md
- 默认生成结果会落到 `data/datasets/<dataset-id>/`。
- 最新一次数据集会通过 `data/latest/train.npz` 和 `data/latest/eval.npz` 暴露给训练流程。
- 若仍需旧式单文件输出，可显式传 `--output` / `--eval_output`。
```

- [ ] **步骤 4：更新 `run_scripts.sh` 命令清单**

将默认数据生成示例改成：

```bash
1. Generate the default train/eval datasets into a timestamped directory
python generate_data.py

4. Generate a smaller legacy single-file dataset
python generate_data.py --output data/train_small.npz --data_generation.num_samples 4000 --train_only
```

- [ ] **步骤 5：人工校对文档一致性**

检查：

- `README.md`
- `README.zh.md`
- `run_scripts.sh`

预期：三处都明确“目录模式是默认，单文件模式是兼容入口”，且训练默认读取 `data/latest/*`。

- [ ] **步骤 6：Commit**

```bash
git add README.md README.zh.md run_scripts.sh
git commit -m "docs: describe dataset directory workflow"
```

## 任务 6：全量验证与收尾

**文件：**
- 修改：无
- 测试：`tests/test_generate_data.py`
- 测试：`tests/test_train_step.py`

- [ ] **步骤 1：运行数据生成主测试集**

运行：`python -m pytest tests/test_generate_data.py -v`
预期：PASS，目录模式、旧模式、元数据、`latest` 同步全部通过。

- [ ] **步骤 2：运行训练相关回归测试**

运行：`python -m pytest tests/test_train_step.py -v`
预期：PASS，训练默认入口改为 `data/latest/*` 后无回归。

- [ ] **步骤 3：抽样执行一个真实 CLI 命令**

运行：`python generate_data.py --train_only --tag smoke`
预期：在 `data/datasets/` 下生成一个带时间戳和标签的目录，目录内至少包含：

```text
train.npz
meta.json
README.txt
```

同时 `data/latest/train.npz` 与 `data/latest/meta.json` 可用。

- [ ] **步骤 4：人工检查输出目录命名与元数据内容**

确认：

- 目录名包含 `train{n}`、`eval0` 或 `eval{n}`、`seq{seq_len}`、`seed{seed}`、`env{env_size}`。
- `meta.json` 中记录的是实际生效值，不是未解析默认值。
- `README.txt` 内容与 `meta.json` 一致。

- [ ] **步骤 5：最终 Commit**

```bash
git add generate_data.py config.yaml tests/test_generate_data.py tests/test_train_step.py README.md README.zh.md run_scripts.sh
git commit -m "feat: organize generated datasets into timestamped directories"
```

## 规格覆盖自检

- 规格中的“默认目录模式”由任务 1 覆盖。
- “目录名包含时间戳、样本数、`seq_len`、`seed`、`env_size`、可选 `tag`”由任务 1 覆盖。
- “目录内统一包含 train/eval/预览产物”由任务 2 覆盖。
- “新增 `meta.json` 与 `README.txt`”由任务 2 覆盖。
- “维护 `data/latest/*` 稳定入口并更新训练默认路径”由任务 3 覆盖。
- “保留显式 `--output` / `--eval_output` 的旧模式兼容”由任务 4 覆盖。
- “更新 README / README.zh / run_scripts”由任务 5 覆盖。
- “验证与真实 smoke run”由任务 6 覆盖。

## 占位符扫描自检

- 无 `TODO`、`TBD`、`后续实现`。
- 每个任务都包含具体文件、测试命令、预期结果和代码片段。
- 没有“类似上一步”式的引用；每个任务都可单独阅读执行。

## 类型与命名一致性自检

- 新参数统一命名为 `output_dir`、`tag`。
- 目录模式帮助函数统一使用 `dataset_id`、`derived_paths`、`created_at`。
- `meta.json` 路径字段统一使用 `train`、`eval`、`train_vis`、`train_anim`、`train_progress`、`eval_progress`。
- `data/latest/*` 统一作为默认训练入口命名。
