"""Generate trajectory datasets and optional preview artifacts."""

import argparse
from datetime import datetime
import json
import os
import re
import shutil
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

from grid_cells.viz.animation import generate_trajectory_animation
from grid_cells.common.config import apply_namespace_overrides, load_config
from grid_cells.common.config_cli import register_config_overrides
from grid_cells.data.dataset import TrajectoryDataset
from grid_cells.data.artifacts import (
    generate_dataset_file as _generate_dataset_file_impl,
    visualize_dataset,
    visualize_dataset_animation,
)
from grid_cells.cells.encoding_utils import prepare_dataset_animation_inputs
from grid_cells.cells.ensemble_utils import (
    get_head_direction_ensembles,
    get_place_cell_ensembles,
)

DEFAULT_SPATIAL_BINS = 32
DEFAULT_DIRECTIONAL_BINS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectory dataset for grid cell training"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz file path. Defaults to training.data_path from config.",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--eval_num_samples", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--env_size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_seed", type=int, default=None)
    parser.add_argument("--vis_output", default=None)
    parser.add_argument("--anim_output", default=None)
    parser.add_argument("--progress_output", default=None)
    parser.add_argument("--eval_progress_output", default=None)
    parser.add_argument("--anim_num_traj", type=int, default=None)
    parser.add_argument("--anim_fps", type=int, default=None)
    parser.add_argument("--anim_step", type=int, default=None)
    parser.add_argument("--anim_workers", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=None)
    parser.add_argument("--spatial_bins", type=int, default=None)
    parser.add_argument("--directional_bins", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for dataset mode. Defaults to data/datasets/<dataset-id>/.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag appended to the derived dataset directory name.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save a visualisation PDF alongside the .npz file",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Generate and save an MP4 animation of sample trajectories",
    )
    parser.add_argument(
        "--eval_output",
        default=None,
        help="Eval split .npz file path. Defaults to training.eval_data_path from config.",
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Generate only the main output file and skip eval split generation",
    )
    parser.add_argument(
        "--visualize_progress",
        action="store_true",
        help="Periodically save a progress preview PNG while trajectories are generated",
    )
    register_config_overrides(
        parser,
        sections=("task", "training", "visualization", "data_generation"),
    )
    return parser.parse_args()


def _resolve_visualization_bins(cfg, args) -> dict:
    """Resolve visualisation bin counts from CLI, config, and safe defaults."""
    vis_cfg = getattr(cfg, "visualization", None)
    bins = {
        "spatial_bins": getattr(args, "spatial_bins", None),
        "directional_bins": getattr(args, "directional_bins", None),
    }
    defaults = {
        "spatial_bins": DEFAULT_SPATIAL_BINS,
        "directional_bins": DEFAULT_DIRECTIONAL_BINS,
    }

    for name, value in bins.items():
        if value is None:
            value = getattr(vis_cfg, name, defaults[name]) if vis_cfg is not None else defaults[name]
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}.")
        bins[name] = value

    return bins


def _resolve_animation_settings(cfg, args) -> dict:
    """Resolve shared animation settings from CLI, config, and legacy keys."""
    vis_cfg = getattr(cfg, "visualization", None)
    training_cfg = getattr(cfg, "training", None)
    defaults = {
        "anim_num_traj": 4,
        "anim_fps": 20,
        "anim_step": 4,
        "anim_workers": 4,
    }
    legacy_names = {
        "anim_num_traj": "eval_anim_num_traj",
        "anim_fps": "eval_anim_fps",
        "anim_step": "eval_anim_step",
        "anim_workers": "eval_anim_workers",
    }
    resolved = {}

    for name, default in defaults.items():
        value = getattr(args, name, None)
        if value is None and vis_cfg is not None and hasattr(vis_cfg, name):
            value = getattr(vis_cfg, name)
        if value is None and training_cfg is not None and hasattr(
            training_cfg,
            legacy_names[name],
        ):
            value = getattr(training_cfg, legacy_names[name])
        if value is None:
            value = default
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}.")
        resolved[name] = value

    return resolved


def visualize(
    dataset: TrajectoryDataset,
    save_path: str,
    spatial_bins: int = DEFAULT_SPATIAL_BINS,
    directional_bins: int = DEFAULT_DIRECTIONAL_BINS,
) -> None:
    """Produce a multi-panel summary figure and save as PDF."""
    return visualize_dataset(
        dataset,
        save_path,
        spatial_bins=spatial_bins,
        directional_bins=directional_bins,
    )


def visualize_animation(
    dataset: TrajectoryDataset,
    save_path: str,
    pc_ensembles,
    hdc_ensembles,
    fps: int = 20,
    max_trajectories: int = 4,
    step: int = 4,
    num_workers: int = 8,
) -> None:
    """Save an eval-style 3-panel animation for generated trajectories."""
    return visualize_dataset_animation(
        dataset,
        save_path,
        pc_ensembles,
        hdc_ensembles,
        prepare_dataset_animation_inputs=prepare_dataset_animation_inputs,
        generate_trajectory_animation=generate_trajectory_animation,
        fps=fps,
        max_trajectories=max_trajectories,
        step=step,
        num_workers=num_workers,
    )


def generate_dataset_file(
    output_path: str,
    num_samples: int,
    seq_len: int,
    env_size: float,
    velocity_noise,
    seed: int,
    pc_ensembles=None,
    hdc_ensembles=None,
    visualize_output: str = None,
    animation_output: str = None,
    animation_num_trajectories: int = 4,
    animation_fps: int = 20,
    animation_step: int = 4,
    anim_workers: int = 8,
    num_workers: int = 1,
    progress_output: str = None,
    progress_every: int = 4,
    spatial_bins: int = DEFAULT_SPATIAL_BINS,
    directional_bins: int = DEFAULT_DIRECTIONAL_BINS,
) -> TrajectoryDataset:
    """Generate one dataset file and optionally emit its visualization artifacts."""
    if spatial_bins is None:
        spatial_bins = DEFAULT_SPATIAL_BINS
    if directional_bins is None:
        directional_bins = DEFAULT_DIRECTIONAL_BINS
    return _generate_dataset_file_impl(
        output_path=output_path,
        num_samples=num_samples,
        seq_len=seq_len,
        env_size=env_size,
        velocity_noise=velocity_noise,
        seed=seed,
        visualize_fn=visualize,
        visualize_animation_fn=visualize_animation,
        pc_ensembles=pc_ensembles,
        hdc_ensembles=hdc_ensembles,
        visualize_output=visualize_output,
        animation_output=animation_output,
        animation_num_trajectories=animation_num_trajectories,
        animation_fps=animation_fps,
        animation_step=animation_step,
        anim_workers=anim_workers,
        num_workers=num_workers,
        progress_output=progress_output,
        progress_every=progress_every,
        spatial_bins=spatial_bins,
        directional_bins=directional_bins,
    )


def _default_artifact_path(base_path: str, suffix: str) -> str:
    base = base_path[:-4] if base_path.endswith(".npz") else base_path
    return f"{base}{suffix}"


def _clean_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", tag.strip().lower()).strip("-")
    return cleaned or None


def _format_env_size(env_size: float) -> str:
    text = f"{float(env_size):g}"
    return text.replace("-", "m").replace(".", "p")


def _build_dataset_id(
    train_samples: int,
    eval_samples: int | None,
    seq_len: int,
    seed: int,
    env_size: float,
    timestamp: str | None = None,
    tag: str | None = None,
) -> str:
    parts = [
        f"train{int(train_samples)}",
        f"seq{int(seq_len)}",
        f"seed{int(seed)}",
        f"env{_format_env_size(env_size)}",
    ]
    if eval_samples is not None:
        parts.insert(1, f"eval{int(eval_samples)}")
    if timestamp is not None:
        parts.append(timestamp)
    cleaned_tag = _clean_tag(tag)
    if cleaned_tag is not None:
        parts.append(cleaned_tag)
    return "-".join(parts)


def _current_dataset_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _dataset_paths_from_dir(output_dir: str, train_only: bool = False) -> dict:
    paths = {
        "output_dir": output_dir,
        "train_output": os.path.join(output_dir, "train.npz"),
        "meta_output": os.path.join(output_dir, "meta.json"),
        "readme_output": os.path.join(output_dir, "README.txt"),
        "train_vis_output": os.path.join(output_dir, "train_vis.pdf"),
        "train_anim_output": os.path.join(output_dir, "train_traj.mp4"),
        "train_progress_output": os.path.join(output_dir, "train_progress.png"),
    }
    paths["eval_output"] = None if train_only else os.path.join(output_dir, "eval.npz")
    paths["eval_progress_output"] = (
        None if train_only else os.path.join(output_dir, "eval_progress.png")
    )
    return paths


def _ensure_empty_output_dir(output_dir: str) -> None:
    if os.path.isdir(output_dir):
        if os.listdir(output_dir):
            raise ValueError(f"Output directory already exists and is not empty: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)


def _relative_dataset_paths(paths: dict) -> dict:
    relative_paths = {}
    for key, value in paths.items():
        if key == "output_dir":
            continue
        relative_paths[key.removesuffix("_output")] = None if value is None else os.path.basename(value)
    return relative_paths


def _build_dataset_metadata(
    dataset_id: str,
    tag: str | None,
    seq_len: int,
    env_size: float,
    seed: int,
    eval_seed: int | None,
    velocity_noise,
    num_samples: int,
    eval_num_samples: int | None,
    paths: dict,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "tag": _clean_tag(tag),
        "task": {
            "seq_len": int(seq_len),
            "env_size": float(env_size),
            "neurons_seed": int(seed),
            "velocity_noise": list(velocity_noise),
        },
        "generation": {
            "train_seed": int(seed),
            "eval_seed": None if eval_seed is None else int(eval_seed),
            "num_samples": int(num_samples),
            "eval_num_samples": None if eval_num_samples is None else int(eval_num_samples),
        },
        "paths": _relative_dataset_paths(paths),
    }


def _write_dataset_metadata(meta_output: str, metadata: dict) -> None:
    with open(meta_output, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)


def _write_dataset_readme(readme_output: str, metadata: dict) -> None:
    lines = [
        f"Dataset artifacts for {metadata['dataset_id']}",
        "",
        "Core artifacts:",
    ]
    for key in ("train", "eval", "meta", "readme"):
        value = metadata["paths"].get(key)
        if value is not None:
            lines.append(f"- {value}")

    optional_names = [
        metadata["paths"].get("train_vis"),
        metadata["paths"].get("train_anim"),
        metadata["paths"].get("train_progress"),
        metadata["paths"].get("eval_progress"),
    ]
    present_optional = [name for name in optional_names if name is not None]
    if present_optional:
        lines.extend(["", "Optional artifacts:"])
        lines.extend(f"- {name}" for name in present_optional)

    with open(readme_output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _remove_path(path: str) -> None:
    if os.path.lexists(path):
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)


def _sync_latest_entry(source_path: str | None, latest_path: str) -> None:
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    if source_path is not None and os.path.abspath(source_path) == os.path.abspath(latest_path):
        return
    _remove_path(latest_path)
    if source_path is None:
        return

    try:
        os.symlink(os.path.abspath(source_path), latest_path)
    except OSError:
        shutil.copy2(source_path, latest_path)


def _sync_latest_dataset_entries(train_path: str, eval_path: str | None, meta_path: str) -> None:
    latest_dir = os.path.join("data", "latest")
    _sync_latest_entry(train_path, os.path.join(latest_dir, "train.npz"))
    _sync_latest_entry(eval_path, os.path.join(latest_dir, "eval.npz"))
    _sync_latest_entry(meta_path, os.path.join(latest_dir, "meta.json"))


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    apply_namespace_overrides(cfg, args)
    training_path_override = getattr(args, "training__data_path", None)
    training_eval_path_override = getattr(args, "training__eval_data_path", None)
    explicit_train_path = args.output if args.output is not None else training_path_override
    explicit_eval_path = (
        args.eval_output if args.eval_output is not None else training_eval_path_override
    )
    explicit_file_mode = any(
        value is not None
        for value in (
            explicit_train_path,
            explicit_eval_path,
        )
    )
    if args.train_only and (
        args.eval_output is not None or training_eval_path_override is not None
    ):
        raise ValueError("Cannot combine --train_only with --eval_output.")
    if explicit_eval_path is not None and explicit_train_path is None:
        raise ValueError("Explicit eval output requires an explicit train output.")

    output_path = None
    eval_output_path = None

    if eval_output_path is not None and os.path.abspath(eval_output_path) == os.path.abspath(
        output_path
    ):
        raise ValueError(
            "Train and eval output paths must be different. Use --train_only to "
            "generate a single split."
        )

    data_generation_cfg = getattr(cfg, "data_generation", SimpleNamespace())
    num_samples = (
        args.num_samples
        if args.num_samples is not None
        else getattr(data_generation_cfg, "num_samples", None)
    )
    if num_samples is None:
        num_samples = cfg.training.steps_per_epoch * cfg.training.batch_size
    eval_num_samples = args.eval_num_samples
    if eval_num_samples is None:
        eval_num_samples = getattr(data_generation_cfg, "eval_num_samples", None)
    if eval_num_samples is None and not args.train_only:
        eval_num_samples = cfg.training.eval_batch_size
    seq_len = args.seq_len if args.seq_len is not None else cfg.task.seq_len
    env_size = args.env_size if args.env_size is not None else cfg.task.env_size
    seed = args.seed if args.seed is not None else cfg.task.neurons_seed

    if explicit_file_mode:
        output_path = explicit_train_path
        eval_output_path = None if args.train_only else explicit_eval_path
        if output_path is None and eval_output_path is None:
            raise ValueError(
                "No output path provided. Set --output or configure training.data_path."
            )
    else:
        timestamp = _current_dataset_timestamp()
        dataset_id = _build_dataset_id(
            train_samples=num_samples,
            eval_samples=None if args.train_only else eval_num_samples,
            seq_len=seq_len,
            seed=seed,
            env_size=env_size,
            timestamp=timestamp,
            tag=args.tag,
        )
        output_dir = args.output_dir or os.path.join("data", "datasets", dataset_id)
        _ensure_empty_output_dir(output_dir)
        dataset_paths = _dataset_paths_from_dir(output_dir, train_only=args.train_only)
        output_path = dataset_paths["train_output"]
        eval_output_path = dataset_paths["eval_output"]

    if (
        output_path is not None
        and eval_output_path is not None
        and os.path.abspath(eval_output_path) == os.path.abspath(output_path)
    ):
        raise ValueError(
            "Train and eval output paths must be different. Use --train_only to "
            "generate a single split."
        )

    animation_settings = _resolve_animation_settings(
        cfg,
        args,
    )
    pc_ensembles = get_place_cell_ensembles(cfg)
    hdc_ensembles = get_head_direction_ensembles(cfg)

    vis_path = None if not args.visualize or output_path is None else getattr(data_generation_cfg, "vis_output", None)
    if args.visualize and output_path is not None and explicit_file_mode and args.vis_output is not None:
        vis_path = args.vis_output
    elif args.visualize and output_path is not None and not explicit_file_mode:
        vis_path = dataset_paths["train_vis_output"]
    elif args.visualize and output_path is not None and vis_path is None:
        vis_path = _default_artifact_path(output_path, "_vis.pdf")

    anim_path = None if not args.animate or output_path is None else getattr(data_generation_cfg, "anim_output", None)
    if args.animate and output_path is not None and explicit_file_mode and args.anim_output is not None:
        anim_path = args.anim_output
    elif args.animate and output_path is not None and not explicit_file_mode:
        anim_path = dataset_paths["train_anim_output"]
    elif args.animate and output_path is not None and anim_path is None:
        anim_path = _default_artifact_path(output_path, "_traj.mp4")

    progress_path = args.progress_output if explicit_file_mode else None
    if progress_path is None and explicit_file_mode:
        progress_path = getattr(data_generation_cfg, "progress_output", None)
    if not args.visualize_progress or output_path is None:
        progress_path = None
    elif not explicit_file_mode:
        progress_path = dataset_paths["train_progress_output"]
    elif progress_path is None:
        progress_path = _default_artifact_path(output_path, "_progress.png")

    metadata_paths = None
    if not explicit_file_mode:
        metadata_paths = {
            "output_dir": dataset_paths["output_dir"],
            "train_output": output_path,
            "eval_output": eval_output_path,
            "meta_output": dataset_paths["meta_output"],
            "readme_output": dataset_paths["readme_output"],
            "train_vis_output": vis_path,
            "train_anim_output": anim_path,
            "train_progress_output": progress_path,
            "eval_progress_output": None,
        }

    visualization_bins = None
    if args.visualize or args.visualize_progress:
        visualization_bins = _resolve_visualization_bins(
            cfg,
            args,
        )

    if output_path is not None:
        generate_dataset_file(
            output_path=output_path,
            num_samples=num_samples,
            seq_len=seq_len,
            env_size=env_size,
            velocity_noise=cfg.task.velocity_noise,
            seed=seed,
            pc_ensembles=pc_ensembles,
            hdc_ensembles=hdc_ensembles,
            visualize_output=vis_path,
            animation_output=anim_path,
            animation_num_trajectories=animation_settings["anim_num_traj"],
            animation_fps=animation_settings["anim_fps"],
            animation_step=animation_settings["anim_step"],
            anim_workers=animation_settings["anim_workers"],
            num_workers=(
                args.num_workers
                if args.num_workers is not None
                else getattr(data_generation_cfg, "num_workers", 8)
            ),
            progress_output=progress_path,
            progress_every=(
                args.progress_every
                if args.progress_every is not None
                else getattr(data_generation_cfg, "progress_every", 4)
            ),
            spatial_bins=None if visualization_bins is None else visualization_bins["spatial_bins"],
            directional_bins=None
            if visualization_bins is None
            else visualization_bins["directional_bins"],
        )

    if eval_output_path is None:
        if not explicit_file_mode:
            metadata = _build_dataset_metadata(
                dataset_id=dataset_id,
                tag=args.tag,
                seq_len=seq_len,
                env_size=env_size,
                seed=seed,
                eval_seed=None,
                velocity_noise=cfg.task.velocity_noise,
                num_samples=num_samples,
                eval_num_samples=None,
                paths=metadata_paths,
            )
            _write_dataset_metadata(dataset_paths["meta_output"], metadata)
            _write_dataset_readme(dataset_paths["readme_output"], metadata)
            _sync_latest_dataset_entries(
                train_path=output_path,
                eval_path=None,
                meta_path=dataset_paths["meta_output"],
            )
        return

    eval_seed = args.eval_seed if args.eval_seed is not None else seed + 1
    eval_progress_path = args.eval_progress_output if explicit_file_mode else None
    if eval_progress_path is None and explicit_file_mode:
        eval_progress_path = getattr(data_generation_cfg, "eval_progress_output", None)
    if not args.visualize_progress:
        eval_progress_path = None
    elif not explicit_file_mode:
        eval_progress_path = dataset_paths["eval_progress_output"]
    elif eval_progress_path is None:
        eval_progress_path = _default_artifact_path(eval_output_path, "_progress.png")

    if metadata_paths is not None:
        metadata_paths["eval_progress_output"] = eval_progress_path

    generate_dataset_file(
        output_path=eval_output_path,
        num_samples=eval_num_samples,
        seq_len=seq_len,
        env_size=env_size,
        velocity_noise=cfg.task.velocity_noise,
        seed=eval_seed,
        pc_ensembles=pc_ensembles,
        hdc_ensembles=hdc_ensembles,
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else getattr(data_generation_cfg, "num_workers", 8)
        ),
        progress_output=eval_progress_path,
        progress_every=(
            args.progress_every
            if args.progress_every is not None
            else getattr(data_generation_cfg, "progress_every", 4)
        ),
        spatial_bins=None if visualization_bins is None else visualization_bins["spatial_bins"],
        directional_bins=None
        if visualization_bins is None
        else visualization_bins["directional_bins"],
    )

    if not explicit_file_mode:
        metadata = _build_dataset_metadata(
            dataset_id=dataset_id,
            tag=args.tag,
            seq_len=seq_len,
            env_size=env_size,
            seed=seed,
            eval_seed=eval_seed,
            velocity_noise=cfg.task.velocity_noise,
            num_samples=num_samples,
            eval_num_samples=eval_num_samples,
            paths=metadata_paths,
        )
        _write_dataset_metadata(dataset_paths["meta_output"], metadata)
        _write_dataset_readme(dataset_paths["readme_output"], metadata)
        _sync_latest_dataset_entries(
            train_path=output_path,
            eval_path=eval_output_path,
            meta_path=dataset_paths["meta_output"],
        )


if __name__ == "__main__":
    main()
