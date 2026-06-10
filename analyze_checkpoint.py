"""Run evaluation and statistical analysis from a saved training checkpoint."""

import argparse
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from grid_cells.analysis.scores import GridScorer
from grid_cells.cells.ensemble_utils import create_cell_ensembles_from_config
from grid_cells.cells.model import GridCellsRNN
from grid_cells.cells.persistence import apply_cell_centers
from grid_cells.common.config import (
    apply_namespace_overrides,
    dict_to_namespace,
    load_config,
    save_config,
)
from grid_cells.common.config_cli import CONFIG_OVERRIDE_SPECS
from train import _build_eval_loader, _evaluate, setup_logger


_TRAINING_OVERRIDE_FIELDS = (
    "eval_num_samples",
    "eval_loader_batch_size",
    "eval_plot_every",
    "eval_num_workers",
    "eval_chunk_size",
    "eval_units_per_page",
    "eval_pdf_dpi",
)

_SAFE_OVERRIDE_KEYS = {
    *(f"training__{field}" for field in _TRAINING_OVERRIDE_FIELDS),
    *(f"visualization__{field}" for field in CONFIG_OVERRIDE_SPECS["visualization"]),
    *(f"analysis__{field}" for field in CONFIG_OVERRIDE_SPECS["analysis"]),
}
_POSTHOC_DEFAULT_FIELDS = {
    "training": _TRAINING_OVERRIDE_FIELDS,
    "visualization": tuple(CONFIG_OVERRIDE_SPECS["visualization"]),
    "analysis": tuple(CONFIG_OVERRIDE_SPECS["analysis"]),
}
_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _add_config_override(parser, section: str, field: str) -> None:
    spec = CONFIG_OVERRIDE_SPECS[section][field]
    parser.add_argument(
        f"--{section}.{field}",
        dest=f"{section}__{field}",
        default=None,
        **spec,
    )


def parse_args(argv=None) -> argparse.Namespace:
    """Parse checkpoint-analysis CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run eval/analysis artifacts from a saved training checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint_epoch_XXXX.pt or checkpoint_final.pt",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for post-hoc artifacts. Defaults to "
        "<run_dir>/ckpt_analysis/<checkpoint_stem>/.",
    )
    parser.add_argument(
        "--eval_data_path",
        default=None,
        help="Optional eval split .npz path overriding the checkpoint config.",
    )

    for field in _TRAINING_OVERRIDE_FIELDS:
        _add_config_override(parser, "training", field)
    for field in CONFIG_OVERRIDE_SPECS["visualization"]:
        _add_config_override(parser, "visualization", field)
    for field in CONFIG_OVERRIDE_SPECS["analysis"]:
        _add_config_override(parser, "analysis", field)

    return parser.parse_args(argv)


def default_output_dir(checkpoint_path: str) -> str:
    """Return the default post-hoc output directory for a checkpoint path."""
    checkpoint = Path(checkpoint_path)
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    return str(run_dir / "ckpt_analysis" / checkpoint.stem)


def _require_checkpoint_payload(payload: dict, checkpoint_path: str) -> None:
    """Validate the minimum checkpoint contract needed for post-hoc analysis."""
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} did not load as a dict payload.")
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(
            f"Unsupported checkpoint schema_version={schema_version!r}; expected 1."
        )
    missing = [
        key
        for key in ("config", "model_state_dict", "cell_centers", "epoch")
        if key not in payload
    ]
    if missing:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing required fields: {', '.join(missing)}."
        )


def load_checkpoint_payload(checkpoint_path: str) -> dict:
    """Load and validate a checkpoint payload on CPU."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    _require_checkpoint_payload(payload, checkpoint_path)
    return payload


def _centers_to_numpy_list(values, expected_shapes, label: str):
    """Convert checkpoint center tensors to validated float32 NumPy arrays."""
    if not isinstance(values, (list, tuple)) or len(values) != len(expected_shapes):
        raise ValueError(
            f"Checkpoint cell_centers.{label} has {len(values) if isinstance(values, (list, tuple)) else 'invalid'} "
            f"entries, expected {len(expected_shapes)}."
        )

    centers = []
    for idx, (value, expected_shape) in enumerate(zip(values, expected_shapes)):
        array = torch.as_tensor(value, dtype=torch.float32).cpu().numpy()
        array = np.asarray(array, dtype=np.float32)
        if array.shape != expected_shape:
            raise ValueError(
                f"Checkpoint cell_centers.{label}[{idx}] has shape {array.shape}, "
                f"expected {expected_shape}."
            )
        centers.append(array)
    return centers


def build_checkpoint_ensembles(cfg, payload: dict):
    """Create ensembles from config and override their centers from the checkpoint."""
    pc_ens, hdc_ens = create_cell_ensembles_from_config(cfg)
    cell_centers = payload["cell_centers"]
    if not isinstance(cell_centers, dict) or "pc" not in cell_centers or "hdc" not in cell_centers:
        raise ValueError("Checkpoint cell_centers must contain 'pc' and 'hdc' entries.")

    pc_centers = _centers_to_numpy_list(
        cell_centers["pc"],
        [(int(n_cells), 2) for n_cells in cfg.task.n_pc],
        "pc",
    )
    hdc_centers = _centers_to_numpy_list(
        cell_centers["hdc"],
        [(int(n_cells),) for n_cells in cfg.task.n_hdc],
        "hdc",
    )
    apply_cell_centers(pc_ens, hdc_ens, pc_centers, hdc_centers)
    return pc_ens, hdc_ens


def build_grid_scorer(cfg) -> GridScorer:
    """Create the grid scorer used by training evaluation."""
    nbins = getattr(getattr(cfg, "visualization", None), "spatial_bins", 20)
    starts = [0.2] * 10
    ends = list(np.linspace(0.4, 1.0, num=10))
    masks_params = list(zip(starts, ends))
    return GridScorer(
        nbins=nbins,
        coords_range=[[-cfg.task.env_size / 2, cfg.task.env_size / 2]] * 2,
        mask_parameters=masks_params,
    )


def _ensure_posthoc_defaults(cfg, output_dir: str) -> None:
    """Set checkpoint-analysis defaults without changing model/task structure."""
    if not hasattr(cfg, "analysis") or cfg.analysis is None:
        cfg.analysis = SimpleNamespace()
    if not hasattr(cfg, "training") or cfg.training is None:
        cfg.training = SimpleNamespace()
    if not hasattr(cfg, "visualization") or cfg.visualization is None:
        cfg.visualization = SimpleNamespace()

    cfg.training.save_dir = output_dir
    cfg.training.eval_every = 1
    cfg.training.eval_plot_every = 1
    cfg.analysis.enabled = True


def _fill_missing_safe_posthoc_fields(cfg) -> None:
    """Backfill safe post-hoc fields missing from older checkpoint configs."""
    defaults = load_config(str(_DEFAULT_CONFIG_PATH))

    for section, fields in _POSTHOC_DEFAULT_FIELDS.items():
        target = getattr(cfg, section, None)
        if target is None:
            target = SimpleNamespace()
            setattr(cfg, section, target)

        source = getattr(defaults, section, None)
        if source is None:
            raise AttributeError(f"Default config is missing section: {section}")

        for field in fields:
            if hasattr(target, field):
                continue
            if not hasattr(source, field):
                raise AttributeError(f"Default config is missing field: {section}.{field}")
            setattr(target, field, copy.deepcopy(getattr(source, field)))


def build_analysis_config(payload: dict, args: argparse.Namespace) -> SimpleNamespace:
    """Build the effective config for one checkpoint-analysis run."""
    cfg = dict_to_namespace(payload["config"])
    output_dir = getattr(args, "output_dir", None) or default_output_dir(args.checkpoint)
    _ensure_posthoc_defaults(cfg, output_dir)
    _fill_missing_safe_posthoc_fields(cfg)
    eval_data_path = getattr(args, "eval_data_path", None)
    if eval_data_path is not None:
        cfg.training.eval_data_path = eval_data_path

    override_args = argparse.Namespace(
        **{
            key: value
            for key, value in vars(args).items()
            if key in _SAFE_OVERRIDE_KEYS and value is not None
        }
    )
    return apply_namespace_overrides(cfg, override_args, skip_keys=())


def analyze_checkpoint(checkpoint_path: str, output_dir: str = None, eval_data_path: str = None, args=None) -> str:
    """Run post-hoc evaluation/analysis for one checkpoint and return the output dir."""
    if args is None:
        args = argparse.Namespace(
            checkpoint=checkpoint_path,
            output_dir=output_dir,
            eval_data_path=eval_data_path,
        )

    payload = load_checkpoint_payload(checkpoint_path)
    cfg = build_analysis_config(payload, args)
    os.makedirs(cfg.training.save_dir, exist_ok=True)
    save_config(cfg, os.path.join(cfg.training.save_dir, "config.yaml"))
    logger = setup_logger(cfg.training.save_dir)
    logger.info("Running checkpoint analysis for %s", checkpoint_path)
    logger.info("Post-hoc output directory: %s", cfg.training.save_dir)

    pc_ens, hdc_ens = build_checkpoint_ensembles(cfg, payload)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model)).to(device)
    model.load_state_dict(payload["model_state_dict"])

    eval_loader = _build_eval_loader(cfg, logger, eval_data_path=getattr(args, "eval_data_path", None))
    _evaluate(
        model,
        pc_ens,
        hdc_ens,
        build_grid_scorer(cfg),
        eval_loader,
        cfg,
        device,
        epoch=int(payload["epoch"]),
        writer=None,
    )
    logger.info("Checkpoint analysis complete.")
    return cfg.training.save_dir


def main(argv=None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    analyze_checkpoint(args.checkpoint, args.output_dir, args.eval_data_path, args=args)


if __name__ == "__main__":
    main()
