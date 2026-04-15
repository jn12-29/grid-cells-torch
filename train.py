"""Train the grid-cells PyTorch model and export run artifacts.

This is the main experiment entrypoint. It loads config, builds datasets and
ensembles, runs training and evaluation, and writes logs, TensorBoard data,
PDF summaries, and optional MP4 outputs under `results/`.

Usage:
    python train.py
    python train.py --config config.yaml
    python train.py --training.epochs 20 --training.batch_size 64
    CUDA_VISIBLE_DEVICES=0 python train.py --run-name debug
"""

import argparse
import logging
import math
import os
import time
from datetime import datetime
from types import SimpleNamespace

import yaml

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

from dataset import get_dataloader
from evaluation import EvaluationHooks, Evaluator
from model import GridCellsRNN
from scores import GridScorer
from training_session import TrainingSession, TrainingSessionHooks
from utils import (
    compute_position_mse,
    decode_position_from_pc_logits,
    get_place_cell_ensembles,
    get_head_direction_ensembles,
    encode_initial_conditions,
    encode_targets,
    generate_eval_animation,
    get_scores_and_plot_from_ratemaps,
    plot_hdc_tuning_curves,
    score_ratemaps,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def resolve_save_dir(
    base_dir: str,
    timestamp_save_dir: bool = True,
    run_name: str = None,
    now: datetime = None,
) -> str:
    """Resolve the final run directory, optionally nesting under a timestamp."""
    if not timestamp_save_dir:
        return base_dir

    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    run_suffix = f"{stamp}_{run_name}" if run_name else stamp
    return os.path.join(base_dir, run_suffix)


def setup_logger(log_dir: str) -> logging.Logger:
    """Configure a logger that writes to stdout and to a file in log_dir."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    logger = logging.getLogger("grid_cells")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logging to %s", log_path)
    return logger


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_namespace(v) if isinstance(v, dict) else v)
    return ns


def _namespace_to_dict(obj):
    """Recursively convert a SimpleNamespace config back to plain dicts."""
    if isinstance(obj, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _namespace_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_namespace_to_dict(v) for v in obj]
    return obj


def _str2bool(value):
    """Parse common textual boolean values for CLI overrides."""
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_step_log_interval(num_steps: int, max_logs_per_epoch: int = 10) -> int:
    """Return a sampling interval that keeps step logs under the cap."""
    if num_steps <= 0:
        return 1
    return max(1, math.ceil(num_steps / max_logs_per_epoch))


def create_summary_writer(cfg, logger: logging.Logger):
    """Create a TensorBoard writer when requested and available."""
    if not getattr(cfg.training, "use_tensorboard", True):
        return None

    if SummaryWriter is None:
        logger.warning(
            "TensorBoard disabled because 'tensorboard' is not installed. "
            "Install it with: pip install tensorboard"
        )
        return None

    log_dir = os.path.join(cfg.training.save_dir, "tensorboard")
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text(
        "run/config",
        "```yaml\n"
        + yaml.safe_dump(_namespace_to_dict(cfg), sort_keys=False)
        + "\n```",
        0,
    )
    logger.info("TensorBoard logging to %s", log_dir)
    return writer


def _make_param_groups(model, cfg):
    """Split model parameters into two groups.

    Decoder heads (pc_heads / hdc_heads) receive the configured weight_decay;
    all other parameters are trained without weight decay.
    Returns (param_groups, decoder_params) where decoder_params is the flat
    list of decoder parameter tensors (used for selective grad clipping).
    """
    decoder_params = (
        list(model.pc_heads.parameters())
        + list(model.hdc_heads.parameters())
        + list(model.bottleneck.parameters())
    )
    decoder_ids = {id(p) for p in decoder_params}
    other_params = [p for p in model.parameters() if id(p) not in decoder_ids]

    param_groups = [
        {"params": other_params, "weight_decay": 0.0},
        {"params": decoder_params, "weight_decay": cfg.training.weight_decay},
    ]
    return param_groups, decoder_params


def build_optimizer(model, cfg):
    """Build the configured optimizer for training.

    Weight decay is applied exclusively to the decoder heads (pc_heads /
    hdc_heads); all other parameters are trained without weight decay.
    Returns (optimizer, decoder_params) so the caller can reuse
    decoder_params for selective gradient clipping.
    """
    param_groups, decoder_params = _make_param_groups(model, cfg)
    optimizer_name = getattr(cfg.training, "optimizer", "rmsprop").lower()

    if optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            # weight_decay is set per-group; pass 0 as the global default
            weight_decay=0.0,
        )
        return optimizer, decoder_params

    if optimizer_name == "adamw":
        beta1, beta2 = getattr(cfg.training, "adamw_betas", [0.9, 0.999])
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg.training.lr,
            betas=(beta1, beta2),
            eps=getattr(cfg.training, "adamw_eps", 1e-8),
            # weight_decay is set per-group; pass 0 as the global default
            weight_decay=0.0,
        )
        return optimizer, decoder_params

    raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")


def load_config(path: str = "config.yaml") -> SimpleNamespace:
    """Load YAML config and return as a nested SimpleNamespace.

    Supports attribute access like cfg.task.env_size.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _dict_to_namespace(raw)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments that can override yaml defaults.

    Dot-notation keys map to nested config sections, e.g.
    --task.env_size 2.2  overrides cfg.task.env_size.
    """
    parser = argparse.ArgumentParser(
        description="Train grid cells RNN (Banino et al., Nature 2018)"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Path to a pre-generated .npz trajectory file "
        "(created by generate_data.py). "
        "Defaults to training.data_path from config; when that file is "
        "missing, train.py falls back to generating trajectories on-the-fly "
        "every epoch.",
    )
    parser.add_argument(
        "--eval_data_path",
        default=None,
        help="Path to a fixed .npz evaluation dataset. "
        "Defaults to training.eval_data_path from config; when that file is "
        "missing, train.py falls back to generating one fixed eval set in memory.",
    )

    # task overrides
    parser.add_argument("--task.env_size", type=float, dest="task__env_size")
    parser.add_argument("--task.n_pc", type=int, nargs="+", dest="task__n_pc")
    parser.add_argument("--task.pc_scale", type=float, nargs="+", dest="task__pc_scale")
    parser.add_argument("--task.n_hdc", type=int, nargs="+", dest="task__n_hdc")
    parser.add_argument(
        "--task.hdc_concentration",
        type=float,
        nargs="+",
        dest="task__hdc_concentration",
    )

    # model overrides
    parser.add_argument("--model.nh_lstm", type=int, dest="model__nh_lstm")
    parser.add_argument("--model.nh_bottleneck", type=int, dest="model__nh_bottleneck")
    parser.add_argument("--model.dropout_rate", type=float, dest="model__dropout_rate")

    # training overrides
    parser.add_argument("--training.epochs", type=int, dest="training__epochs")
    parser.add_argument(
        "--training.steps_per_epoch", type=int, dest="training__steps_per_epoch"
    )
    parser.add_argument("--training.batch_size", type=int, dest="training__batch_size")
    parser.add_argument("--training.lr", type=float, dest="training__lr")
    parser.add_argument("--training.optimizer", type=str, dest="training__optimizer")
    parser.add_argument("--training.eval_every", type=int, dest="training__eval_every")
    parser.add_argument("--training.save_dir", type=str, dest="training__save_dir")
    parser.add_argument("--training.run_name", type=str, dest="training__run_name")
    parser.add_argument(
        "--training.adamw_betas",
        type=float,
        nargs=2,
        dest="training__adamw_betas",
    )
    parser.add_argument("--training.adamw_eps", type=float, dest="training__adamw_eps")
    parser.add_argument(
        "--training.tensorboard_log_every",
        type=int,
        dest="training__tensorboard_log_every",
    )
    parser.add_argument(
        "--training.timestamp_save_dir",
        type=_str2bool,
        dest="training__timestamp_save_dir",
    )
    parser.add_argument(
        "--training.use_tqdm", type=_str2bool, dest="training__use_tqdm"
    )
    parser.add_argument(
        "--training.use_tensorboard", type=_str2bool, dest="training__use_tensorboard"
    )
    parser.add_argument(
        "--training.eval_num_workers", type=int, dest="training__eval_num_workers"
    )
    parser.add_argument(
        "--training.eval_chunk_size", type=int, dest="training__eval_chunk_size"
    )
    parser.add_argument(
        "--training.eval_units_per_page", type=int, dest="training__eval_units_per_page"
    )
    parser.add_argument(
        "--training.eval_loader_batch_size",
        type=int,
        dest="training__eval_loader_batch_size",
    )
    parser.add_argument(
        "--training.eval_plot_every",
        type=int,
        dest="training__eval_plot_every",
    )
    parser.add_argument(
        "--training.eval_pdf_dpi",
        type=int,
        dest="training__eval_pdf_dpi",
    )
    parser.add_argument(
        "--visualization.anim_num_traj",
        type=int,
        dest="visualization__anim_num_traj",
    )
    parser.add_argument(
        "--visualization.anim_fps",
        type=int,
        dest="visualization__anim_fps",
    )
    parser.add_argument(
        "--visualization.anim_step",
        type=int,
        dest="visualization__anim_step",
    )
    parser.add_argument(
        "--visualization.anim_workers",
        type=int,
        dest="visualization__anim_workers",
    )

    return parser.parse_args()


def _apply_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> SimpleNamespace:
    """Apply non-None CLI overrides to the config namespace in-place."""
    for key, value in vars(args).items():
        if key in ("config", "data_path", "eval_data_path") or value is None:
            continue
        section, attr = key.split("__", 1)
        setattr(getattr(cfg, section), attr, value)
    return cfg


def _get_animation_setting(cfg: SimpleNamespace, name: str, default):
    """Resolve unified animation config, with fallback to legacy eval keys."""
    vis_cfg = getattr(cfg, "visualization", None)
    if vis_cfg is not None and hasattr(vis_cfg, name):
        return getattr(vis_cfg, name)

    legacy_name = {
        "anim_num_traj": "eval_anim_num_traj",
        "anim_fps": "eval_anim_fps",
        "anim_step": "eval_anim_step",
        "anim_workers": "eval_anim_workers",
    }[name]
    training_cfg = getattr(cfg, "training", None)
    if training_cfg is not None and hasattr(training_cfg, legacy_name):
        return getattr(training_cfg, legacy_name)
    return default


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    model, pc_ens, hdc_ens, scorer, eval_loader, cfg, device, epoch, writer=None
):
    """Collect bottleneck activations, compute grid scores, and optionally save PDF + MP4."""
    hooks = EvaluationHooks(
        encode_initial_conditions=encode_initial_conditions,
        compute_position_mse=compute_position_mse,
        decode_position_from_pc_logits=decode_position_from_pc_logits,
        get_scores_and_plot_from_ratemaps=get_scores_and_plot_from_ratemaps,
        plot_hdc_tuning_curves=plot_hdc_tuning_curves,
        generate_eval_animation=generate_eval_animation,
        score_ratemaps=score_ratemaps,
        get_animation_setting=lambda name, default: _get_animation_setting(
            cfg,
            name,
            default,
        ),
    )
    evaluator = Evaluator(
        model=model,
        pc_ens=pc_ens,
        hdc_ens=hdc_ens,
        scorer=scorer,
        eval_loader=eval_loader,
        cfg=cfg,
        device=device,
        hooks=hooks,
    )
    evaluator.run(epoch=epoch, writer=writer)


def _build_eval_loader(cfg, logger: logging.Logger, eval_data_path: str = None):
    """Build one fixed evaluation loader, preferring a configured dataset file."""
    resolved_eval_data_path = eval_data_path
    if resolved_eval_data_path is None:
        resolved_eval_data_path = getattr(cfg.training, "eval_data_path", None)

    eval_batch_size = getattr(
        cfg.training, "eval_loader_batch_size", cfg.training.batch_size
    )

    if resolved_eval_data_path and os.path.exists(resolved_eval_data_path):
        logger.info("Loading eval trajectories from %s", resolved_eval_data_path)
        return get_dataloader(
            cfg,
            data_path=resolved_eval_data_path,
            shuffle=False,
            batch_size=eval_batch_size,
        )

    if resolved_eval_data_path:
        logger.warning(
            "Eval dataset %s not found; generating one fixed in-memory eval set with %d trajectories",
            resolved_eval_data_path,
            cfg.training.eval_batch_size,
        )
    else:
        logger.info(
            "Generating one fixed in-memory eval set with %d trajectories",
            cfg.training.eval_batch_size,
        )

    return get_dataloader(
        cfg,
        num_samples=cfg.training.eval_batch_size,
        shuffle=False,
        batch_size=eval_batch_size,
    )


def _build_train_loader(
    cfg,
    logger: logging.Logger,
    pc_ens,
    hdc_ens,
    data_path: str = None,
):
    """Build one fixed training loader, preferring a configured dataset file."""
    resolved_data_path = data_path
    if resolved_data_path is None:
        resolved_data_path = getattr(cfg.training, "data_path", None)

    if resolved_data_path and os.path.exists(resolved_data_path):
        logger.info("Loading trajectories from %s", resolved_data_path)
        return get_dataloader(
            cfg,
            data_path=resolved_data_path,
            pc_ens=pc_ens,
            hdc_ens=hdc_ens,
        )

    if resolved_data_path:
        logger.warning(
            "Training trajectory file %s not found; falling back to on-the-fly generation.",
            resolved_data_path,
        )

    return None


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg, data_path: str = None, eval_data_path: str = None):
    """Run the full training loop described in Banino et al., Nature 2018."""
    hooks = TrainingSessionHooks(
        resolve_save_dir=resolve_save_dir,
        setup_logger=setup_logger,
        create_summary_writer=create_summary_writer,
        build_optimizer=build_optimizer,
        build_train_loader=_build_train_loader,
        build_eval_loader=_build_eval_loader,
        evaluate=_evaluate,
        get_step_log_interval=get_step_log_interval,
        tqdm=tqdm,
    )
    session = TrainingSession(
        cfg,
        hooks=hooks,
        data_path=data_path,
        eval_data_path=eval_data_path,
    )
    session.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    train(cfg, data_path=args.data_path, eval_data_path=args.eval_data_path)
