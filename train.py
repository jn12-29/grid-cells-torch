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
from model import GridCellsRNN
from scores import GridScorer
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

    return parser.parse_args()


def _apply_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> SimpleNamespace:
    """Apply non-None CLI overrides to the config namespace in-place."""
    for key, value in vars(args).items():
        if key in ("config", "data_path", "eval_data_path") or value is None:
            continue
        section, attr = key.split("__", 1)
        setattr(getattr(cfg, section), attr, value)
    return cfg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    model, pc_ens, hdc_ens, scorer, eval_loader, cfg, device, epoch, writer=None
):
    """Collect bottleneck activations, compute grid scores, and optionally save PDF + MP4."""
    model_was_training = model.training
    model.eval()

    # Determine whether to save visualisations this eval (PDF + animation).
    # eval_plot_every=N means "save every N evals"; 0 means never.
    # epoch is 0 for pre-training eval, then eval_every, 2*eval_every, ...
    plot_every = getattr(cfg.training, "eval_plot_every", 1)
    eval_every = cfg.training.eval_every
    eval_index = epoch // max(1, eval_every)  # 0, 1, 2, ...
    save_pdf = (plot_every > 0) and (eval_index % plot_every == 0)

    ratemap_sums, ratemap_counts = scorer.allocate_ratemap_accumulators(
        cfg.model.nh_bottleneck
    )
    eval_start = time.time()
    eval_pos_mse_sum = 0.0
    eval_pos_batches = 0

    # Animation data collected from the first batch only.
    anim_data = None
    num_anim_traj = getattr(cfg.training, "eval_anim_num_traj", 1)

    # HDC tuning curve accumulators (all batches).
    hd_list   = []
    hdc_list  = []

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            init_cond = encode_initial_conditions(batch, pc_ens, hdc_ens).to(device)
            pc_logits, hdc_logits, bottleneck, _ = model(
                init_cond, batch["ego_vel"], training=False
            )
            pos_mse = compute_position_mse(pc_logits, batch["target_pos"], pc_ens)
            eval_pos_mse_sum += float(pos_mse.item())
            eval_pos_batches += 1
            scorer.accumulate_ratemaps(
                batch["target_pos"].detach().cpu().numpy(),
                bottleneck.detach().cpu().numpy(),
                ratemap_sums,
                ratemap_counts,
            )

            # Accumulate HDC data for tuning curves (flatten B×T).
            if save_pdf:
                hd_np  = batch["target_hd"].detach().cpu().numpy()           # (B, T, 1)
                hdc_np = torch.softmax(hdc_logits[0], dim=-1).detach().cpu().numpy()  # (B, T, n_hdc)
                hd_list.append(hd_np.reshape(-1))
                hdc_list.append(hdc_np.reshape(-1, hdc_np.shape[-1]))

            # Collect animation data from the first batch.
            if save_pdf and anim_data is None:
                n_anim = min(num_anim_traj, batch["target_pos"].shape[0])
                pred_pos_batch = decode_position_from_pc_logits(pc_logits, pc_ens)
                pc_probs  = torch.softmax(pc_logits[0], dim=-1)
                hdc_probs = torch.softmax(hdc_logits[0], dim=-1)
                anim_data = {
                    "target_pos": batch["target_pos"][:n_anim].detach().cpu().numpy(),
                    "pred_pos":   pred_pos_batch[:n_anim].detach().cpu().numpy(),
                    "pc_acts":    pc_probs[:n_anim].detach().cpu().numpy(),
                    "hdc_acts":   hdc_probs[:n_anim].detach().cpu().numpy(),
                }

    infer_seconds = time.time() - eval_start
    ratemaps = scorer.finalize_ratemaps(ratemap_sums, ratemap_counts)

    score_start = time.time()
    num_workers = getattr(cfg.training, "eval_num_workers", 0)
    chunk_size = getattr(cfg.training, "eval_chunk_size", 32)

    if save_pdf:
        save_dir = cfg.training.save_dir
        filename = f"rates_and_sac_epoch_{epoch:04d}.pdf"
        scores = get_scores_and_plot_from_ratemaps(
            scorer,
            ratemaps,
            save_dir,
            filename,
            num_workers=num_workers,
            chunk_size=chunk_size,
            units_per_page=getattr(cfg.training, "eval_units_per_page", 128),
            pdf_dpi=getattr(cfg.training, "eval_pdf_dpi", 72),
        )
        score_60 = scores[0]
        score_90 = scores[1]

        # HDC directional tuning curves PDF.
        logger = logging.getLogger("grid_cells")
        if hd_list:
            try:
                hdc_pdf_path = os.path.join(save_dir, f"hdc_tuning_epoch_{epoch:04d}.pdf")
                plot_hdc_tuning_curves(
                    np.concatenate(hd_list),
                    np.concatenate(hdc_list, axis=0),
                    n_bins=getattr(cfg.visualization, "directional_bins", 20),
                    save_path=hdc_pdf_path,
                    pdf_dpi=getattr(cfg.training, "eval_pdf_dpi", 100),
                )
                logger.info("HDC tuning curves saved to %s", hdc_pdf_path)
            except Exception as exc:
                logger.warning("HDC tuning curve plot failed: %s", exc)

        # Generate eval animation MP4 alongside the PDF.
        if anim_data is not None:
            anim_path = os.path.join(save_dir, f"eval_animation_epoch_{epoch:04d}.mp4")
            pc_centers = np.concatenate([ens.means for ens in pc_ens], axis=0)
            try:
                generate_eval_animation(
                    anim_data["target_pos"],
                    anim_data["pred_pos"],
                    anim_data["pc_acts"],
                    anim_data["hdc_acts"],
                    pc_centers,
                    cfg.task.env_size,
                    anim_path,
                    fps=getattr(cfg.training, "eval_anim_fps", 20),
                    step=getattr(cfg.training, "eval_anim_step", 4),
                    num_workers=getattr(cfg.training, "eval_anim_workers", 4),
                )
            except Exception as exc:
                logger.warning("eval animation failed: %s", exc)
    else:
        score_60, score_90, _, _, _ = score_ratemaps(
            scorer,
            ratemaps,
            num_workers=num_workers,
            chunk_size=chunk_size,
        )

    score_seconds = time.time() - score_start
    eval_seconds = time.time() - eval_start
    eval_pos_mse = eval_pos_mse_sum / max(eval_pos_batches, 1)

    logger = logging.getLogger("grid_cells")
    logger.info(
        "eval epoch=%d  pos_mse=%.6f  grid_score_60 max=%.4f  grid_score_90 max=%.4f"
        "  infer=%.1fs  score=%.1fs  total=%.1fs  pdf=%s",
        epoch,
        eval_pos_mse,
        float(score_60.max()),
        float(score_90.max()),
        infer_seconds,
        score_seconds,
        eval_seconds,
        save_pdf,
    )

    if writer is not None:
        writer.add_scalar("eval/pos_mse", eval_pos_mse, epoch)
        writer.add_scalar("eval/grid_score_60_max", float(score_60.max()), epoch)
        writer.add_scalar("eval/grid_score_90_max", float(score_90.max()), epoch)
        writer.add_scalar("eval/seconds", eval_seconds, epoch)
        writer.add_scalar("eval/infer_seconds", infer_seconds, epoch)
        writer.add_scalar("eval/score_seconds", score_seconds, epoch)

    if model_was_training:
        model.train()


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
    cfg.training.save_dir = resolve_save_dir(
        cfg.training.save_dir,
        timestamp_save_dir=getattr(cfg.training, "timestamp_save_dir", True),
        run_name=getattr(cfg.training, "run_name", None),
    )
    logger = setup_logger(cfg.training.save_dir)
    writer = create_summary_writer(cfg, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Optimizer: %s", getattr(cfg.training, "optimizer", "rmsprop"))
    logger.info("Run directory: %s", cfg.training.save_dir)
    logger.info("Progress bar enabled: %s", getattr(cfg.training, "use_tqdm", True))
    logger.info(
        "Eval workers=%s  chunk_size=%s  units_per_page=%s",
        getattr(cfg.training, "eval_num_workers", 0),
        getattr(cfg.training, "eval_chunk_size", 32),
        getattr(cfg.training, "eval_units_per_page", 128),
    )

    # 1. Create ensembles
    pc_ens = get_place_cell_ensembles(cfg)
    hdc_ens = get_head_direction_ensembles(cfg)

    # 2. Create model
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model)).to(device)

    # 3. Optimizer (weight decay restricted to decoder heads)
    optimizer, decoder_params = build_optimizer(model, cfg)

    if getattr(cfg.training, "use_tqdm", True) and tqdm is None:
        logger.warning("tqdm is not installed; falling back to plain logs")

    # 4. GridScorer for evaluation
    nbins = getattr(getattr(cfg, "visualization", None), "spatial_bins", 20)
    starts = [0.2] * 10
    ends = list(np.linspace(0.4, 1.0, num=10))
    masks_params = list(zip(starts, ends))
    scorer = GridScorer(
        nbins,
        [
            [-cfg.task.env_size / 2, cfg.task.env_size / 2],
            [-cfg.task.env_size / 2, cfg.task.env_size / 2],
        ],
        masks_params,
    )

    # 5. Create save directory
    os.makedirs(cfg.training.save_dir, exist_ok=True)

    # 6. Training loop
    _fixed_loader = _build_train_loader(
        cfg,
        logger,
        pc_ens,
        hdc_ens,
        data_path=data_path,
    )

    _fixed_eval_loader = _build_eval_loader(
        cfg,
        logger,
        eval_data_path=eval_data_path,
    )

    global_step = 0

    # Pre-training evaluation (baseline before any weight updates)
    _evaluate(
        model,
        pc_ens,
        hdc_ens,
        scorer,
        _fixed_eval_loader,
        cfg,
        device,
        epoch=0,
        writer=writer,
    )

    try:
        for epoch in range(cfg.training.epochs):
            # Use pre-generated data if provided, otherwise generate fresh each epoch
            dataloader = (
                _fixed_loader if _fixed_loader is not None else get_dataloader(cfg)
            )
            num_steps = len(dataloader)
            step_log_interval = max(
                1,
                getattr(cfg.training, "tensorboard_log_every", 0)
                or get_step_log_interval(num_steps),
            )
            model.train()
            loss_acc = []
            pos_mse_acc = []
            epoch_start = time.time()

            use_tqdm = getattr(cfg.training, "use_tqdm", True) and tqdm is not None
            progress = (
                tqdm(
                    dataloader,
                    total=num_steps,
                    desc=f"epoch {epoch + 1}/{cfg.training.epochs}",
                    leave=False,
                    dynamic_ncols=True,
                )
                if use_tqdm
                else dataloader
            )

            for step, batch in enumerate(progress):
                optimizer.zero_grad()

                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                if "init_cond" in batch:
                    init_cond = batch["init_cond"].float()
                    pc_targets = [batch[f"pc_targets_{i}"] for i in range(len(pc_ens))]
                    hdc_targets = [
                        batch[f"hdc_targets_{i}"] for i in range(len(hdc_ens))
                    ]
                else:
                    init_cond = encode_initial_conditions(batch, pc_ens, hdc_ens).to(
                        device
                    )
                    pc_targets, hdc_targets = encode_targets(batch, pc_ens, hdc_ens)

                pc_logits, hdc_logits, _, _ = model(
                    init_cond, batch["ego_vel"], training=True
                )
                pos_mse = compute_position_mse(pc_logits, batch["target_pos"], pc_ens)

                loss = sum(
                    ens.loss(logits, targets)
                    for ens, logits, targets in zip(pc_ens, pc_logits, pc_targets)
                )
                loss += sum(
                    ens.loss(logits, targets)
                    for ens, logits, targets in zip(hdc_ens, hdc_logits, hdc_targets)
                )

                loss.backward()
                torch.nn.utils.clip_grad_value_(decoder_params, cfg.training.grad_clip)
                optimizer.step()

                loss_value = loss.item()
                pos_mse_value = float(pos_mse.item())
                loss_acc.append(loss_value)
                pos_mse_acc.append(pos_mse_value)
                global_step += 1

                if use_tqdm:
                    progress.set_postfix(
                        loss=f"{loss_value:.4f}",
                        avg=f"{np.mean(loss_acc):.4f}",
                        pos_mse=f"{pos_mse_value:.6f}",
                    )

                should_log_step = ((step + 1) % step_log_interval == 0) or (
                    step == num_steps - 1
                )
                if writer is not None and should_log_step:
                    writer.add_scalar("train/loss_step", loss_value, global_step)
                    writer.add_scalar("train/pos_mse_step", pos_mse_value, global_step)

            epoch_mean = float(np.mean(loss_acc))
            epoch_std = float(np.std(loss_acc))
            epoch_pos_mse = float(np.mean(pos_mse_acc))
            epoch_time = time.time() - epoch_start

            logger.info(
                "epoch=%4d  loss mean=%.4f  std=%.4f  pos_mse=%.6f  seconds=%.1f",
                epoch,
                epoch_mean,
                epoch_std,
                epoch_pos_mse,
                epoch_time,
            )

            if writer is not None:
                writer.add_scalar("train/loss_mean", epoch_mean, epoch)
                writer.add_scalar("train/loss_std", epoch_std, epoch)
                writer.add_scalar("train/pos_mse_mean", epoch_pos_mse, epoch)
                writer.add_scalar("train/epoch_seconds", epoch_time, epoch)

            if (epoch + 1) % cfg.training.eval_every == 0:
                _evaluate(
                    model,
                    pc_ens,
                    hdc_ens,
                    scorer,
                    _fixed_eval_loader,
                    cfg,
                    device,
                    epoch + 1,
                    writer=writer,
                )
    finally:
        if writer is not None:
            writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    train(cfg, data_path=args.data_path, eval_data_path=args.eval_data_path)
