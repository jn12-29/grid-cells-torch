"""Training-session orchestration for grid-cell experiments.

This module encapsulates the experiment lifecycle so ``train.py`` can remain a
thin public entrypoint while preserving the existing helper functions and CLI.

Usage:
    session = TrainingSession(cfg, hooks=hooks)
    session.run()
"""

from dataclasses import dataclass
import math
import os
import time
from typing import Callable

import numpy as np
import torch

from grid_cells.data.dataset import get_dataloader
from grid_cells.cells.encoding_utils import (
    compute_head_direction_mae_rad,
    compute_position_mse,
    encode_initial_conditions,
    encode_targets,
)
from grid_cells.cells.contracts import validate_cell_code_contract
from grid_cells.cells.ensemble_utils import (
    get_cell_ensembles,
)
from grid_cells.cells.model import GridCellsRNN
from grid_cells.analysis.scores import GridScorer


_METRIC_RATIO_EPS = 1e-12
_GRAD_MONITOR_MODULE_NAMES = (
    "state_init",
    "cell_init",
    "lstm",
    "bottleneck",
    "pc_heads",
    "hdc_heads",
)


def _metric_ratio(numerator: float, denominator: float) -> float:
    """Return a stable diagnostic ratio for nonnegative scalar metrics."""
    return numerator / max(denominator, _METRIC_RATIO_EPS)


def _grad_monitor_param_groups(model):
    """Return model submodules tracked by the gradient monitor."""
    groups = {}
    for module_name in _GRAD_MONITOR_MODULE_NAMES:
        module = getattr(model, module_name, None)
        if module is None or not hasattr(module, "parameters"):
            continue
        params = list(module.parameters())
        if params:
            groups[module_name] = params
    return groups


def _collect_grad_group_stats(params, norm_type: float, clip_threshold: float = None):
    """Return aggregate gradient summary stats for one parameter group."""
    grads = [
        param.grad.detach()
        for param in params
        if param.grad is not None and param.grad.numel() > 0
    ]
    if not grads:
        return None

    max_abs = 0.0
    sum_values = 0.0
    sum_abs = 0.0
    sum_squares = 0.0
    numel = 0
    clip_exceed_count = 0
    per_param_norms = []
    for grad in grads:
        grad_abs = grad.abs()
        max_abs = max(max_abs, float(grad_abs.max().item()))
        sum_values += float(grad.sum().item())
        sum_abs += float(grad_abs.sum().item())
        sum_squares += float(grad.square().sum().item())
        numel += int(grad.numel())
        if clip_threshold is not None:
            clip_exceed_count += int((grad_abs > clip_threshold).sum().item())
        if norm_type != float("inf"):
            per_param_norms.append(torch.linalg.vector_norm(grad, ord=norm_type))

    if numel <= 0:
        return None

    if norm_type == float("inf"):
        grad_norm = max_abs
    else:
        per_param_norms = torch.stack(per_param_norms)
        grad_norm = float(
            torch.linalg.vector_norm(per_param_norms, ord=norm_type).item()
        )

    grad_rms = math.sqrt(sum_squares / numel)
    stats = {
        "norm": float(grad_norm),
        "rms": float(grad_rms),
        "mean_abs": float(sum_abs / numel),
        "mean": float(sum_values / numel),
        "max_abs": float(max_abs),
        "numel": float(numel),
    }
    if clip_threshold is not None:
        stats["clip_exceed_count"] = float(clip_exceed_count)
    return stats


def _collect_grad_monitor_stats(model, norm_type: float, clip_threshold: float):
    """Return gradient stats for all monitored model submodules with gradients."""
    stats = {}
    for module_name, params in _grad_monitor_param_groups(model).items():
        group_stats = _collect_grad_group_stats(params, norm_type, clip_threshold)
        if group_stats is not None:
            stats[module_name] = group_stats
    return stats


def _snapshot_grad_monitor_grads(model):
    """Clone monitored gradients so post-clip stats can count changed elements."""
    snapshots = {}
    for module_name, params in _grad_monitor_param_groups(model).items():
        module_snapshot = []
        for param in params:
            if param.grad is None or param.grad.numel() <= 0:
                continue
            module_snapshot.append((param, param.grad.detach().clone()))
        if module_snapshot:
            snapshots[module_name] = module_snapshot
    return snapshots


def _collect_grad_monitor_change_stats(grad_snapshots, clip_threshold: float):
    """Return changed-gradient counts and post-clip boundary counts."""
    tolerance = max(1e-12, clip_threshold * 1e-6)
    stats = {}
    for module_name, module_snapshot in grad_snapshots.items():
        numel = 0
        changed_count = 0
        post_saturated_count = 0
        for param, pre_grad in module_snapshot:
            if param.grad is None or param.grad.shape != pre_grad.shape:
                continue
            post_grad = param.grad.detach()
            post_abs = post_grad.abs()
            numel += int(pre_grad.numel())
            changed_count += int(post_grad.ne(pre_grad).sum().item())
            post_saturated_count += int(
                (post_abs - clip_threshold).abs().le(tolerance).sum().item()
            )
        if numel > 0:
            stats[module_name] = {
                "numel": float(numel),
                "changed_count": float(changed_count),
                "post_saturated_count": float(post_saturated_count),
            }
    return stats


def _merge_grad_monitor_stats(pre_clip_stats, post_clip_stats, change_stats=None):
    """Combine pre/post clipping stats into the public TensorBoard metric shape."""
    merged = {}
    for module_name, pre_stats in pre_clip_stats.items():
        post_stats = post_clip_stats.get(module_name)
        if post_stats is None:
            continue
        numel = max(
            float(pre_stats.get("numel", post_stats.get("numel", 0.0))),
            _METRIC_RATIO_EPS,
        )
        module_change_stats = (change_stats or {}).get(module_name, {})
        merged[module_name] = {
            "pre_clip_norm": pre_stats["norm"],
            "post_clip_norm": post_stats["norm"],
            "pre_clip_rms": pre_stats["rms"],
            "post_clip_rms": post_stats["rms"],
            "pre_clip_mean_abs": pre_stats["mean_abs"],
            "post_clip_mean_abs": post_stats["mean_abs"],
            "pre_clip_mean": pre_stats["mean"],
            "post_clip_mean": post_stats["mean"],
            "pre_clip_max_abs": pre_stats["max_abs"],
            "post_clip_max_abs": post_stats["max_abs"],
            "pre_clip_exceed_frac": _metric_ratio(
                pre_stats.get("clip_exceed_count", 0.0),
                numel,
            ),
            "clip_changed_frac": _metric_ratio(
                module_change_stats.get("changed_count", 0.0),
                numel,
            ),
            "post_clip_saturated_frac": _metric_ratio(
                module_change_stats.get("post_saturated_count", 0.0),
                numel,
            ),
            "clip_ratio": _metric_ratio(post_stats["norm"], pre_stats["norm"]),
        }
    return merged


def _accumulate_grad_monitor_stats(accumulator, grad_stats) -> None:
    """Append one sampled grad-stat dict into an epoch accumulator."""
    for module_name, module_stats in grad_stats.items():
        metric_accumulator = accumulator.setdefault(module_name, {})
        for metric_name, value in module_stats.items():
            metric_accumulator.setdefault(metric_name, []).append(value)


def _mean_grad_monitor_stats(accumulator):
    """Return per-module mean values from sampled gradient stats."""
    return {
        module_name: {
            metric_name: float(np.mean(values))
            for metric_name, values in module_stats.items()
            if values
        }
        for module_name, module_stats in accumulator.items()
    }


def _write_grad_monitor_scalars(writer, grad_stats, step: int, suffix: str) -> None:
    """Write gradient monitor scalars under train/grad/<module>/... tags."""
    if writer is None:
        return
    for module_name, module_stats in grad_stats.items():
        for metric_name, value in module_stats.items():
            writer.add_scalar(
                f"train/grad/{module_name}/{metric_name}_{suffix}",
                value,
                step,
            )


def _format_grad_monitor_log(grad_stats) -> str:
    """Return a compact epoch-level gradient diagnostic for train.log."""
    parts = []
    for module_name, module_stats in grad_stats.items():
        parts.append(
            (
                "{} pre={:.4g} post={:.4g} rms={:.4g}->{:.4g} "
                "mean_abs={:.4g}->{:.4g} mean={:.4g}->{:.4g} "
                "max_abs={:.4g}->{:.4g} pre_exceed={:.4g} "
                "changed={:.4g} saturated={:.4g} ratio={:.4g}"
            ).format(
                module_name,
                module_stats["pre_clip_norm"],
                module_stats["post_clip_norm"],
                module_stats["pre_clip_rms"],
                module_stats["post_clip_rms"],
                module_stats["pre_clip_mean_abs"],
                module_stats["post_clip_mean_abs"],
                module_stats["pre_clip_mean"],
                module_stats["post_clip_mean"],
                module_stats["pre_clip_max_abs"],
                module_stats["post_clip_max_abs"],
                module_stats["pre_clip_exceed_frac"],
                module_stats["clip_changed_frac"],
                module_stats["post_clip_saturated_frac"],
                module_stats["clip_ratio"],
            )
        )
    return "; ".join(parts)


def _normalize_grad_clip_mode(mode: str) -> str:
    """Return the canonical gradient clipping mode name."""
    mode = str(mode).lower()
    if mode in {"value", "valueclip"}:
        return "value"
    if mode in {"norm", "normclip"}:
        return "norm"
    raise ValueError(f"Unsupported gradient clipping mode: {mode}")


def _select_grad_clip_params(model, decoder_params, scope: str):
    """Return the model parameters selected for gradient clipping."""
    scope = str(scope).lower()
    if scope in {"all", "model"}:
        return list(model.parameters())
    if scope == "decoder":
        return list(decoder_params)
    raise ValueError(f"Unsupported gradient clipping scope: {scope}")


def _resolve_grad_clip_norm_type(norm_type):
    """Resolve a CLI/YAML norm type value for torch gradient norm clipping."""
    if isinstance(norm_type, str):
        normalized = norm_type.lower()
        if normalized in {"inf", "infinity"}:
            return float("inf")
        return float(norm_type)
    return float(norm_type)


def _resolve_grad_monitor_norm_type(cfg):
    """Return the norm type used for diagnostic gradient monitor stats."""
    mode = str(getattr(cfg.training, "grad_clip_mode", "norm")).lower()
    if mode in {"norm", "normclip"}:
        return _resolve_grad_clip_norm_type(
            getattr(cfg.training, "grad_clip_norm_type", 2.0)
        )
    return 2.0


def _clip_gradients(model, decoder_params, cfg) -> None:
    """Clip gradients using the configured method and parameter scope."""
    clip_value = float(cfg.training.grad_clip)
    if not np.isfinite(clip_value):
        raise ValueError("training.grad_clip must be finite.")
    if clip_value < 0.0:
        raise ValueError("training.grad_clip must be non-negative.")

    mode = _normalize_grad_clip_mode(getattr(cfg.training, "grad_clip_mode", "norm"))
    params = _select_grad_clip_params(
        model,
        decoder_params,
        getattr(cfg.training, "grad_clip_scope", "decoder"),
    )

    if mode == "value":
        torch.nn.utils.clip_grad_value_(params, clip_value)
        return

    norm_type = _resolve_grad_clip_norm_type(
        getattr(cfg.training, "grad_clip_norm_type", 2.0)
    )
    torch.nn.utils.clip_grad_norm_(params, clip_value, norm_type=norm_type)


@dataclass
class TrainingSessionHooks:
    """Inject train-module helpers so the public wrappers keep current behavior."""

    resolve_save_dir: Callable
    setup_logger: Callable
    save_config: Callable
    create_summary_writer: Callable
    build_optimizer: Callable
    build_lr_scheduler: Callable
    build_train_loader: Callable
    build_eval_loader: Callable
    evaluate: Callable
    save_checkpoint: Callable
    get_step_log_interval: Callable
    tqdm: object


class TrainingSession:
    """Run one end-to-end training session including periodic evaluation."""

    def __init__(self, cfg, hooks: TrainingSessionHooks, data_path: str = None, eval_data_path: str = None):
        self.cfg = cfg
        self.hooks = hooks
        self.data_path = data_path
        self.eval_data_path = eval_data_path

    def run(self) -> None:
        """Execute the full training loop described by the config."""
        self.cfg.training.save_dir = self.hooks.resolve_save_dir(
            self.cfg.training.save_dir,
            timestamp_save_dir=getattr(self.cfg.training, "timestamp_save_dir", True),
            run_name=getattr(self.cfg.training, "run_name", None),
        )
        logger = self.hooks.setup_logger(self.cfg.training.save_dir)
        config_path = os.path.join(self.cfg.training.save_dir, "config.yaml")
        self.hooks.save_config(self.cfg, config_path)
        logger.info("Saved effective config to %s", config_path)
        writer = self.hooks.create_summary_writer(self.cfg, logger)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Using device: %s", device)
        logger.info("Optimizer: %s", getattr(self.cfg.training, "optimizer", "rmsprop"))
        logger.info("Learning-rate scheduler: %s", getattr(self.cfg.training, "lr_scheduler", "cosine"))
        logger.info(
            "Gradient clipping: mode=%s  scope=%s  value=%s  norm_type=%s",
            getattr(self.cfg.training, "grad_clip_mode", "norm"),
            getattr(self.cfg.training, "grad_clip_scope", "decoder"),
            self.cfg.training.grad_clip,
            getattr(self.cfg.training, "grad_clip_norm_type", 2.0),
        )
        logger.info("Run directory: %s", self.cfg.training.save_dir)
        logger.info(
            "Progress bar enabled: %s",
            getattr(self.cfg.training, "use_tqdm", True),
        )
        logger.info(
            "Eval workers=%s  chunk_size=%s  units_per_page=%s",
            getattr(self.cfg.training, "eval_num_workers", 0),
            getattr(self.cfg.training, "eval_chunk_size", 32),
            getattr(self.cfg.training, "eval_units_per_page", 128),
        )

        pc_ens, hdc_ens = get_cell_ensembles(self.cfg)
        validate_cell_code_contract(self.cfg, pc_ens, hdc_ens, logger)
        model = GridCellsRNN(pc_ens, hdc_ens, **vars(self.cfg.model)).to(device)
        optimizer, decoder_params = self.hooks.build_optimizer(model, self.cfg)
        lr_scheduler = self.hooks.build_lr_scheduler(optimizer, self.cfg)

        if getattr(self.cfg.training, "use_tqdm", True) and self.hooks.tqdm is None:
            logger.warning("tqdm is not installed; falling back to plain logs")

        scorer = self._build_scorer()
        os.makedirs(self.cfg.training.save_dir, exist_ok=True)

        fixed_loader = self.hooks.build_train_loader(
            self.cfg,
            logger,
            pc_ens,
            hdc_ens,
            data_path=self.data_path,
        )
        fixed_eval_loader = self.hooks.build_eval_loader(
            self.cfg,
            logger,
            eval_data_path=self.eval_data_path,
        )

        global_step = 0
        self.hooks.evaluate(
            model,
            pc_ens,
            hdc_ens,
            scorer,
            fixed_eval_loader,
            self.cfg,
            device,
            epoch=0,
            writer=writer,
        )

        try:
            completed_epochs = 0
            for epoch in range(self.cfg.training.epochs):
                global_step = self._run_epoch(
                    epoch=epoch,
                    global_step=global_step,
                    logger=logger,
                    writer=writer,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    decoder_params=decoder_params,
                    pc_ens=pc_ens,
                    hdc_ens=hdc_ens,
                    scorer=scorer,
                    fixed_loader=fixed_loader,
                    fixed_eval_loader=fixed_eval_loader,
                )
                completed_epochs = epoch + 1
                checkpoint_every = getattr(self.cfg.training, "checkpoint_every", 0)
                if checkpoint_every > 0 and completed_epochs % checkpoint_every == 0:
                    checkpoint_path = self.hooks.save_checkpoint(
                        model,
                        optimizer,
                        lr_scheduler,
                        self.cfg,
                        pc_ens,
                        hdc_ens,
                        epoch=completed_epochs,
                        global_step=global_step,
                        filename=f"checkpoint_epoch_{completed_epochs:04d}.pt",
                    )
                    logger.info("Saved checkpoint to %s", checkpoint_path)

            if completed_epochs > 0 and getattr(
                self.cfg.training,
                "save_final_checkpoint",
                True,
            ):
                checkpoint_path = self.hooks.save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    self.cfg,
                    pc_ens,
                    hdc_ens,
                    epoch=completed_epochs,
                    global_step=global_step,
                    filename="checkpoint_final.pt",
                )
                logger.info("Saved final checkpoint to %s", checkpoint_path)
        finally:
            if writer is not None:
                writer.close()

    def _build_scorer(self) -> GridScorer:
        """Create the grid scorer used during evaluation."""
        nbins = getattr(getattr(self.cfg, "visualization", None), "spatial_bins", 20)
        starts = [0.2] * 10
        ends = list(np.linspace(0.4, 1.0, num=10))
        masks_params = list(zip(starts, ends))
        return GridScorer(
            nbins,
            [
                [-self.cfg.task.env_size / 2, self.cfg.task.env_size / 2],
                [-self.cfg.task.env_size / 2, self.cfg.task.env_size / 2],
            ],
            masks_params,
        )

    def _run_epoch(
        self,
        epoch: int,
        global_step: int,
        logger,
        writer,
        device,
        model,
        optimizer,
        lr_scheduler,
        decoder_params,
        pc_ens,
        hdc_ens,
        scorer,
        fixed_loader,
        fixed_eval_loader,
    ) -> int:
        """Run one training epoch and trigger evaluation if configured."""
        dataloader = fixed_loader if fixed_loader is not None else get_dataloader(self.cfg)
        num_steps = len(dataloader)
        step_log_interval = max(
            1,
            getattr(self.cfg.training, "tensorboard_log_every", 0)
            or self.hooks.get_step_log_interval(num_steps),
        )
        model.train()
        loss_acc = []
        pos_mse_acc = []
        first_pos_mse_acc = []
        hd_mae_acc = []
        first_hd_mae_acc = []
        grad_stats_acc = {}
        grad_norm_type = _resolve_grad_monitor_norm_type(self.cfg)
        grad_clip_threshold = abs(float(self.cfg.training.grad_clip))
        epoch_start = time.time()

        use_tqdm = getattr(self.cfg.training, "use_tqdm", True) and self.hooks.tqdm is not None
        progress = (
            self.hooks.tqdm(
                dataloader,
                total=num_steps,
                desc=f"epoch {epoch + 1}/{self.cfg.training.epochs}",
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
                hdc_targets = [batch[f"hdc_targets_{i}"] for i in range(len(hdc_ens))]
            else:
                init_cond = encode_initial_conditions(batch, pc_ens, hdc_ens).to(device)
                pc_targets, hdc_targets = encode_targets(batch, pc_ens, hdc_ens)

            pc_logits, hdc_logits, _, _ = model(init_cond, batch["ego_vel"], training=True)
            first_pos_loss_multiplier = float(
                getattr(self.cfg.training, "first_pos_loss_multiplier", 1.0)
            )
            if not np.isfinite(first_pos_loss_multiplier):
                raise ValueError("training.first_pos_loss_multiplier must be finite.")
            if first_pos_loss_multiplier < 0.0:
                raise ValueError(
                    "training.first_pos_loss_multiplier must be non-negative."
                )
            pc_timestep_weights = None
            if first_pos_loss_multiplier != 1.0:
                seq_len = pc_logits[0].shape[1]
                pc_timestep_weights = torch.ones(
                    seq_len,
                    dtype=pc_logits[0].dtype,
                    device=pc_logits[0].device,
                )
                pc_timestep_weights[0] = first_pos_loss_multiplier

            with torch.no_grad():
                pos_mse = compute_position_mse(pc_logits, batch["target_pos"], pc_ens)
                first_pos_mse = compute_position_mse(
                    [logits[:, :1, :] for logits in pc_logits],
                    batch["target_pos"][:, :1, :],
                    pc_ens,
                )
                hd_mae = compute_head_direction_mae_rad(
                    hdc_logits,
                    batch["target_hd"],
                    hdc_ens,
                )
                first_hd_mae = compute_head_direction_mae_rad(
                    [logits[:, :1, :] for logits in hdc_logits],
                    batch["target_hd"][:, :1, :],
                    hdc_ens,
                )

            loss = sum(
                ens.loss(logits, targets, timestep_weights=pc_timestep_weights)
                for ens, logits, targets in zip(pc_ens, pc_logits, pc_targets)
            )
            loss += sum(
                ens.loss(logits, targets)
                for ens, logits, targets in zip(hdc_ens, hdc_logits, hdc_targets)
            )

            should_log_step = ((step + 1) % step_log_interval == 0) or (step == num_steps - 1)
            loss.backward()
            grad_stats = {}
            if should_log_step:
                pre_clip_grad_stats = _collect_grad_monitor_stats(
                    model,
                    grad_norm_type,
                    grad_clip_threshold,
                )
                pre_clip_grad_snapshots = _snapshot_grad_monitor_grads(model)
            _clip_gradients(model, decoder_params, self.cfg)
            if should_log_step:
                post_clip_grad_stats = _collect_grad_monitor_stats(
                    model,
                    grad_norm_type,
                    grad_clip_threshold,
                )
                grad_change_stats = _collect_grad_monitor_change_stats(
                    pre_clip_grad_snapshots,
                    grad_clip_threshold,
                )
                grad_stats = _merge_grad_monitor_stats(
                    pre_clip_grad_stats,
                    post_clip_grad_stats,
                    grad_change_stats,
                )
                _accumulate_grad_monitor_stats(grad_stats_acc, grad_stats)
            optimizer.step()

            loss_value = loss.item()
            pos_mse_value = float(pos_mse.item())
            first_pos_mse_value = float(first_pos_mse.item())
            hd_mae_value = float(hd_mae.item())
            first_hd_mae_value = float(first_hd_mae.item())
            loss_acc.append(loss_value)
            pos_mse_acc.append(pos_mse_value)
            first_pos_mse_acc.append(first_pos_mse_value)
            hd_mae_acc.append(hd_mae_value)
            first_hd_mae_acc.append(first_hd_mae_value)
            global_step += 1

            if use_tqdm:
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    avg=f"{np.mean(loss_acc):.4f}",
                    pos_mse=f"{pos_mse_value:.6f}",
                    first_pos=f"{first_pos_mse_value:.6f}",
                    hd_mae=f"{hd_mae_value:.4f}",
                    first_hd=f"{first_hd_mae_value:.4f}",
                )

            if writer is not None and should_log_step:
                writer.add_scalar("train/loss_step", loss_value, global_step)
                writer.add_scalar("train/pos_mse_step", pos_mse_value, global_step)
                writer.add_scalar("train/first_pos_mse_step", first_pos_mse_value, global_step)
                writer.add_scalar("train/hd_mae_rad_step", hd_mae_value, global_step)
                writer.add_scalar("train/first_hd_mae_rad_step", first_hd_mae_value, global_step)
                _write_grad_monitor_scalars(writer, grad_stats, global_step, "step")

        epoch_mean = float(np.mean(loss_acc))
        epoch_std = float(np.std(loss_acc))
        epoch_pos_mse = float(np.mean(pos_mse_acc))
        epoch_first_pos_mse = float(np.mean(first_pos_mse_acc))
        epoch_hd_mae = float(np.mean(hd_mae_acc))
        epoch_first_hd_mae = float(np.mean(first_hd_mae_acc))
        epoch_first_pos_ratio = _metric_ratio(epoch_first_pos_mse, epoch_pos_mse)
        epoch_first_hd_ratio = _metric_ratio(epoch_first_hd_mae, epoch_hd_mae)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_grad_stats = _mean_grad_monitor_stats(grad_stats_acc)

        logger.info(
            "epoch=%4d  loss mean=%.4f  std=%.4f  pos_mse=%.6f  "
            "first_pos_mse=%.6f  first_pos_mse_ratio=%.4f  hd_mae_rad=%.4f  "
            "first_hd_mae_rad=%.4f  first_hd_mae_rad_ratio=%.4f  "
            "lr=%.6g  seconds=%.1f",
            epoch,
            epoch_mean,
            epoch_std,
            epoch_pos_mse,
            epoch_first_pos_mse,
            epoch_first_pos_ratio,
            epoch_hd_mae,
            epoch_first_hd_mae,
            epoch_first_hd_ratio,
            current_lr,
            epoch_time,
        )
        if epoch_grad_stats:
            logger.info(
                "epoch=%4d  grad sampled_means %s",
                epoch,
                _format_grad_monitor_log(epoch_grad_stats),
            )

        if writer is not None:
            writer.add_scalar("train/loss_mean", epoch_mean, epoch)
            writer.add_scalar("train/loss_std", epoch_std, epoch)
            writer.add_scalar("train/pos_mse_mean", epoch_pos_mse, epoch)
            writer.add_scalar("train/first_pos_mse_mean", epoch_first_pos_mse, epoch)
            writer.add_scalar("train/first_pos_mse_ratio", epoch_first_pos_ratio, epoch)
            writer.add_scalar("train/hd_mae_rad_mean", epoch_hd_mae, epoch)
            writer.add_scalar("train/first_hd_mae_rad_mean", epoch_first_hd_mae, epoch)
            writer.add_scalar("train/first_hd_mae_rad_ratio", epoch_first_hd_ratio, epoch)
            writer.add_scalar(
                "train/first_pos_loss_multiplier",
                float(getattr(self.cfg.training, "first_pos_loss_multiplier", 1.0)),
                epoch,
            )
            writer.add_scalar("train/lr", current_lr, epoch)
            writer.add_scalar("train/epoch_seconds", epoch_time, epoch)
            _write_grad_monitor_scalars(writer, epoch_grad_stats, epoch, "mean")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if (epoch + 1) % self.cfg.training.eval_every == 0:
            self.hooks.evaluate(
                model,
                pc_ens,
                hdc_ens,
                scorer,
                fixed_eval_loader,
                self.cfg,
                device,
                epoch + 1,
                writer=writer,
            )

        return global_step
