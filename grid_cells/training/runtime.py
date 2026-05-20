"""Training runtime helpers shared by the train entrypoint wrappers."""

from datetime import datetime
import logging
import math
import os
import re

import torch

from grid_cells.common.config import namespace_to_dict
from grid_cells.data.dataset import get_dataloader
from grid_cells.training.evaluation import EvaluationHooks, Evaluator


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

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging to %s", log_path)
    return logger


def get_step_log_interval(num_steps: int, max_logs_per_epoch: int = 10) -> int:
    """Return a sampling interval that keeps step logs under the cap."""
    if num_steps <= 0:
        return 1
    return max(1, math.ceil(num_steps / max_logs_per_epoch))


def _make_param_groups(model, cfg):
    """Split model parameters into decoder and non-decoder groups."""
    decoder_params = (
        list(model.pc_heads.parameters())
        + list(model.hdc_heads.parameters())
        + list(model.bottleneck.parameters())
    )
    decoder_ids = {id(param) for param in decoder_params}
    other_params = [param for param in model.parameters() if id(param) not in decoder_ids]

    param_groups = [
        {"params": other_params, "weight_decay": 0.0},
        {"params": decoder_params, "weight_decay": cfg.training.weight_decay},
    ]
    return param_groups, decoder_params


def build_optimizer(model, cfg):
    """Build the configured optimizer for training."""
    param_groups, decoder_params = _make_param_groups(model, cfg)
    optimizer_name = getattr(cfg.training, "optimizer", "rmsprop").lower()

    if optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
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
            weight_decay=0.0,
        )
        return optimizer, decoder_params

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            weight_decay=0.0,
        )
        return optimizer, decoder_params

    raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")


def build_lr_scheduler(optimizer, cfg):
    """Build the configured epoch-level learning-rate scheduler."""
    scheduler_name = getattr(cfg.training, "lr_scheduler", "cosine").lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=getattr(cfg.training, "lr_min", 0.0),
        )

    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(cfg.training, "lr_step_size", 1),
            gamma=getattr(cfg.training, "lr_gamma", 0.1),
        )

    raise ValueError(f"Unsupported learning-rate scheduler: {cfg.training.lr_scheduler}")


def _copy_tensors_to_cpu(value):
    """Recursively copy tensors to CPU without mutating live training objects."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tensors_to_cpu(item) for item in value)
    return value


def _validate_checkpoint_filename(filename: str) -> None:
    """Reject path escapes and names outside the checkpoint contract."""
    if filename != os.path.basename(filename) or "/" in filename or "\\" in filename:
        raise ValueError(f"Checkpoint filename must be a basename: {filename}")
    if filename == "checkpoint_final.pt":
        return
    if re.fullmatch(r"checkpoint_epoch_\d{4,}\.pt", filename):
        return
    raise ValueError(f"Unsupported checkpoint filename: {filename}")


def build_checkpoint_payload(
    model,
    optimizer,
    lr_scheduler,
    cfg,
    pc_ens,
    hdc_ens,
    epoch: int,
    global_step: int,
):
    """Build a CPU-loadable checkpoint payload for later analysis."""
    scheduler_state_dict = (
        None if lr_scheduler is None else _copy_tensors_to_cpu(lr_scheduler.state_dict())
    )

    return {
        "schema_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": _copy_tensors_to_cpu(model.state_dict()),
        "optimizer_state_dict": _copy_tensors_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": scheduler_state_dict,
        "config": namespace_to_dict(cfg),
        "cell_centers": {
            "pc": [
                torch.as_tensor(ensemble.means, dtype=torch.float32).cpu()
                for ensemble in pc_ens
            ],
            "hdc": [
                torch.as_tensor(ensemble.means, dtype=torch.float32).reshape(-1).cpu()
                for ensemble in hdc_ens
            ],
        },
    }


def save_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    cfg,
    pc_ens,
    hdc_ens,
    epoch: int,
    global_step: int,
    filename: str,
) -> str:
    """Atomically save one training checkpoint under the run directory."""
    _validate_checkpoint_filename(filename)
    checkpoint_dir = os.path.join(cfg.training.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    final_path = os.path.join(checkpoint_dir, filename)
    tmp_path = final_path + ".tmp"
    payload = build_checkpoint_payload(
        model,
        optimizer,
        lr_scheduler,
        cfg,
        pc_ens,
        hdc_ens,
        epoch,
        global_step,
    )
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path


def _resolve_split_path(cfg, explicit_path: str, split_name: str, fallback_attr: str) -> str:
    """Resolve a train/eval split path, preferring configured dataset directories."""
    if explicit_path:
        return explicit_path

    datadir = getattr(cfg.training, "datadir", None)
    if datadir:
        split_path = os.path.join(datadir, f"{split_name}.npz")
        if os.path.exists(split_path):
            return split_path

    return getattr(cfg.training, fallback_attr, None)


def resolve_animation_setting(cfg, name: str, default):
    """Resolve unified animation config from visualization settings."""
    vis_cfg = getattr(cfg, "visualization", None)
    if vis_cfg is not None and hasattr(vis_cfg, name):
        return getattr(vis_cfg, name)
    return default


def build_eval_loader(cfg, logger: logging.Logger, eval_data_path: str = None, get_dataloader_fn=None):
    """Build one fixed evaluation loader, preferring a configured dataset file."""
    get_dataloader_fn = get_dataloader_fn or get_dataloader
    resolved_eval_data_path = _resolve_split_path(
        cfg,
        eval_data_path,
        split_name="eval",
        fallback_attr="eval_data_path",
    )

    eval_loader_batch_size = getattr(
        cfg.training,
        "eval_loader_batch_size",
        cfg.training.batch_size,
    )

    if resolved_eval_data_path and os.path.exists(resolved_eval_data_path):
        logger.info("Loading eval trajectories from %s", resolved_eval_data_path)
        return get_dataloader_fn(
            cfg,
            data_path=resolved_eval_data_path,
            shuffle=False,
            batch_size=eval_loader_batch_size,
        )

    if resolved_eval_data_path:
        logger.warning(
            "Eval dataset %s not found; generating one fixed in-memory eval set with %d trajectories",
            resolved_eval_data_path,
            cfg.training.eval_num_samples,
        )
    else:
        logger.info(
            "Generating one fixed in-memory eval set with %d trajectories",
            cfg.training.eval_num_samples,
        )

    return get_dataloader_fn(
        cfg,
        num_samples=cfg.training.eval_num_samples,
        shuffle=False,
        batch_size=eval_loader_batch_size,
    )


def build_train_loader(
    cfg,
    logger: logging.Logger,
    pc_ens,
    hdc_ens,
    data_path: str = None,
    get_dataloader_fn=None,
):
    """Build one fixed training loader, preferring a configured dataset file."""
    get_dataloader_fn = get_dataloader_fn or get_dataloader
    resolved_data_path = _resolve_split_path(
        cfg,
        data_path,
        split_name="train",
        fallback_attr="data_path",
    )

    if resolved_data_path and os.path.exists(resolved_data_path):
        logger.info("Loading trajectories from %s", resolved_data_path)
        return get_dataloader_fn(
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


def run_evaluation(
    model,
    pc_ens,
    hdc_ens,
    scorer,
    eval_loader,
    cfg,
    device,
    epoch,
    *,
    encode_initial_conditions,
    compute_position_mse,
    compute_head_direction_mae_rad,
    decode_position_from_pc_logits,
    get_scores_and_plot_from_ratemaps,
    plot_hdc_tuning_curves,
    generate_eval_animation,
    score_ratemaps,
    writer=None,
):
    """Collect bottleneck activations, compute grid scores, and export artifacts."""
    hooks = EvaluationHooks(
        encode_initial_conditions=encode_initial_conditions,
        compute_position_mse=compute_position_mse,
        compute_head_direction_mae_rad=compute_head_direction_mae_rad,
        decode_position_from_pc_logits=decode_position_from_pc_logits,
        get_scores_and_plot_from_ratemaps=get_scores_and_plot_from_ratemaps,
        plot_hdc_tuning_curves=plot_hdc_tuning_curves,
        generate_eval_animation=generate_eval_animation,
        score_ratemaps=score_ratemaps,
        get_animation_setting=lambda name, default: resolve_animation_setting(
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
