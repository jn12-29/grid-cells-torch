"""Training runtime helpers shared by the train entrypoint wrappers."""

from datetime import datetime
import logging
import math
import os

import torch

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

    raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")


def resolve_animation_setting(cfg, name: str, default):
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


def build_eval_loader(cfg, logger: logging.Logger, eval_data_path: str = None, get_dataloader_fn=None):
    """Build one fixed evaluation loader, preferring a configured dataset file."""
    get_dataloader_fn = get_dataloader_fn or get_dataloader
    resolved_eval_data_path = eval_data_path
    if resolved_eval_data_path is None:
        resolved_eval_data_path = getattr(cfg.training, "eval_data_path", None)

    eval_batch_size = getattr(
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

    return get_dataloader_fn(
        cfg,
        num_samples=cfg.training.eval_batch_size,
        shuffle=False,
        batch_size=eval_batch_size,
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
    resolved_data_path = data_path
    if resolved_data_path is None:
        resolved_data_path = getattr(cfg.training, "data_path", None)

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
