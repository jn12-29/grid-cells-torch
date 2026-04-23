"""Train the grid-cells PyTorch model and export run artifacts."""

import argparse
import logging
import os
from types import SimpleNamespace

import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

from grid_cells.viz.animation import generate_eval_animation
from grid_cells.common.config import (
    apply_namespace_overrides,
    dict_to_namespace,
    load_config as _load_config_impl,
    namespace_to_dict,
    str2bool,
)
from grid_cells.data.dataset import get_dataloader
from grid_cells.cells.encoding_utils import (
    compute_position_mse,
    decode_position_from_pc_logits,
    encode_initial_conditions,
)
from grid_cells.cells.ensemble_utils import (
    get_head_direction_ensembles,
    get_place_cell_ensembles,
)
from grid_cells.analysis.scoring_utils import (
    get_scores_and_plot_from_ratemaps,
    plot_hdc_tuning_curves,
    score_ratemaps,
)
from grid_cells.training.cli import parse_train_args
from grid_cells.training.session import TrainingSession, TrainingSessionHooks
from grid_cells.training.runtime import (
    build_eval_loader as _build_eval_loader_impl,
    build_optimizer,
    build_train_loader as _build_train_loader_impl,
    get_step_log_interval,
    resolve_animation_setting,
    resolve_save_dir,
    run_evaluation,
    setup_logger,
)


def _dict_to_namespace(data: dict) -> SimpleNamespace:
    """Backward-compatible wrapper for nested namespace conversion."""
    return dict_to_namespace(data)


def _namespace_to_dict(obj):
    """Backward-compatible wrapper for namespace serialization."""
    return namespace_to_dict(obj)


def _str2bool(value):
    """Backward-compatible wrapper for CLI boolean parsing."""
    return str2bool(value)


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


def load_config(path: str = "config.yaml") -> SimpleNamespace:
    """Load YAML config and return it as a nested namespace."""
    return _load_config_impl(path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments that can override yaml defaults."""
    return parse_train_args()


def _apply_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> SimpleNamespace:
    """Apply non-None CLI overrides to the config namespace in-place."""
    return apply_namespace_overrides(cfg, args)


def _get_animation_setting(cfg: SimpleNamespace, name: str, default):
    """Resolve unified animation config, with fallback to legacy eval keys."""
    return resolve_animation_setting(cfg, name, default)


def _evaluate(
    model,
    pc_ens,
    hdc_ens,
    scorer,
    eval_loader,
    cfg,
    device,
    epoch,
    writer=None,
):
    """Collect bottleneck activations, compute grid scores, and export artifacts."""
    return run_evaluation(
        model,
        pc_ens,
        hdc_ens,
        scorer,
        eval_loader,
        cfg,
        device,
        epoch,
        encode_initial_conditions=encode_initial_conditions,
        compute_position_mse=compute_position_mse,
        decode_position_from_pc_logits=decode_position_from_pc_logits,
        get_scores_and_plot_from_ratemaps=get_scores_and_plot_from_ratemaps,
        plot_hdc_tuning_curves=plot_hdc_tuning_curves,
        generate_eval_animation=generate_eval_animation,
        score_ratemaps=score_ratemaps,
        writer=writer,
    )


def _build_eval_loader(cfg, logger: logging.Logger, eval_data_path: str = None):
    """Build one fixed evaluation loader, preferring a configured dataset file."""
    return _build_eval_loader_impl(
        cfg,
        logger,
        eval_data_path=eval_data_path,
        get_dataloader_fn=get_dataloader,
    )


def _build_train_loader(
    cfg,
    logger: logging.Logger,
    pc_ens,
    hdc_ens,
    data_path: str = None,
):
    """Build one fixed training loader, preferring a configured dataset file."""
    return _build_train_loader_impl(
        cfg,
        logger,
        pc_ens,
        hdc_ens,
        data_path=data_path,
        get_dataloader_fn=get_dataloader,
    )


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


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    train(cfg, data_path=args.data_path, eval_data_path=args.eval_data_path)
