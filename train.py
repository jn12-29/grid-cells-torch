"""
Training script for the Grid Cells PyTorch reimplementation.

PyTorch port of the original TensorFlow training loop from:
  Banino et al., "Vector-based navigation using grid-like representations
  in artificial agents", Nature 2018.
"""

import logging
import os
import yaml
import argparse
from types import SimpleNamespace

import numpy as np
import torch

from dataset import get_dataloader
from model import GridCellsRNN
from scores import GridScorer
from utils import (
    get_place_cell_ensembles,
    get_head_direction_ensembles,
    encode_initial_conditions,
    encode_targets,
    get_scores_and_plot,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str) -> logging.Logger:
    """Configure a logger that writes to stdout and to a file in log_dir."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")

    logger = logging.getLogger("grid_cells")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger  # already configured (e.g. called twice)

    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
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
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data_path", default=None,
                        help="Path to a pre-generated .npz trajectory file "
                             "(created by generate_data.py). "
                             "If omitted, trajectories are generated on-the-fly "
                             "every epoch.")

    # task overrides
    parser.add_argument("--task.env_size", type=float, dest="task__env_size")
    parser.add_argument("--task.n_pc", type=int, nargs="+", dest="task__n_pc")
    parser.add_argument("--task.pc_scale", type=float, nargs="+", dest="task__pc_scale")
    parser.add_argument("--task.n_hdc", type=int, nargs="+", dest="task__n_hdc")
    parser.add_argument("--task.hdc_concentration", type=float, nargs="+",
                        dest="task__hdc_concentration")

    # model overrides
    parser.add_argument("--model.nh_lstm", type=int, dest="model__nh_lstm")
    parser.add_argument("--model.nh_bottleneck", type=int, dest="model__nh_bottleneck")
    parser.add_argument("--model.dropout_rate", type=float, dest="model__dropout_rate")

    # training overrides
    parser.add_argument("--training.epochs", type=int, dest="training__epochs")
    parser.add_argument("--training.steps_per_epoch", type=int,
                        dest="training__steps_per_epoch")
    parser.add_argument("--training.batch_size", type=int, dest="training__batch_size")
    parser.add_argument("--training.lr", type=float, dest="training__lr")
    parser.add_argument("--training.eval_every", type=int, dest="training__eval_every")
    parser.add_argument("--training.save_dir", type=str, dest="training__save_dir")

    return parser.parse_args()


def _apply_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> SimpleNamespace:
    """Apply non-None CLI overrides to the config namespace in-place."""
    for key, value in vars(args).items():
        if key in ("config", "data_path") or value is None:
            continue
        section, attr = key.split("__", 1)
        setattr(getattr(cfg, section), attr, value)
    return cfg


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(model, pc_ens, hdc_ens, scorer, cfg, device, epoch):
    """Collect bottleneck activations, compute grid scores, and save a PDF."""
    model_was_training = model.training
    model.eval()

    all_bottleneck = []
    all_pos = []

    eval_loader = get_dataloader(cfg)
    n_eval_batches = cfg.training.eval_batch_size // cfg.training.batch_size

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= n_eval_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            init_cond = encode_initial_conditions(batch, pc_ens, hdc_ens).to(device)
            _, _, bottleneck, _ = model(init_cond, batch["ego_vel"], training=False)
            all_bottleneck.append(bottleneck.cpu())
            all_pos.append(batch["target_pos"].cpu())

    bottleneck_np = torch.cat(all_bottleneck, dim=0).numpy()  # (N, T, nh_bottleneck)
    pos_np = torch.cat(all_pos, dim=0).numpy()                # (N, T, 2)

    save_dir = cfg.training.save_dir
    filename = f"rates_and_sac_epoch_{epoch:04d}.pdf"
    scores = get_scores_and_plot(scorer, pos_np, bottleneck_np, save_dir, filename)
    logger = logging.getLogger("grid_cells")
    logger.info("eval epoch=%d  grid_score_60 max=%.4f  grid_score_90 max=%.4f",
                epoch, scores[0].max(), scores[1].max())

    if model_was_training:
        model.train()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg, data_path: str = None):
    """Run the full training loop described in Banino et al., Nature 2018."""
    logger = setup_logger(cfg.training.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Create ensembles
    pc_ens = get_place_cell_ensembles(cfg)
    hdc_ens = get_head_direction_ensembles(cfg)

    # 2. Create model
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model)).to(device)

    # 3. Optimizer: RMSprop
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=cfg.training.lr,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )

    # 4. GridScorer for evaluation
    starts = [0.2] * 10
    ends = list(np.linspace(0.4, 1.0, num=10))
    masks_params = list(zip(starts, ends))
    scorer = GridScorer(
        20,
        [
            [-cfg.task.env_size / 2, cfg.task.env_size / 2],
            [-cfg.task.env_size / 2, cfg.task.env_size / 2],
        ],
        masks_params,
    )

    # 5. Create save directory
    os.makedirs(cfg.training.save_dir, exist_ok=True)

    # 6. Training loop
    if data_path is not None:
        logger.info("Loading trajectories from %s", data_path)
        # Pre-load once; same dataset is reused every epoch (shuffled by DataLoader)
        _fixed_loader = get_dataloader(cfg, data_path=data_path)
    else:
        _fixed_loader = None

    for epoch in range(cfg.training.epochs):
        # Use pre-generated data if provided, otherwise generate fresh each epoch
        dataloader = _fixed_loader if _fixed_loader is not None else get_dataloader(cfg)
        model.train()
        loss_acc = []

        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            init_cond = encode_initial_conditions(batch, pc_ens, hdc_ens).to(device)
            pc_targets, hdc_targets = encode_targets(batch, pc_ens, hdc_ens)

            pc_logits, hdc_logits, _, _ = model(
                init_cond, batch["ego_vel"], training=True
            )

            loss = sum(
                ens.loss(logits, targets)
                for ens, logits, targets in zip(pc_ens, pc_logits, pc_targets)
            )
            loss += sum(
                ens.loss(logits, targets)
                for ens, logits, targets in zip(hdc_ens, hdc_logits, hdc_targets)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.training.grad_clip
            )
            optimizer.step()
            loss_acc.append(loss.item())
            logger.debug("epoch=%4d  step=%4d  loss=%.4f", epoch, step, loss.item())

        logger.info("epoch=%4d  loss mean=%.4f  std=%.4f",
                    epoch, np.mean(loss_acc), np.std(loss_acc))

        if epoch % cfg.training.eval_every == 0:
            _evaluate(model, pc_ens, hdc_ens, scorer, cfg, device, epoch)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    train(cfg, data_path=args.data_path)
