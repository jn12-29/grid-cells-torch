"""CLI parser for the training entrypoint."""

import argparse

from grid_cells.common.config_cli import register_config_overrides


def parse_train_args() -> argparse.Namespace:
    """Parse CLI arguments that can override yaml defaults."""
    parser = argparse.ArgumentParser(
        description="Train grid cells RNN (Banino et al., Nature 2018)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Explicit train split .npz path. When omitted, train.py first "
        "checks training.datadir/train.npz, then falls back to "
        "training.data_path, then on-the-fly generation.",
    )
    parser.add_argument(
        "--eval_data_path",
        default=None,
        help="Explicit eval split .npz path. When omitted, train.py first "
        "checks training.datadir/eval.npz, then falls back to "
        "training.eval_data_path, then one fixed in-memory eval set.",
    )
    register_config_overrides(
        parser,
        sections=("task", "model", "training", "visualization"),
    )

    return parser.parse_args()
