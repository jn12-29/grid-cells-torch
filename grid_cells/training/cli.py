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
    register_config_overrides(
        parser,
        sections=("task", "model", "training", "visualization"),
    )

    return parser.parse_args()
