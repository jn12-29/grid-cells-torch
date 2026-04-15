"""Tests for generate_data helpers and CLI flow."""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import generate_data


def test_generate_dataset_file_writes_npz(tmp_path):
    """Helper should generate the requested dataset file."""
    output_path = tmp_path / "train.npz"

    dataset = generate_data.generate_dataset_file(
        output_path=str(output_path),
        num_samples=6,
        seq_len=5,
        env_size=2.2,
        velocity_noise=[0.0, 0.0, 0.0],
        seed=3,
    )

    assert output_path.exists()
    assert dataset.num_samples == 6


def test_main_can_generate_train_and_eval_splits(tmp_path, monkeypatch):
    """CLI entry point should support writing train and eval files in one run."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = SimpleNamespace(
        task=SimpleNamespace(seq_len=6, env_size=2.2, neurons_seed=5, velocity_noise=[0.0, 0.0, 0.0]),
        training=SimpleNamespace(steps_per_epoch=2, batch_size=3, eval_batch_size=4),
    )

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: SimpleNamespace(
            config="config.yaml",
            output=str(train_path),
            num_samples=None,
            seq_len=None,
            env_size=None,
            seed=None,
            visualize=False,
            vis_output=None,
            eval_output=str(eval_path),
            eval_num_samples=None,
            eval_seed=None,
        ),
    )

    generate_data.main()

    train_meta = np.load(train_path, allow_pickle=False)["meta"].item()
    eval_meta = np.load(eval_path, allow_pickle=False)["meta"].item()
    assert '"num_samples": 6' in train_meta
    assert '"num_samples": 4' in eval_meta
