"""Regression tests for dataset generation helpers and CLI flow.

Usage:
    pytest tests/test_generate_data.py
    pytest tests/test_generate_data.py -k animation
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import generate_data
from ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble


def _make_cfg(train_path: str, eval_path: str = None):
    return SimpleNamespace(
        task=SimpleNamespace(
            seq_len=6,
            env_size=2.2,
            neurons_seed=5,
            velocity_noise=[0.0, 0.0, 0.0],
            targets_type="softmax",
            lstm_init_type="softmax",
            n_pc=[8],
            pc_scale=[0.35],
            n_hdc=[6],
            hdc_concentration=[20.0],
        ),
        training=SimpleNamespace(
            steps_per_epoch=2,
            batch_size=3,
            eval_batch_size=4,
            data_path=train_path,
            eval_data_path=eval_path,
        ),
        visualization=SimpleNamespace(
            spatial_bins=32,
            directional_bins=20,
            anim_num_traj=4,
            anim_fps=20,
            anim_step=4,
            anim_workers=4,
        ),
    )


def _make_args(**overrides):
    defaults = dict(
        config="config.yaml",
        output=None,
        num_samples=None,
        seq_len=None,
        env_size=None,
        seed=None,
        visualize=False,
        animate=False,
        vis_output=None,
        anim_output=None,
        anim_num_traj=None,
        anim_fps=None,
        anim_step=None,
        anim_workers=None,
        num_workers=1,
        visualize_progress=False,
        progress_output=None,
        eval_progress_output=None,
        progress_every=4,
        spatial_bins=None,
        directional_bins=None,
        eval_output=None,
        train_only=False,
        eval_num_samples=None,
        eval_seed=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


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


def test_generate_dataset_file_writes_progress_preview(tmp_path):
    """Progress visualisation should be written when requested."""
    output_path = tmp_path / "train.npz"
    progress_path = tmp_path / "train_progress.png"

    generate_data.generate_dataset_file(
        output_path=str(output_path),
        num_samples=6,
        seq_len=5,
        env_size=2.2,
        velocity_noise=[0.0, 0.0, 0.0],
        seed=3,
        progress_output=str(progress_path),
        progress_every=1,
    )

    assert progress_path.exists()


def test_generate_dataset_file_can_write_animation(tmp_path, monkeypatch):
    """Animation export should be called with the requested MP4 path."""
    output_path = tmp_path / "train.npz"
    anim_path = tmp_path / "train_traj.mp4"
    calls = []
    pc_ens = [PlaceCellEnsemble(4, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(6, concentration=20.0, seed=0)]

    def fake_visualize_animation(
        dataset,
        save_path,
        pc_ensembles,
        hdc_ensembles,
        fps=20,
        max_trajectories=4,
        step=4,
        num_workers=8,
    ):
        calls.append(
            (
                dataset.num_samples,
                save_path,
                len(pc_ensembles),
                len(hdc_ensembles),
                fps,
                max_trajectories,
                step,
                num_workers,
            )
        )
        anim_path.write_bytes(b"fake-mp4")

    monkeypatch.setattr(generate_data, "visualize_animation", fake_visualize_animation)

    generate_data.generate_dataset_file(
        output_path=str(output_path),
        num_samples=6,
        seq_len=5,
        env_size=2.2,
        velocity_noise=[0.0, 0.0, 0.0],
        seed=3,
        pc_ensembles=pc_ens,
        hdc_ensembles=hdc_ens,
        animation_output=str(anim_path),
        animation_num_trajectories=3,
        animation_fps=12,
        animation_step=2,
        anim_workers=5,
    )

    assert anim_path.exists()
    assert calls == [(6, str(anim_path), 1, 1, 12, 3, 2, 5)]


def test_resolve_visualization_bins_prefers_cli_then_config():
    """CLI overrides should win over config for visualization bins."""
    cfg = SimpleNamespace(
        visualization=SimpleNamespace(spatial_bins=41, directional_bins=17)
    )
    args = SimpleNamespace(spatial_bins=75, directional_bins=None)

    resolved = generate_data._resolve_visualization_bins(cfg, args)

    assert resolved == {"spatial_bins": 75, "directional_bins": 17}


def test_resolve_visualization_bins_rejects_non_positive_values():
    """Visualization bins must remain positive integers."""
    cfg = SimpleNamespace(visualization=SimpleNamespace(spatial_bins=0))
    args = SimpleNamespace(spatial_bins=None, directional_bins=None)

    with pytest.raises(ValueError, match="spatial_bins must be a positive integer"):
        generate_data._resolve_visualization_bins(cfg, args)


def test_resolve_animation_settings_prefers_cli_then_config_then_legacy():
    """Animation settings should use CLI overrides, then visualization, then legacy eval keys."""
    cfg = SimpleNamespace(
        visualization=SimpleNamespace(anim_num_traj=4, anim_fps=24, anim_step=5),
        training=SimpleNamespace(eval_anim_workers=7),
    )
    args = SimpleNamespace(
        anim_num_traj=2,
        anim_fps=None,
        anim_step=None,
        anim_workers=None,
    )

    resolved = generate_data._resolve_animation_settings(cfg, args)

    assert resolved == {
        "anim_num_traj": 2,
        "anim_fps": 24,
        "anim_step": 5,
        "anim_workers": 7,
    }


def test_resolve_animation_settings_rejects_non_positive_values():
    """Animation settings must remain positive integers."""
    cfg = SimpleNamespace(visualization=SimpleNamespace(anim_step=0), training=SimpleNamespace())
    args = SimpleNamespace(
        anim_num_traj=None,
        anim_fps=None,
        anim_step=None,
        anim_workers=None,
    )

    with pytest.raises(ValueError, match="anim_step must be a positive integer"):
        generate_data._resolve_animation_settings(cfg, args)


def test_visualize_animation_prepares_eval_style_inputs(tmp_path, monkeypatch):
    """Animation export should prepare eval-style inputs and call the shared renderer."""
    output_path = tmp_path / "train.npz"
    anim_path = tmp_path / "train_traj.mp4"
    dataset = generate_data.generate_dataset_file(
        output_path=str(output_path),
        num_samples=4,
        seq_len=5,
        env_size=2.2,
        velocity_noise=[0.0, 0.0, 0.0],
        seed=3,
    )
    pc_ens = [PlaceCellEnsemble(5, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(6, concentration=20.0, seed=0)]
    captured = {}

    def fake_generate_trajectory_animation(
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        env_size,
        save_path,
        fps=20,
        step=4,
        num_workers=4,
        title_prefix="Trajectory",
        pred_label="predicted",
    ):
        captured.update(
            {
                "target_pos": target_pos,
                "pred_pos": pred_pos,
                "pc_acts": pc_acts,
                "hdc_acts": hdc_acts,
                "pc_centers": pc_centers,
                "hdc_centers": hdc_centers,
                "env_size": env_size,
                "save_path": save_path,
                "fps": fps,
                "step": step,
                "num_workers": num_workers,
                "title_prefix": title_prefix,
                "pred_label": pred_label,
            }
        )
        anim_path.write_bytes(b"fake-mp4")

    monkeypatch.setattr(generate_data, "generate_trajectory_animation", fake_generate_trajectory_animation)

    generate_data.visualize_animation(
        dataset,
        str(anim_path),
        pc_ens,
        hdc_ens,
        fps=12,
        max_trajectories=3,
        step=2,
        num_workers=2,
    )

    assert anim_path.exists()
    assert captured["target_pos"].shape == (3, 5, 2)
    assert captured["pred_pos"].shape == (3, 5, 2)
    assert captured["pc_acts"].shape == (3, 5, 5)
    assert captured["hdc_acts"].shape == (3, 5, 6)
    assert captured["pc_centers"].shape == (5, 2)
    assert captured["hdc_centers"].shape == (6,)
    assert captured["save_path"] == str(anim_path)
    assert captured["fps"] == 12
    assert captured["step"] == 2
    assert captured["num_workers"] == 2
    assert captured["title_prefix"] == "Data traj"
    assert captured["pred_label"] == "decoded"


def test_main_defaults_output_to_config_train_and_eval_paths(tmp_path, monkeypatch):
    """CLI should default to config train/eval output paths."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = _make_cfg(str(train_path), str(eval_path))

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(generate_data, "parse_args", lambda: _make_args())

    generate_data.main()

    train_meta = np.load(train_path, allow_pickle=False)["meta"].item()
    eval_meta = np.load(eval_path, allow_pickle=False)["meta"].item()
    assert '"num_samples": 6' in train_meta
    assert '"num_samples": 4' in eval_meta


def test_main_train_only_skips_default_eval_output(tmp_path, monkeypatch):
    """CLI should allow generating only the main split."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = _make_cfg(str(train_path), str(eval_path))

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(generate_data, "parse_args", lambda: _make_args(train_only=True))

    generate_data.main()

    assert train_path.exists()
    assert not eval_path.exists()


def test_main_can_generate_train_and_eval_splits(tmp_path, monkeypatch):
    """CLI entry point should support writing train and eval files in one run."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = _make_cfg(str(train_path), str(tmp_path / "config-eval.npz"))

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output=str(train_path), eval_output=str(eval_path)),
    )

    generate_data.main()

    train_meta = np.load(train_path, allow_pickle=False)["meta"].item()
    eval_meta = np.load(eval_path, allow_pickle=False)["meta"].item()
    assert '"num_samples": 6' in train_meta
    assert '"num_samples": 4' in eval_meta


def test_main_rejects_same_train_and_eval_output_paths(tmp_path, monkeypatch):
    """Train and eval outputs should not resolve to the same path."""
    shared_path = tmp_path / "shared.npz"
    cfg = _make_cfg(str(shared_path), str(shared_path))

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(generate_data, "parse_args", lambda: _make_args())

    with pytest.raises(ValueError, match="Train and eval output paths must be different"):
        generate_data.main()


def test_main_rejects_train_only_with_eval_output(tmp_path, monkeypatch):
    """Conflicting train-only/eval flags should be rejected."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = _make_cfg(str(train_path), str(eval_path))

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(eval_output=str(eval_path), train_only=True),
    )

    with pytest.raises(ValueError, match="Cannot combine --train_only with --eval_output"):
        generate_data.main()
