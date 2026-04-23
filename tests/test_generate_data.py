"""Regression tests for dataset generation helpers and CLI flow.

Usage:
    pytest tests/test_generate_data.py
    pytest tests/test_generate_data.py -k animation
"""
import os
import sys
import json
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import generate_data
from grid_cells.cells.ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble


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
        data_generation=SimpleNamespace(
            num_samples=None,
            eval_num_samples=None,
            num_workers=8,
            progress_every=4,
            vis_output=None,
            anim_output=None,
            progress_output=None,
            eval_progress_output=None,
        ),
    )


def _make_args(**overrides):
    defaults = dict(
        config="config.yaml",
        output=None,
        num_samples=None,
        eval_num_samples=None,
        seq_len=None,
        env_size=None,
        seed=None,
        eval_seed=None,
        vis_output=None,
        anim_output=None,
        progress_output=None,
        eval_progress_output=None,
        anim_num_traj=None,
        anim_fps=None,
        anim_step=None,
        anim_workers=None,
        num_workers=None,
        progress_every=None,
        spatial_bins=None,
        directional_bins=None,
        output_dir=None,
        tag=None,
        visualize=False,
        animate=False,
        visualize_progress=False,
        eval_output=None,
        train_only=False,
        task__seq_len=None,
        task__env_size=None,
        task__neurons_seed=None,
        training__data_path=None,
        training__eval_data_path=None,
        visualization__spatial_bins=None,
        visualization__directional_bins=None,
        visualization__anim_num_traj=None,
        visualization__anim_fps=None,
        visualization__anim_step=None,
        visualization__anim_workers=None,
        data_generation__num_samples=None,
        data_generation__eval_num_samples=None,
        data_generation__num_workers=None,
        data_generation__progress_every=None,
        data_generation__vis_output=None,
        data_generation__anim_output=None,
        data_generation__progress_output=None,
        data_generation__eval_progress_output=None,
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
        spatial_bins=32,
        directional_bins=20,
    )

    assert progress_path.exists()


def test_generate_dataset_file_writes_progress_preview_with_default_bins(tmp_path):
    """Direct helper callers should still get safe bin defaults for progress previews."""
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


def test_resolve_visualization_bins_falls_back_to_safe_defaults():
    """Visualization bins should fall back to safe defaults for older configs."""
    cfg = SimpleNamespace(visualization=SimpleNamespace(spatial_bins=32))
    args = SimpleNamespace(spatial_bins=None, directional_bins=None)

    resolved = generate_data._resolve_visualization_bins(cfg, args)

    assert resolved == {"spatial_bins": 32, "directional_bins": 20}


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


def test_main_explicit_outputs_use_legacy_single_file_mode(tmp_path, monkeypatch):
    """Explicit output paths should preserve the legacy single-file mode."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = _make_cfg(str(tmp_path / "config-train.npz"), str(tmp_path / "config-eval.npz"))

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


def test_main_explicit_train_output_does_not_touch_latest_or_default_eval(
    tmp_path,
    monkeypatch,
):
    """Single-file mode with only --output should generate only train and avoid data/latest."""
    train_path = tmp_path / "train-only-single.npz"
    cfg = _make_cfg(
        str(tmp_path / "config-train.npz"),
        str(tmp_path / "data" / "latest" / "eval.npz"),
    )
    latest_dir = tmp_path / "data" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    stale_eval = latest_dir / "eval.npz"
    stale_meta = latest_dir / "meta.json"
    stale_eval.write_bytes(b"stale-eval")
    stale_meta.write_text('{"stale": true}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output=str(train_path)),
    )

    generate_data.main()

    assert train_path.exists()
    assert stale_eval.read_bytes() == b"stale-eval"
    assert stale_meta.read_text(encoding="utf-8") == '{"stale": true}'


def test_main_rejects_eval_output_without_explicit_train_output(tmp_path, monkeypatch):
    """Explicit eval output should require an explicit train output too."""
    eval_path = tmp_path / "eval-only.npz"
    cfg = _make_cfg(
        str(tmp_path / "data" / "latest" / "train.npz"),
        str(tmp_path / "config-eval.npz"),
    )

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(eval_output=str(eval_path)),
    )

    with pytest.raises(ValueError, match="Explicit eval output requires an explicit train output"):
        generate_data.main()


def test_main_rejects_training_eval_path_override_without_explicit_train_output(
    tmp_path,
    monkeypatch,
):
    """training.eval_data_path override alone should require explicit train output too."""
    eval_path = tmp_path / "eval-only.npz"
    cfg = _make_cfg(
        str(tmp_path / "data" / "latest" / "train.npz"),
        str(tmp_path / "config-eval.npz"),
    )

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(training__eval_data_path=str(eval_path)),
    )

    with pytest.raises(ValueError, match="Explicit eval output requires an explicit train output"):
        generate_data.main()


def test_main_accepts_legacy_single_file_cli_arguments(tmp_path, monkeypatch):
    """Legacy flat CLI flags should still parse and drive single-file outputs."""
    train_path = tmp_path / "legacy-train.npz"
    eval_path = tmp_path / "legacy-eval.npz"
    vis_path = tmp_path / "legacy-vis.pdf"
    anim_path = tmp_path / "legacy-traj.mp4"
    progress_path = tmp_path / "legacy-progress.png"
    cfg = _make_cfg(str(tmp_path / "config-train.npz"), str(tmp_path / "config-eval.npz"))

    def fake_visualize(dataset, save_path, spatial_bins=32, directional_bins=20):
        assert save_path == str(vis_path)
        vis_path.write_bytes(b"fake-pdf")

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
        assert save_path == str(anim_path)
        anim_path.write_bytes(b"fake-mp4")

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(generate_data, "visualize", fake_visualize)
    monkeypatch.setattr(generate_data, "visualize_animation", fake_visualize_animation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_data.py",
            "--output",
            str(train_path),
            "--eval_output",
            str(eval_path),
            "--num_samples",
            "5",
            "--eval_num_samples",
            "2",
            "--seq_len",
            "7",
            "--env_size",
            "3.0",
            "--seed",
            "11",
            "--vis_output",
            str(vis_path),
            "--anim_output",
            str(anim_path),
            "--progress_output",
            str(progress_path),
            "--visualize",
            "--animate",
            "--visualize_progress",
        ],
    )

    generate_data.main()

    assert train_path.exists()
    assert eval_path.exists()
    assert vis_path.exists()
    assert anim_path.exists()
    assert progress_path.exists()
    train_meta = np.load(train_path, allow_pickle=False)["meta"].item()
    eval_meta = np.load(eval_path, allow_pickle=False)["meta"].item()
    train_data = np.load(train_path, allow_pickle=False)
    eval_data = np.load(eval_path, allow_pickle=False)
    assert '"num_samples": 5' in train_meta
    assert '"num_samples": 2' in eval_meta
    assert '"seq_len": 7' in train_meta
    assert '"seq_len": 7' in eval_meta
    assert '"env_size": 3.0' in train_meta
    assert '"env_size": 3.0' in eval_meta

    ref_train_path = tmp_path / "ref-train.npz"
    ref_eval_path = tmp_path / "ref-eval.npz"
    generate_data.generate_dataset_file(
        output_path=str(ref_train_path),
        num_samples=5,
        seq_len=7,
        env_size=3.0,
        velocity_noise=[0.0, 0.0, 0.0],
        seed=11,
    )
    generate_data.generate_dataset_file(
        output_path=str(ref_eval_path),
        num_samples=2,
        seq_len=7,
        env_size=3.0,
        velocity_noise=[0.0, 0.0, 0.0],
        seed=12,
    )
    ref_train_data = np.load(ref_train_path, allow_pickle=False)
    ref_eval_data = np.load(ref_eval_path, allow_pickle=False)
    assert np.array_equal(train_data["target_pos"], ref_train_data["target_pos"])
    assert np.array_equal(eval_data["target_pos"], ref_eval_data["target_pos"])


def test_main_legacy_cli_visualization_bins_are_applied(tmp_path, monkeypatch):
    """Legacy flat visualization bin flags should reach the visualize callback."""
    train_path = tmp_path / "legacy-bins-train.npz"
    cfg = _make_cfg(str(tmp_path / "config-train.npz"), None)
    captured = {}

    def fake_visualize(dataset, save_path, spatial_bins=32, directional_bins=20):
        captured["save_path"] = save_path
        captured["spatial_bins"] = spatial_bins
        captured["directional_bins"] = directional_bins
        tmp_path.joinpath("legacy-bins-vis.pdf").write_bytes(b"fake-pdf")

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(generate_data, "visualize", fake_visualize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_data.py",
            "--output",
            str(train_path),
            "--train_only",
            "--visualize",
            "--spatial_bins",
            "41",
            "--directional_bins",
            "23",
        ],
    )

    generate_data.main()

    assert captured["save_path"] == str(tmp_path / "legacy-bins-train_vis.pdf")
    assert captured["spatial_bins"] == 41
    assert captured["directional_bins"] == 23


def test_main_legacy_cli_animation_settings_are_applied(tmp_path, monkeypatch):
    """Legacy flat animation flags should reach the animation callback."""
    train_path = tmp_path / "legacy-anim-train.npz"
    cfg = _make_cfg(str(tmp_path / "config-train.npz"), None)
    captured = {}

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
        captured["save_path"] = save_path
        captured["fps"] = fps
        captured["max_trajectories"] = max_trajectories
        captured["step"] = step
        captured["num_workers"] = num_workers
        tmp_path.joinpath("legacy-anim-traj.mp4").write_bytes(b"fake-mp4")

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(generate_data, "visualize_animation", fake_visualize_animation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_data.py",
            "--output",
            str(train_path),
            "--train_only",
            "--animate",
            "--anim_num_traj",
            "3",
            "--anim_fps",
            "12",
            "--anim_step",
            "2",
            "--anim_workers",
            "5",
        ],
    )

    generate_data.main()

    assert captured["save_path"] == str(tmp_path / "legacy-anim-train_traj.mp4")
    assert captured["fps"] == 12
    assert captured["max_trajectories"] == 3
    assert captured["step"] == 2
    assert captured["num_workers"] == 5


def test_main_training_section_overrides_use_legacy_single_file_mode(
    tmp_path,
    monkeypatch,
):
    """training.* CLI overrides should trigger the legacy single-file mode."""
    train_path = tmp_path / "override-train.npz"
    eval_path = tmp_path / "override-eval.npz"
    cfg = _make_cfg(str(tmp_path / "config-train.npz"), str(tmp_path / "config-eval.npz"))
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(
            training__data_path=str(train_path),
            training__eval_data_path=str(eval_path),
        ),
    )

    generate_data.main()

    train_meta = np.load(train_path, allow_pickle=False)["meta"].item()
    eval_meta = np.load(eval_path, allow_pickle=False)["meta"].item()
    assert '"num_samples": 6' in train_meta
    assert '"num_samples": 4' in eval_meta
    assert not (tmp_path / "data" / "datasets").exists()


def test_main_directory_mode_defaults_to_dataset_id_directory(tmp_path, monkeypatch):
    """Without explicit output files, CLI should derive a dataset directory and standard files."""
    train_path = tmp_path / "config-train.npz"
    eval_path = tmp_path / "config-eval.npz"
    cfg = _make_cfg(str(train_path), str(eval_path))
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(generate_data, "parse_args", lambda: _make_args())

    generate_data.main()

    datasets_dir = tmp_path / "data" / "datasets"
    created_dirs = [p for p in datasets_dir.iterdir() if p.is_dir()]
    assert len(created_dirs) == 1
    output_dir = created_dirs[0]
    assert "train6" in output_dir.name
    assert "eval4" in output_dir.name
    assert "seq6" in output_dir.name
    assert "seed5" in output_dir.name
    assert "env2p2" in output_dir.name
    assert (output_dir / "train.npz").exists()
    assert (output_dir / "eval.npz").exists()
    assert (output_dir / "meta.json").exists()
    assert (output_dir / "README.txt").exists()


def test_main_directory_mode_uses_timestamped_default_dirs_for_repeated_runs(
    tmp_path,
    monkeypatch,
):
    """Default directory mode should avoid collisions for repeated runs within one second."""
    cfg = _make_cfg(
        str(tmp_path / "config-train.npz"),
        str(tmp_path / "config-eval.npz"),
    )
    timestamps = iter(["20260423-100001-123456", "20260423-100001-123457"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(generate_data, "parse_args", lambda: _make_args())
    monkeypatch.setattr(
        generate_data,
        "_current_dataset_timestamp",
        lambda: next(timestamps),
        raising=False,
    )

    generate_data.main()
    generate_data.main()

    datasets_dir = tmp_path / "data" / "datasets"
    created_dirs = sorted(p.name for p in datasets_dir.iterdir() if p.is_dir())
    assert len(created_dirs) == 2
    assert created_dirs[0] != created_dirs[1]
    assert "train6" in created_dirs[0]
    assert "eval4" in created_dirs[0]
    assert "seq6" in created_dirs[0]
    assert "seed5" in created_dirs[0]
    assert "env2p2" in created_dirs[0]
    assert "20260423-100001-123456" in created_dirs[0]
    assert "20260423-100001-123457" in created_dirs[1]


def test_main_directory_mode_honors_explicit_output_dir(tmp_path, monkeypatch):
    """Directory mode should write standard files directly into an explicit output_dir."""
    cfg = _make_cfg(
        str(tmp_path / "config-train.npz"),
        str(tmp_path / "config-eval.npz"),
    )
    output_dir = tmp_path / "baseline-dir"

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output_dir=str(output_dir), tag="baseline"),
    )

    generate_data.main()

    assert output_dir.exists()
    assert (output_dir / "train.npz").exists()
    assert (output_dir / "eval.npz").exists()
    assert (output_dir / "meta.json").exists()
    assert (output_dir / "README.txt").exists()


def test_main_directory_mode_writes_effective_metadata(tmp_path, monkeypatch):
    """Directory mode should record effective run parameters and relative paths in meta.json."""
    cfg = _make_cfg(
        str(tmp_path / "config-train.npz"),
        str(tmp_path / "config-eval.npz"),
    )
    output_dir = tmp_path / "metadata-dir"

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(
            output_dir=str(output_dir),
            tag="baseline",
            task__seq_len=8,
            task__env_size=3.5,
            data_generation__num_samples=9,
            data_generation__eval_num_samples=2,
        ),
    )

    generate_data.main()

    meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["tag"] == "baseline"
    assert meta["task"]["seq_len"] == 8
    assert meta["task"]["env_size"] == 3.5
    assert meta["task"]["velocity_noise"] == [0.0, 0.0, 0.0]
    assert meta["generation"]["train_seed"] == 5
    assert meta["generation"]["eval_seed"] == 6
    assert meta["generation"]["num_samples"] == 9
    assert meta["generation"]["eval_num_samples"] == 2
    assert meta["paths"]["train"] == "train.npz"
    assert meta["paths"]["eval"] == "eval.npz"


def test_main_directory_mode_uses_fixed_artifact_names_and_readme_listing(
    tmp_path,
    monkeypatch,
):
    """Directory mode should place optional artifacts under fixed dataset-relative names."""
    cfg = _make_cfg(
        str(tmp_path / "config-train.npz"),
        str(tmp_path / "config-eval.npz"),
    )
    output_dir = tmp_path / "artifact-dir"
    captured = {}

    def fake_visualize(dataset, save_path, spatial_bins=32, directional_bins=20):
        captured["train_vis"] = save_path
        output_dir.joinpath(os.path.basename(save_path)).write_bytes(b"fake-pdf")

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
        captured["train_anim"] = save_path
        output_dir.joinpath(os.path.basename(save_path)).write_bytes(b"fake-mp4")

    monkeypatch.setattr(generate_data, "visualize", fake_visualize)
    monkeypatch.setattr(generate_data, "visualize_animation", fake_visualize_animation)
    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(
            output_dir=str(output_dir),
            visualize=True,
            animate=True,
            visualize_progress=True,
        ),
    )

    generate_data.main()

    assert captured["train_vis"] == str(output_dir / "train_vis.pdf")
    assert captured["train_anim"] == str(output_dir / "train_traj.mp4")
    assert (output_dir / "train_vis.pdf").exists()
    assert (output_dir / "train_traj.mp4").exists()
    assert (output_dir / "train_progress.png").exists()
    assert (output_dir / "eval_progress.png").exists()

    readme_text = (output_dir / "README.txt").read_text(encoding="utf-8")
    assert "train.npz" in readme_text
    assert "eval.npz" in readme_text
    assert "meta.json" in readme_text
    assert "train_vis.pdf" in readme_text
    assert "train_traj.mp4" in readme_text
    assert "train_progress.png" in readme_text
    assert "eval_progress.png" in readme_text


def test_main_directory_mode_updates_latest_stable_entries(tmp_path, monkeypatch):
    """Directory mode should sync train/eval/meta into data/latest after success."""
    cfg = _make_cfg(
        str(tmp_path / "config-train.npz"),
        str(tmp_path / "config-eval.npz"),
    )
    output_dir = tmp_path / "dataset-dir"
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output_dir=str(output_dir)),
    )

    generate_data.main()

    latest_dir = tmp_path / "data" / "latest"
    latest_train = latest_dir / "train.npz"
    latest_eval = latest_dir / "eval.npz"
    latest_meta = latest_dir / "meta.json"
    assert latest_train.exists()
    assert latest_eval.exists()
    assert latest_meta.exists()

    latest_train_data = np.load(latest_train, allow_pickle=False)
    latest_eval_data = np.load(latest_eval, allow_pickle=False)
    dataset_train_data = np.load(output_dir / "train.npz", allow_pickle=False)
    dataset_eval_data = np.load(output_dir / "eval.npz", allow_pickle=False)
    assert np.array_equal(latest_train_data["target_pos"], dataset_train_data["target_pos"])
    assert np.array_equal(latest_eval_data["target_pos"], dataset_eval_data["target_pos"])
    assert json.loads(latest_meta.read_text(encoding="utf-8")) == json.loads(
        (output_dir / "meta.json").read_text(encoding="utf-8")
    )


def test_main_directory_mode_train_only_updates_latest_and_clears_eval(tmp_path, monkeypatch):
    """train_only should refresh latest train/meta and remove any stale latest eval entry."""
    cfg = _make_cfg(
        str(tmp_path / "config-train.npz"),
        str(tmp_path / "config-eval.npz"),
    )
    output_dir = tmp_path / "train-only-dir"
    latest_dir = tmp_path / "data" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    stale_eval = latest_dir / "eval.npz"
    stale_eval.write_bytes(b"stale")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output_dir=str(output_dir), train_only=True),
    )

    generate_data.main()

    latest_train = latest_dir / "train.npz"
    latest_meta = latest_dir / "meta.json"
    assert latest_train.exists()
    assert latest_meta.exists()
    assert not stale_eval.exists()

    latest_train_data = np.load(latest_train, allow_pickle=False)
    dataset_train_data = np.load(output_dir / "train.npz", allow_pickle=False)
    assert np.array_equal(latest_train_data["target_pos"], dataset_train_data["target_pos"])
    assert json.loads(latest_meta.read_text(encoding="utf-8")) == json.loads(
        (output_dir / "meta.json").read_text(encoding="utf-8")
    )


def test_main_directory_mode_latest_output_dir_does_not_self_reference(tmp_path, monkeypatch):
    """Using data/latest as output_dir should leave concrete files in place, not self-links."""
    cfg = _make_cfg(
        str(tmp_path / "config-train.npz"),
        str(tmp_path / "config-eval.npz"),
    )
    output_dir = tmp_path / "data" / "latest"
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output_dir=str(output_dir)),
    )

    generate_data.main()

    train_path = output_dir / "train.npz"
    eval_path = output_dir / "eval.npz"
    meta_path = output_dir / "meta.json"
    assert train_path.exists()
    assert eval_path.exists()
    assert meta_path.exists()
    assert not train_path.is_symlink()
    assert not eval_path.is_symlink()
    assert not meta_path.is_symlink()


def test_main_directory_mode_latest_falls_back_to_copy_when_symlink_fails(
    tmp_path,
    monkeypatch,
):
    """latest sync should copy files when symlink creation fails."""
    cfg = _make_cfg(
        str(tmp_path / "config-train.npz"),
        str(tmp_path / "config-eval.npz"),
    )
    output_dir = tmp_path / "dataset-dir"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output_dir=str(output_dir)),
    )

    def fail_symlink(src, dst):
        raise OSError("symlink unsupported")

    monkeypatch.setattr(generate_data.os, "symlink", fail_symlink)

    generate_data.main()

    latest_dir = tmp_path / "data" / "latest"
    latest_train = latest_dir / "train.npz"
    latest_meta = latest_dir / "meta.json"
    assert latest_train.exists()
    assert latest_meta.exists()
    assert not latest_train.is_symlink()
    assert not latest_meta.is_symlink()
    assert latest_train.read_bytes() == (output_dir / "train.npz").read_bytes()
    assert latest_meta.read_text(encoding="utf-8") == (output_dir / "meta.json").read_text(
        encoding="utf-8"
    )


def test_main_directory_mode_ignores_legacy_artifact_paths(tmp_path, monkeypatch):
    """Directory mode should ignore legacy artifact path flags and keep outputs inside output_dir."""
    cfg = _make_cfg(
        str(tmp_path / "config-train.npz"),
        str(tmp_path / "config-eval.npz"),
    )
    output_dir = tmp_path / "artifact-dir"
    outside_vis = tmp_path / "outside-vis.pdf"
    outside_anim = tmp_path / "outside-traj.mp4"
    outside_progress = tmp_path / "outside-progress.png"
    outside_eval_progress = tmp_path / "outside-eval-progress.png"
    captured = {}

    def fake_visualize(dataset, save_path, spatial_bins=32, directional_bins=20):
        captured["train_vis"] = save_path
        output_dir.joinpath(os.path.basename(save_path)).write_bytes(b"fake-pdf")

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
        captured["train_anim"] = save_path
        output_dir.joinpath(os.path.basename(save_path)).write_bytes(b"fake-mp4")

    monkeypatch.setattr(generate_data, "visualize", fake_visualize)
    monkeypatch.setattr(generate_data, "visualize_animation", fake_visualize_animation)
    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(
            output_dir=str(output_dir),
            visualize=True,
            animate=True,
            visualize_progress=True,
            vis_output=str(outside_vis),
            anim_output=str(outside_anim),
            progress_output=str(outside_progress),
            eval_progress_output=str(outside_eval_progress),
        ),
    )

    generate_data.main()

    assert captured["train_vis"] == str(output_dir / "train_vis.pdf")
    assert captured["train_anim"] == str(output_dir / "train_traj.mp4")
    assert (output_dir / "train_vis.pdf").exists()
    assert (output_dir / "train_traj.mp4").exists()
    assert (output_dir / "train_progress.png").exists()
    assert (output_dir / "eval_progress.png").exists()
    assert not outside_vis.exists()
    assert not outside_anim.exists()
    assert not outside_progress.exists()
    assert not outside_eval_progress.exists()


def test_main_train_only_skips_default_eval_output(tmp_path, monkeypatch):
    """Directory mode should allow generating only the train split."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = _make_cfg(str(train_path), str(eval_path))
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(generate_data, "parse_args", lambda: _make_args(train_only=True))

    generate_data.main()

    datasets_dir = tmp_path / "data" / "datasets"
    created_dirs = [p for p in datasets_dir.iterdir() if p.is_dir()]
    assert len(created_dirs) == 1
    output_dir = created_dirs[0]
    assert (output_dir / "train.npz").exists()
    assert not (output_dir / "eval.npz").exists()
    assert (output_dir / "meta.json").exists()
    assert (output_dir / "README.txt").exists()
    meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["generation"]["eval_num_samples"] is None
    assert meta["paths"]["eval"] is None


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
    """Legacy single-file mode should reject shared train/eval output paths."""
    shared_path = tmp_path / "shared.npz"
    cfg = _make_cfg(str(shared_path), str(shared_path))

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output=str(shared_path), eval_output=str(shared_path)),
    )

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


def test_main_rejects_train_only_with_training_eval_path_override(tmp_path, monkeypatch):
    """train_only should also reject an explicit training.eval_data_path override."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = _make_cfg(str(train_path), str(eval_path))

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(
            train_only=True,
            training__eval_data_path=str(eval_path),
        ),
    )

    with pytest.raises(ValueError, match="Cannot combine --train_only with --eval_output"):
        generate_data.main()


def test_main_uses_data_generation_defaults_and_cli_section_overrides(
    tmp_path,
    monkeypatch,
):
    """Data-generation defaults should come from config and allow section overrides."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = _make_cfg(str(train_path), str(eval_path))
    cfg.data_generation.num_samples = 11
    cfg.data_generation.eval_num_samples = 7
    cfg.data_generation.num_workers = 2
    cfg.data_generation.progress_every = 1
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(
            data_generation__num_samples=13,
            task__seq_len=5,
        ),
    )

    generate_data.main()

    datasets_dir = tmp_path / "data" / "datasets"
    created_dirs = [p for p in datasets_dir.iterdir() if p.is_dir()]
    assert len(created_dirs) == 1
    output_dir = created_dirs[0]
    train_meta = np.load(output_dir / "train.npz", allow_pickle=False)["meta"].item()
    eval_meta = np.load(output_dir / "eval.npz", allow_pickle=False)["meta"].item()
    assert '"num_samples": 13' in train_meta
    assert '"num_samples": 7' in eval_meta


def test_main_pure_generation_skips_visualization_bin_resolution(tmp_path, monkeypatch):
    """Pure NPZ generation should not require visualization bins in config."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = _make_cfg(str(train_path), str(eval_path))
    cfg.visualization = SimpleNamespace(
        anim_num_traj=4,
        anim_fps=20,
        anim_step=4,
        anim_workers=4,
    )

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: _make_args(output=str(train_path), eval_output=str(eval_path)),
    )

    generate_data.main()

    assert train_path.exists()
    assert eval_path.exists()
