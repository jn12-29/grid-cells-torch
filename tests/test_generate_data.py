"""Tests for generate_data helpers and CLI flow."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

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

    def fake_visualize_animation(
        dataset,
        save_path,
        fps=20,
        max_trajectories=16,
        num_workers=8,
        chunk_size=32,
    ):
        calls.append(
            (dataset.num_samples, save_path, fps, max_trajectories, num_workers, chunk_size)
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
        animation_output=str(anim_path),
        animation_fps=12,
        anim_workers=5,
        anim_chunk_size=7,
    )

    assert anim_path.exists()
    assert calls == [(6, str(anim_path), 12, 16, 5, 7)]


def test_visualize_animation_renders_frames_and_encodes_mp4(tmp_path, monkeypatch):
    """Animation export should render frame chunks and invoke ffmpeg encoding."""
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

    render_calls = []
    encode_calls = []

    def fake_render_animation_chunk(task):
        full_pos, colors, env_size, seq_len, start_frame, end_frame, frames_dir = task
        render_calls.append((start_frame, end_frame, full_pos.shape[0], seq_len, len(colors)))
        for frame_idx in range(start_frame, end_frame):
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
            with open(frame_path, "wb") as handle:
                handle.write(b"png")
        return end_frame - start_frame

    def fake_encode_animation_frames(frames_dir, save_path, fps, num_frames):
        encode_calls.append((frames_dir, save_path, fps, num_frames))
        with open(save_path, "wb") as handle:
            handle.write(b"fake-mp4")

    class DummyFuture:
        def __init__(self, result_value):
            self._result_value = result_value

        def result(self):
            return self._result_value

    class DummyExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return DummyFuture(fn(*args, **kwargs))

    monkeypatch.setattr(generate_data, "_render_animation_chunk", fake_render_animation_chunk)
    monkeypatch.setattr(generate_data, "_encode_animation_frames", fake_encode_animation_frames)
    monkeypatch.setattr(generate_data, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(generate_data, "as_completed", lambda futures: futures)

    generate_data.visualize_animation(
        dataset,
        str(anim_path),
        fps=12,
        max_trajectories=4,
        num_workers=2,
        chunk_size=2,
    )

    assert anim_path.exists()
    assert render_calls == [
        (0, 2, 4, 5, 4),
        (2, 4, 4, 5, 4),
        (4, 6, 4, 5, 4),
    ]
    assert encode_calls[0][1:] == (str(anim_path), 12, 6)


def test_main_defaults_output_to_config_train_and_eval_paths(tmp_path, monkeypatch):
    """CLI should default to config train/eval output paths."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = SimpleNamespace(
        task=SimpleNamespace(
            seq_len=6,
            env_size=2.2,
            neurons_seed=5,
            velocity_noise=[0.0, 0.0, 0.0],
        ),
        training=SimpleNamespace(
            steps_per_epoch=2,
            batch_size=3,
            eval_batch_size=4,
            data_path=str(train_path),
            eval_data_path=str(eval_path),
        ),
    )

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: SimpleNamespace(
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
            anim_fps=20,
            anim_workers=8,
            anim_chunk_size=32,
            num_workers=1,
            visualize_progress=False,
            progress_output=None,
            eval_progress_output=None,
            progress_every=4,
            eval_output=None,
            train_only=False,
            eval_num_samples=None,
            eval_seed=None,
        ),
    )

    generate_data.main()

    train_meta = np.load(train_path, allow_pickle=False)["meta"].item()
    eval_meta = np.load(eval_path, allow_pickle=False)["meta"].item()
    assert '"num_samples": 6' in train_meta
    assert '"num_samples": 4' in eval_meta


def test_main_train_only_skips_default_eval_output(tmp_path, monkeypatch):
    """CLI should allow generating only the main split."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = SimpleNamespace(
        task=SimpleNamespace(
            seq_len=6,
            env_size=2.2,
            neurons_seed=5,
            velocity_noise=[0.0, 0.0, 0.0],
        ),
        training=SimpleNamespace(
            steps_per_epoch=2,
            batch_size=3,
            eval_batch_size=4,
            data_path=str(train_path),
            eval_data_path=str(eval_path),
        ),
    )

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: SimpleNamespace(
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
            anim_fps=20,
            anim_workers=8,
            anim_chunk_size=32,
            num_workers=1,
            visualize_progress=False,
            progress_output=None,
            eval_progress_output=None,
            progress_every=4,
            eval_output=None,
            train_only=True,
            eval_num_samples=None,
            eval_seed=None,
        ),
    )

    generate_data.main()

    assert train_path.exists()
    assert not eval_path.exists()


def test_main_can_generate_train_and_eval_splits(tmp_path, monkeypatch):
    """CLI entry point should support writing train and eval files in one run."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = SimpleNamespace(
        task=SimpleNamespace(seq_len=6, env_size=2.2, neurons_seed=5, velocity_noise=[0.0, 0.0, 0.0]),
        training=SimpleNamespace(
            steps_per_epoch=2,
            batch_size=3,
            eval_batch_size=4,
            data_path=str(train_path),
            eval_data_path=str(tmp_path / "config-eval.npz"),
        ),
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
            animate=False,
            vis_output=None,
            anim_output=None,
            anim_fps=20,
            anim_workers=8,
            anim_chunk_size=32,
            num_workers=1,
            visualize_progress=False,
            progress_output=None,
            eval_progress_output=None,
            progress_every=4,
            eval_output=str(eval_path),
            train_only=False,
            eval_num_samples=None,
            eval_seed=None,
        ),
    )

    generate_data.main()

    train_meta = np.load(train_path, allow_pickle=False)["meta"].item()
    eval_meta = np.load(eval_path, allow_pickle=False)["meta"].item()
    assert '"num_samples": 6' in train_meta
    assert '"num_samples": 4' in eval_meta


def test_main_rejects_same_train_and_eval_output_paths(tmp_path, monkeypatch):
    """Train and eval outputs should not resolve to the same path."""
    shared_path = tmp_path / "shared.npz"
    cfg = SimpleNamespace(
        task=SimpleNamespace(
            seq_len=6,
            env_size=2.2,
            neurons_seed=5,
            velocity_noise=[0.0, 0.0, 0.0],
        ),
        training=SimpleNamespace(
            steps_per_epoch=2,
            batch_size=3,
            eval_batch_size=4,
            data_path=str(shared_path),
            eval_data_path=str(shared_path),
        ),
    )

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: SimpleNamespace(
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
            anim_fps=20,
            anim_workers=8,
            anim_chunk_size=32,
            num_workers=1,
            visualize_progress=False,
            progress_output=None,
            eval_progress_output=None,
            progress_every=4,
            eval_output=None,
            train_only=False,
            eval_num_samples=None,
            eval_seed=None,
        ),
    )

    with pytest.raises(ValueError, match="Train and eval output paths must be different"):
        generate_data.main()


def test_main_rejects_train_only_with_eval_output(tmp_path, monkeypatch):
    """Conflicting train-only/eval flags should be rejected."""
    train_path = tmp_path / "train.npz"
    eval_path = tmp_path / "eval.npz"
    cfg = SimpleNamespace(
        task=SimpleNamespace(
            seq_len=6,
            env_size=2.2,
            neurons_seed=5,
            velocity_noise=[0.0, 0.0, 0.0],
        ),
        training=SimpleNamespace(
            steps_per_epoch=2,
            batch_size=3,
            eval_batch_size=4,
            data_path=str(train_path),
            eval_data_path=str(eval_path),
        ),
    )

    monkeypatch.setattr(generate_data, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        generate_data,
        "parse_args",
        lambda: SimpleNamespace(
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
            anim_fps=20,
            anim_workers=8,
            anim_chunk_size=32,
            num_workers=1,
            visualize_progress=False,
            progress_output=None,
            eval_progress_output=None,
            progress_every=4,
            eval_output=str(eval_path),
            train_only=True,
            eval_num_samples=None,
            eval_seed=None,
        ),
    )

    with pytest.raises(ValueError, match="Cannot combine --train_only with --eval_output"):
        generate_data.main()
