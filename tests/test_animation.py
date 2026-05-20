"""Tests for shared animation rendering helpers.

Usage:
    pytest tests/test_animation.py
"""

import os

import numpy as np

from grid_cells.viz.animation import AnimationRenderer, _render_joined_trajectory_animation


def test_animation_renderer_joins_multiple_trajectories_into_one_file(monkeypatch, tmp_path):
    """Multi-trajectory renders should encode one sequential video."""
    captured = {}

    class DummyExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, task):
            class Future:
                def result(self_inner):
                    return fn(task)

            return Future()

    def fake_render_chunk(task):
        frames_subset = task[9]
        frames_dir = task[10]
        for out_idx, _ in frames_subset:
            open(os.path.join(frames_dir, f"frame_{out_idx:06d}.png"), "wb").close()
        return len(frames_subset)

    def fake_encode(frames_dir, save_path, fps, n_frames):
        captured["save_path"] = save_path
        captured["fps"] = fps
        captured["n_frames"] = n_frames
        captured["frames"] = sorted(os.listdir(frames_dir))

    monkeypatch.setattr("grid_cells.viz.animation.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("grid_cells.viz.animation.ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr("grid_cells.viz.animation.as_completed", lambda futures: futures)
    monkeypatch.setattr(
        "grid_cells.viz.animation._render_trajectory_animation_chunk",
        fake_render_chunk,
    )
    monkeypatch.setattr(AnimationRenderer, "encode_animation_frames", staticmethod(fake_encode))

    renderer = AnimationRenderer(env_size=2.2, fps=12, step=2, num_workers=2)
    target_pos = np.zeros((2, 5, 2), dtype=np.float32)
    pred_pos = np.zeros((2, 5, 2), dtype=np.float32)
    pc_acts = np.ones((2, 5, 3), dtype=np.float32)
    hdc_acts = np.ones((2, 5, 4), dtype=np.float32)
    pc_centers = np.zeros((3, 2), dtype=np.float32)
    hdc_centers = np.linspace(0.0, 2 * np.pi, 4, endpoint=False, dtype=np.float32)
    out_path = tmp_path / "eval_animation_epoch_0002.mp4"

    renderer.render(
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        str(out_path),
    )

    assert captured["save_path"] == str(out_path)
    assert captured["fps"] == 12
    assert captured["n_frames"] == 6
    assert captured["frames"] == [f"frame_{idx:06d}.png" for idx in range(6)]


def test_joined_animation_scales_hdc_axis_from_full_batch(monkeypatch, tmp_path):
    """Joined renders should initialize HDC scale from every trajectory."""
    captured = {}

    class DummyWriter:
        def __init__(self, *args, **kwargs):
            pass

        def saving(self, fig, path, dpi):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def grab_frame(self):
            pass

    def fake_build(
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        env_size,
        pc_vmax,
        title_prefix,
        pred_label,
        hdc_ylim_max=None,
    ):
        import matplotlib.pyplot as plt

        captured["initial_hdc_max"] = float(hdc_acts.max())
        captured["hdc_ylim_max"] = hdc_ylim_max
        return plt.figure(), {}, {}

    monkeypatch.setattr("matplotlib.animation.FFMpegWriter", DummyWriter)
    monkeypatch.setattr(AnimationRenderer, "build_animation_artists", staticmethod(fake_build))
    monkeypatch.setattr(
        AnimationRenderer,
        "update_animation_frame",
        staticmethod(lambda *args: None),
    )

    target_pos = np.zeros((2, 2, 2), dtype=np.float32)
    pred_pos = np.zeros((2, 2, 2), dtype=np.float32)
    pc_acts = np.ones((2, 2, 3), dtype=np.float32)
    hdc_acts = np.ones((2, 2, 4), dtype=np.float32)
    hdc_acts[1, 1, 2] = 7.0
    pc_centers = np.zeros((3, 2), dtype=np.float32)
    hdc_centers = np.linspace(0.0, 2 * np.pi, 4, endpoint=False, dtype=np.float32)

    _render_joined_trajectory_animation(
        (
            target_pos,
            pred_pos,
            pc_acts,
            hdc_acts,
            pc_centers,
            hdc_centers,
            2.2,
            float(pc_acts.max()),
            12,
            1,
            str(tmp_path / "joined.mp4"),
            "Eval traj",
            "predicted",
        )
    )

    assert captured["initial_hdc_max"] == 1.0
    assert captured["hdc_ylim_max"] == 7.0
