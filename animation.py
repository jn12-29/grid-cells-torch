"""Shared trajectory animation rendering utilities.

This module centralizes the eval-style 3-panel animation pipeline used by both
training evaluation exports and dataset-generation previews while keeping the
legacy function entrypoints available via ``utils.py``.

Usage:
    renderer = AnimationRenderer(env_size=2.2, fps=20, step=4)
    renderer.render(target_pos, pred_pos, pc_acts, hdc_acts, pc_centers, hdc_centers, "out.mp4")
"""

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


class AnimationRenderer:
    """Render trajectory animations with a consistent 3-panel layout."""

    def __init__(
        self,
        env_size: float,
        fps: int = 20,
        step: int = 4,
        num_workers: int = 4,
        title_prefix: str = "Trajectory",
        pred_label: str = "predicted",
    ) -> None:
        self.env_size = float(env_size)
        self.fps = max(1, int(fps))
        self.step = max(1, int(step))
        self.num_workers = max(1, int(num_workers))
        self.title_prefix = title_prefix
        self.pred_label = pred_label

    @staticmethod
    def sort_hdc_for_animation(hdc_acts: np.ndarray, hdc_centers: np.ndarray):
        """Sort HDC centers around the circle so polar bars render in angular order."""
        hdc_centers = np.asarray(hdc_centers, dtype=np.float32).reshape(-1)
        if hdc_acts.shape[-1] != hdc_centers.shape[0]:
            raise ValueError(
                "hdc_acts and hdc_centers must agree on the number of head-direction cells."
            )
        order = np.argsort(np.mod(hdc_centers, 2 * np.pi))
        return hdc_acts[..., order], hdc_centers[order]

    @staticmethod
    def make_animation_title(
        title_prefix: str,
        traj_idx: int,
        step_idx: int,
        total_steps: int,
    ) -> str:
        """Build a consistent animation title across writer and chunked paths."""
        return f"{title_prefix} #{traj_idx}  -  step {step_idx} / {total_steps}"

    @staticmethod
    def build_frame_chunks(frame_indices, num_workers: int):
        """Split sequential frame indices into evenly sized worker chunks."""
        chunk_size = max(1, (len(frame_indices) + num_workers - 1) // max(num_workers, 1))
        return [
            [(j, frame_indices[j]) for j in range(start, min(start + chunk_size, len(frame_indices)))]
            for start in range(0, len(frame_indices), chunk_size)
        ]

    @staticmethod
    def save_animation_frame(fig, path: str) -> None:
        """Persist one rendered animation frame, preferring a fast canvas copy path."""
        try:
            from PIL import Image
        except ImportError:
            Image = None

        if Image is not None:
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            Image.fromarray(rgba[:, :, :3]).save(path, optimize=False, compress_level=1)
            return

        fig.savefig(path, dpi=110)

    @staticmethod
    def encode_animation_frames(frames_dir: str, save_path: str, fps: int, n_frames: int) -> None:
        """Encode sequentially-numbered PNG frames into an MP4 via ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(frames_dir, "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-progress",
            "pipe:1",
            "-nostats",
            save_path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        bar = (
            _tqdm(total=n_frames, desc=f"encode:{os.path.basename(save_path)}", unit="frame")
            if _tqdm is not None
            else None
        )
        last = 0
        try:
            if proc.stdout:
                for line in proc.stdout:
                    if not line.strip().startswith("frame="):
                        continue
                    try:
                        cur = min(n_frames, int(line.strip().split("=", 1)[1]))
                    except ValueError:
                        continue
                    if bar is not None and cur > last:
                        bar.update(cur - last)
                    last = cur
            stderr = proc.stderr.read() if proc.stderr else ""
            rc = proc.wait()
            if bar is not None and last < n_frames:
                bar.update(n_frames - last)
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed:\n{stderr}")
        finally:
            if bar is not None:
                bar.close()

    @staticmethod
    def build_animation_artists(
        target_pos: np.ndarray,
        pred_pos: np.ndarray,
        pc_acts: np.ndarray,
        hdc_acts: np.ndarray,
        pc_centers: np.ndarray,
        hdc_centers: np.ndarray,
        env_size: float,
        pc_vmax: float,
        title_prefix: str,
        pred_label: str,
    ):
        """Create the shared 3-panel animation figure and artists."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        hdc_acts, hdc_centers = AnimationRenderer.sort_hdc_for_animation(
            hdc_acts,
            hdc_centers,
        )
        total_steps = target_pos.shape[0]
        n_hdc = hdc_acts.shape[1]
        half = env_size / 2.0
        safe_vmax = max(float(pc_vmax), 1e-8)
        hdc_width = (2 * np.pi / max(n_hdc, 1)) * 0.9

        fig = plt.figure(figsize=(13, 4.5), facecolor="#1a1a2e")
        fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.1, wspace=0.35)

        ax_traj = fig.add_subplot(1, 3, 1)
        ax_pc = fig.add_subplot(1, 3, 2)
        ax_hdc = fig.add_subplot(1, 3, 3, projection="polar")

        for ax in (ax_traj, ax_pc):
            ax.set_facecolor("#0d0d1a")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
        ax_hdc.set_facecolor("#0d0d1a")
        ax_hdc.spines["polar"].set_edgecolor("#444466")

        title_txt = fig.suptitle(
            AnimationRenderer.make_animation_title(title_prefix, 0, 0, total_steps),
            color="#ccccee",
            fontsize=11,
        )

        ax_traj.add_patch(
            mpatches.FancyBboxPatch(
                (-half, -half),
                env_size,
                env_size,
                boxstyle="square,pad=0",
                linewidth=1.2,
                edgecolor="#5566aa",
                facecolor="none",
            )
        )
        ax_traj.set_xlim(-half * 1.08, half * 1.08)
        ax_traj.set_ylim(-half * 1.08, half * 1.08)
        ax_traj.set_aspect("equal")
        ax_traj.set_title("Trajectory", color="#aaaacc", fontsize=9)
        ax_traj.tick_params(colors="#666688", labelsize=7)

        (line_actual,) = ax_traj.plot([], [], lw=1.0, color="#4488ff", alpha=0.7, label="actual")
        (dot_actual,) = ax_traj.plot([], [], "o", ms=5, color="#ffdd44", zorder=5)
        (line_pred,) = ax_traj.plot(
            [], [], lw=1.0, color="#ff6644", alpha=0.7, linestyle="--", label=pred_label
        )
        (dot_pred,) = ax_traj.plot([], [], "s", ms=4, color="#ffaa44", zorder=5)
        ax_traj.legend(
            fontsize=6,
            loc="upper right",
            labelcolor="#aaaacc",
            facecolor="#1a1a2e",
            edgecolor="#444466",
        )

        ax_pc.set_xlim(-half, half)
        ax_pc.set_ylim(-half, half)
        ax_pc.set_aspect("equal")
        ax_pc.set_title("Place cells", color="#aaaacc", fontsize=9)
        ax_pc.tick_params(colors="#666688", labelsize=7)
        sc_pc = ax_pc.scatter(
            pc_centers[:, 0],
            pc_centers[:, 1],
            c=pc_acts[0],
            cmap="hot",
            s=18,
            vmin=0.0,
            vmax=safe_vmax,
        )
        fig.colorbar(sc_pc, ax=ax_pc, fraction=0.046, pad=0.04).ax.tick_params(
            colors="#aaaacc", labelsize=6
        )

        bars_hdc = ax_hdc.bar(
            hdc_centers,
            hdc_acts[0],
            width=hdc_width,
            color="#4fc3f7",
            alpha=0.85,
            align="center",
        )
        ax_hdc.set_ylim(0, max(float(hdc_acts.max()), 1.0 / max(n_hdc, 1)))
        ax_hdc.set_title("Head direction cells", color="#aaaacc", fontsize=9, pad=8)
        ax_hdc.tick_params(colors="#666688", labelsize=7)
        ax_hdc.yaxis.label.set_color("#666688")

        artists = {
            "title_txt": title_txt,
            "line_actual": line_actual,
            "dot_actual": dot_actual,
            "line_pred": line_pred,
            "dot_pred": dot_pred,
            "sc_pc": sc_pc,
            "bars_hdc": bars_hdc,
        }
        arrays = {
            "target_pos": target_pos,
            "pred_pos": pred_pos,
            "pc_acts": pc_acts,
            "hdc_acts": hdc_acts,
            "total_steps": total_steps,
            "title_prefix": title_prefix,
        }
        return fig, artists, arrays

    @staticmethod
    def update_animation_frame(artists: dict, arrays: dict, traj_idx: int, t: int) -> None:
        """Update the shared 3-panel animation figure to one timestep."""
        target_pos = arrays["target_pos"]
        pred_pos = arrays["pred_pos"]
        pc_acts = arrays["pc_acts"]
        hdc_acts = arrays["hdc_acts"]
        total_steps = arrays["total_steps"]
        title_prefix = arrays["title_prefix"]

        artists["title_txt"].set_text(
            AnimationRenderer.make_animation_title(title_prefix, traj_idx, t, total_steps)
        )
        artists["line_actual"].set_data(target_pos[: t + 1, 0], target_pos[: t + 1, 1])
        artists["dot_actual"].set_data([target_pos[t, 0]], [target_pos[t, 1]])
        artists["line_pred"].set_data(pred_pos[: t + 1, 0], pred_pos[: t + 1, 1])
        artists["dot_pred"].set_data([pred_pos[t, 0]], [pred_pos[t, 1]])
        artists["sc_pc"].set_array(pc_acts[t])
        for bar, h in zip(artists["bars_hdc"], hdc_acts[t]):
            bar.set_height(h)

    def _traj_path(self, save_path: str, num_trajectories: int, traj_idx: int) -> str:
        """Build the output path for one trajectory animation."""
        if num_trajectories == 1:
            return save_path
        base, ext = os.path.splitext(save_path)
        return f"{base}_traj{traj_idx:04d}{ext}"

    def render(
        self,
        target_pos: np.ndarray,
        pred_pos: np.ndarray,
        pc_acts: np.ndarray,
        hdc_acts: np.ndarray,
        pc_centers: np.ndarray,
        hdc_centers: np.ndarray,
        save_path: str,
    ) -> None:
        """Generate one animation file per trajectory in the provided batch."""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        num_trajectories = target_pos.shape[0]
        total_steps = target_pos.shape[1]
        if num_trajectories == 0 or total_steps == 0:
            raise ValueError("Cannot animate: empty trajectory data.")

        pc_vmax = float(pc_acts.max())
        num_workers = self.num_workers
        if num_workers > 1 and shutil.which("ffmpeg") is None:
            num_workers = 1

        if num_workers <= 1:
            for traj_idx in range(num_trajectories):
                saved = _render_trajectory_animation(
                    (
                        traj_idx,
                        target_pos[traj_idx],
                        pred_pos[traj_idx],
                        pc_acts[traj_idx],
                        hdc_acts[traj_idx],
                        pc_centers,
                        hdc_centers,
                        self.env_size,
                        pc_vmax,
                        self.fps,
                        self.step,
                        self._traj_path(save_path, num_trajectories, traj_idx),
                        self.title_prefix,
                        self.pred_label,
                    )
                )
                print(f"Animation saved to {saved}")
            return

        all_frame_indices = list(range(0, total_steps, self.step))
        n_frames = len(all_frame_indices)
        frame_chunks = self.build_frame_chunks(all_frame_indices, num_workers=num_workers)
        actual_workers = min(num_workers, len(frame_chunks))

        for traj_idx in range(num_trajectories):
            out_path = self._traj_path(save_path, num_trajectories, traj_idx)
            tasks = [
                (
                    traj_idx,
                    target_pos[traj_idx],
                    pred_pos[traj_idx],
                    pc_acts[traj_idx],
                    hdc_acts[traj_idx],
                    pc_centers,
                    hdc_centers,
                    self.env_size,
                    pc_vmax,
                    chunk,
                    "",
                    self.title_prefix,
                    self.pred_label,
                )
                for chunk in frame_chunks
            ]

            render_bar = (
                _tqdm(total=n_frames, desc=f"render:{os.path.basename(out_path)}", unit="frame")
                if _tqdm is not None
                else None
            )

            with tempfile.TemporaryDirectory(prefix="eval_anim_") as frames_dir:
                tasks = [task[:-3] + (frames_dir,) + task[-2:] for task in tasks]

                try:
                    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                        futures = [
                            executor.submit(_render_trajectory_animation_chunk, task)
                            for task in tasks
                        ]
                        for future in as_completed(futures):
                            rendered = future.result()
                            if render_bar is not None:
                                render_bar.update(rendered)
                finally:
                    if render_bar is not None:
                        render_bar.close()

                self.encode_animation_frames(frames_dir, out_path, fps=self.fps, n_frames=n_frames)

            print(f"Animation saved to {out_path}")


def _render_trajectory_animation(args) -> str:
    """Render a single 3-panel trajectory animation using a direct writer path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    try:
        from tqdm.auto import tqdm as _local_tqdm
    except ImportError:
        _local_tqdm = None

    (
        traj_idx,
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        env_size,
        pc_vmax,
        fps,
        step,
        out_path,
        title_prefix,
        pred_label,
    ) = args

    frame_indices = list(range(0, target_pos.shape[0], max(1, step)))
    fig, artists, arrays = AnimationRenderer.build_animation_artists(
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
    )

    def _make_writer(path):
        try:
            writer = FFMpegWriter(
                fps=fps,
                metadata={"title": f"{title_prefix} #{traj_idx}"},
            )
            return writer, path
        except Exception:
            gif_path = path.replace(".mp4", ".gif")
            return PillowWriter(fps=fps), gif_path

    writer, out_path = _make_writer(out_path)
    pbar = (
        _local_tqdm(frame_indices, desc=f"render:{os.path.basename(out_path)}", unit="frame")
        if _local_tqdm is not None
        else frame_indices
    )

    with writer.saving(fig, out_path, dpi=110):
        for t in pbar:
            AnimationRenderer.update_animation_frame(artists, arrays, traj_idx, t)
            writer.grab_frame()

    plt.close(fig)
    return out_path


def _render_trajectory_animation_chunk(task) -> int:
    """Render one worker chunk of a 3-panel trajectory animation to PNG frames."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (
        traj_idx,
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        env_size,
        pc_vmax,
        frames_subset,
        frames_dir,
        title_prefix,
        pred_label,
    ) = task

    fig, artists, arrays = AnimationRenderer.build_animation_artists(
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
    )

    try:
        for out_idx, t in frames_subset:
            AnimationRenderer.update_animation_frame(artists, arrays, traj_idx, t)
            AnimationRenderer.save_animation_frame(
                fig,
                os.path.join(frames_dir, f"frame_{out_idx:06d}.png"),
            )
    finally:
        plt.close(fig)

    return len(frames_subset)


def generate_trajectory_animation(
    target_pos: np.ndarray,
    pred_pos: np.ndarray,
    pc_acts: np.ndarray,
    hdc_acts: np.ndarray,
    pc_centers: np.ndarray,
    hdc_centers: np.ndarray,
    env_size: float,
    save_path: str,
    fps: int = 20,
    step: int = 4,
    num_workers: int = 4,
    title_prefix: str = "Trajectory",
    pred_label: str = "predicted",
) -> None:
    """Backward-compatible function entrypoint for trajectory animations."""
    renderer = AnimationRenderer(
        env_size=env_size,
        fps=fps,
        step=step,
        num_workers=num_workers,
        title_prefix=title_prefix,
        pred_label=pred_label,
    )
    renderer.render(
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        save_path,
    )


def generate_eval_animation(
    target_pos: np.ndarray,
    pred_pos: np.ndarray,
    pc_acts: np.ndarray,
    hdc_acts: np.ndarray,
    pc_centers: np.ndarray,
    hdc_centers: np.ndarray,
    env_size: float,
    save_path: str,
    fps: int = 20,
    step: int = 4,
    num_workers: int = 4,
) -> None:
    """Backward-compatible wrapper for eval animation exports."""
    generate_trajectory_animation(
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        env_size,
        save_path,
        fps=fps,
        step=step,
        num_workers=num_workers,
        title_prefix="Eval traj",
        pred_label="predicted",
    )
