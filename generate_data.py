"""
Generate and optionally visualize trajectory data for grid cell training.

Usage
-----
# Generate trajectories with default config, save to training.data_path
python generate_data.py

# Visualize without saving
python generate_data.py --output data/train.npz --visualize

# Generate a smaller evaluation set
python generate_data.py --output data/eval.npz --num_samples 4000

# Export an MP4 animation of sample trajectories
python generate_data.py --output data/train.npz --animate

# Generate train/eval splits in one run
python generate_data.py --output data/train.npz --eval_output data/eval.npz

# Use a custom config
python generate_data.py --config my_config.yaml --output data/train.npz
"""

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # headless-safe; override below if displaying
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

# Allow running from the grid-cells-torch directory
sys.path.insert(0, os.path.dirname(__file__))

from dataset import TrajectoryDataset
from train import load_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectory dataset for grid cell training"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz file path. Defaults to training.data_path from config, "
        "e.g. data/train.npz",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of trajectories to generate. "
        "Defaults to steps_per_epoch × batch_size from config.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Steps per trajectory (overrides config)",
    )
    parser.add_argument(
        "--env_size",
        type=float,
        default=None,
        help="Environment side length in metres (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config neurons_seed)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and save a visualisation PDF alongside the .npz file",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Generate and save an MP4 animation of sample trajectories",
    )
    parser.add_argument(
        "--eval_output",
        default=None,
        help="Optional second .npz file path for a fixed evaluation split, "
        "e.g. data/eval.npz",
    )
    parser.add_argument(
        "--eval_num_samples",
        type=int,
        default=None,
        help="Number of trajectories for the optional eval split. "
        "Defaults to training.eval_batch_size from config.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=None,
        help="Random seed for the optional eval split. Defaults to seed + 1.",
    )
    parser.add_argument(
        "--vis_output",
        default=None,
        help="Path for the visualisation PDF. "
        "Defaults to <output>.pdf (replacing .npz extension)",
    )
    parser.add_argument(
        "--anim_output",
        default=None,
        help="Path for the trajectory animation MP4. " "Defaults to <output>_traj.mp4",
    )
    parser.add_argument(
        "--anim_fps",
        type=int,
        default=20,
        help="Frames per second for the trajectory animation MP4",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes for chunked trajectory generation",
    )
    parser.add_argument(
        "--visualize_progress",
        action="store_true",
        help="Periodically save a progress preview PNG while trajectories are generated",
    )
    parser.add_argument(
        "--progress_output",
        default=None,
        help="Path for the progress preview PNG. Defaults to <output>_progress.png",
    )
    parser.add_argument(
        "--eval_progress_output",
        default=None,
        help="Path for the eval split progress preview PNG. "
        "Defaults to <eval_output>_progress.png",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=4,
        help="Refresh the progress preview every N completed chunks",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def visualize(dataset: TrajectoryDataset, save_path: str) -> None:
    """Produce a multi-panel summary figure and save as PDF.

    Panels
    ------
    1. Sample trajectories (up to 16) plotted in the environment
    2. Position coverage heatmap (where the agent spends time)
    3. Translational speed distribution
    4. Angular velocity distribution
    5. Head direction rose chart (polar histogram)
    6. Trajectory length sanity check (should all be seq_len steps)
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages

    half = dataset.env_size / 2.0
    pos_all = dataset._data["target_pos"]  # (N, T, 2)
    vel_all = dataset._data["ego_vel"]  # (N, T, 3)
    hd_all = dataset._data["target_hd"]  # (N, T, 1)
    N, T, _ = pos_all.shape

    speeds = np.sqrt(vel_all[:, :, 0] ** 2 + vel_all[:, :, 1] ** 2).ravel()  # m/s
    omegas = vel_all[:, :, 2].ravel()  # rad/s
    hds = hd_all.ravel()  # rad

    with PdfPages(save_path) as pdf:

        # ------------------------------------------------------------------
        # Page 1: overview
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

        # --- panel 1: sample trajectories ---
        ax_traj = fig.add_subplot(gs[0, :2])
        n_show = min(16, N)
        cmap = plt.cm.tab20
        for i in range(n_show):
            xy = pos_all[i]  # (T, 2)
            ax_traj.plot(
                xy[:, 0], xy[:, 1], color=cmap(i / n_show), alpha=0.7, linewidth=0.8
            )
            ax_traj.scatter(xy[0, 0], xy[0, 1], color=cmap(i / n_show), s=20, zorder=3)
        ax_traj.set_xlim(-half, half)
        ax_traj.set_ylim(-half, half)
        ax_traj.set_aspect("equal")
        ax_traj.set_title(f"Sample trajectories (n={n_show})")
        ax_traj.set_xlabel("x (m)")
        ax_traj.set_ylabel("y (m)")
        rect = plt.Rectangle(
            (-half, -half),
            dataset.env_size,
            dataset.env_size,
            fill=False,
            edgecolor="k",
            linewidth=1.5,
        )
        ax_traj.add_patch(rect)

        # --- panel 2: position coverage heatmap ---
        ax_heat = fig.add_subplot(gs[0, 2])
        nbins = 40
        heatmap, xedges, yedges = np.histogram2d(
            pos_all[:, :, 0].ravel(),
            pos_all[:, :, 1].ravel(),
            bins=nbins,
            range=[[-half, half], [-half, half]],
        )
        im = ax_heat.imshow(
            heatmap.T,
            origin="lower",
            extent=[-half, half, -half, half],
            cmap="hot",
            aspect="equal",
        )
        plt.colorbar(im, ax=ax_heat, label="visit count")
        ax_heat.set_title("Position coverage")
        ax_heat.set_xlabel("x (m)")
        ax_heat.set_ylabel("y (m)")

        # --- panel 3: speed distribution ---
        ax_spd = fig.add_subplot(gs[0, 3])
        ax_spd.hist(speeds, bins=60, color="steelblue", edgecolor="none", density=True)
        ax_spd.axvline(
            speeds.mean(),
            color="tomato",
            linestyle="--",
            label=f"mean={speeds.mean():.3f} m/s",
        )
        ax_spd.set_title("Translational speed")
        ax_spd.set_xlabel("speed (m/s)")
        ax_spd.set_ylabel("density")
        ax_spd.legend(fontsize=8)

        # --- panel 4: angular velocity distribution ---
        ax_omg = fig.add_subplot(gs[1, 0])
        ax_omg.hist(
            omegas, bins=60, color="mediumpurple", edgecolor="none", density=True
        )
        ax_omg.axvline(
            omegas.mean(),
            color="tomato",
            linestyle="--",
            label=f"mean={omegas.mean():.3f} rad/s",
        )
        ax_omg.set_title("Angular velocity")
        ax_omg.set_xlabel("ω (rad/s)")
        ax_omg.set_ylabel("density")
        ax_omg.legend(fontsize=8)

        # --- panel 5: head direction rose (polar histogram) ---
        ax_hd = fig.add_subplot(gs[1, 1], projection="polar")
        n_bins = 36
        counts, bin_edges = np.histogram(hds, bins=n_bins, range=(-np.pi, np.pi))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        width = 2 * np.pi / n_bins
        ax_hd.bar(
            bin_centers,
            counts,
            width=width,
            bottom=0,
            color="seagreen",
            alpha=0.8,
            edgecolor="none",
        )
        ax_hd.set_title("Head direction distribution", pad=12)
        ax_hd.set_theta_zero_location("E")
        ax_hd.set_theta_direction(1)

        # --- panel 6: displacement vs recorded velocity sanity check ---
        # For non-reflection steps, |Δpos| / dt should equal recorded speed
        ax_san = fig.add_subplot(gs[1, 2:])
        sample_idx = min(200, N)
        pos_sample = pos_all[:sample_idx]  # (M, T, 2)
        vel_sample = vel_all[:sample_idx, :, :2]  # (M, T, 2)
        init_pos = dataset._data["init_pos"][:sample_idx]  # (M, 2)
        full_pos = np.concatenate(
            [init_pos[:, np.newaxis, :], pos_sample], axis=1
        )  # (M, T+1, 2)
        displace = np.linalg.norm(np.diff(full_pos, axis=1), axis=-1).ravel()  # (M*T,)
        recorded = np.linalg.norm(vel_sample, axis=-1).ravel() * TrajectoryDataset._DT
        ax_san.scatter(recorded, displace, s=1, alpha=0.15, color="gray")
        # ideal line
        lim = max(displace.max(), recorded.max()) * 1.05
        ax_san.plot([0, lim], [0, lim], "r--", linewidth=1, label="ideal (y=x)")
        ax_san.set_xlim(0, lim)
        ax_san.set_ylim(0, lim)
        ax_san.set_title(
            "Velocity–displacement consistency\n(non-reflection steps cluster on y=x)"
        )
        ax_san.set_xlabel("recorded |v|·dt  (m)")
        ax_san.set_ylabel("|Δpos|  (m)")
        ax_san.legend(fontsize=8)

        # stats box
        n_total = len(displace)
        n_matched = int(np.sum(np.abs(displace - recorded) < 1e-4))
        ax_san.text(
            0.98,
            0.05,
            f"Exact match: {n_matched}/{n_total} ({100*n_matched/n_total:.1f}%)\n"
            f"(mismatches = boundary reflection steps)",
            transform=ax_san.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.suptitle(
            f"Trajectory dataset summary\n"
            f"{N} trajectories × {T} steps  |  "
            f"env {dataset.env_size} m × {dataset.env_size} m  |  "
            f"dt={TrajectoryDataset._DT} s",
            fontsize=13,
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Visualisation saved to {save_path}")


def visualize_animation(
    dataset: TrajectoryDataset,
    save_path: str,
    fps: int = 20,
    max_trajectories: int = 16,
) -> None:
    """Save an MP4 animation showing trajectories unfolding over time."""
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "MP4 animation export requires ffmpeg to be installed and available on PATH."
        )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    pos_all = dataset._data["target_pos"]
    init_pos = dataset._data["init_pos"]
    num_show = min(max_trajectories, len(pos_all))
    full_pos = np.concatenate(
        [init_pos[:num_show, np.newaxis, :], pos_all[:num_show]],
        axis=1,
    )
    num_frames = full_pos.shape[1]
    half = dataset.env_size / 2.0
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(num_show, 1), endpoint=False))

    fig, ax = plt.subplots(figsize=(7, 7))
    line_artists = [
        ax.plot([], [], color=colors[i], alpha=0.75, linewidth=1.5)[0]
        for i in range(num_show)
    ]
    point_artist = ax.scatter([], [], s=32, c=colors[:num_show], zorder=3)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top")

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Sample trajectories over time (n={num_show})")
    ax.add_patch(
        plt.Rectangle(
            (-half, -half),
            dataset.env_size,
            dataset.env_size,
            fill=False,
            edgecolor="k",
            linewidth=1.5,
        )
    )

    def init_frame():
        empty_offsets = np.empty((0, 2), dtype=np.float32)
        point_artist.set_offsets(empty_offsets)
        for line in line_artists:
            line.set_data([], [])
        time_text.set_text("")
        return [*line_artists, point_artist, time_text]

    def update_frame(frame_idx: int):
        for traj_idx, line in enumerate(line_artists):
            xy = full_pos[traj_idx, : frame_idx + 1]
            line.set_data(xy[:, 0], xy[:, 1])

        point_artist.set_offsets(full_pos[:, frame_idx, :])
        time_text.set_text(
            f"step {max(frame_idx - 1, 0)}/{dataset.seq_len}"
            f"\ntime {max(frame_idx - 1, 0) * TrajectoryDataset._DT:.2f}s"
        )
        return [*line_artists, point_artist, time_text]

    anim = animation.FuncAnimation(
        fig,
        update_frame,
        init_func=init_frame,
        frames=num_frames,
        interval=1000 / max(fps, 1),
        blit=False,
    )
    anim.save(save_path, writer=animation.FFMpegWriter(fps=max(fps, 1), bitrate=1800))
    plt.close(fig)
    print(f"Animation saved to {save_path}")


class GenerationProgressPreview:
    """Accumulate lightweight dataset stats and periodically save a preview PNG."""

    def __init__(
        self,
        total_samples: int,
        env_size: float,
        save_path: str,
        refresh_every: int = 4,
    ) -> None:
        self.total_samples = total_samples
        self.env_size = env_size
        self.save_path = save_path
        self.refresh_every = max(1, int(refresh_every))
        self.completed_chunks = 0
        self.sample_trajectories = []
        self.heatmap_bins = 40
        self.heatmap = np.zeros(
            (self.heatmap_bins, self.heatmap_bins), dtype=np.float64
        )
        self.speed_edges = np.linspace(0.0, 1.0, 61)
        self.speed_hist = np.zeros(len(self.speed_edges) - 1, dtype=np.int64)
        self.omega_edges = np.linspace(-8.0, 8.0, 61)
        self.omega_hist = np.zeros(len(self.omega_edges) - 1, dtype=np.int64)
        self.hd_edges = np.linspace(-np.pi, np.pi, 37)
        self.hd_hist = np.zeros(len(self.hd_edges) - 1, dtype=np.int64)
        self.last_completed_samples = 0
        self.last_chunk_range = None

    def update(self, event: dict, elapsed: float, throughput: float) -> None:
        chunk = event["chunk_data"]
        self.completed_chunks += 1
        self.last_completed_samples = event["completed_samples"]
        self.last_chunk_range = (event["chunk_start"], event["chunk_end"])

        if len(self.sample_trajectories) < 16:
            remaining = 16 - len(self.sample_trajectories)
            self.sample_trajectories.extend(list(chunk["target_pos"][:remaining]))

        half = self.env_size / 2.0
        heatmap, _, _ = np.histogram2d(
            chunk["target_pos"][:, :, 0].ravel(),
            chunk["target_pos"][:, :, 1].ravel(),
            bins=self.heatmap_bins,
            range=[[-half, half], [-half, half]],
        )
        self.heatmap += heatmap

        speeds = np.linalg.norm(chunk["ego_vel"][:, :, :2], axis=-1).ravel()
        omegas = chunk["ego_vel"][:, :, 2].ravel()
        hds = chunk["target_hd"].ravel()
        self.speed_hist += np.histogram(speeds, bins=self.speed_edges)[0]
        self.omega_hist += np.histogram(omegas, bins=self.omega_edges)[0]
        self.hd_hist += np.histogram(hds, bins=self.hd_edges)[0]

        if self.completed_chunks % self.refresh_every == 0:
            self.write(elapsed, throughput)

    def write(self, elapsed: float, throughput: float) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
        half = self.env_size / 2.0
        progress = self.last_completed_samples / max(self.total_samples, 1)
        eta_seconds = (
            max(self.total_samples - self.last_completed_samples, 0) / throughput
            if throughput > 0.0
            else float("inf")
        )

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        ax = axes[0, 0]
        cmap = plt.cm.tab20
        for idx, xy in enumerate(self.sample_trajectories):
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=cmap(idx / max(len(self.sample_trajectories), 1)),
                alpha=0.75,
                linewidth=0.9,
            )
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_title(f"Sample trajectories ({len(self.sample_trajectories)})")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.add_patch(
            plt.Rectangle(
                (-half, -half),
                self.env_size,
                self.env_size,
                fill=False,
                edgecolor="k",
                linewidth=1.2,
            )
        )

        ax = axes[0, 1]
        im = ax.imshow(
            self.heatmap.T,
            origin="lower",
            extent=[-half, half, -half, half],
            cmap="hot",
            aspect="equal",
        )
        fig.colorbar(im, ax=ax, label="visit count")
        ax.set_title("Position coverage so far")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        ax = axes[0, 2]
        ax.bar(
            self.speed_edges[:-1],
            self.speed_hist,
            width=np.diff(self.speed_edges),
            align="edge",
            color="steelblue",
            edgecolor="none",
        )
        ax.set_title("Speed histogram so far")
        ax.set_xlabel("speed (m/s)")
        ax.set_ylabel("count")

        ax = axes[1, 0]
        ax.bar(
            self.omega_edges[:-1],
            self.omega_hist,
            width=np.diff(self.omega_edges),
            align="edge",
            color="mediumpurple",
            edgecolor="none",
        )
        ax.set_title("Angular velocity histogram so far")
        ax.set_xlabel("omega (rad/s)")
        ax.set_ylabel("count")

        axes[1, 1].remove()
        centers = 0.5 * (self.hd_edges[:-1] + self.hd_edges[1:])
        width = self.hd_edges[1] - self.hd_edges[0]
        ax = fig.add_subplot(2, 3, 5, projection="polar")
        ax.bar(
            centers,
            self.hd_hist,
            width=width,
            color="seagreen",
            edgecolor="none",
            alpha=0.8,
        )
        ax.set_title("Head direction so far", pad=12)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)

        ax = axes[1, 2]
        eta_text = (
            f"{eta_seconds:.1f}s" if np.isfinite(eta_seconds) else "estimating..."
        )
        ax.axis("off")
        lines = [
            "Generation progress",
            "",
            f"completed: {self.last_completed_samples}/{self.total_samples}",
            f"progress: {progress * 100:.1f}%",
            f"chunks done: {self.completed_chunks}",
            f"throughput: {throughput:.1f} traj/s",
            f"elapsed: {elapsed:.1f}s",
            f"eta: {eta_text}",
        ]
        if self.last_chunk_range is not None:
            lines.append(
                f"last chunk: [{self.last_chunk_range[0]}, {self.last_chunk_range[1]})"
            )
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12)

        fig.suptitle("Trajectory generation preview", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(self.save_path, dpi=150)
        plt.close(fig)


class GenerationMonitor:
    """Update console progress and optional preview images from chunk callbacks."""

    def __init__(
        self,
        label: str,
        total_samples: int,
        env_size: float,
        progress_output: str = None,
        progress_every: int = 4,
    ) -> None:
        self.label = label
        self.total_samples = total_samples
        self.start_time = time.time()
        self.progress_bar = (
            tqdm(total=total_samples, desc=label, unit="traj")
            if tqdm is not None
            else None
        )
        self.preview = None
        if progress_output is not None:
            self.preview = GenerationProgressPreview(
                total_samples=total_samples,
                env_size=env_size,
                save_path=progress_output,
                refresh_every=progress_every,
            )

    def update(self, event: dict) -> None:
        chunk_count = event["chunk_end"] - event["chunk_start"]
        elapsed = time.time() - self.start_time
        throughput = event["completed_samples"] / elapsed if elapsed > 0.0 else 0.0
        remaining = max(self.total_samples - event["completed_samples"], 0)
        eta_seconds = remaining / throughput if throughput > 0.0 else float("inf")

        if self.progress_bar is not None:
            self.progress_bar.update(chunk_count)
            postfix = {"traj/s": f"{throughput:.1f}"}
            if np.isfinite(eta_seconds):
                postfix["eta"] = f"{eta_seconds:.1f}s"
            self.progress_bar.set_postfix(postfix)

        if self.preview is not None:
            self.preview.update(event, elapsed=elapsed, throughput=throughput)

    def finalize(self) -> None:
        elapsed = time.time() - self.start_time
        throughput = self.total_samples / elapsed if elapsed > 0.0 else 0.0
        if self.preview is not None:
            self.preview.write(elapsed=elapsed, throughput=throughput)
            print(f"Progress preview saved to {self.preview.save_path}")
        if self.progress_bar is not None:
            self.progress_bar.close()


def generate_dataset_file(
    output_path: str,
    num_samples: int,
    seq_len: int,
    env_size: float,
    velocity_noise,
    seed: int,
    visualize_output: str = None,
    animation_output: str = None,
    animation_fps: int = 20,
    num_workers: int = 1,
    progress_output: str = None,
    progress_every: int = 4,
) -> TrajectoryDataset:
    """Generate one dataset file and optionally emit its visualization PDF."""
    print(
        f"Generating {num_samples} trajectories for {output_path} "
        f"(seq_len={seq_len}, env_size={env_size} m, seed={seed}, "
        f"workers={num_workers}) ..."
    )
    monitor = GenerationMonitor(
        label=f"gen:{os.path.basename(output_path)}",
        total_samples=num_samples,
        env_size=env_size,
        progress_output=progress_output,
        progress_every=progress_every,
    )
    try:
        dataset = TrajectoryDataset(
            num_samples=num_samples,
            seq_len=seq_len,
            env_size=env_size,
            velocity_noise=velocity_noise,
            seed=seed,
            num_workers=num_workers,
            progress_callback=monitor.update,
        )
    finally:
        monitor.finalize()
    dataset.save(output_path)

    if visualize_output is not None:
        visualize(dataset, visualize_output)

    if animation_output is not None:
        visualize_animation(dataset, animation_output, fps=animation_fps)

    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load config
    cfg = load_config(args.config)
    output_path = args.output or getattr(cfg.training, "data_path", None)
    if output_path is None:
        raise ValueError(
            "No output path provided. Set --output or configure training.data_path."
        )

    # Resolve generation parameters (CLI overrides config)
    num_samples = args.num_samples or (
        cfg.training.steps_per_epoch * cfg.training.batch_size
    )
    seq_len = args.seq_len or cfg.task.seq_len
    env_size = args.env_size or cfg.task.env_size
    seed = args.seed if args.seed is not None else cfg.task.neurons_seed

    vis_path = None
    if args.visualize:
        vis_path = args.vis_output
        if vis_path is None:
            base = output_path
            vis_path = (base[:-4] if base.endswith(".npz") else base) + "_vis.pdf"

    anim_path = None
    if args.animate:
        anim_path = args.anim_output
        if anim_path is None:
            base = output_path
            anim_path = (base[:-4] if base.endswith(".npz") else base) + "_traj.mp4"

    progress_path = None
    if args.visualize_progress:
        progress_path = args.progress_output
        if progress_path is None:
            base = output_path
            progress_path = (
                base[:-4] if base.endswith(".npz") else base
            ) + "_progress.png"

    generate_dataset_file(
        output_path=output_path,
        num_samples=num_samples,
        seq_len=seq_len,
        env_size=env_size,
        velocity_noise=cfg.task.velocity_noise,
        seed=seed,
        visualize_output=vis_path,
        animation_output=anim_path,
        animation_fps=args.anim_fps,
        num_workers=args.num_workers,
        progress_output=progress_path,
        progress_every=args.progress_every,
    )

    if args.eval_output is not None:
        eval_num_samples = args.eval_num_samples or cfg.training.eval_batch_size
        eval_seed = args.eval_seed if args.eval_seed is not None else seed + 1
        eval_progress_path = None
        if args.visualize_progress:
            eval_progress_path = args.eval_progress_output
            if eval_progress_path is None:
                base = args.eval_output
                eval_progress_path = (
                    base[:-4] if base.endswith(".npz") else base
                ) + "_progress.png"
        generate_dataset_file(
            output_path=args.eval_output,
            num_samples=eval_num_samples,
            seq_len=seq_len,
            env_size=env_size,
            velocity_noise=cfg.task.velocity_noise,
            seed=eval_seed,
            num_workers=args.num_workers,
            progress_output=eval_progress_path,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
