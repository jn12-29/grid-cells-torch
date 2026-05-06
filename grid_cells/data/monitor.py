"""Progress monitoring and preview generation for dataset synthesis."""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


class GenerationProgressPreview:
    """Accumulate lightweight dataset stats and periodically save a preview PNG."""

    def __init__(
        self,
        total_samples: int,
        env_size: float,
        save_path: str,
        refresh_every: int = 4,
        spatial_bins: int = 32,
        directional_bins: int = 20,
    ) -> None:
        self.total_samples = total_samples
        self.env_size = env_size
        self.save_path = save_path
        self.refresh_every = max(1, int(refresh_every))
        self.completed_chunks = 0
        self.sample_trajectories = []
        self.heatmap_bins = int(spatial_bins)
        self.heatmap = np.zeros((self.heatmap_bins, self.heatmap_bins), dtype=np.float64)
        self.speed_edges = np.linspace(0.0, 1.0, int(spatial_bins) + 1)
        self.speed_hist = np.zeros(len(self.speed_edges) - 1, dtype=np.int64)
        self.omega_edges = np.linspace(-8.0, 8.0, int(directional_bins) + 1)
        self.omega_hist = np.zeros(len(self.omega_edges) - 1, dtype=np.int64)
        self.hd_edges = np.linspace(-np.pi, np.pi, int(directional_bins) + 1)
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
        eta_text = f"{eta_seconds:.1f}s" if np.isfinite(eta_seconds) else "estimating..."
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
        spatial_bins: int = 32,
        directional_bins: int = 20,
    ) -> None:
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
                spatial_bins=spatial_bins,
                directional_bins=directional_bins,
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
