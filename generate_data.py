"""
Generate and optionally visualize trajectory data for grid cell training.

Usage
-----
# Generate 100 000 trajectories with default config, save to data/train.npz
python generate_data.py --output data/train.npz

# Visualize without saving
python generate_data.py --output data/train.npz --visualize

# Generate a smaller evaluation set
python generate_data.py --output data/eval.npz --num_samples 4000

# Use a custom config
python generate_data.py --config my_config.yaml --output data/train.npz
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")          # headless-safe; override below if displaying
import matplotlib.pyplot as plt
import numpy as np

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
        "--config", default="config.yaml",
        help="Path to YAML config (default: config.yaml)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .npz file path, e.g. data/train.npz"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of trajectories to generate. "
             "Defaults to steps_per_epoch × batch_size from config."
    )
    parser.add_argument(
        "--seq_len", type=int, default=None,
        help="Steps per trajectory (overrides config)"
    )
    parser.add_argument(
        "--env_size", type=float, default=None,
        help="Environment side length in metres (overrides config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config neurons_seed)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate and save a visualisation PDF alongside the .npz file"
    )
    parser.add_argument(
        "--vis_output", default=None,
        help="Path for the visualisation PDF. "
             "Defaults to <output>.pdf (replacing .npz extension)"
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

    half     = dataset.env_size / 2.0
    pos_all  = dataset._data["target_pos"]   # (N, T, 2)
    vel_all  = dataset._data["ego_vel"]      # (N, T, 3)
    hd_all   = dataset._data["target_hd"]    # (N, T, 1)
    N, T, _  = pos_all.shape

    speeds = np.sqrt(vel_all[:, :, 0] ** 2 + vel_all[:, :, 1] ** 2).ravel()  # m/s
    omegas = vel_all[:, :, 2].ravel()                                          # rad/s
    hds    = hd_all.ravel()                                                    # rad

    with PdfPages(save_path) as pdf:

        # ------------------------------------------------------------------
        # Page 1: overview
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(18, 12))
        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

        # --- panel 1: sample trajectories ---
        ax_traj = fig.add_subplot(gs[0, :2])
        n_show  = min(16, N)
        cmap    = plt.cm.tab20
        for i in range(n_show):
            xy = pos_all[i]                   # (T, 2)
            ax_traj.plot(xy[:, 0], xy[:, 1],
                         color=cmap(i / n_show), alpha=0.7, linewidth=0.8)
            ax_traj.scatter(xy[0, 0], xy[0, 1],
                            color=cmap(i / n_show), s=20, zorder=3)
        ax_traj.set_xlim(-half, half)
        ax_traj.set_ylim(-half, half)
        ax_traj.set_aspect("equal")
        ax_traj.set_title(f"Sample trajectories (n={n_show})")
        ax_traj.set_xlabel("x (m)")
        ax_traj.set_ylabel("y (m)")
        rect = plt.Rectangle((-half, -half), dataset.env_size, dataset.env_size,
                              fill=False, edgecolor="k", linewidth=1.5)
        ax_traj.add_patch(rect)

        # --- panel 2: position coverage heatmap ---
        ax_heat = fig.add_subplot(gs[0, 2])
        nbins   = 40
        heatmap, xedges, yedges = np.histogram2d(
            pos_all[:, :, 0].ravel(),
            pos_all[:, :, 1].ravel(),
            bins=nbins,
            range=[[-half, half], [-half, half]],
        )
        im = ax_heat.imshow(
            heatmap.T, origin="lower",
            extent=[-half, half, -half, half],
            cmap="hot", aspect="equal",
        )
        plt.colorbar(im, ax=ax_heat, label="visit count")
        ax_heat.set_title("Position coverage")
        ax_heat.set_xlabel("x (m)")
        ax_heat.set_ylabel("y (m)")

        # --- panel 3: speed distribution ---
        ax_spd = fig.add_subplot(gs[0, 3])
        ax_spd.hist(speeds, bins=60, color="steelblue", edgecolor="none", density=True)
        ax_spd.axvline(speeds.mean(), color="tomato", linestyle="--",
                       label=f"mean={speeds.mean():.3f} m/s")
        ax_spd.set_title("Translational speed")
        ax_spd.set_xlabel("speed (m/s)")
        ax_spd.set_ylabel("density")
        ax_spd.legend(fontsize=8)

        # --- panel 4: angular velocity distribution ---
        ax_omg = fig.add_subplot(gs[1, 0])
        ax_omg.hist(omegas, bins=60, color="mediumpurple", edgecolor="none", density=True)
        ax_omg.axvline(omegas.mean(), color="tomato", linestyle="--",
                       label=f"mean={omegas.mean():.3f} rad/s")
        ax_omg.set_title("Angular velocity")
        ax_omg.set_xlabel("ω (rad/s)")
        ax_omg.set_ylabel("density")
        ax_omg.legend(fontsize=8)

        # --- panel 5: head direction rose (polar histogram) ---
        ax_hd  = fig.add_subplot(gs[1, 1], projection="polar")
        n_bins = 36
        counts, bin_edges = np.histogram(hds, bins=n_bins, range=(-np.pi, np.pi))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        width       = 2 * np.pi / n_bins
        ax_hd.bar(bin_centers, counts, width=width, bottom=0,
                  color="seagreen", alpha=0.8, edgecolor="none")
        ax_hd.set_title("Head direction distribution", pad=12)
        ax_hd.set_theta_zero_location("E")
        ax_hd.set_theta_direction(1)

        # --- panel 6: displacement vs recorded velocity sanity check ---
        # For non-reflection steps, |Δpos| / dt should equal recorded speed
        ax_san = fig.add_subplot(gs[1, 2:])
        sample_idx = min(200, N)
        pos_sample = pos_all[:sample_idx]       # (M, T, 2)
        vel_sample = vel_all[:sample_idx, :, :2]  # (M, T, 2)
        init_pos   = dataset._data["init_pos"][:sample_idx]  # (M, 2)
        full_pos   = np.concatenate(
            [init_pos[:, np.newaxis, :], pos_sample], axis=1
        )  # (M, T+1, 2)
        displace   = np.linalg.norm(np.diff(full_pos, axis=1), axis=-1).ravel()  # (M*T,)
        recorded   = np.linalg.norm(vel_sample, axis=-1).ravel() * TrajectoryDataset._DT
        ax_san.scatter(recorded, displace, s=1, alpha=0.15, color="gray")
        # ideal line
        lim = max(displace.max(), recorded.max()) * 1.05
        ax_san.plot([0, lim], [0, lim], "r--", linewidth=1, label="ideal (y=x)")
        ax_san.set_xlim(0, lim)
        ax_san.set_ylim(0, lim)
        ax_san.set_title("Velocity–displacement consistency\n(non-reflection steps cluster on y=x)")
        ax_san.set_xlabel("recorded |v|·dt  (m)")
        ax_san.set_ylabel("|Δpos|  (m)")
        ax_san.legend(fontsize=8)

        # stats box
        n_total   = len(displace)
        n_matched = int(np.sum(np.abs(displace - recorded) < 1e-4))
        ax_san.text(
            0.98, 0.05,
            f"Exact match: {n_matched}/{n_total} ({100*n_matched/n_total:.1f}%)\n"
            f"(mismatches = boundary reflection steps)",
            transform=ax_san.transAxes,
            ha="right", va="bottom", fontsize=8,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # Resolve generation parameters (CLI overrides config)
    num_samples = args.num_samples or (
        cfg.training.steps_per_epoch * cfg.training.batch_size
    )
    seq_len  = args.seq_len  or cfg.task.seq_len
    env_size = args.env_size or cfg.task.env_size
    seed     = args.seed     if args.seed is not None else cfg.task.neurons_seed

    print(
        f"Generating {num_samples} trajectories  "
        f"(seq_len={seq_len}, env_size={env_size} m, seed={seed}) ..."
    )

    dataset = TrajectoryDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        env_size=env_size,
        velocity_noise=cfg.task.velocity_noise,
        seed=seed,
    )

    # Save
    dataset.save(args.output)

    # Optionally visualize
    if args.visualize:
        vis_path = args.vis_output
        if vis_path is None:
            base = args.output
            vis_path = (base[:-4] if base.endswith(".npz") else base) + "_vis.pdf"
        visualize(dataset, vis_path)


if __name__ == "__main__":
    main()
