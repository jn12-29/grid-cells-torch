"""Visualization helpers for generated trajectory datasets."""

import matplotlib.pyplot as plt
import numpy as np

from grid_cells.data.dataset import TrajectoryDataset


def visualize_dataset(
    dataset: TrajectoryDataset,
    save_path: str,
    spatial_bins: int = 32,
    directional_bins: int = 20,
) -> None:
    """Produce a multi-panel dataset summary figure and save it as PDF."""
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages

    half = dataset.env_size / 2.0
    pos_all = dataset._data["target_pos"]
    vel_all = dataset._data["ego_vel"]
    hd_all = dataset._data["target_hd"]
    num_trajectories, seq_len, _ = pos_all.shape

    speeds = np.sqrt(vel_all[:, :, 0] ** 2 + vel_all[:, :, 1] ** 2).ravel()
    omegas = vel_all[:, :, 2].ravel()
    hds = hd_all.ravel()

    with PdfPages(save_path) as pdf:
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

        ax_traj = fig.add_subplot(gs[0, :2])
        n_show = min(16, num_trajectories)
        cmap = plt.cm.tab20
        for idx in range(n_show):
            xy = pos_all[idx]
            ax_traj.plot(
                xy[:, 0],
                xy[:, 1],
                color=cmap(idx / n_show),
                alpha=0.7,
                linewidth=0.8,
            )
            ax_traj.scatter(xy[0, 0], xy[0, 1], color=cmap(idx / n_show), s=20, zorder=3)
        ax_traj.set_xlim(-half, half)
        ax_traj.set_ylim(-half, half)
        ax_traj.set_aspect("equal")
        ax_traj.set_title(f"Sample trajectories (n={n_show})")
        ax_traj.set_xlabel("x (m)")
        ax_traj.set_ylabel("y (m)")
        ax_traj.add_patch(
            plt.Rectangle(
                (-half, -half),
                dataset.env_size,
                dataset.env_size,
                fill=False,
                edgecolor="k",
                linewidth=1.5,
            )
        )

        ax_heat = fig.add_subplot(gs[0, 2])
        heatmap, _, _ = np.histogram2d(
            pos_all[:, :, 0].ravel(),
            pos_all[:, :, 1].ravel(),
            bins=spatial_bins,
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

        ax_spd = fig.add_subplot(gs[0, 3])
        ax_spd.hist(
            speeds,
            bins=spatial_bins,
            color="steelblue",
            edgecolor="none",
            density=True,
        )
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

        ax_omega = fig.add_subplot(gs[1, 0])
        ax_omega.hist(
            omegas,
            bins=directional_bins,
            color="mediumpurple",
            edgecolor="none",
            density=True,
        )
        ax_omega.axvline(
            omegas.mean(),
            color="tomato",
            linestyle="--",
            label=f"mean={omegas.mean():.3f} rad/s",
        )
        ax_omega.set_title("Angular velocity")
        ax_omega.set_xlabel("ω (rad/s)")
        ax_omega.set_ylabel("density")
        ax_omega.legend(fontsize=8)

        ax_hd = fig.add_subplot(gs[1, 1], projection="polar")
        counts, bin_edges = np.histogram(
            hds,
            bins=directional_bins,
            range=(-np.pi, np.pi),
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        width = 2 * np.pi / directional_bins
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

        ax_sanity = fig.add_subplot(gs[1, 2:])
        sample_idx = min(200, num_trajectories)
        pos_sample = pos_all[:sample_idx]
        vel_sample = vel_all[:sample_idx, :, :2]
        init_pos = dataset._data["init_pos"][:sample_idx]
        full_pos = np.concatenate([init_pos[:, np.newaxis, :], pos_sample], axis=1)
        displace = np.linalg.norm(np.diff(full_pos, axis=1), axis=-1).ravel()
        recorded = np.linalg.norm(vel_sample, axis=-1).ravel() * TrajectoryDataset._DT
        ax_sanity.scatter(recorded, displace, s=1, alpha=0.15, color="gray")
        limit = max(displace.max(), recorded.max()) * 1.05
        ax_sanity.plot([0, limit], [0, limit], "r--", linewidth=1, label="ideal (y=x)")
        ax_sanity.set_xlim(0, limit)
        ax_sanity.set_ylim(0, limit)
        ax_sanity.set_title(
            "Velocity-displacement consistency\n(non-reflection steps cluster on y=x)"
        )
        ax_sanity.set_xlabel("recorded |v|·dt  (m)")
        ax_sanity.set_ylabel("|Δpos|  (m)")
        ax_sanity.legend(fontsize=8)

        n_total = len(displace)
        n_matched = int(np.sum(np.abs(displace - recorded) < 1e-4))
        ax_sanity.text(
            0.98,
            0.05,
            f"Exact match: {n_matched}/{n_total} ({100 * n_matched / n_total:.1f}%)\n"
            "(mismatches = boundary reflection steps)",
            transform=ax_sanity.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.suptitle(
            "Trajectory dataset summary\n"
            f"{num_trajectories} trajectories x {seq_len} steps  |  "
            f"env {dataset.env_size} m x {dataset.env_size} m  |  "
            f"dt={TrajectoryDataset._DT} s",
            fontsize=13,
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Visualisation saved to {save_path}")


def visualize_dataset_animation(
    dataset: TrajectoryDataset,
    save_path: str,
    pc_ensembles,
    hdc_ensembles,
    *,
    prepare_dataset_animation_inputs,
    generate_trajectory_animation,
    fps: int = 20,
    max_trajectories: int = 4,
    step: int = 4,
    num_workers: int = 8,
) -> None:
    """Save an eval-style animation for generated trajectories."""
    anim_inputs = prepare_dataset_animation_inputs(
        dataset,
        pc_ensembles,
        hdc_ensembles,
        max_trajectories=max_trajectories,
    )
    generate_trajectory_animation(
        anim_inputs["target_pos"],
        anim_inputs["pred_pos"],
        anim_inputs["pc_acts"],
        anim_inputs["hdc_acts"],
        anim_inputs["pc_centers"],
        anim_inputs["hdc_centers"],
        dataset.env_size,
        save_path,
        fps=fps,
        step=step,
        num_workers=num_workers,
        title_prefix="Data traj",
        pred_label="decoded",
    )
