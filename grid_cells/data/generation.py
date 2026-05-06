"""Dataset generation workflow helpers."""

import os

from grid_cells.data.dataset import TrajectoryDataset
from grid_cells.data.monitor import GenerationMonitor


def generate_dataset_file(
    output_path: str,
    num_samples: int,
    seq_len: int,
    env_size: float,
    velocity_noise,
    seed: int,
    *,
    motion_params: dict = None,
    visualize_fn,
    visualize_animation_fn=None,
    pc_ensembles=None,
    hdc_ensembles=None,
    visualize_output: str = None,
    animation_output: str = None,
    animation_num_trajectories: int = 4,
    animation_fps: int = 20,
    animation_step: int = 4,
    anim_workers: int = 8,
    num_workers: int = 1,
    progress_output: str = None,
    progress_every: int = 4,
    spatial_bins: int = 32,
    directional_bins: int = 20,
) -> TrajectoryDataset:
    """Generate one dataset file and optionally emit its visualization artifacts."""
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
        spatial_bins=spatial_bins,
        directional_bins=directional_bins,
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
            **(motion_params or {}),
        )
    finally:
        monitor.finalize()
    dataset.save(output_path)

    if visualize_output is not None:
        visualize_fn(
            dataset,
            visualize_output,
            spatial_bins=spatial_bins,
            directional_bins=directional_bins,
        )

    if animation_output is not None:
        if pc_ensembles is None or hdc_ensembles is None:
            raise ValueError(
                "Animation export requires place-cell and head-direction ensembles."
            )
        if visualize_animation_fn is None:
            raise ValueError("Animation export requires a visualize_animation function.")
        visualize_animation_fn(
            dataset,
            animation_output,
            pc_ensembles=pc_ensembles,
            hdc_ensembles=hdc_ensembles,
            fps=animation_fps,
            max_trajectories=animation_num_trajectories,
            step=animation_step,
            num_workers=anim_workers,
        )

    return dataset
