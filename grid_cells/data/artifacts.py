"""Backward-compatible facade for dataset generation artifacts."""

from grid_cells.data.generation import generate_dataset_file
from grid_cells.data.monitor import GenerationMonitor, GenerationProgressPreview
from grid_cells.data.visualization import visualize_dataset, visualize_dataset_animation

__all__ = [
    "GenerationMonitor",
    "GenerationProgressPreview",
    "generate_dataset_file",
    "visualize_dataset",
    "visualize_dataset_animation",
]
