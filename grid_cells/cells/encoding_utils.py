"""Encoding and decoding helpers shared by training and dataset tooling."""

from typing import List, Tuple

import torch

from grid_cells.cells.encoding import EnsembleEncoder
from grid_cells.cells.decoding import (
    compute_head_direction_mae_rad,
    compute_position_mse,
    decode_head_direction_from_hdc_logits,
    decode_position_from_pc_activations,
    decode_position_from_pc_activations_top_k,
    decode_position_from_pc_logits,
)
from grid_cells.cells.ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble


def encode_initial_conditions(
    batch: dict,
    pc_ensembles: List[PlaceCellEnsemble],
    hdc_ensembles: List[HeadDirectionCellEnsemble],
) -> torch.Tensor:
    """Encode initial position and head direction into cell activations."""
    encoder = EnsembleEncoder(pc_ensembles, hdc_ensembles)
    return torch.from_numpy(
        encoder.encode_initial_conditions(batch["init_pos"], batch["init_hd"])
    )


def encode_targets(
    batch: dict,
    pc_ensembles: List[PlaceCellEnsemble],
    hdc_ensembles: List[HeadDirectionCellEnsemble],
) -> Tuple[list, list]:
    """Encode target trajectories into place-cell and HDC activations."""
    encoder = EnsembleEncoder(pc_ensembles, hdc_ensembles)
    encoded_targets = encoder.encode_targets(batch["target_pos"], batch["target_hd"])
    return encoded_targets.pc_targets, encoded_targets.hdc_targets


def prepare_dataset_animation_inputs(
    dataset,
    pc_ensembles: List[PlaceCellEnsemble],
    hdc_ensembles: List[HeadDirectionCellEnsemble],
    max_trajectories: int = 4,
) -> dict:
    """Build eval-style animation payloads directly from a generated dataset."""
    encoder = EnsembleEncoder(pc_ensembles, hdc_ensembles)
    return encoder.prepare_animation_inputs(
        dataset,
        max_trajectories=max_trajectories,
    ).as_dict()
