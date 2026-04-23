"""Encoding and decoding helpers shared by training and dataset tooling."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from grid_cells.cells.encoding import EnsembleEncoder
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


def decode_position_from_pc_logits(
    pc_logits: List[torch.Tensor],
    pc_ensembles: List[PlaceCellEnsemble],
) -> torch.Tensor:
    """Decode positions from place-cell logits via a weighted mean."""
    if not pc_logits or not pc_ensembles:
        raise ValueError("At least one place-cell ensemble is required to decode positions.")

    decoded_positions = []
    for logits, ensemble in zip(pc_logits, pc_ensembles):
        probs = F.softmax(logits, dim=-1)
        means = torch.as_tensor(
            ensemble.means,
            dtype=logits.dtype,
            device=logits.device,
        )
        decoded_positions.append(torch.matmul(probs, means))

    return torch.stack(decoded_positions, dim=0).mean(dim=0)


def compute_position_mse(
    pc_logits: List[torch.Tensor],
    target_pos: torch.Tensor,
    pc_ensembles: List[PlaceCellEnsemble],
) -> torch.Tensor:
    """Compute mean-squared error between decoded and ground-truth positions."""
    pred_pos = decode_position_from_pc_logits(pc_logits, pc_ensembles)
    return F.mse_loss(pred_pos, target_pos)


def decode_position_from_pc_activations(
    pc_acts: np.ndarray,
    pc_centers: np.ndarray,
) -> np.ndarray:
    """Decode positions from place-cell probabilities via weighted means."""
    return np.einsum("...n,nc->...c", pc_acts, pc_centers)


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
