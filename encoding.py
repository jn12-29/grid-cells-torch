"""Centralize ensemble-based encoding helpers shared across the project.

This module keeps place-cell and head-direction-cell encoding logic in one
location so training, dataset workers, and animation preparation can reuse the
same implementation while preserving the repository's existing public helpers.

Usage:
    encoder = EnsembleEncoder(pc_ensembles, hdc_ensembles)
    init_cond = encoder.encode_initial_conditions(init_pos, init_hd)
    targets = encoder.encode_targets(target_pos, target_hd)
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble


def decode_position_from_pc_activations(
    pc_acts: np.ndarray,
    pc_centers: np.ndarray,
) -> np.ndarray:
    """Decode positions from place-cell activation probabilities via weighted means."""
    pc_acts = np.asarray(pc_acts, dtype=np.float32)
    pc_centers = np.asarray(pc_centers, dtype=np.float32)
    if pc_acts.ndim < 2:
        raise ValueError("pc_acts must include a cell dimension.")
    if pc_centers.ndim != 2 or pc_centers.shape[1] != 2:
        raise ValueError("pc_centers must have shape (n_pc, 2).")
    if pc_acts.shape[-1] != pc_centers.shape[0]:
        raise ValueError(
            "pc_acts and pc_centers must agree on the number of place cells."
        )
    return np.tensordot(pc_acts, pc_centers, axes=([-1], [0])).astype(np.float32)


@dataclass
class EncodedTargets:
    """Container for encoded supervision targets from all ensembles."""

    pc_targets: List[np.ndarray]
    hdc_targets: List[np.ndarray]


@dataclass
class AnimationInputs:
    """Container for eval-style animation payloads built from a dataset."""

    target_pos: np.ndarray
    pred_pos: np.ndarray
    pc_acts: np.ndarray
    hdc_acts: np.ndarray
    pc_centers: np.ndarray
    hdc_centers: np.ndarray

    def as_dict(self) -> dict:
        """Return the legacy dict payload expected by existing callers."""
        return {
            "target_pos": self.target_pos,
            "pred_pos": self.pred_pos,
            "pc_acts": self.pc_acts,
            "hdc_acts": self.hdc_acts,
            "pc_centers": self.pc_centers,
            "hdc_centers": self.hdc_centers,
        }


class EnsembleEncoder:
    """Encode trajectories and initial conditions for place/HDC ensembles."""

    def __init__(
        self,
        pc_ensembles: List[PlaceCellEnsemble],
        hdc_ensembles: List[HeadDirectionCellEnsemble],
    ) -> None:
        self.pc_ensembles = list(pc_ensembles)
        self.hdc_ensembles = list(hdc_ensembles)

    @staticmethod
    def _to_numpy(values, dtype=np.float32) -> np.ndarray:
        """Convert numpy/tensor inputs into detached CPU numpy arrays."""
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().to(torch.float32).numpy()
        return np.asarray(values, dtype=dtype)

    def _prepare_init_values(self, values, name: str, feature_dim: int) -> np.ndarray:
        """Normalize init values to shape (batch, 1, feature_dim)."""
        values = self._to_numpy(values)
        if values.ndim == 2:
            values = values[:, np.newaxis, :]
        if values.ndim != 3 or values.shape[-1] != feature_dim:
            raise ValueError(
                f"{name} must have shape (batch, {feature_dim}) or "
                f"(batch, 1, {feature_dim}), got {values.shape}."
            )
        return values.astype(np.float32, copy=False)

    def _prepare_target_values(self, values, name: str, feature_dim: int) -> np.ndarray:
        """Normalize target values to shape (batch, seq_len, feature_dim)."""
        values = self._to_numpy(values)
        if values.ndim == 2:
            values = values[np.newaxis, :, :]
        if values.ndim != 3 or values.shape[-1] != feature_dim:
            raise ValueError(
                f"{name} must have shape (batch, seq_len, {feature_dim}) or "
                f"(seq_len, {feature_dim}), got {values.shape}."
            )
        return values.astype(np.float32, copy=False)

    def encode_initial_conditions(self, init_pos, init_hd) -> np.ndarray:
        """Encode initial position and heading into concatenated ensemble codes."""
        init_pos_np = self._prepare_init_values(init_pos, "init_pos", 2)
        init_hd_np = self._prepare_init_values(init_hd, "init_hd", 1)

        parts = []
        for ens in self.pc_ensembles:
            parts.append(ens.get_init(init_pos_np)[:, 0, :])
        for ens in self.hdc_ensembles:
            parts.append(ens.get_init(init_hd_np)[:, 0, :])

        if not parts:
            raise ValueError("At least one ensemble is required to encode initial conditions.")

        return np.concatenate(parts, axis=-1).astype(np.float32)

    def encode_targets(self, target_pos, target_hd) -> EncodedTargets:
        """Encode trajectory targets for all place and head-direction ensembles."""
        target_pos_np = self._prepare_target_values(target_pos, "target_pos", 2)
        target_hd_np = self._prepare_target_values(target_hd, "target_hd", 1)

        return EncodedTargets(
            pc_targets=[ens.get_targets(target_pos_np) for ens in self.pc_ensembles],
            hdc_targets=[ens.get_targets(target_hd_np) for ens in self.hdc_ensembles],
        )

    def concat_pc_centers(self) -> np.ndarray:
        """Concatenate place-cell centres for downstream decoding/plotting."""
        if not self.pc_ensembles:
            raise ValueError("At least one place-cell ensemble is required.")
        return np.concatenate([ens.means for ens in self.pc_ensembles], axis=0).astype(
            np.float32
        )

    def concat_hdc_centers(self) -> np.ndarray:
        """Concatenate HDC preferred angles for downstream plotting."""
        if not self.hdc_ensembles:
            raise ValueError("At least one head-direction ensemble is required.")
        return np.concatenate(
            [ens.means.reshape(-1) for ens in self.hdc_ensembles], axis=0
        ).astype(np.float32)

    def prepare_animation_inputs(self, dataset, max_trajectories: int = 4) -> AnimationInputs:
        """Build the shared eval-style animation payload from a dataset."""
        target_pos_all = dataset._data["target_pos"]
        target_hd_all = dataset._data["target_hd"]
        num_show = min(int(max_trajectories), len(target_pos_all))
        if num_show <= 0:
            raise ValueError("Cannot animate an empty dataset.")

        target_pos = target_pos_all[:num_show]
        target_hd = target_hd_all[:num_show]
        encoded_targets = self.encode_targets(target_pos, target_hd)
        pc_acts = np.concatenate(encoded_targets.pc_targets, axis=-1)
        hdc_acts = np.concatenate(encoded_targets.hdc_targets, axis=-1)
        pc_centers = self.concat_pc_centers()
        hdc_centers = self.concat_hdc_centers()
        pred_pos = decode_position_from_pc_activations(pc_acts, pc_centers)

        return AnimationInputs(
            target_pos=target_pos,
            pred_pos=pred_pos,
            pc_acts=pc_acts,
            hdc_acts=hdc_acts,
            pc_centers=pc_centers,
            hdc_centers=hdc_centers,
        )
