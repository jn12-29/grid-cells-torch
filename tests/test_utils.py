"""Tests for encoding, decoding, and metric helpers.

Usage:
    pytest tests/test_utils.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset import TrajectoryDataset
from ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble
from utils import (
    compute_position_mse,
    decode_position_from_pc_activations,
    decode_position_from_pc_logits,
    prepare_dataset_animation_inputs,
)


def test_decode_position_from_pc_logits_returns_weighted_mean():
    """Uniform logits should decode to the mean of place-cell centres."""
    ensemble = PlaceCellEnsemble(2, stdev=0.35, pos_min=0.0, pos_max=1.0, seed=0)
    ensemble.means = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32).numpy()
    logits = [torch.zeros(1, 1, 2)]

    decoded = decode_position_from_pc_logits(logits, [ensemble])

    assert torch.allclose(decoded, torch.tensor([[[1.0, 1.0]]]))


def test_compute_position_mse_matches_decoded_targets():
    """Position MSE should be zero when decoded positions match the targets."""
    ensemble = PlaceCellEnsemble(2, stdev=0.35, pos_min=0.0, pos_max=1.0, seed=0)
    ensemble.means = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32).numpy()
    logits = [torch.tensor([[[12.0, -12.0]]])]
    target_pos = torch.tensor([[[0.0, 0.0]]])

    mse = compute_position_mse(logits, target_pos, [ensemble])

    assert torch.isclose(mse, torch.tensor(0.0), atol=1e-6)


def test_decode_position_from_pc_activations_returns_weighted_mean():
    """Numpy place-cell probabilities should decode via the same weighted mean rule."""
    pc_acts = np.array([[[0.25, 0.75]]], dtype=np.float32)
    pc_centers = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)

    decoded = decode_position_from_pc_activations(pc_acts, pc_centers)

    assert np.allclose(decoded, np.array([[[1.5, 1.5]]], dtype=np.float32))


def test_prepare_dataset_animation_inputs_builds_eval_style_payload(tmp_path):
    """Generated datasets should be convertible into the shared 3-panel animation inputs."""
    output_path = tmp_path / "train.npz"
    dataset = TrajectoryDataset(
        num_samples=3,
        seq_len=4,
        env_size=2.2,
        velocity_noise=(0.0, 0.0, 0.0),
        seed=3,
    )
    dataset.save(output_path)
    pc_ens = [PlaceCellEnsemble(5, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(6, concentration=20.0, seed=0)]

    anim_inputs = prepare_dataset_animation_inputs(
        dataset,
        pc_ens,
        hdc_ens,
        max_trajectories=2,
    )

    assert anim_inputs["target_pos"].shape == (2, 4, 2)
    assert anim_inputs["pred_pos"].shape == (2, 4, 2)
    assert anim_inputs["pc_acts"].shape == (2, 4, 5)
    assert anim_inputs["hdc_acts"].shape == (2, 4, 6)
    assert anim_inputs["pc_centers"].shape == (5, 2)
    assert anim_inputs["hdc_centers"].shape == (6,)
