"""Tests for encoding/decoding utility helpers."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ensembles import PlaceCellEnsemble
from utils import compute_position_mse, decode_position_from_pc_logits


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
