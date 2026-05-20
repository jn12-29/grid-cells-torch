"""Tests for encoding, decoding, and metric helpers.

Usage:
    pytest tests/test_utils.py
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grid_cells.data.dataset import TrajectoryDataset
from grid_cells.cells.ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble
from grid_cells.cells.encoding_utils import (
    compute_head_direction_mae_rad,
    compute_position_mse,
    decode_position_from_pc_activations,
    decode_head_direction_from_hdc_logits,
    decode_position_from_pc_logits,
    prepare_dataset_animation_inputs,
)
from grid_cells.cells.decoding import (
    decode_head_direction_from_hdc_scores,
    decode_position_from_pc_scores,
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
    """Numpy place-cell activations should decode via normalized weighted mean."""
    pc_acts = np.array([[[2.0, 6.0]]], dtype=np.float32)
    pc_centers = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)

    decoded = decode_position_from_pc_activations(pc_acts, pc_centers)

    assert np.allclose(decoded, np.array([[[1.5, 1.5]]], dtype=np.float32))


def test_decode_position_from_tiny_pc_activations_preserves_relative_weights():
    """Tiny positive activations should not be clamped before normalization."""
    pc_acts = np.array([[[1e-12, 3e-12]]], dtype=np.float32)
    pc_centers = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)

    decoded = decode_position_from_pc_activations(pc_acts, pc_centers)

    assert np.allclose(decoded, np.array([[[1.5, 1.5]]], dtype=np.float32))


def test_decode_position_from_pc_scores_analytic_recovers_position():
    """Analytic PC decoding should invert raw Gaussian log-scores."""
    ensemble = PlaceCellEnsemble(4, stdev=0.2, pos_min=-1.0, pos_max=1.0, seed=0)
    ensemble.means = np.array(
        [[-0.8, -0.8], [0.8, -0.7], [-0.7, 0.9], [0.9, 0.8]],
        dtype=np.float32,
    )
    positions = np.array(
        [[[0.2, -0.1], [-0.35, 0.4]], [[0.0, 0.0], [0.55, 0.25]]],
        dtype=np.float32,
    )

    decoded = decode_position_from_pc_scores(ensemble.unnor_logpdf(positions), ensemble)

    assert np.allclose(decoded, positions, atol=1e-5)


def test_decode_position_from_pc_logits_analytic_recovers_position():
    """Analytic PC-logit decoding should invert softmax-compatible log-scores."""
    ensemble = PlaceCellEnsemble(4, stdev=0.2, pos_min=-1.0, pos_max=1.0, seed=0)
    ensemble.means = np.array(
        [[-0.8, -0.8], [0.8, -0.7], [-0.7, 0.9], [0.9, 0.8]],
        dtype=np.float32,
    )
    positions = np.array([[[0.25, -0.15], [-0.45, 0.3]]], dtype=np.float32)
    logits = [torch.from_numpy(ensemble.unnor_logpdf(positions))]

    decoded = decode_position_from_pc_logits(logits, [ensemble])

    assert torch.allclose(decoded, torch.from_numpy(positions), atol=1e-5)


def test_decode_position_from_pc_logits_analytic_recovers_scaled_log_scores():
    """PC-logit analytic decoding should be invariant to positive score scale."""
    ensemble = PlaceCellEnsemble(4, stdev=0.2, pos_min=-1.0, pos_max=1.0, seed=0)
    ensemble.means = np.array(
        [[-0.8, -0.8], [0.8, -0.7], [-0.7, 0.9], [0.9, 0.8]],
        dtype=np.float32,
    )
    positions = np.array([[[0.25, -0.15], [-0.45, 0.3]]], dtype=np.float32)
    raw_scores = torch.from_numpy(ensemble.unnor_logpdf(positions)).double()
    target = torch.from_numpy(positions).double()

    for scale in (1e-8, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 2.0):
        logits = [raw_scores * scale + 7.0]
        decoded = decode_position_from_pc_logits(logits, [ensemble])

        assert torch.allclose(decoded, target, atol=2e-4)


def test_decode_position_from_pc_logits_falls_back_for_scale_ambiguous_geometry():
    """Scale-invariant PC decoding should fallback when alpha is unidentifiable."""
    ensemble = PlaceCellEnsemble(3, stdev=0.2, pos_min=-1.0, pos_max=1.0, seed=0)
    ensemble.means = np.array(
        [[1.0, 0.0], [-0.5, 0.8660254], [-0.5, -0.8660254]],
        dtype=np.float32,
    )
    logits = torch.tensor([[[2.0, -1.0, 0.5]]], dtype=torch.float32)

    decoded = decode_position_from_pc_logits([logits], [ensemble])
    expected = torch.softmax(logits, dim=-1) @ torch.from_numpy(ensemble.means)

    assert torch.allclose(decoded, expected, atol=1e-6)


def test_decode_position_from_pc_logits_falls_back_for_off_manifold_logits():
    """Off-manifold logits should use weighted decoding instead of exploding."""
    ensemble = PlaceCellEnsemble(5, stdev=0.2, pos_min=-1.0, pos_max=1.0, seed=0)
    ensemble.means = np.array(
        [[-0.8, -0.8], [0.8, -0.7], [-0.7, 0.9], [0.9, 0.8], [0.1, -0.2]],
        dtype=np.float32,
    )
    logits = torch.tensor([[[10.0, -10.0, 7.0, -3.0, 0.0]]], dtype=torch.float32)

    decoded = decode_position_from_pc_logits([logits], [ensemble])
    expected = torch.softmax(logits, dim=-1) @ torch.from_numpy(ensemble.means)

    assert torch.allclose(decoded, expected, atol=1e-6)


def test_decode_position_from_normalized_pc_logits_uses_sigmoid_weights():
    """Normalized PC logits should decode from independent sigmoid activations."""
    ensemble = PlaceCellEnsemble(
        2,
        stdev=0.35,
        pos_min=0.0,
        pos_max=1.0,
        seed=0,
        soft_targets="normalized",
    )
    ensemble.means = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    logits = torch.tensor([[[0.0, 2.0]]], dtype=torch.float32)

    decoded = decode_position_from_pc_logits([logits], [ensemble])
    sigmoid_weights = torch.sigmoid(logits)
    expected = (
        sigmoid_weights / sigmoid_weights.sum(dim=-1, keepdim=True)
    ) @ torch.from_numpy(ensemble.means)
    softmax_expected = torch.softmax(logits, dim=-1) @ torch.from_numpy(ensemble.means)

    assert torch.allclose(decoded, expected, atol=1e-6)
    assert not torch.allclose(decoded, softmax_expected, atol=1e-3)


def test_decode_position_from_tiny_normalized_pc_logits_preserves_relative_weights():
    """Tiny positive sigmoid sums should still decode as normalized weights."""
    ensemble = PlaceCellEnsemble(
        2,
        stdev=0.35,
        pos_min=0.0,
        pos_max=1.0,
        seed=0,
        soft_targets="normalized",
    )
    ensemble.means = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    logits = torch.tensor([[[-30.0, -29.0]]], dtype=torch.float32)

    decoded = decode_position_from_pc_logits([logits], [ensemble])
    sigmoid_weights = torch.sigmoid(logits)
    assert sigmoid_weights.sum() < torch.finfo(logits.dtype).eps
    expected = (
        sigmoid_weights / sigmoid_weights.sum(dim=-1, keepdim=True)
    ) @ torch.as_tensor(ensemble.means, dtype=logits.dtype)

    assert torch.allclose(decoded, expected, atol=1e-6)


def test_decode_position_from_underflowed_normalized_pc_logits_preserves_relative_logits():
    """Saturated negative normalized PC logits should decode from stable sigmoid weights."""
    ensemble = PlaceCellEnsemble(
        2,
        stdev=0.35,
        pos_min=0.0,
        pos_max=1.0,
        seed=0,
        soft_targets="normalized",
    )
    ensemble.means = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    logits = torch.tensor([[[-1000.0, -999.0]]], dtype=torch.float32)

    decoded = decode_position_from_pc_logits([logits], [ensemble])
    assert torch.count_nonzero(torch.sigmoid(logits)) == 0
    expected = torch.softmax(torch.nn.functional.logsigmoid(logits), dim=-1) @ torch.from_numpy(
        ensemble.means
    )

    assert torch.allclose(decoded, expected, atol=1e-6)
    assert not torch.allclose(decoded, torch.zeros_like(decoded), atol=1e-3)


def test_decode_position_from_pc_scores_rejects_rank_deficient_geometry():
    """Raw-score analytic decoding should not pseudo-invert unsupported PC layouts."""
    ensemble = PlaceCellEnsemble(2, stdev=0.2, pos_min=-1.0, pos_max=1.0, seed=0)
    ensemble.means = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    positions = np.array([[[0.2, 0.3]]], dtype=np.float32)

    with pytest.raises(ValueError, match="full-rank analytic geometry"):
        decode_position_from_pc_scores(ensemble.unnor_logpdf(positions), ensemble)


def test_decode_position_from_pc_logits_rejects_invalid_decode_type():
    """Direct decoder use should not silently accept a misspelled decode_type."""
    ensemble = PlaceCellEnsemble(2, stdev=0.35, pos_min=0.0, pos_max=1.0, seed=0)
    ensemble.decode_type = "bad-mode"
    logits = [torch.zeros(1, 1, 2)]

    with pytest.raises(ValueError, match="Unsupported ensemble decode_type"):
        decode_position_from_pc_logits(logits, [ensemble])


def test_decode_head_direction_from_hdc_scores_analytic_recovers_angle():
    """Analytic HDC decoding should invert raw Von Mises log-scores."""
    ensemble = HeadDirectionCellEnsemble(5, concentration=4.0, seed=0)
    headings = np.array([[[-3.0], [-0.2], [3.05]]], dtype=np.float32)

    decoded = decode_head_direction_from_hdc_scores(ensemble.unnor_logpdf(headings), ensemble)
    err = np.angle(np.exp(1j * (decoded - headings)))

    assert np.allclose(err, 0.0, atol=1e-5)


def test_compute_head_direction_mae_rad_wraps_circular_error():
    """Heading MAE should compare angles on the circle, not the real line."""
    ensemble = HeadDirectionCellEnsemble(5, concentration=4.0, seed=0)
    headings = np.array([[[np.pi - 0.01]]], dtype=np.float32)
    logits = [torch.from_numpy(ensemble.unnor_logpdf(headings))]
    target = torch.tensor([[[-np.pi + 0.01]]], dtype=torch.float32)

    decoded = decode_head_direction_from_hdc_logits(logits, [ensemble])
    mae = compute_head_direction_mae_rad(logits, target, [ensemble])

    assert torch.allclose(decoded, torch.from_numpy(headings), atol=1e-5)
    assert torch.isclose(mae, torch.tensor(0.02), atol=1e-5)


def test_compute_head_direction_mae_rad_rejects_squeezed_targets():
    """Heading metrics should fail clearly when target_hd loses its final dimension."""
    ensemble = HeadDirectionCellEnsemble(5, concentration=4.0, seed=0)
    headings = np.array([[[0.25]]], dtype=np.float32)
    logits = [torch.from_numpy(ensemble.unnor_logpdf(headings))]
    squeezed_target = torch.tensor([[0.25]], dtype=torch.float32)

    with pytest.raises(ValueError, match="target_hd must have shape"):
        compute_head_direction_mae_rad(logits, squeezed_target, [ensemble])


def test_decode_head_direction_from_normalized_hdc_logits_uses_sigmoid_weights():
    """Normalized HDC logits should decode from independent sigmoid activations."""
    ensemble = HeadDirectionCellEnsemble(
        2,
        concentration=4.0,
        seed=0,
        soft_targets="normalized",
    )
    ensemble.means = np.array([0.0, np.pi / 2.0], dtype=np.float32)
    logits = torch.tensor([[[0.0, 2.0]]], dtype=torch.float32)

    decoded = decode_head_direction_from_hdc_logits([logits], [ensemble])
    sigmoid_weights = torch.sigmoid(logits)
    sigmoid_weights = sigmoid_weights / sigmoid_weights.sum(dim=-1, keepdim=True)
    means = torch.from_numpy(ensemble.means)
    expected = torch.atan2(
        sigmoid_weights @ torch.sin(means),
        sigmoid_weights @ torch.cos(means),
    ).unsqueeze(-1)
    softmax_weights = torch.softmax(logits, dim=-1)
    softmax_expected = torch.atan2(
        softmax_weights @ torch.sin(means),
        softmax_weights @ torch.cos(means),
    ).unsqueeze(-1)

    assert torch.allclose(decoded, expected, atol=1e-6)
    assert not torch.allclose(decoded, softmax_expected, atol=1e-3)


def test_decode_head_direction_from_underflowed_normalized_hdc_logits_preserves_relative_logits():
    """Saturated negative normalized HDC logits should decode from stable sigmoid weights."""
    ensemble = HeadDirectionCellEnsemble(
        2,
        concentration=4.0,
        seed=0,
        soft_targets="normalized",
    )
    ensemble.means = np.array([0.0, np.pi / 2.0], dtype=np.float32)
    logits = torch.tensor([[[-1000.0, -999.0]]], dtype=torch.float32)

    decoded = decode_head_direction_from_hdc_logits([logits], [ensemble])
    assert torch.count_nonzero(torch.sigmoid(logits)) == 0
    weights = torch.softmax(torch.nn.functional.logsigmoid(logits), dim=-1)
    means = torch.from_numpy(ensemble.means)
    expected = torch.atan2(
        weights @ torch.sin(means),
        weights @ torch.cos(means),
    ).unsqueeze(-1)

    assert torch.allclose(decoded, expected, atol=1e-6)
    assert not torch.allclose(decoded, torch.zeros_like(decoded), atol=1e-3)


def test_normalized_hdc_targets_are_unit_peak_bce_labels():
    """Normalized HDC targets should be bounded independent-cell activations."""
    ensemble = HeadDirectionCellEnsemble(
        12,
        concentration=20.0,
        seed=0,
        soft_targets="normalized",
    )
    headings = np.array(
        [[[ensemble.means[0]], [0.0], [np.pi]]],
        dtype=np.float32,
    )

    targets = ensemble.get_targets(headings)
    loss = ensemble.loss(torch.full((1, 3, 12), 10.0), targets)

    assert np.isclose(targets[0, 0, 0], 1.0, atol=1e-6)
    assert targets.min() >= 0.0
    assert targets.max() <= 1.0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


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
    assert np.allclose(anim_inputs["pred_pos"], anim_inputs["target_pos"], atol=1e-5)
    assert anim_inputs["pc_acts"].shape == (2, 4, 5)
    assert anim_inputs["hdc_acts"].shape == (2, 4, 6)
    assert anim_inputs["pc_centers"].shape == (5, 2)
    assert anim_inputs["hdc_centers"].shape == (6,)
