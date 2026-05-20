"""Tests for grid-cell statistical analysis helpers.

Usage:
    pytest tests/test_spatial_stats.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grid_cells.analysis.scores import GridScorer
from grid_cells.analysis.spatial_stats import (
    STAT_FIELDS,
    analyze_bottleneck_spatial_stats,
    benjamini_hochberg,
    circular_shift_activations,
    sample_circular_shifts,
    save_spatial_stats_artifacts,
    shuffle_p_values,
    split_half_correlations,
)


def make_scorer(nbins=5):
    return GridScorer(
        nbins,
        [[-1.0, 1.0], [-1.0, 1.0]],
        [(0.2, 0.6), (0.3, 0.8)],
    )


def test_benjamini_hochberg_q_values_are_monotonic_and_ordered():
    """BH correction should preserve the standard adjusted p-value contract."""
    p_values = np.array([0.01, 0.04, 0.03, 0.20])

    q_values = benjamini_hochberg(p_values)

    assert np.allclose(q_values, np.array([0.04, 0.05333333, 0.05333333, 0.20]))
    ordered_q = q_values[np.argsort(p_values)]
    assert np.all(np.diff(ordered_q) >= -1e-12)


def test_circular_shift_shuffle_preserves_shape_and_values():
    """Circular shift shuffling should not alter activation values or shape."""
    activations = np.arange(2 * 6 * 3).reshape(2, 6, 3)
    shifts = sample_circular_shifts(
        n_trajectories=2,
        seq_len=6,
        n_units=3,
        min_shift_fraction=0.25,
        rng=np.random.default_rng(0),
    )

    shifted = circular_shift_activations(activations, shifts)

    assert shifted.shape == activations.shape
    assert np.array_equal(np.sort(shifted.reshape(-1)), np.sort(activations.reshape(-1)))
    assert np.all(np.minimum(shifts, 6 - shifts) >= 2)
    assert shifts.shape == (2, 3)


def test_circular_shift_matches_per_unit_rolls():
    """Vectorized per-unit shifts should match direct numpy roll semantics."""
    activations = np.arange(2 * 5 * 3).reshape(2, 5, 3)
    shifts = np.array([[0, 1, 2], [3, 4, 0]])

    shifted = circular_shift_activations(activations, shifts)
    expected = np.empty_like(activations)
    for traj_index in range(activations.shape[0]):
        for unit_index in range(activations.shape[2]):
            expected[traj_index, :, unit_index] = np.roll(
                activations[traj_index, :, unit_index],
                shifts[traj_index, unit_index],
            )

    assert np.array_equal(shifted, expected)


def test_shuffle_p_values_ignore_nan_null_samples():
    """NaN null samples should not make p-values artificially small."""
    observed = np.array([0.5, 0.5, np.nan])
    null_distribution = np.array(
        [
            [0.4, np.nan, 0.1],
            [0.6, np.nan, 0.2],
            [np.nan, np.nan, 0.3],
        ]
    )

    p_values = shuffle_p_values(observed, null_distribution)

    assert np.allclose(p_values[:2], np.array([2.0 / 3.0, np.nan]), equal_nan=True)
    assert np.isnan(p_values[2])


def test_split_half_reliability_is_high_for_identical_maps():
    """Identical non-constant maps should return a perfect split-half correlation."""
    ratemap = np.array([[0.0, 1.0, np.nan], [2.0, 3.0, 4.0]], dtype=np.float64)
    ratemaps_a = np.stack([ratemap, ratemap * 2.0], axis=0)
    ratemaps_b = ratemaps_a.copy()

    corr = split_half_correlations(ratemaps_a, ratemaps_b)

    assert np.allclose(corr, np.ones(2))


def test_small_synthetic_analysis_returns_required_fields():
    """A small synthetic eval subset should produce a complete stats table."""
    rng = np.random.default_rng(1)
    scorer = make_scorer()
    n_trajectories = 4
    seq_len = 8
    positions = rng.uniform(-0.9, 0.9, size=(n_trajectories, seq_len, 2))
    hd = rng.uniform(-np.pi, np.pi, size=(n_trajectories, seq_len, 1))
    activations = rng.normal(size=(n_trajectories, seq_len, 3))

    result = analyze_bottleneck_spatial_stats(
        scorer,
        positions,
        hd,
        activations,
        num_shuffles=2,
        random_seed=3,
    )

    assert result["fields"] == STAT_FIELDS
    assert len(result["rows"]) == 3
    assert set(STAT_FIELDS).issubset(result["rows"][0])
    assert result["arrays"]["unit_id"].shape == (3,)
    assert result["null_grid_score_60"].shape == (2, 3)
    assert result["directional_tuning"].shape == (3, 20)
    assert result["summary"]["n_units"] == 3
    assert result["summary"]["analysis_num_shuffles"] == 2


def test_save_spatial_stats_artifacts_writes_contract_files(tmp_path):
    """Statistical analysis output should include CSV, NPZ, and summary JSON."""
    rng = np.random.default_rng(2)
    scorer = make_scorer()
    positions = rng.uniform(-0.9, 0.9, size=(3, 6, 2))
    hd = rng.uniform(-np.pi, np.pi, size=(3, 6, 1))
    activations = rng.normal(size=(3, 6, 2))
    result = analyze_bottleneck_spatial_stats(
        scorer,
        positions,
        hd,
        activations,
        num_shuffles=1,
        random_seed=4,
    )

    paths = save_spatial_stats_artifacts(result, str(tmp_path), epoch=7)

    assert os.path.basename(paths["csv"]) == "grid_stats_epoch_0007.csv"
    assert os.path.basename(paths["npz"]) == "grid_stats_epoch_0007.npz"
    assert os.path.basename(paths["summary_json"]) == "grid_stats_summary_epoch_0007.json"
    assert os.path.exists(paths["csv"])
    assert os.path.exists(paths["npz"])
    assert os.path.exists(paths["summary_json"])
