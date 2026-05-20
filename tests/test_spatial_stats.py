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
    fit_grid_scale_gmm_bic,
    grid_scale_discreteness_test,
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


def test_grid_scale_discreteness_test_uses_banino_shuffle_contract():
    """Discreteness should use fixed bins and jitter scales by half the minimum scale."""
    scales = np.array([10.0, 10.2, 20.0, 20.1, 35.8, 36.0])

    result = grid_scale_discreteness_test(
        scales,
        n_bins=13,
        value_range=(10.0, 36.0),
        num_shuffles=11,
        random_seed=7,
    )

    assert result["score"] is not None
    assert result["null_scores"].shape == (11,)
    assert 0.0 < result["p_value"] <= 1.0
    assert result["null_mean"] is not None
    assert result["null_std"] is not None


def test_fit_grid_scale_gmm_bic_selects_clustered_scale_model():
    """GMM/BIC should recover multiple clusters for separated 1-D scale samples."""
    scales = np.array(
        [
            9.8,
            10.0,
            10.2,
            10.1,
            19.7,
            20.0,
            20.3,
            20.2,
            31.8,
            32.0,
            32.2,
            32.1,
        ]
    )

    result = fit_grid_scale_gmm_bic(scales, max_components=5, random_seed=3)

    assert result["best_components"] >= 2
    assert "1" in result["bic"]
    assert len(result["means"]) == result["best_components"]
    assert np.all(np.diff(result["means"]) >= 0)


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
    assert result["arrays"]["grid_scale_m"].shape == (3,)
    assert result["arrays"]["grid_scale_peak_count"].shape == (3,)
    assert result["arrays"]["is_grid_threshold"].shape == (3,)
    assert result["null_grid_score_60"].shape == (2, 3)
    assert result["directional_tuning"].shape == (3, 20)
    assert result["summary"]["n_units"] == 3
    assert result["summary"]["analysis_num_shuffles"] == 2
    assert "median_grid_fdr_scale_m" in result["summary"]
    assert "grid_fdr_scale_discreteness_p" in result["summary"]
    assert "grid_fdr_scale_gmm_best_components" in result["summary"]
    assert "grid_threshold_scale_discreteness_p" in result["summary"]
    assert "grid_threshold_scale_gmm_best_components" in result["summary"]
    assert result["grid_fdr_scale_discreteness_null"].shape == (500,)
    assert result["grid_threshold_scale_discreteness_null"].shape == (500,)
    assert result["summary"]["analysis_compute_grid_selectivity"] is True
    assert result["summary"]["analysis_compute_grid_geometry"] is True
    assert result["summary"]["analysis_compute_shuffle_significance"] is True
    assert result["summary"]["analysis_compute_split_half"] is True
    assert result["summary"]["analysis_compute_hd_selectivity"] is True


def test_analysis_modules_can_run_hd_selectivity_only():
    """Selective analysis should preserve schema while skipping disabled modules."""
    rng = np.random.default_rng(5)
    scorer = make_scorer()
    positions = rng.uniform(-0.9, 0.9, size=(4, 8, 2))
    hd = rng.uniform(-np.pi, np.pi, size=(4, 8, 1))
    activations = rng.normal(size=(4, 8, 3))

    result = analyze_bottleneck_spatial_stats(
        scorer,
        positions,
        hd,
        activations,
        num_shuffles=3,
        compute_grid_selectivity=False,
        compute_grid_geometry=False,
        compute_shuffle_significance=False,
        compute_split_half=False,
        compute_hd_selectivity=True,
    )

    assert result["fields"] == STAT_FIELDS
    assert len(result["rows"]) == 3
    assert np.all(np.isnan(result["arrays"]["grid_score_60"]))
    assert np.all(np.isnan(result["arrays"]["grid_scale_m"]))
    assert np.all(np.isnan(result["arrays"]["grid_score_60_p"]))
    assert np.all(np.isnan(result["arrays"]["split_half_corr"]))
    assert np.all(np.isfinite(result["arrays"]["hd_mrl"]))
    assert np.all(np.isnan(result["arrays"]["hd_p"]))
    assert np.all(np.isnan(result["null_grid_score_60"]))
    assert np.all(np.isnan(result["null_hd_mrl"]))
    assert result["directional_tuning"].shape == (3, 20)
    assert result["summary"]["analysis_compute_grid_selectivity"] is False
    assert result["summary"]["analysis_compute_grid_geometry"] is False
    assert result["summary"]["analysis_compute_shuffle_significance"] is False
    assert result["summary"]["analysis_compute_split_half"] is False
    assert result["summary"]["analysis_compute_hd_selectivity"] is True


def test_analysis_modules_can_run_grid_geometry_only():
    """Geometry-only analysis should produce grid scale without grid-score stats."""
    class GeometryScorer:
        _coords_range = [[-1.0, 1.0], [-1.0, 1.0]]
        _nbins = 5

        def allocate_ratemap_accumulators(self, n_units):
            return np.zeros((n_units, 5, 5), dtype=np.float64), np.zeros((5, 5))

        def accumulate_ratemaps(self, positions, activations, sums, counts):
            sums += 1.0
            counts += 1.0

        def finalize_ratemaps(self, sums, counts):
            return sums

        def calculate_sac_batch(self, ratemaps):
            sacs = np.zeros((ratemaps.shape[0], 11, 11), dtype=np.float64)
            center = np.array([5, 5])
            offsets = np.array(
                [[0, 2], [0, -2], [2, 0], [-2, 0], [2, 2], [-2, -2]]
            )
            sacs[:, center[0], center[1]] = 1.0
            for offset in offsets:
                coord = center + offset
                sacs[:, coord[0], coord[1]] = 0.8
            return sacs

        def calculate_grid_scales(self, sacs):
            return make_scorer().calculate_grid_scales(sacs)

    positions = np.zeros((2, 4, 2), dtype=np.float64)
    hd = np.zeros((2, 4, 1), dtype=np.float64)
    activations = np.zeros((2, 4, 3), dtype=np.float64)

    result = analyze_bottleneck_spatial_stats(
        GeometryScorer(),
        positions,
        hd,
        activations,
        num_shuffles=0,
        compute_grid_selectivity=False,
        compute_grid_geometry=True,
        compute_shuffle_significance=False,
        compute_split_half=False,
        compute_hd_selectivity=False,
        scale_discreteness_num_shuffles=5,
        scale_gmm_max_components=3,
    )

    assert np.all(np.isnan(result["arrays"]["grid_score_60"]))
    assert np.all(result["arrays"]["grid_scale_valid"])
    assert np.all(result["arrays"]["grid_scale_peak_count"] == 6)
    assert np.all(np.isfinite(result["arrays"]["grid_scale_m"]))
    assert result["summary"]["n_grid_scale_valid"] == 3
    assert result["summary"]["n_grid_fdr_scale_valid"] == 0
    assert result["summary"]["n_grid_threshold_scale_valid"] == 0
    assert result["grid_fdr_scale_discreteness_null"].shape == (5,)
    assert result["grid_threshold_scale_discreteness_null"].shape == (5,)
    assert result["summary"]["analysis_compute_grid_geometry"] is True
    assert result["summary"]["analysis_compute_grid_selectivity"] is False


def test_grid_threshold_population_stats_run_in_parallel_with_fdr_stats():
    """Threshold-selected grid units should get their own population scale analysis."""
    class ThresholdScorer:
        _coords_range = [[-1.0, 1.0], [-1.0, 1.0]]
        _nbins = 5

        def allocate_ratemap_accumulators(self, n_units):
            return np.zeros((n_units, 5, 5), dtype=np.float64), np.zeros((5, 5))

        def accumulate_ratemaps(self, positions, activations, sums, counts):
            sums += 1.0
            counts += 1.0

        def finalize_ratemaps(self, sums, counts):
            return sums

        def get_scores_batch(self, ratemaps):
            scores_60 = np.array([0.5, 0.1, 0.8], dtype=np.float64)
            scores_90 = np.zeros(3, dtype=np.float64)
            sacs = self.calculate_sac_batch(ratemaps)
            return scores_60, scores_90, [None] * 3, [None] * 3, sacs

        def calculate_sac_batch(self, ratemaps):
            sacs = np.zeros((ratemaps.shape[0], 11, 11), dtype=np.float64)
            center = np.array([5, 5])
            offsets = np.array(
                [[0, 2], [0, -2], [2, 0], [-2, 0], [2, 2], [-2, -2]]
            )
            sacs[:, center[0], center[1]] = 1.0
            for offset in offsets:
                coord = center + offset
                sacs[:, coord[0], coord[1]] = 0.8
            return sacs

        def calculate_grid_scales(self, sacs):
            return make_scorer().calculate_grid_scales(sacs)

    positions = np.zeros((2, 4, 2), dtype=np.float64)
    hd = np.zeros((2, 4, 1), dtype=np.float64)
    activations = np.zeros((2, 4, 3), dtype=np.float64)

    result = analyze_bottleneck_spatial_stats(
        ThresholdScorer(),
        positions,
        hd,
        activations,
        num_shuffles=1,
        fdr_alpha=1.0,
        compute_grid_selectivity=True,
        compute_grid_geometry=True,
        compute_shuffle_significance=True,
        compute_split_half=False,
        compute_hd_selectivity=False,
        gridness_threshold=0.37,
        scale_discreteness_num_shuffles=3,
        scale_gmm_max_components=3,
    )

    assert np.array_equal(
        result["arrays"]["is_grid_threshold"],
        np.array([True, False, True]),
    )
    assert result["summary"]["n_grid_fdr"] == 3
    assert result["summary"]["n_grid_threshold"] == 2
    assert result["summary"]["n_grid_fdr_scale_valid"] == 3
    assert result["summary"]["n_grid_threshold_scale_valid"] == 2
    assert result["summary"]["grid_threshold_scale_discreteness"] is not None
    assert result["summary"]["grid_threshold_scale_gmm_best_components"] is not None
    assert result["grid_threshold_scale_discreteness_null"].shape == (3,)


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
    with np.load(paths["npz"]) as payload:
        assert "grid_fdr_scale_discreteness_null" in payload
        assert "grid_threshold_scale_discreteness_null" in payload
