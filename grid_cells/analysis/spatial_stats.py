"""Statistical analysis for bottleneck-unit spatial and directional selectivity.

The functions here keep grid-score shuffle significance, split-half
reliability, and bottleneck head-direction selectivity separate from the
training evaluation loop.

Usage:
    result = analyze_bottleneck_spatial_stats(scorer, positions, hd, activations)
    save_spatial_stats_artifacts(result, "results/run", epoch=10)
"""

import csv
import json
import os
from dataclasses import dataclass

import numpy as np


STAT_FIELDS = [
    "unit_id",
    "grid_score_60",
    "grid_score_90",
    "grid_score_60_p",
    "grid_score_60_q",
    "grid_score_90_p",
    "grid_score_90_q",
    "grid_scale_bins",
    "grid_scale_m",
    "grid_scale_peak_count",
    "grid_scale_valid",
    "null_60_mean",
    "null_60_std",
    "null_60_95",
    "split_half_corr",
    "grid_score_60_half_a",
    "grid_score_60_half_b",
    "hd_mrl",
    "hd_preferred_direction_rad",
    "hd_p",
    "hd_q",
    "is_grid_fdr",
    "is_grid_threshold",
    "is_reliable_grid",
    "is_hd_selective",
    "unit_class",
]


@dataclass(frozen=True)
class SpatialAnalysisModules:
    """Enabled bottleneck-unit statistical analysis modules."""

    compute_grid_selectivity: bool = True
    compute_grid_geometry: bool = True
    compute_shuffle_significance: bool = True
    compute_split_half: bool = True
    compute_hd_selectivity: bool = True


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg FDR q-values for a 1-D p-value array."""
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.full(p_values.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(p_values)
    finite_p = p_values[finite_mask]
    if finite_p.size == 0:
        return q_values

    order = np.argsort(finite_p)
    ordered_p = finite_p[order]
    ranks = np.arange(1, ordered_p.size + 1, dtype=np.float64)
    ordered_q = ordered_p * ordered_p.size / ranks
    ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
    ordered_q = np.clip(ordered_q, 0.0, 1.0)

    finite_indices = np.flatnonzero(finite_mask)
    q_values[finite_indices[order]] = ordered_q
    return q_values


def sample_circular_shifts(
    n_trajectories,
    seq_len,
    n_units=None,
    min_shift_fraction=0.1,
    rng=None,
):
    """Sample non-trivial circular shifts for each trajectory and optionally unit."""
    rng = rng or np.random.default_rng()
    shape = (n_trajectories,) if n_units is None else (n_trajectories, n_units)
    if seq_len <= 1:
        return np.zeros(shape, dtype=np.int64)

    min_shift = int(np.ceil(float(min_shift_fraction) * seq_len))
    min_shift = max(1, min(min_shift, seq_len - 1))
    candidates = np.asarray(
        [shift for shift in range(1, seq_len) if min(shift, seq_len - shift) >= min_shift],
        dtype=np.int64,
    )
    if candidates.size == 0:
        candidates = np.arange(1, seq_len, dtype=np.int64)
    return rng.choice(candidates, size=shape, replace=True)


def circular_shift_activations(activations, shifts):
    """Circularly shift every trajectory activation sequence and preserve shape."""
    activations = np.asarray(activations)
    shifts = np.asarray(shifts, dtype=np.int64)
    if activations.ndim != 3:
        raise ValueError("activations must have shape (n_trajectories, seq_len, n_units)")
    if shifts.shape not in {
        (activations.shape[0],),
        (activations.shape[0], activations.shape[2]),
    }:
        raise ValueError("shifts must have shape (n_trajectories,) or (n_trajectories, n_units)")

    if shifts.ndim == 1:
        shifts = shifts[:, np.newaxis]

    seq_len = activations.shape[1]
    time_index = (
        np.arange(seq_len, dtype=np.int64)[np.newaxis, :, np.newaxis]
        - shifts[:, np.newaxis, :]
    ) % seq_len
    if time_index.shape[-1] == 1:
        time_index = np.broadcast_to(time_index, activations.shape)
    return np.take_along_axis(activations, time_index, axis=1)


def split_half_correlations(ratemaps_a, ratemaps_b):
    """Compute per-unit Pearson correlations between two rate-map batches."""
    ratemaps_a = np.asarray(ratemaps_a, dtype=np.float64)
    ratemaps_b = np.asarray(ratemaps_b, dtype=np.float64)
    if ratemaps_a.shape != ratemaps_b.shape or ratemaps_a.ndim != 3:
        raise ValueError("ratemaps_a and ratemaps_b must share shape (n_units, x_bins, y_bins)")

    corr = np.full(ratemaps_a.shape[0], np.nan, dtype=np.float64)
    for unit_index in range(ratemaps_a.shape[0]):
        a = ratemaps_a[unit_index].ravel()
        b = ratemaps_b[unit_index].ravel()
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 2:
            continue
        a_valid = a[valid]
        b_valid = b[valid]
        a_centered = a_valid - a_valid.mean()
        b_centered = b_valid - b_valid.mean()
        denom = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
        if denom <= 0:
            corr[unit_index] = 1.0 if np.allclose(a_valid, b_valid) else np.nan
        else:
            corr[unit_index] = np.sum(a_centered * b_centered) / denom
    return corr


def analyze_bottleneck_spatial_stats(
    scorer,
    positions,
    head_directions,
    activations,
    *,
    n_direction_bins=20,
    num_shuffles=200,
    fdr_alpha=0.05,
    min_shift_fraction=0.1,
    split_half_min_corr=0.3,
    split_method="odd_even",
    random_seed=0,
    compute_grid_selectivity=True,
    compute_grid_geometry=True,
    compute_shuffle_significance=True,
    compute_split_half=True,
    compute_hd_selectivity=True,
    scale_discreteness_num_shuffles=500,
    scale_discreteness_bins=13,
    scale_discreteness_min_bins=10.0,
    scale_discreteness_max_bins=36.0,
    scale_gmm_max_components=8,
    gridness_threshold=0.37,
):
    """Compute selected bottleneck-unit statistical analysis modules."""
    positions, head_directions, activations = _prepare_inputs(
        positions,
        head_directions,
        activations,
    )
    n_trajectories, seq_len, n_units = activations.shape
    rng = np.random.default_rng(random_seed)
    modules = SpatialAnalysisModules(
        compute_grid_selectivity=bool(compute_grid_selectivity),
        compute_grid_geometry=bool(compute_grid_geometry),
        compute_shuffle_significance=bool(compute_shuffle_significance),
        compute_split_half=bool(compute_split_half),
        compute_hd_selectivity=bool(compute_hd_selectivity),
    )

    observed_ratemaps = None
    observed_sacs = None
    observed_grid_score_60 = _nan_vector(n_units)
    observed_grid_score_90 = _nan_vector(n_units)
    if modules.compute_grid_selectivity or modules.compute_grid_geometry:
        observed_ratemaps = _calculate_ratemaps(scorer, positions, activations)
    if modules.compute_grid_selectivity:
        observed_grid_score_60, observed_grid_score_90, observed_sacs = _score_ratemaps(
            scorer,
            observed_ratemaps,
        )

    grid_scale_bins = _nan_vector(n_units)
    grid_scale_m = _nan_vector(n_units)
    grid_scale_peak_count = np.zeros(n_units, dtype=np.int64)
    grid_scale_valid = np.zeros(n_units, dtype=bool)
    if modules.compute_grid_geometry:
        if observed_sacs is None:
            observed_sacs = _calculate_sacs(scorer, observed_ratemaps)
        grid_scale_bins, grid_scale_peak_count = scorer.calculate_grid_scales(
            observed_sacs
        )
        grid_scale_m = grid_scale_bins * _spatial_bin_size_m(scorer)
        grid_scale_valid = np.isfinite(grid_scale_bins)

    if modules.compute_split_half:
        half = _compute_split_half_stats(
            scorer,
            positions,
            activations,
            split_method=split_method,
            random_seed=random_seed,
        )
    else:
        half = _empty_split_half_stats(n_units)

    tuning = _nan_matrix(n_units, int(n_direction_bins))
    hd_mrl = _nan_vector(n_units)
    hd_preferred_direction = _nan_vector(n_units)
    if modules.compute_hd_selectivity:
        tuning, hd_mrl, hd_preferred_direction = directional_tuning(
            head_directions,
            activations,
            n_bins=n_direction_bins,
        )

    null_60 = _nan_matrix(int(num_shuffles), n_units)
    null_90 = _nan_matrix(int(num_shuffles), n_units)
    null_hd = _nan_matrix(int(num_shuffles), n_units)
    if (
        modules.compute_shuffle_significance
        and int(num_shuffles) > 0
        and (modules.compute_grid_selectivity or modules.compute_hd_selectivity)
    ):
        null_60, null_90, null_hd = _shuffle_null_distributions(
            scorer,
            positions,
            head_directions,
            activations,
            n_direction_bins=n_direction_bins,
            num_shuffles=num_shuffles,
            min_shift_fraction=min_shift_fraction,
            rng=rng,
            compute_grid_selectivity=modules.compute_grid_selectivity,
            compute_hd_selectivity=modules.compute_hd_selectivity,
        )

    grid_60_p = _nan_vector(n_units)
    grid_90_p = _nan_vector(n_units)
    grid_60_q = _nan_vector(n_units)
    grid_90_q = _nan_vector(n_units)
    if modules.compute_grid_selectivity and modules.compute_shuffle_significance:
        grid_60_p = shuffle_p_values(observed_grid_score_60, null_60)
        grid_90_p = shuffle_p_values(observed_grid_score_90, null_90)
        grid_60_q = benjamini_hochberg(grid_60_p)
        grid_90_q = benjamini_hochberg(grid_90_p)

    hd_p = _nan_vector(n_units)
    hd_q = _nan_vector(n_units)
    if modules.compute_hd_selectivity and modules.compute_shuffle_significance:
        hd_p = shuffle_p_values(hd_mrl, null_hd)
        hd_q = benjamini_hochberg(hd_p)

    is_grid_fdr = grid_60_q <= fdr_alpha
    is_grid_threshold = observed_grid_score_60 >= float(gridness_threshold)
    is_reliable_grid = is_grid_fdr & (half["split_half_corr"] >= split_half_min_corr)
    is_hd_selective = hd_q <= fdr_alpha
    unit_class = _classify_units(is_grid_fdr, is_hd_selective)

    arrays = {
        "unit_id": np.arange(n_units, dtype=np.int64),
        "grid_score_60": observed_grid_score_60,
        "grid_score_90": observed_grid_score_90,
        "grid_score_60_p": grid_60_p,
        "grid_score_60_q": grid_60_q,
        "grid_score_90_p": grid_90_p,
        "grid_score_90_q": grid_90_q,
        "grid_scale_bins": grid_scale_bins,
        "grid_scale_m": grid_scale_m,
        "grid_scale_peak_count": grid_scale_peak_count,
        "grid_scale_valid": grid_scale_valid,
        "null_60_mean": _nanmean_axis0(null_60),
        "null_60_std": _nanstd_axis0(null_60),
        "null_60_95": _nanpercentile_axis0(null_60, 95.0),
        "split_half_corr": half["split_half_corr"],
        "grid_score_60_half_a": half["grid_score_60_half_a"],
        "grid_score_60_half_b": half["grid_score_60_half_b"],
        "hd_mrl": hd_mrl,
        "hd_preferred_direction_rad": hd_preferred_direction,
        "hd_p": hd_p,
        "hd_q": hd_q,
        "is_grid_fdr": is_grid_fdr,
        "is_grid_threshold": is_grid_threshold,
        "is_reliable_grid": is_reliable_grid,
        "is_hd_selective": is_hd_selective,
        "unit_class": unit_class,
    }
    fdr_scale_population = _grid_scale_population_stats(
        arrays["grid_scale_bins"],
        arrays["is_grid_fdr"],
        random_seed=random_seed,
        scale_discreteness_num_shuffles=scale_discreteness_num_shuffles,
        scale_discreteness_bins=scale_discreteness_bins,
        scale_discreteness_range=(
            scale_discreteness_min_bins,
            scale_discreteness_max_bins,
        ),
        scale_gmm_max_components=scale_gmm_max_components,
        spatial_bin_size_m=(
            _spatial_bin_size_m(scorer) if modules.compute_grid_geometry else np.nan
        ),
    )
    threshold_scale_population = _grid_scale_population_stats(
        arrays["grid_scale_bins"],
        arrays["is_grid_threshold"],
        random_seed=random_seed,
        scale_discreteness_num_shuffles=scale_discreteness_num_shuffles,
        scale_discreteness_bins=scale_discreteness_bins,
        scale_discreteness_range=(
            scale_discreteness_min_bins,
            scale_discreteness_max_bins,
        ),
        scale_gmm_max_components=scale_gmm_max_components,
        spatial_bin_size_m=(
            _spatial_bin_size_m(scorer) if modules.compute_grid_geometry else np.nan
        ),
    )
    summary = _summary(
        arrays,
        num_shuffles,
        modules,
        fdr_scale_population=fdr_scale_population,
        threshold_scale_population=threshold_scale_population,
        gridness_threshold=gridness_threshold,
    )
    rows = [_row_from_arrays(arrays, index) for index in range(n_units)]

    return {
        "fields": STAT_FIELDS,
        "rows": rows,
        "arrays": arrays,
        "summary": summary,
        "null_grid_score_60": null_60,
        "null_grid_score_90": null_90,
        "null_hd_mrl": null_hd,
        "grid_fdr_scale_discreteness_null": fdr_scale_population[
            "discreteness_null_scores"
        ],
        "grid_threshold_scale_discreteness_null": threshold_scale_population[
            "discreteness_null_scores"
        ],
        "directional_tuning": tuning,
    }


def directional_tuning(head_directions, activations, n_bins=20):
    """Compute directional tuning, mean resultant length, and preferred angle."""
    head_directions = np.asarray(head_directions, dtype=np.float64).reshape(-1)
    activations = np.asarray(activations, dtype=np.float64)
    activations = activations.reshape(-1, activations.shape[-1])
    angles = (head_directions + np.pi) % (2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, int(n_bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_index = np.digitize(angles, edges[1:-1])

    n_units = activations.shape[-1]
    tuning = np.full((n_units, int(n_bins)), np.nan, dtype=np.float64)
    for bin_id in range(int(n_bins)):
        mask = bin_index == bin_id
        if np.any(mask):
            tuning[:, bin_id] = np.nanmean(activations[mask], axis=0)

    weights = _nonnegative_tuning_weights(tuning)
    vectors = weights @ np.exp(1j * centers)
    totals = weights.sum(axis=1)
    mrl = np.zeros(n_units, dtype=np.float64)
    preferred = np.full(n_units, np.nan, dtype=np.float64)
    valid = totals > 0
    mrl[valid] = np.abs(vectors[valid]) / totals[valid]
    preferred[valid] = np.angle(vectors[valid])
    return tuning, mrl, preferred


def save_spatial_stats_artifacts(result, save_dir, epoch):
    """Write CSV, NPZ, and JSON statistical artifacts for one eval epoch."""
    os.makedirs(save_dir, exist_ok=True)
    stem = f"grid_stats_epoch_{int(epoch):04d}"
    csv_path = os.path.join(save_dir, stem + ".csv")
    npz_path = os.path.join(save_dir, stem + ".npz")
    json_path = os.path.join(save_dir, f"grid_stats_summary_epoch_{int(epoch):04d}.json")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result["fields"])
        writer.writeheader()
        writer.writerows(result["rows"])

    npz_payload = {key: value for key, value in result["arrays"].items()}
    npz_payload.update(
        {
            "null_grid_score_60": result["null_grid_score_60"],
            "null_grid_score_90": result["null_grid_score_90"],
            "null_hd_mrl": result["null_hd_mrl"],
            "grid_fdr_scale_discreteness_null": result[
                "grid_fdr_scale_discreteness_null"
            ],
            "grid_threshold_scale_discreteness_null": result[
                "grid_threshold_scale_discreteness_null"
            ],
            "directional_tuning": result["directional_tuning"],
        }
    )
    np.savez_compressed(npz_path, **npz_payload)

    with open(json_path, "w") as handle:
        json.dump(result["summary"], handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {"csv": csv_path, "npz": npz_path, "summary_json": json_path}


def _prepare_inputs(positions, head_directions, activations):
    positions = np.asarray(positions)
    head_directions = np.asarray(head_directions)
    activations = np.asarray(activations)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (n_trajectories, seq_len, 2)")
    if activations.ndim != 3:
        raise ValueError("activations must have shape (n_trajectories, seq_len, n_units)")
    if head_directions.ndim == 3 and head_directions.shape[-1] == 1:
        head_directions = head_directions[..., 0]
    if head_directions.shape != positions.shape[:2]:
        raise ValueError("head_directions must have shape (n_trajectories, seq_len)")
    if activations.shape[:2] != positions.shape[:2]:
        raise ValueError("activations and positions must share trajectory and time dimensions")
    return positions, head_directions, activations


def _calculate_ratemaps(scorer, positions, activations):
    n_units = activations.shape[-1]
    sums, counts = scorer.allocate_ratemap_accumulators(n_units)
    scorer.accumulate_ratemaps(positions, activations, sums, counts)
    return scorer.finalize_ratemaps(sums, counts)


def _score_ratemaps(scorer, ratemaps):
    if hasattr(scorer, "get_scores_batch"):
        score_60, score_90, _, _, sacs = scorer.get_scores_batch(ratemaps)
        return (
            np.asarray(score_60, dtype=np.float64),
            np.asarray(score_90, dtype=np.float64),
            np.asarray(sacs, dtype=np.float64),
        )

    scores = [scorer.get_scores(ratemap) for ratemap in ratemaps]
    return (
        np.asarray([score[0] for score in scores], dtype=np.float64),
        np.asarray([score[1] for score in scores], dtype=np.float64),
        np.asarray([score[4] for score in scores], dtype=np.float64),
    )


def _calculate_sacs(scorer, ratemaps):
    if hasattr(scorer, "calculate_sac_batch"):
        return scorer.calculate_sac_batch(ratemaps)
    return np.asarray([scorer.calculate_sac(ratemap) for ratemap in ratemaps])


def _spatial_bin_size_m(scorer):
    x_range = scorer._coords_range[0]
    y_range = scorer._coords_range[1]
    x_bin = (float(x_range[1]) - float(x_range[0])) / float(scorer._nbins)
    y_bin = (float(y_range[1]) - float(y_range[0])) / float(scorer._nbins)
    return float((x_bin + y_bin) / 2.0)


def _compute_split_half_stats(
    scorer,
    positions,
    activations,
    *,
    split_method,
    random_seed,
):
    n_trajectories = positions.shape[0]
    half_a, half_b = _split_indices(n_trajectories, split_method, random_seed)
    n_units = activations.shape[-1]
    if half_a.size == 0 or half_b.size == 0:
        return {
            "split_half_corr": np.full(n_units, np.nan, dtype=np.float64),
            "grid_score_60_half_a": np.full(n_units, np.nan, dtype=np.float64),
            "grid_score_60_half_b": np.full(n_units, np.nan, dtype=np.float64),
        }

    ratemaps_a = _calculate_ratemaps(scorer, positions[half_a], activations[half_a])
    ratemaps_b = _calculate_ratemaps(scorer, positions[half_b], activations[half_b])
    score_60_a, _, _ = _score_ratemaps(scorer, ratemaps_a)
    score_60_b, _, _ = _score_ratemaps(scorer, ratemaps_b)
    return {
        "split_half_corr": split_half_correlations(ratemaps_a, ratemaps_b),
        "grid_score_60_half_a": score_60_a,
        "grid_score_60_half_b": score_60_b,
    }


def _split_indices(n_trajectories, split_method, random_seed):
    indices = np.arange(n_trajectories)
    if split_method == "odd_even":
        return indices[::2], indices[1::2]
    if split_method == "random":
        rng = np.random.default_rng(random_seed)
        shuffled = rng.permutation(indices)
        midpoint = n_trajectories // 2
        return shuffled[:midpoint], shuffled[midpoint:]
    raise ValueError(f"Unsupported split_method: {split_method}")


def _shuffle_null_distributions(
    scorer,
    positions,
    head_directions,
    activations,
    *,
    n_direction_bins,
    num_shuffles,
    min_shift_fraction,
    rng,
    compute_grid_selectivity=True,
    compute_hd_selectivity=True,
):
    n_trajectories, seq_len, n_units = activations.shape
    null_60 = np.full((int(num_shuffles), n_units), np.nan, dtype=np.float64)
    null_90 = np.full((int(num_shuffles), n_units), np.nan, dtype=np.float64)
    null_hd = np.full((int(num_shuffles), n_units), np.nan, dtype=np.float64)

    for shuffle_index in range(int(num_shuffles)):
        shifts = sample_circular_shifts(
            n_trajectories,
            seq_len,
            n_units=n_units,
            min_shift_fraction=min_shift_fraction,
            rng=rng,
        )
        shifted = circular_shift_activations(activations, shifts)
        if compute_grid_selectivity:
            shuffled_ratemaps = _calculate_ratemaps(scorer, positions, shifted)
            null_60[shuffle_index], null_90[shuffle_index], _ = _score_ratemaps(
                scorer,
                shuffled_ratemaps,
            )
        if compute_hd_selectivity:
            _, null_hd[shuffle_index], _ = directional_tuning(
                head_directions,
                shifted,
                n_bins=n_direction_bins,
            )
    return null_60, null_90, null_hd


def shuffle_p_values(observed, null_distribution):
    """Compute one-sided shuffle p-values, ignoring invalid null samples per unit."""
    observed = np.asarray(observed, dtype=np.float64)
    null_distribution = np.asarray(null_distribution, dtype=np.float64)
    if null_distribution.shape[0] == 0:
        return np.full(observed.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(null_distribution) & np.isfinite(observed)[np.newaxis, :]
    finite_count = np.sum(finite, axis=0)
    extreme_count = np.sum(finite & (null_distribution >= observed[np.newaxis, :]), axis=0)
    p_values = np.full(observed.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(observed) & (finite_count > 0)
    p_values[valid] = (1.0 + extreme_count[valid]) / (1.0 + finite_count[valid])
    return p_values


def _nonnegative_tuning_weights(tuning):
    weights = np.asarray(tuning, dtype=np.float64)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    minima = np.min(weights, axis=1, keepdims=True)
    weights = np.where(minima < 0.0, weights - minima, weights)
    return weights


def _classify_units(is_grid_fdr, is_hd_selective):
    unit_class = np.full(is_grid_fdr.shape, "non_spatial", dtype="U16")
    unit_class[is_grid_fdr & ~is_hd_selective] = "grid_only"
    unit_class[~is_grid_fdr & is_hd_selective] = "hd_only"
    unit_class[is_grid_fdr & is_hd_selective] = "grid_and_hd"
    return unit_class


def _nan_vector(size):
    return np.full(int(size), np.nan, dtype=np.float64)


def _nan_matrix(rows, cols):
    return np.full((int(rows), int(cols)), np.nan, dtype=np.float64)


def _empty_split_half_stats(n_units):
    return {
        "split_half_corr": _nan_vector(n_units),
        "grid_score_60_half_a": _nan_vector(n_units),
        "grid_score_60_half_b": _nan_vector(n_units),
    }


def _row_from_arrays(arrays, index):
    row = {}
    for field in STAT_FIELDS:
        value = arrays[field][index]
        if isinstance(value, np.generic):
            value = value.item()
        row[field] = value
    return row


def _summary(
    arrays,
    num_shuffles,
    modules: SpatialAnalysisModules,
    *,
    fdr_scale_population,
    threshold_scale_population,
    gridness_threshold,
):
    grid_scale_bins = np.asarray(arrays["grid_scale_bins"], dtype=np.float64)
    grid_scale_m = np.asarray(arrays["grid_scale_m"], dtype=np.float64)
    is_grid_fdr = np.asarray(arrays["is_grid_fdr"], dtype=bool)
    grid_fdr_scale_m = np.where(is_grid_fdr, grid_scale_m, np.nan)
    is_grid_threshold = np.asarray(arrays["is_grid_threshold"], dtype=bool)
    grid_threshold_scale_m = np.where(is_grid_threshold, grid_scale_m, np.nan)
    summary = {
        "n_units": int(arrays["unit_id"].size),
        "n_grid_fdr": int(np.sum(arrays["is_grid_fdr"])),
        "n_grid_threshold": int(np.sum(arrays["is_grid_threshold"])),
        "n_reliable_grid": int(np.sum(arrays["is_reliable_grid"])),
        "n_hd_selective": int(np.sum(arrays["is_hd_selective"])),
        "n_grid_and_hd": int(np.sum(arrays["unit_class"] == "grid_and_hd")),
        "median_grid_score_60": _safe_nanmedian(arrays["grid_score_60"]),
        "n_grid_scale_valid": int(np.sum(arrays["grid_scale_valid"])),
        "median_grid_scale_m": _safe_nanmedian(grid_scale_m),
        "mean_grid_scale_m": _safe_nanmean(grid_scale_m),
        "grid_scale_discreteness": _scale_discreteness(grid_scale_bins),
        "n_grid_fdr_scale_valid": int(np.sum(np.isfinite(grid_fdr_scale_m))),
        "median_grid_fdr_scale_m": _safe_nanmedian(grid_fdr_scale_m),
        "mean_grid_fdr_scale_m": _safe_nanmean(grid_fdr_scale_m),
        "n_grid_threshold_scale_valid": int(np.sum(np.isfinite(grid_threshold_scale_m))),
        "median_grid_threshold_scale_m": _safe_nanmedian(grid_threshold_scale_m),
        "mean_grid_threshold_scale_m": _safe_nanmean(grid_threshold_scale_m),
        "median_split_half_corr": _safe_nanmedian(arrays["split_half_corr"]),
        "analysis_num_shuffles": int(num_shuffles),
        "analysis_gridness_threshold": float(gridness_threshold),
        "analysis_compute_grid_selectivity": bool(modules.compute_grid_selectivity),
        "analysis_compute_grid_geometry": bool(modules.compute_grid_geometry),
        "analysis_compute_shuffle_significance": bool(
            modules.compute_shuffle_significance
        ),
        "analysis_compute_split_half": bool(modules.compute_split_half),
        "analysis_compute_hd_selectivity": bool(modules.compute_hd_selectivity),
    }
    summary.update(_prefixed_scale_population_summary("grid_fdr", fdr_scale_population))
    summary.update(
        _prefixed_scale_population_summary(
            "grid_threshold",
            threshold_scale_population,
        )
    )
    return summary


def _grid_scale_population_stats(
    grid_scale_bins,
    include_mask,
    *,
    random_seed,
    scale_discreteness_num_shuffles,
    scale_discreteness_bins,
    scale_discreteness_range,
    scale_gmm_max_components,
    spatial_bin_size_m,
):
    grid_scale_bins = np.asarray(grid_scale_bins, dtype=np.float64)
    include_mask = np.asarray(include_mask, dtype=bool)
    included_scale_bins = np.where(include_mask, grid_scale_bins, np.nan)
    discreteness = grid_scale_discreteness_test(
        included_scale_bins,
        n_bins=scale_discreteness_bins,
        value_range=scale_discreteness_range,
        num_shuffles=scale_discreteness_num_shuffles,
        random_seed=random_seed,
    )
    gmm = fit_grid_scale_gmm_bic(
        included_scale_bins,
        max_components=scale_gmm_max_components,
        random_seed=random_seed,
    )
    gmm_means_bins = gmm["means"]
    gmm_means_m = None
    if gmm_means_bins is not None and np.isfinite(spatial_bin_size_m):
        gmm_means_m = [float(mean * spatial_bin_size_m) for mean in gmm_means_bins]
    return {
        "discreteness": discreteness,
        "discreteness_num_shuffles": int(scale_discreteness_num_shuffles),
        "discreteness_null_scores": discreteness["null_scores"],
        "gmm": gmm,
        "gmm_means_m": gmm_means_m,
    }


def _prefixed_scale_population_summary(prefix, scale_population):
    discreteness = scale_population["discreteness"]
    gmm = scale_population["gmm"]
    return {
        f"{prefix}_scale_discreteness": discreteness["score"],
        f"{prefix}_scale_discreteness_p": discreteness["p_value"],
        f"{prefix}_scale_discreteness_null_mean": discreteness["null_mean"],
        f"{prefix}_scale_discreteness_null_std": discreteness["null_std"],
        f"{prefix}_scale_discreteness_num_shuffles": int(
            scale_population["discreteness_num_shuffles"]
        ),
        f"{prefix}_scale_gmm_best_components": gmm["best_components"],
        f"{prefix}_scale_gmm_best_bic": gmm["best_bic"],
        f"{prefix}_scale_gmm_component_means_bins": gmm["means"],
        f"{prefix}_scale_gmm_component_means_m": scale_population["gmm_means_m"],
        f"{prefix}_scale_gmm_component_weights": gmm["weights"],
        f"{prefix}_scale_gmm_bic": gmm["bic"],
    }


def _safe_nanmedian(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.median(finite))


def _safe_nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.mean(finite))


def _scale_discreteness(values, n_bins=13, value_range=None):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    counts, _ = np.histogram(finite, bins=int(n_bins), range=value_range)
    return float(np.std(counts))


def grid_scale_discreteness_test(
    values,
    *,
    n_bins=13,
    value_range=(10.0, 36.0),
    num_shuffles=500,
    random_seed=0,
):
    """Compute Banino-style grid-scale discreteness and shuffle p-value."""
    finite = _finite_values(values)
    n_bins = int(n_bins)
    num_shuffles = int(num_shuffles)
    if finite.size == 0 or n_bins <= 0:
        return _empty_discreteness_result(num_shuffles)

    value_range = (float(value_range[0]), float(value_range[1]))
    observed = _scale_discreteness(finite, n_bins=n_bins, value_range=value_range)
    if num_shuffles <= 0 or finite.size == 0:
        return {
            "score": observed,
            "p_value": None,
            "null_mean": None,
            "null_std": None,
            "null_scores": np.empty(0, dtype=np.float64),
        }

    smallest_scale = float(np.min(finite))
    if not np.isfinite(smallest_scale) or smallest_scale <= 0.0:
        return {
            "score": observed,
            "p_value": None,
            "null_mean": None,
            "null_std": None,
            "null_scores": np.empty(0, dtype=np.float64),
        }

    rng = np.random.default_rng(random_seed)
    jitter = rng.uniform(
        -0.5 * smallest_scale,
        0.5 * smallest_scale,
        size=(num_shuffles, finite.size),
    )
    shuffled = finite[np.newaxis, :] + jitter
    counts = np.stack(
        [
            np.histogram(sample, bins=n_bins, range=value_range)[0]
            for sample in shuffled
        ],
        axis=0,
    )
    null_scores = np.std(counts, axis=1)
    p_value = (1.0 + np.sum(null_scores >= observed)) / (1.0 + num_shuffles)
    return {
        "score": observed,
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
        "null_scores": null_scores.astype(np.float64, copy=False),
    }


def fit_grid_scale_gmm_bic(
    values,
    *,
    max_components=8,
    random_seed=0,
    max_iter=200,
    tol=1e-6,
):
    """Fit 1-D Gaussian mixtures and choose the component count with lowest BIC."""
    finite = _finite_values(values)
    max_components = int(max_components)
    if finite.size == 0 or max_components <= 0:
        return _empty_gmm_result()

    max_components = min(max_components, finite.size)
    bic = {}
    fits = {}
    for n_components in range(1, max_components + 1):
        fit = _fit_1d_gmm(
            finite,
            n_components=n_components,
            random_seed=random_seed,
            max_iter=max_iter,
            tol=tol,
        )
        bic_value = _gmm_bic(fit["log_likelihood"], finite.size, n_components)
        bic[str(n_components)] = float(bic_value)
        fits[n_components] = fit

    best_components = min(fits, key=lambda n_components: bic[str(n_components)])
    best_fit = fits[best_components]
    order = np.argsort(best_fit["means"])
    means = best_fit["means"][order]
    weights = best_fit["weights"][order]
    variances = best_fit["variances"][order]
    return {
        "best_components": int(best_components),
        "best_bic": float(bic[str(best_components)]),
        "bic": bic,
        "means": [float(value) for value in means],
        "weights": [float(value) for value in weights],
        "variances": [float(value) for value in variances],
    }


def _finite_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _empty_discreteness_result(num_shuffles):
    return {
        "score": None,
        "p_value": None,
        "null_mean": None,
        "null_std": None,
        "null_scores": np.full(max(0, int(num_shuffles)), np.nan, dtype=np.float64),
    }


def _empty_gmm_result():
    return {
        "best_components": None,
        "best_bic": None,
        "bic": {},
        "means": None,
        "weights": None,
        "variances": None,
    }


def _fit_1d_gmm(values, *, n_components, random_seed, max_iter, tol):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(random_seed + int(n_components))
    quantiles = (np.arange(n_components, dtype=np.float64) + 0.5) / n_components
    means = np.quantile(values, quantiles)
    if np.unique(means).size < n_components:
        means = rng.choice(values, size=n_components, replace=True)

    variance_floor = max(float(np.var(values)) * 1e-6, 1e-6)
    variances = np.full(
        n_components,
        max(float(np.var(values)), variance_floor),
        dtype=np.float64,
    )
    weights = np.full(n_components, 1.0 / n_components, dtype=np.float64)
    prev_log_likelihood = -np.inf

    for _ in range(int(max_iter)):
        log_prob = _gmm_component_log_prob(values, weights, means, variances)
        log_norm = _logsumexp(log_prob, axis=1)
        log_likelihood = float(np.sum(log_norm))
        responsibilities = np.exp(log_prob - log_norm[:, np.newaxis])
        nk = np.sum(responsibilities, axis=0) + 1e-12
        weights = nk / values.size
        means = (responsibilities.T @ values) / nk
        diffs = values[:, np.newaxis] - means[np.newaxis, :]
        variances = np.sum(responsibilities * diffs**2, axis=0) / nk
        variances = np.maximum(variances, variance_floor)
        if abs(log_likelihood - prev_log_likelihood) < float(tol):
            break
        prev_log_likelihood = log_likelihood

    log_prob = _gmm_component_log_prob(values, weights, means, variances)
    log_likelihood = float(np.sum(_logsumexp(log_prob, axis=1)))
    return {
        "weights": weights,
        "means": means,
        "variances": variances,
        "log_likelihood": log_likelihood,
    }


def _gmm_component_log_prob(values, weights, means, variances):
    values = values[:, np.newaxis]
    return (
        np.log(weights[np.newaxis, :] + 1e-300)
        - 0.5 * np.log(2.0 * np.pi * variances[np.newaxis, :])
        - 0.5 * (values - means[np.newaxis, :]) ** 2 / variances[np.newaxis, :]
    )


def _logsumexp(values, axis=None):
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_values)
    summed = np.sum(stable, axis=axis, keepdims=True)
    result = max_values + np.log(summed)
    if axis is None:
        return result.squeeze()
    return np.squeeze(result, axis=axis)


def _gmm_bic(log_likelihood, n_samples, n_components):
    n_parameters = (n_components - 1) + n_components + n_components
    return -2.0 * float(log_likelihood) + n_parameters * np.log(float(n_samples))


def _nanmean_axis0(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 0 or not np.any(np.isfinite(values)):
        return np.full(values.shape[1:], np.nan, dtype=np.float64)
    return np.nanmean(values, axis=0)


def _nanstd_axis0(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 0 or not np.any(np.isfinite(values)):
        return np.full(values.shape[1:], np.nan, dtype=np.float64)
    return np.nanstd(values, axis=0)


def _nanpercentile_axis0(values, percentile):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 0 or not np.any(np.isfinite(values)):
        return np.full(values.shape[1:], np.nan, dtype=np.float64)
    return np.nanpercentile(values, percentile, axis=0)
