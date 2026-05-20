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

import numpy as np


STAT_FIELDS = [
    "unit_id",
    "grid_score_60",
    "grid_score_90",
    "grid_score_60_p",
    "grid_score_60_q",
    "grid_score_90_p",
    "grid_score_90_q",
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
    "is_reliable_grid",
    "is_hd_selective",
    "unit_class",
]


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
):
    """Compute v1 grid-cell statistical analysis for bottleneck units."""
    positions, head_directions, activations = _prepare_inputs(
        positions,
        head_directions,
        activations,
    )
    n_trajectories, seq_len, n_units = activations.shape
    rng = np.random.default_rng(random_seed)

    observed_ratemaps = _calculate_ratemaps(scorer, positions, activations)
    observed_grid_score_60, observed_grid_score_90 = _score_ratemaps(
        scorer,
        observed_ratemaps,
    )

    half = _compute_split_half_stats(
        scorer,
        positions,
        activations,
        split_method=split_method,
        random_seed=random_seed,
    )
    tuning, hd_mrl, hd_preferred_direction = directional_tuning(
        head_directions,
        activations,
        n_bins=n_direction_bins,
    )

    null_60, null_90, null_hd = _shuffle_null_distributions(
        scorer,
        positions,
        head_directions,
        activations,
        n_direction_bins=n_direction_bins,
        num_shuffles=num_shuffles,
        min_shift_fraction=min_shift_fraction,
        rng=rng,
    )

    grid_60_p = shuffle_p_values(observed_grid_score_60, null_60)
    grid_90_p = shuffle_p_values(observed_grid_score_90, null_90)
    hd_p = shuffle_p_values(hd_mrl, null_hd)
    grid_60_q = benjamini_hochberg(grid_60_p)
    grid_90_q = benjamini_hochberg(grid_90_p)
    hd_q = benjamini_hochberg(hd_p)

    is_grid_fdr = grid_60_q <= fdr_alpha
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
        "is_reliable_grid": is_reliable_grid,
        "is_hd_selective": is_hd_selective,
        "unit_class": unit_class,
    }
    summary = _summary(arrays, num_shuffles)
    rows = [_row_from_arrays(arrays, index) for index in range(n_units)]

    return {
        "fields": STAT_FIELDS,
        "rows": rows,
        "arrays": arrays,
        "summary": summary,
        "null_grid_score_60": null_60,
        "null_grid_score_90": null_90,
        "null_hd_mrl": null_hd,
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
        score_60, score_90, _, _, _ = scorer.get_scores_batch(ratemaps)
        return np.asarray(score_60, dtype=np.float64), np.asarray(score_90, dtype=np.float64)

    scores = [scorer.get_scores(ratemap) for ratemap in ratemaps]
    return (
        np.asarray([score[0] for score in scores], dtype=np.float64),
        np.asarray([score[1] for score in scores], dtype=np.float64),
    )


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
    score_60_a, _ = _score_ratemaps(scorer, ratemaps_a)
    score_60_b, _ = _score_ratemaps(scorer, ratemaps_b)
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
        shuffled_ratemaps = _calculate_ratemaps(scorer, positions, shifted)
        null_60[shuffle_index], null_90[shuffle_index] = _score_ratemaps(
            scorer,
            shuffled_ratemaps,
        )
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


def _row_from_arrays(arrays, index):
    row = {}
    for field in STAT_FIELDS:
        value = arrays[field][index]
        if isinstance(value, np.generic):
            value = value.item()
        row[field] = value
    return row


def _summary(arrays, num_shuffles):
    return {
        "n_units": int(arrays["unit_id"].size),
        "n_grid_fdr": int(np.sum(arrays["is_grid_fdr"])),
        "n_reliable_grid": int(np.sum(arrays["is_reliable_grid"])),
        "n_hd_selective": int(np.sum(arrays["is_hd_selective"])),
        "n_grid_and_hd": int(np.sum(arrays["unit_class"] == "grid_and_hd")),
        "median_grid_score_60": _safe_nanmedian(arrays["grid_score_60"]),
        "median_split_half_corr": _safe_nanmedian(arrays["split_half_corr"]),
        "analysis_num_shuffles": int(num_shuffles),
    }


def _safe_nanmedian(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.median(finite))


def _nanmean_axis0(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 0:
        return np.full(values.shape[1:], np.nan, dtype=np.float64)
    return np.nanmean(values, axis=0)


def _nanstd_axis0(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 0:
        return np.full(values.shape[1:], np.nan, dtype=np.float64)
    return np.nanstd(values, axis=0)


def _nanpercentile_axis0(values, percentile):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 0:
        return np.full(values.shape[1:], np.nan, dtype=np.float64)
    return np.nanpercentile(values, percentile, axis=0)
