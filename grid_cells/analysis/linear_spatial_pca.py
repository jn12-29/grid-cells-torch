"""Linear spatial-PCA gridness baseline for bottleneck hidden states.

Usage:
    result = analyze_linear_spatial_pca_gridness(scorer, positions, activations)
    save_linear_spatial_pca_artifacts(result, "results/run/eval_stats", epoch=10)
"""

import csv
import json
import os

import numpy as np

from grid_cells.analysis.spatial_stats import (
    _calculate_ratemaps,
    _score_ratemaps,
    _spatial_bin_size_m,
    circular_shift_activations,
    sample_circular_shifts,
)


LINEAR_SPATIAL_PCA_FIELDS = [
    "mode_id",
    "selected_rank",
    "fit_variance_explained",
    "grid_score_60_fit",
    "grid_score_90_fit",
    "grid_score_60_test",
    "grid_score_90_test",
    "grid_scale_bins",
    "grid_scale_m",
    "grid_scale_valid",
    "loading_participation_ratio",
]


def analyze_linear_spatial_pca_gridness(
    scorer,
    positions,
    activations,
    *,
    top_k=16,
    fit_fraction=0.5,
    num_shuffles=0,
    min_shift_fraction=0.1,
    random_seed=0,
):
    """Evaluate held-out gridness of spatial-PCA hidden-state projections."""
    positions, activations = _prepare_inputs(positions, activations)
    n_trajectories, seq_len, n_units = activations.shape
    top_k = max(1, min(int(top_k), n_units))
    fit_indices, test_indices = _fit_test_indices(
        n_trajectories,
        fit_fraction=fit_fraction,
        random_seed=random_seed,
    )

    fit = _fit_spatial_pca_directions(
        scorer,
        positions[fit_indices],
        activations[fit_indices],
        top_k=top_k,
    )
    observed = _evaluate_modes(
        scorer,
        positions[test_indices],
        activations[test_indices],
        fit["center"],
        fit["W_top"],
    )

    null_T = np.full(int(num_shuffles), np.nan, dtype=np.float64)
    null_top_scores = np.full((int(num_shuffles), top_k), np.nan, dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    for shuffle_index in range(int(num_shuffles)):
        shifts = sample_circular_shifts(
            n_trajectories,
            seq_len,
            n_units=None,
            min_shift_fraction=min_shift_fraction,
            rng=rng,
        )
        shifted = circular_shift_activations(activations, shifts)
        shuffled_fit = _fit_spatial_pca_directions(
            scorer,
            positions[fit_indices],
            shifted[fit_indices],
            top_k=top_k,
        )
        shuffled_observed = _evaluate_modes(
            scorer,
            positions[test_indices],
            shifted[test_indices],
            shuffled_fit["center"],
            shuffled_fit["W_top"],
        )
        null_top_scores[shuffle_index] = shuffled_observed["grid_score_60"]
        null_T[shuffle_index] = _max_finite(shuffled_observed["grid_score_60"])

    T_real = _max_finite(observed["grid_score_60"])
    p_value = _shuffle_p_value(T_real, null_T)
    loading_participation_ratio = _loading_participation_ratio(fit["W_top"])
    mode_order = _mode_order_by_score(observed["grid_score_60"])
    arrays = {
        "mode_id": np.arange(top_k, dtype=np.int64),
        "selected_rank": np.arange(1, top_k + 1, dtype=np.int64),
        "fit_variance_explained": fit["variance_explained"],
        "fit_cumulative_variance_explained": np.cumsum(fit["variance_explained"]),
        "grid_score_60_fit": fit["grid_score_60"],
        "grid_score_90_fit": fit["grid_score_90"],
        "grid_score_60_test": observed["grid_score_60"],
        "grid_score_90_test": observed["grid_score_90"],
        "grid_scale_bins": observed["grid_scale_bins"],
        "grid_scale_m": observed["grid_scale_m"],
        "grid_scale_valid": observed["grid_scale_valid"],
        "loading_participation_ratio": loading_participation_ratio,
        "mode_order_by_test_grid_score": mode_order,
    }
    rows = [_row_from_arrays(arrays, index) for index in range(top_k)]
    W_top = fit["W_top"]
    orthogonality_error = float(
        np.max(np.abs(W_top.T @ W_top - np.eye(top_k)))
    )

    summary = {
        "version": "v1_spatial_pca_baseline",
        "fit_method": "spatial_bin_mean_svd",
        "transform_constraint": "orthogonal_top_k",
        "not_grid_score_optimized": True,
        "shuffle_method": "trajectory_shared_circular_shift_refit",
        "split_method": "random_trajectory",
        "n_trajectories": int(n_trajectories),
        "n_fit_trajectories": int(fit_indices.size),
        "n_test_trajectories": int(test_indices.size),
        "n_hidden_units": int(n_units),
        "top_k": int(top_k),
        "fit_fraction": float(fit_fraction),
        "num_shuffles": int(num_shuffles),
        "min_shift_fraction": float(min_shift_fraction),
        "random_seed": int(random_seed),
        "T_real": _json_float(T_real),
        "p_value": _json_float(p_value),
        "null_T_mean": _json_float(_finite_mean(null_T)) if null_T.size else None,
        "null_T_std": _json_float(_finite_std(null_T)) if null_T.size else None,
        "orthogonality_error": orthogonality_error,
    }
    return {
        "fields": LINEAR_SPATIAL_PCA_FIELDS,
        "rows": rows,
        "arrays": arrays,
        "summary": summary,
        "W_top": W_top,
        "fit_center": fit["center"],
        "fit_indices": fit_indices,
        "test_indices": test_indices,
        "null_T": null_T,
        "null_top_scores": null_top_scores,
        "fit_ratemaps": fit["ratemaps"],
        "test_ratemaps": observed["ratemaps"],
        "test_sacs": observed["sacs"],
        "mode_order_by_test_grid_score": mode_order,
    }


def save_linear_spatial_pca_artifacts(
    result,
    save_dir,
    epoch,
    *,
    save_plots=True,
    plot_top_n=16,
    pdf_dpi=100,
):
    """Write CSV, NPZ, and JSON artifacts for linear spatial-PCA modes."""
    os.makedirs(save_dir, exist_ok=True)
    stem = f"linear_spatial_pca_modes_epoch_{int(epoch):04d}"
    csv_path = os.path.join(save_dir, stem + ".csv")
    npz_path = os.path.join(save_dir, stem + ".npz")
    json_path = os.path.join(
        save_dir,
        f"linear_spatial_pca_summary_epoch_{int(epoch):04d}.json",
    )

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result["fields"])
        writer.writeheader()
        writer.writerows(result["rows"])

    npz_payload = {key: value for key, value in result["arrays"].items()}
    npz_payload.update(
        {
            "W_top": result["W_top"],
            "fit_center": result["fit_center"],
            "fit_indices": result["fit_indices"],
            "test_indices": result["test_indices"],
            "null_T": result["null_T"],
            "null_top_scores": result["null_top_scores"],
        }
    )
    for key in ("fit_ratemaps", "test_ratemaps", "test_sacs"):
        if key in result:
            npz_payload[key] = result[key]
    np.savez_compressed(npz_path, **npz_payload)

    with open(json_path, "w") as handle:
        json.dump(result["summary"], handle, indent=2, sort_keys=True)
        handle.write("\n")

    paths = {"csv": csv_path, "npz": npz_path, "summary_json": json_path}
    if save_plots:
        plot_paths = _save_linear_spatial_pca_plots(
            result,
            save_dir=save_dir,
            epoch=epoch,
            plot_top_n=plot_top_n,
            pdf_dpi=pdf_dpi,
        )
        paths.update(plot_paths)
    return paths


def _prepare_inputs(positions, activations):
    positions = np.asarray(positions)
    activations = np.asarray(activations)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (n_trajectories, seq_len, 2)")
    if activations.ndim != 3:
        raise ValueError("activations must have shape (n_trajectories, seq_len, n_units)")
    if activations.shape[:2] != positions.shape[:2]:
        raise ValueError("activations and positions must share trajectory and time dimensions")
    return positions, activations


def _fit_test_indices(n_trajectories, *, fit_fraction, random_seed):
    indices = np.arange(n_trajectories, dtype=np.int64)
    if n_trajectories <= 1:
        return indices, np.empty(0, dtype=np.int64)

    fit_fraction = float(fit_fraction)
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between 0 and 1")

    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(indices)
    n_fit = int(round(n_trajectories * fit_fraction))
    n_fit = max(1, min(n_fit, n_trajectories - 1))
    return np.sort(shuffled[:n_fit]), np.sort(shuffled[n_fit:])


def _fit_spatial_pca_directions(scorer, positions, activations, *, top_k):
    n_units = activations.shape[-1]
    top_k = max(1, min(int(top_k), n_units))
    ratemaps = _calculate_ratemaps(scorer, positions, activations)
    bin_means = np.moveaxis(ratemaps, 0, -1).reshape(-1, n_units)
    occupied = np.all(np.isfinite(bin_means), axis=1)
    bin_means = bin_means[occupied]

    if bin_means.size == 0:
        center = np.nanmean(activations.reshape(-1, n_units), axis=0)
        center = np.nan_to_num(center, nan=0.0)
        W_top = _complete_orthonormal_rows(np.empty((0, n_units)), n_units, top_k)
        fit_scores = _empty_mode_scores(top_k)
        return {
            "W_top": W_top,
            "center": center,
            "variance_explained": np.zeros(top_k, dtype=np.float64),
            "ratemaps": np.full((top_k,) + ratemaps.shape[1:], np.nan, dtype=np.float64),
            "grid_score_60": fit_scores["grid_score_60"],
            "grid_score_90": fit_scores["grid_score_90"],
        }

    center = np.mean(bin_means, axis=0)
    centered = bin_means - center[np.newaxis, :]
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    W_top = _complete_orthonormal_rows(vh, n_units, top_k)

    variance = singular_values ** 2
    total = float(np.sum(variance))
    explained = np.zeros(top_k, dtype=np.float64)
    if total > 0.0:
        explained[: min(top_k, variance.size)] = variance[:top_k] / total
    fit_scores = _evaluate_modes(scorer, positions, activations, center, W_top)
    return {
        "W_top": W_top,
        "center": center,
        "variance_explained": explained,
        "ratemaps": fit_scores["ratemaps"],
        "grid_score_60": fit_scores["grid_score_60"],
        "grid_score_90": fit_scores["grid_score_90"],
    }


def _complete_orthonormal_rows(rows, n_units, top_k):
    directions = []
    for row in np.asarray(rows, dtype=np.float64):
        _append_orthonormal_direction(directions, row)
        if len(directions) == top_k:
            break
    for unit_index in range(n_units):
        basis = np.zeros(n_units, dtype=np.float64)
        basis[unit_index] = 1.0
        _append_orthonormal_direction(directions, basis)
        if len(directions) == top_k:
            break
    if len(directions) < top_k:
        raise ValueError("could not construct enough orthonormal directions")
    return np.stack(directions[:top_k], axis=1)


def _append_orthonormal_direction(directions, candidate):
    vector = np.asarray(candidate, dtype=np.float64).copy()
    for direction in directions:
        vector -= np.dot(direction, vector) * direction
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        directions.append(vector / norm)


def _evaluate_modes(scorer, positions, activations, center, W_top):
    top_k = W_top.shape[1]
    if positions.shape[0] == 0:
        return _empty_mode_scores(top_k)

    projected = (activations - center[np.newaxis, np.newaxis, :]) @ W_top
    ratemaps = _calculate_ratemaps(scorer, positions, projected)
    score_60, score_90, sacs = _score_ratemaps(scorer, ratemaps)
    scale_bins = np.full(top_k, np.nan, dtype=np.float64)
    if hasattr(scorer, "calculate_grid_scales"):
        scale_bins, _ = scorer.calculate_grid_scales(sacs)
    scale_m = scale_bins * _spatial_bin_size_m(scorer)
    return {
        "ratemaps": ratemaps,
        "sacs": sacs,
        "grid_score_60": score_60,
        "grid_score_90": score_90,
        "grid_scale_bins": scale_bins,
        "grid_scale_m": scale_m,
        "grid_scale_valid": np.isfinite(scale_bins),
    }


def _empty_mode_scores(top_k):
    return {
        "ratemaps": np.empty((top_k, 0, 0), dtype=np.float64),
        "sacs": np.empty((top_k, 0, 0), dtype=np.float64),
        "grid_score_60": np.full(top_k, np.nan, dtype=np.float64),
        "grid_score_90": np.full(top_k, np.nan, dtype=np.float64),
        "grid_scale_bins": np.full(top_k, np.nan, dtype=np.float64),
        "grid_scale_m": np.full(top_k, np.nan, dtype=np.float64),
        "grid_scale_valid": np.zeros(top_k, dtype=bool),
    }


def _max_finite(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else np.nan


def _shuffle_p_value(observed, null_distribution):
    null_distribution = np.asarray(null_distribution, dtype=np.float64)
    finite = np.isfinite(null_distribution)
    if not np.isfinite(observed) or not np.any(finite):
        return np.nan
    return float((1.0 + np.sum(null_distribution[finite] >= observed)) / (1.0 + finite.sum()))


def _row_from_arrays(arrays, index):
    row = {}
    for field in LINEAR_SPATIAL_PCA_FIELDS:
        value = arrays[field][index]
        row[field] = value.item() if hasattr(value, "item") else value
    return row


def _loading_participation_ratio(W_top):
    squared = np.asarray(W_top, dtype=np.float64) ** 2
    denom = np.sum(squared ** 2, axis=0)
    ratio = np.full(W_top.shape[1], np.nan, dtype=np.float64)
    valid = denom > 0.0
    ratio[valid] = 1.0 / denom[valid]
    return ratio


def _mode_order_by_score(scores):
    scores = np.asarray(scores, dtype=np.float64)
    sortable = np.where(np.isfinite(scores), scores, -np.inf)
    return np.argsort(-sortable, kind="mergesort").astype(np.int64)


def _save_linear_spatial_pca_plots(result, *, save_dir, epoch, plot_top_n, pdf_dpi):
    required = {"fit_ratemaps", "test_ratemaps", "test_sacs", "W_top"}
    if not required.issubset(result):
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    overview_pdf = os.path.join(
        save_dir,
        f"linear_spatial_pca_overview_epoch_{int(epoch):04d}.pdf",
    )
    overview_png = os.path.join(
        save_dir,
        f"linear_spatial_pca_overview_epoch_{int(epoch):04d}.png",
    )
    mode_maps_pdf = os.path.join(
        save_dir,
        f"linear_spatial_pca_mode_maps_epoch_{int(epoch):04d}.pdf",
    )

    _plot_linear_spatial_pca_overview(
        result,
        pdf_path=overview_pdf,
        png_path=overview_png,
        dpi=pdf_dpi,
        plt=plt,
    )
    _plot_linear_spatial_pca_mode_maps(
        result,
        pdf_path=mode_maps_pdf,
        plot_top_n=plot_top_n,
        dpi=pdf_dpi,
        plt=plt,
        PdfPages=PdfPages,
    )
    return {
        "overview_pdf": overview_pdf,
        "overview_png": overview_png,
        "mode_maps_pdf": mode_maps_pdf,
    }


def _plot_linear_spatial_pca_overview(result, *, pdf_path, png_path, dpi, plt):
    arrays = result.get("arrays", {})
    summary = result.get("summary", {})
    W_top = np.asarray(result["W_top"], dtype=np.float64)
    mode_ids = np.asarray(arrays.get("mode_id", np.arange(W_top.shape[1])))
    fit_var = np.asarray(
        arrays.get("fit_variance_explained", np.full(W_top.shape[1], np.nan)),
        dtype=np.float64,
    )
    cumulative = np.asarray(
        arrays.get("fit_cumulative_variance_explained", np.cumsum(fit_var)),
        dtype=np.float64,
    )
    test_scores = np.asarray(
        arrays.get("grid_score_60_test", np.full(W_top.shape[1], np.nan)),
        dtype=np.float64,
    )
    null_T = np.asarray(result.get("null_T", np.empty(0)), dtype=np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    fig.suptitle("Linear spatial-PCA baseline diagnostics (v1, not grid-score optimized)")

    ax = axes[0, 0]
    ax.bar(mode_ids, test_scores, color="#4c78a8")
    ax.set_title("Held-out grid score 60 by mode")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Grid score 60 (test)")
    ax.axhline(0.0, color="0.4", linewidth=0.8)

    ax = axes[0, 1]
    ax.bar(mode_ids, fit_var, color="#59a14f", label="Per-mode")
    ax.plot(mode_ids, cumulative, color="#f28e2b", marker="o", label="Cumulative")
    ax.set_title("Fit variance explained")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Fraction")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 2]
    finite_null = null_T[np.isfinite(null_T)]
    if finite_null.size:
        ax.hist(finite_null, bins=min(20, max(5, finite_null.size)), color="#bab0ac")
        T_real = summary.get("T_real")
        if T_real is not None and np.isfinite(float(T_real)):
            ax.axvline(float(T_real), color="#e15759", linewidth=2.0, label="T_real")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No shuffle null samples", ha="center", va="center")
    ax.set_title(f"T null distribution  p={summary.get('p_value')}")
    ax.set_xlabel("Max held-out grid score 60")
    ax.set_ylabel("Shuffles")

    ax = axes[1, 0]
    ax.scatter(fit_var, test_scores, color="#b07aa1")
    ax.set_title("Fit variance vs held-out gridness")
    ax.set_xlabel("Fit variance explained")
    ax.set_ylabel("Grid score 60 (test)")
    ax.axhline(0.0, color="0.4", linewidth=0.8)

    ax = axes[1, 1]
    im = ax.imshow(W_top.T, aspect="auto", cmap="coolwarm")
    ax.set_title("W_top.T loadings")
    ax.set_xlabel("Hidden unit")
    ax.set_ylabel("Mode")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 2]
    order = np.asarray(
        result.get("mode_order_by_test_grid_score", _mode_order_by_score(test_scores)),
        dtype=np.int64,
    )
    ordered_scores = test_scores[order] if order.size else np.empty(0)
    ordered_labels = mode_ids[order] if order.size else np.empty(0)
    ax.bar(np.arange(ordered_scores.size), ordered_scores, color="#edc948")
    ax.set_title("Modes sorted by held-out gridness")
    ax.set_xlabel("Sorted rank")
    ax.set_ylabel("Grid score 60 (test)")
    if ordered_scores.size <= 20:
        ax.set_xticks(np.arange(ordered_scores.size))
        ax.set_xticklabels([str(int(label)) for label in ordered_labels], rotation=45)
    ax.axhline(0.0, color="0.4", linewidth=0.8)

    fig.savefig(pdf_path, dpi=dpi)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


def _plot_linear_spatial_pca_mode_maps(
    result,
    *,
    pdf_path,
    plot_top_n,
    dpi,
    plt,
    PdfPages,
):
    arrays = result.get("arrays", {})
    W_top = np.asarray(result["W_top"], dtype=np.float64)
    fit_maps = np.asarray(result["fit_ratemaps"], dtype=np.float64)
    test_maps = np.asarray(result["test_ratemaps"], dtype=np.float64)
    test_sacs = np.asarray(result["test_sacs"], dtype=np.float64)
    n_modes = W_top.shape[1]
    order = np.asarray(
        result.get(
            "mode_order_by_test_grid_score",
            _mode_order_by_score(arrays.get("grid_score_60_test", np.full(n_modes, np.nan))),
        ),
        dtype=np.int64,
    )
    n_plot = max(0, min(int(plot_top_n), n_modes, order.size))

    with PdfPages(pdf_path) as pdf:
        if n_plot == 0:
            fig, ax = plt.subplots(figsize=(7.0, 3.0), constrained_layout=True)
            ax.axis("off")
            ax.text(0.5, 0.5, "No linear spatial-PCA modes to plot", ha="center", va="center")
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
            return

        for start in range(0, n_plot, 4):
            page_modes = order[start : start + 4]
            fig, axes = plt.subplots(
                len(page_modes),
                4,
                figsize=(11.0, 2.55 * len(page_modes)),
                constrained_layout=True,
                squeeze=False,
            )
            fig.suptitle("Linear spatial-PCA mode maps (v1 baseline)")
            for row_index, mode_index in enumerate(page_modes):
                _plot_mode_row(
                    axes[row_index],
                    int(mode_index),
                    arrays=arrays,
                    fit_maps=fit_maps,
                    test_maps=test_maps,
                    test_sacs=test_sacs,
                    W_top=W_top,
                    plt=plt,
                )
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)


def _plot_mode_row(axes, mode_index, *, arrays, fit_maps, test_maps, test_sacs, W_top, plt):
    fit_var = _array_value(arrays, "fit_variance_explained", mode_index)
    fit_60 = _array_value(arrays, "grid_score_60_fit", mode_index)
    test_60 = _array_value(arrays, "grid_score_60_test", mode_index)
    scale = _array_value(arrays, "grid_scale_m", mode_index)
    title = (
        f"mode {mode_index}  fit var={_format_float(fit_var)}  "
        f"fit/test g60={_format_float(fit_60)}/{_format_float(test_60)}  "
        f"scale={_format_float(scale)} m"
    )

    _imshow_or_placeholder(axes[0], fit_maps, mode_index, cmap="viridis")
    axes[0].set_title(title, fontsize=8)
    axes[0].set_ylabel("fit map")
    _imshow_or_placeholder(axes[1], test_maps, mode_index, cmap="viridis")
    axes[1].set_title("held-out test map", fontsize=8)
    _imshow_or_placeholder(axes[2], test_sacs, mode_index, cmap="magma")
    axes[2].set_title("held-out SAC", fontsize=8)
    axes[3].bar(np.arange(W_top.shape[0]), W_top[:, mode_index], color="#4c78a8")
    axes[3].set_title("loading weights", fontsize=8)
    axes[3].set_xlabel("hidden unit")
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[3].tick_params(labelsize=7)


def _imshow_or_placeholder(ax, maps, index, *, cmap):
    if index >= maps.shape[0] or maps.ndim != 3 or maps.shape[1] == 0 or maps.shape[2] == 0:
        ax.text(0.5, 0.5, "No held-out map", ha="center", va="center")
        return
    ax.imshow(maps[index], origin="lower", cmap=cmap)


def _array_value(arrays, key, index):
    values = arrays.get(key)
    if values is None:
        return np.nan
    values = np.asarray(values)
    if index >= values.shape[0]:
        return np.nan
    return values[index]


def _format_float(value):
    value = float(value) if np.isfinite(value) else np.nan
    return f"{value:.3g}" if np.isfinite(value) else "nan"


def _json_float(value):
    return float(value) if np.isfinite(value) else None


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def _finite_std(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.std(finite)) if finite.size else np.nan
