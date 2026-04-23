"""Ratemap scoring and plotting helpers."""

from concurrent.futures import ProcessPoolExecutor
import io
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch


def _render_page_to_png_bytes(args):
    """Render one page of ratemaps and SACs to PNG bytes."""
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import numpy as _np

    (
        ratemaps_page,
        sac_page,
        score_60_page,
        mask_60_page,
        page_indices,
        rows,
        cols,
        cmap,
        dpi,
        nbins,
        plotting_sac_mask,
    ) = args

    center = nbins - 1
    fig = _plt.figure(figsize=(24, rows * 4))

    for panel_idx, index in enumerate(page_indices):
        title = "%d (%.2f)" % (int(index), score_60_page[panel_idx])

        ax_rm = _plt.subplot(rows * 2, cols, panel_idx + 1)
        ax_rm.imshow(ratemaps_page[panel_idx], interpolation="none", cmap=cmap)
        ax_rm.axis("off")
        ax_rm.set_title(title)

        ax_sac = _plt.subplot(rows * 2, cols, rows * cols + panel_idx + 1)
        useful_sac = sac_page[panel_idx] * plotting_sac_mask
        ax_sac.imshow(useful_sac, interpolation="none", cmap=cmap)
        mask_min, mask_max = mask_60_page[panel_idx]
        ax_sac.add_artist(
            _plt.Circle(
                (center, center),
                mask_min * nbins,
                fill=False,
                edgecolor="k",
            )
        )
        ax_sac.add_artist(
            _plt.Circle(
                (center, center),
                mask_max * nbins,
                fill=False,
                edgecolor="k",
            )
        )
        ax_sac.axis("off")
        ax_sac.set_title(title)

    buffer = _io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    _plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def _score_ratemap_chunk(args):
    """Worker helper for scoring a chunk of rate maps."""
    scorer, ratemap_chunk = args
    score_60, score_90, max_60_mask, max_90_mask, sacs = scorer.get_scores_batch(
        ratemap_chunk
    )
    return {
        "score_60": np.asarray(score_60),
        "score_90": np.asarray(score_90),
        "max_60_mask": max_60_mask,
        "max_90_mask": max_90_mask,
        "sacs": np.asarray(sacs),
    }


def score_ratemaps(scorer, ratemaps, num_workers: int = 0, chunk_size: int = 32):
    """Score rate maps, optionally in parallel across unit chunks."""
    n_units = ratemaps.shape[0]
    chunk_size = max(1, chunk_size)
    chunks = [
        (scorer, ratemaps[start : start + chunk_size])
        for start in range(0, n_units, chunk_size)
    ]

    if num_workers and num_workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            chunk_results = list(executor.map(_score_ratemap_chunk, chunks))
    else:
        chunk_results = [_score_ratemap_chunk(chunk) for chunk in chunks]

    score_60 = np.concatenate([result["score_60"] for result in chunk_results], axis=0)
    score_90 = np.concatenate([result["score_90"] for result in chunk_results], axis=0)
    max_60_mask = [mask for result in chunk_results for mask in result["max_60_mask"]]
    max_90_mask = [mask for result in chunk_results for mask in result["max_90_mask"]]
    sacs = np.concatenate([result["sacs"] for result in chunk_results], axis=0)

    return score_60, score_90, max_60_mask, max_90_mask, sacs


def get_scores_and_plot_from_ratemaps(
    scorer,
    ratemaps,
    directory: str,
    filename: str,
    cm: str = "jet",
    sort_by_score_60: bool = True,
    num_workers: int = 0,
    chunk_size: int = 32,
    units_per_page: int = 128,
    pdf_dpi: int = 72,
):
    """Compute grid scores from precomputed rate maps and save a paginated PDF."""
    ratemaps = np.asarray(ratemaps)
    n_units = ratemaps.shape[0]
    score_60, score_90, max_60_mask, max_90_mask, sac = score_ratemaps(
        scorer,
        ratemaps,
        num_workers=num_workers,
        chunk_size=chunk_size,
    )

    ordering = np.argsort(-np.array(score_60)) if sort_by_score_60 else np.arange(n_units)
    units_per_page = max(1, units_per_page)
    cols = min(16, units_per_page)

    page_args = []
    for page_start in range(0, n_units, units_per_page):
        page_indices = ordering[page_start : page_start + units_per_page]
        n_page = len(page_indices)
        rows = int(np.ceil(n_page / cols))
        page_args.append(
            (
                ratemaps[page_indices],
                sac[page_indices],
                np.asarray(score_60)[page_indices],
                [max_60_mask[i] for i in page_indices],
                page_indices,
                rows,
                cols,
                cm,
                pdf_dpi,
                scorer._nbins,
                scorer._plotting_sac_mask,
            )
        )

    if num_workers > 1 and len(page_args) > 1:
        with ProcessPoolExecutor(max_workers=min(num_workers, len(page_args))) as executor:
            png_bytes_list = list(executor.map(_render_page_to_png_bytes, page_args))
    else:
        png_bytes_list = [_render_page_to_png_bytes(args) for args in page_args]

    os.makedirs(directory, exist_ok=True)
    with PdfPages(os.path.join(directory, filename)) as pdf:
        for png_bytes in png_bytes_list:
            img = plt.imread(io.BytesIO(png_bytes))
            height, width = img.shape[:2]
            fig = plt.figure(figsize=(width / pdf_dpi, height / pdf_dpi), dpi=pdf_dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, dpi=pdf_dpi)
            plt.close(fig)

    return (
        np.asarray(score_60),
        np.asarray(score_90),
        np.asarray([np.mean(mask) for mask in max_60_mask]),
        np.asarray([np.mean(mask) for mask in max_90_mask]),
    )


def plot_hdc_tuning_curves(
    hd_angles: np.ndarray,
    hdc_acts: np.ndarray,
    n_bins: int = 20,
    save_path: str = "hdc_tuning.pdf",
    pdf_dpi: int = 100,
) -> None:
    """Plot head-direction-cell directional tuning curves and save to PDF."""
    hd_angles = np.asarray(hd_angles, dtype=np.float64).ravel()
    hdc_acts = np.asarray(hdc_acts, dtype=np.float64)
    if hdc_acts.ndim == 1:
        hdc_acts = hdc_acts[:, np.newaxis]
    _, n_hdc = hdc_acts.shape

    hd_angles = (hd_angles + np.pi) % (2 * np.pi) - np.pi
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bar_width = 2 * np.pi / n_bins

    bin_idx = np.digitize(hd_angles, edges[1:-1])
    tuning = np.zeros((n_hdc, n_bins), dtype=np.float64)
    for bin_id in range(n_bins):
        mask = bin_idx == bin_id
        if mask.sum() > 0:
            tuning[:, bin_id] = hdc_acts[mask].mean(axis=0)

    cols = min(12, n_hdc)
    rows = max(1, int(np.ceil(n_hdc / cols)))
    fig, axes = plt.subplots(
        rows,
        cols,
        subplot_kw={"projection": "polar"},
        figsize=(cols * 2.2, rows * 2.6 + 0.6),
        squeeze=False,
    )
    fig.suptitle("HDC directional tuning curves", fontsize=11)

    vmax = float(tuning.max()) if tuning.max() > 0 else 1.0
    for index in range(n_hdc):
        row, col = divmod(index, cols)
        ax = axes[row, col]
        colors = plt.cm.plasma(tuning[index] / vmax)
        ax.bar(
            centers,
            tuning[index],
            width=bar_width,
            color=colors,
            align="center",
            alpha=0.85,
        )
        ax.set_ylim(0, vmax * 1.1)
        ax.set_title(f"HDC {index}", fontsize=7, pad=3)
        ax.tick_params(labelsize=5)
        ax.set_yticklabels([])

    for index in range(n_hdc, rows * cols):
        row, col = divmod(index, cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, dpi=pdf_dpi)
    plt.close(fig)


def get_scores_and_plot(
    scorer,
    data_abs_xy,
    activations,
    directory: str,
    filename: str,
    cm: str = "jet",
    sort_by_score_60: bool = True,
    num_workers: int = 0,
    chunk_size: int = 32,
    units_per_page: int = 128,
):
    """Compute grid scores from activations and save rate-map plots to PDF."""
    if isinstance(data_abs_xy, torch.Tensor):
        data_abs_xy = data_abs_xy.detach().cpu().numpy()
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()

    n_units = activations.shape[-1]
    sums, counts = scorer.allocate_ratemap_accumulators(n_units)
    scorer.accumulate_ratemaps(data_abs_xy, activations, sums, counts)
    ratemaps = scorer.finalize_ratemaps(sums, counts)

    return get_scores_and_plot_from_ratemaps(
        scorer,
        ratemaps,
        directory,
        filename,
        cm=cm,
        sort_by_score_60=sort_by_score_60,
        num_workers=num_workers,
        chunk_size=chunk_size,
        units_per_page=units_per_page,
    )
