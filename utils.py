"""Shared helpers for encoding, decoding, plotting, and media export.

Most of the training and evaluation pipeline passes through this module: it
builds ensembles, prepares model inputs and targets, computes decoded position
metrics, and writes PDFs or MP4 summaries from evaluation outputs.

Usage:
    pc_ens = get_place_cell_ensembles(cfg)
    hdc_ens = get_head_direction_ensembles(cfg)
    init_cond = encode_initial_conditions(batch, pc_ens, hdc_ens)
    pos_mse = compute_position_mse(pc_logits, target_pos, pc_ens)
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn.functional as F

from animation import generate_eval_animation as _generate_eval_animation_impl
from animation import generate_trajectory_animation as _generate_trajectory_animation_impl
from encoding import EnsembleEncoder
from encoding import decode_position_from_pc_activations as _decode_pc_acts_impl
from ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble


# ---------------------------------------------------------------------------
# Ensemble factory helpers
# ---------------------------------------------------------------------------

def get_place_cell_ensembles(cfg) -> List[PlaceCellEnsemble]:
    """Create a list of PlaceCellEnsemble objects from a config object.

    Args:
        cfg: config object with attribute cfg.task containing:
             env_size, neurons_seed, targets_type, lstm_init_type,
             n_pc (list), pc_scale (list).

    Returns:
        List of PlaceCellEnsemble instances.
    """
    t = cfg.task
    return [
        PlaceCellEnsemble(
            n,
            stdev=s,
            pos_min=-t.env_size / 2,
            pos_max=t.env_size / 2,
            seed=t.neurons_seed,
            soft_targets=t.targets_type,
            soft_init=t.lstm_init_type,
        )
        for n, s in zip(t.n_pc, t.pc_scale)
    ]


def get_head_direction_ensembles(cfg) -> List[HeadDirectionCellEnsemble]:
    """Create a list of HeadDirectionCellEnsemble objects from a config object.

    Args:
        cfg: config object with attribute cfg.task containing:
             neurons_seed, targets_type, lstm_init_type,
             n_hdc (list), hdc_concentration (list).

    Returns:
        List of HeadDirectionCellEnsemble instances.
    """
    t = cfg.task
    return [
        HeadDirectionCellEnsemble(
            n,
            concentration=con,
            seed=t.neurons_seed,
            soft_targets=t.targets_type,
            soft_init=t.lstm_init_type,
        )
        for n, con in zip(t.n_hdc, t.hdc_concentration)
    ]


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_initial_conditions(
    batch: dict,
    pc_ensembles: List[PlaceCellEnsemble],
    hdc_ensembles: List[HeadDirectionCellEnsemble],
) -> torch.Tensor:
    """Encode initial position and head direction into cell activations.

    Calls get_init() on each ensemble (which expects numpy input with a seq
    dimension of 1) and concatenates results into a single Tensor.

    Args:
        batch: dict from DataLoader with keys:
               'init_pos' — torch.Tensor of shape (B, 2)
               'init_hd'  — torch.Tensor of shape (B, 1)
        pc_ensembles:  list of PlaceCellEnsemble instances.
        hdc_ensembles: list of HeadDirectionCellEnsemble instances.

    Returns:
        torch.Tensor of shape (B, sum_n_cells) — concatenated activations
        for all place-cell and head-direction-cell ensembles.
    """
    encoder = EnsembleEncoder(pc_ensembles, hdc_ensembles)
    return torch.from_numpy(
        encoder.encode_initial_conditions(batch["init_pos"], batch["init_hd"])
    )


def encode_targets(
    batch: dict,
    pc_ensembles: List[PlaceCellEnsemble],
    hdc_ensembles: List[HeadDirectionCellEnsemble],
) -> Tuple[list, list]:
    """Encode target trajectories into place-cell and head-direction-cell activations.

    Args:
        batch: dict from DataLoader with keys:
               'target_pos' — torch.Tensor of shape (B, T, 2)
               'target_hd'  — torch.Tensor of shape (B, T, 1)
        pc_ensembles:  list of PlaceCellEnsemble instances.
        hdc_ensembles: list of HeadDirectionCellEnsemble instances.

    Returns:
        Tuple (pc_targets, hdc_targets) where:
          pc_targets  — list of numpy arrays, each (B, T, n_pc_i)
          hdc_targets — list of numpy arrays, each (B, T, n_hdc_i)
    """
    encoder = EnsembleEncoder(pc_ensembles, hdc_ensembles)
    encoded_targets = encoder.encode_targets(batch["target_pos"], batch["target_hd"])
    return encoded_targets.pc_targets, encoded_targets.hdc_targets


def decode_position_from_pc_logits(
    pc_logits: List[torch.Tensor],
    pc_ensembles: List[PlaceCellEnsemble],
) -> torch.Tensor:
    """Decode positions from place-cell logits via a weighted mean of cell centres."""
    if not pc_logits or not pc_ensembles:
        raise ValueError("At least one place-cell ensemble is required to decode positions.")

    decoded_positions = []
    for logits, ensemble in zip(pc_logits, pc_ensembles):
        probs = F.softmax(logits, dim=-1)
        means = torch.as_tensor(
            ensemble.means,
            dtype=logits.dtype,
            device=logits.device,
        )
        decoded_positions.append(torch.matmul(probs, means))

    return torch.stack(decoded_positions, dim=0).mean(dim=0)


def compute_position_mse(
    pc_logits: List[torch.Tensor],
    target_pos: torch.Tensor,
    pc_ensembles: List[PlaceCellEnsemble],
) -> torch.Tensor:
    """Compute mean-squared error between decoded and ground-truth positions."""
    pred_pos = decode_position_from_pc_logits(pc_logits, pc_ensembles)
    return F.mse_loss(pred_pos, target_pos)


# ---------------------------------------------------------------------------
# Scoring and plotting
# ---------------------------------------------------------------------------

def _render_page_to_png_bytes(args):
    """Render one page of ratemaps + SACs to PNG bytes.

    Layout matches the original: all ratemaps fill the top half of the figure
    (subplot positions 1..rows*cols) and all SACs fill the bottom half
    (positions rows*cols+1..2*rows*cols), using figsize=(24, rows*4).

    Top-level so it can be pickled by ProcessPoolExecutor.
    """
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import numpy as _np

    (ratemaps_page, sac_page, score_60_page, mask_60_page,
     page_indices, rows, cols, cmap, dpi, nbins, plotting_sac_mask) = args

    n_page = len(page_indices)
    center = nbins - 1

    fig = _plt.figure(figsize=(24, rows * 4))

    for panel_idx in range(n_page):
        index = page_indices[panel_idx]
        title = "%d (%.2f)" % (int(index), score_60_page[panel_idx])

        # Ratemap — top half
        ax_rm = _plt.subplot(rows * 2, cols, panel_idx + 1)
        ax_rm.imshow(ratemaps_page[panel_idx], interpolation="none", cmap=cmap)
        ax_rm.axis("off")
        ax_rm.set_title(title)

        # SAC — bottom half
        ax_sac = _plt.subplot(rows * 2, cols, rows * cols + panel_idx + 1)
        useful_sac = sac_page[panel_idx] * plotting_sac_mask
        ax_sac.imshow(useful_sac, interpolation="none", cmap=cmap)
        mask_min, mask_max = mask_60_page[panel_idx]
        ax_sac.add_artist(_plt.Circle(
            (center, center), mask_min * nbins, fill=False, edgecolor="k",
        ))
        ax_sac.add_artist(_plt.Circle(
            (center, center), mask_max * nbins, fill=False, edgecolor="k",
        ))
        ax_sac.axis("off")
        ax_sac.set_title(title)

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    _plt.close(fig)
    buf.seek(0)
    return buf.read()


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
        (scorer, ratemaps[start:start + chunk_size])
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
    """Compute grid scores from precomputed rate maps and save a paginated PDF.

    Each page is rendered to PNG bytes (fast Agg rasteriser) and the pages are
    assembled into a single PDF.  When num_workers > 1 the pages are rendered in
    parallel, which is useful when there are many pages.
    """
    ratemaps = np.asarray(ratemaps)
    n_units = ratemaps.shape[0]
    score_60, score_90, max_60_mask, max_90_mask, sac = score_ratemaps(
        scorer,
        ratemaps,
        num_workers=num_workers,
        chunk_size=chunk_size,
    )

    if sort_by_score_60:
        ordering = np.argsort(-np.array(score_60))
    else:
        ordering = np.arange(n_units)

    units_per_page = max(1, units_per_page)
    cols = min(16, units_per_page)

    # Build per-page argument tuples for the worker
    page_args = []
    for page_start in range(0, n_units, units_per_page):
        page_indices = ordering[page_start : page_start + units_per_page]
        n_page = len(page_indices)
        rows = int(np.ceil(n_page / cols))
        page_args.append((
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
        ))

    # Render pages — parallel when multiple workers are requested and there are
    # multiple pages; otherwise render serially to avoid fork overhead.
    n_pages = len(page_args)
    if num_workers > 1 and n_pages > 1:
        with ProcessPoolExecutor(max_workers=min(num_workers, n_pages)) as executor:
            png_bytes_list = list(executor.map(_render_page_to_png_bytes, page_args))
    else:
        png_bytes_list = [_render_page_to_png_bytes(a) for a in page_args]

    # Assemble PNG pages into a single PDF
    os.makedirs(directory, exist_ok=True)
    with PdfPages(os.path.join(directory, filename)) as pdf:
        for png_bytes in png_bytes_list:
            img = plt.imread(io.BytesIO(png_bytes))
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(w / pdf_dpi, h / pdf_dpi), dpi=pdf_dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, dpi=pdf_dpi)
            plt.close(fig)

    return (
        np.asarray(score_60),
        np.asarray(score_90),
        np.asarray([np.mean(m) for m in max_60_mask]),
        np.asarray([np.mean(m) for m in max_90_mask]),
    )

# ---------------------------------------------------------------------------
# HDC directional tuning curves
# ---------------------------------------------------------------------------


def plot_hdc_tuning_curves(
    hd_angles: np.ndarray,
    hdc_acts: np.ndarray,
    n_bins: int = 20,
    save_path: str = "hdc_tuning.pdf",
    pdf_dpi: int = 100,
) -> None:
    """Plot head-direction-cell directional tuning curves and save to PDF.

    Each HDC unit gets one polar bar chart showing its mean softmax probability
    as a function of the animal's ground-truth head direction.

    Args:
        hd_angles : (N,) flat array of head directions in radians.
        hdc_acts  : (N, n_hdc) flat array of HDC softmax probabilities.
        n_bins    : number of directional bins (default 20 -> 18 degrees each).
        save_path : output PDF path.
        pdf_dpi   : rasterisation DPI.
    """
    hd_angles = np.asarray(hd_angles, dtype=np.float64).ravel()
    hdc_acts  = np.asarray(hdc_acts,  dtype=np.float64)
    if hdc_acts.ndim == 1:
        hdc_acts = hdc_acts[:, np.newaxis]
    _, n_hdc = hdc_acts.shape

    # Normalise angles to [-pi, pi]
    hd_angles = (hd_angles + np.pi) % (2 * np.pi) - np.pi

    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bar_width = 2 * np.pi / n_bins

    # Per-bin mean activation for each unit
    bin_idx = np.digitize(hd_angles, edges[1:-1])   # 0 ... n_bins-1
    tuning = np.zeros((n_hdc, n_bins), dtype=np.float64)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            tuning[:, b] = hdc_acts[mask].mean(axis=0)

    cols = min(12, n_hdc)
    rows = max(1, int(np.ceil(n_hdc / cols)))
    fig, axes = plt.subplots(
        rows, cols,
        subplot_kw={"projection": "polar"},
        figsize=(cols * 2.2, rows * 2.6 + 0.6),
        squeeze=False,
    )
    fig.suptitle("HDC directional tuning curves", fontsize=11)

    vmax = float(tuning.max()) if tuning.max() > 0 else 1.0
    for i in range(n_hdc):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        colors = plt.cm.plasma(tuning[i] / vmax)
        ax.bar(centers, tuning[i], width=bar_width,
               color=colors, align="center", alpha=0.85)
        ax.set_ylim(0, vmax * 1.1)
        ax.set_title(f"HDC {i}", fontsize=7, pad=3)
        ax.tick_params(labelsize=5)
        ax.set_yticklabels([])

    for i in range(n_hdc, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, dpi=pdf_dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Eval animation (3-panel MP4: trajectory / PC activation / HDC activation)
# ---------------------------------------------------------------------------


def decode_position_from_pc_activations(
    pc_acts: np.ndarray,
    pc_centers: np.ndarray,
) -> np.ndarray:
    """Decode positions from place-cell activation probabilities via weighted means."""
    return _decode_pc_acts_impl(pc_acts, pc_centers)


def prepare_dataset_animation_inputs(
    dataset,
    pc_ensembles: List[PlaceCellEnsemble],
    hdc_ensembles: List[HeadDirectionCellEnsemble],
    max_trajectories: int = 4,
) -> dict:
    """Build eval-style animation inputs directly from a generated dataset."""
    encoder = EnsembleEncoder(pc_ensembles, hdc_ensembles)
    return encoder.prepare_animation_inputs(
        dataset,
        max_trajectories=max_trajectories,
    ).as_dict()


def generate_trajectory_animation(
    target_pos: np.ndarray,
    pred_pos: np.ndarray,
    pc_acts: np.ndarray,
    hdc_acts: np.ndarray,
    pc_centers: np.ndarray,
    hdc_centers: np.ndarray,
    env_size: float,
    save_path: str,
    fps: int = 20,
    step: int = 4,
    num_workers: int = 4,
    title_prefix: str = "Trajectory",
    pred_label: str = "predicted",
) -> None:
    """Generate a 3-panel trajectory animation MP4 or GIF."""
    _generate_trajectory_animation_impl(
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        env_size,
        save_path,
        fps=fps,
        step=step,
        num_workers=num_workers,
        title_prefix=title_prefix,
        pred_label=pred_label,
    )


def generate_eval_animation(
    target_pos: np.ndarray,
    pred_pos: np.ndarray,
    pc_acts: np.ndarray,
    hdc_acts: np.ndarray,
    pc_centers: np.ndarray,
    hdc_centers: np.ndarray,
    env_size: float,
    save_path: str,
    fps: int = 20,
    step: int = 4,
    num_workers: int = 4,
) -> None:
    """Backward-compatible wrapper for eval animation exports."""
    _generate_eval_animation_impl(
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        env_size,
        save_path,
        fps=fps,
        step=step,
        num_workers=num_workers,
    )


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
    """Compute grid scores and save rate-map / SAC plots to a PDF.

    Args:
        scorer:       GridScorer instance.
        data_abs_xy:  (N, T, 2) Tensor or numpy array — absolute positions.
        activations:  (N, T, n_units) Tensor or numpy array — unit activations.
        directory:    output directory (created if absent).
        filename:     PDF filename (relative to directory).
        cm:           matplotlib colormap name.
        sort_by_score_60: if True, sort panels by 60-degree grid score descending.

    Returns:
        Tuple of four numpy arrays:
          (score_60, score_90, separability_60, separability_90)
    """
    # Convert Tensors to numpy if necessary
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
