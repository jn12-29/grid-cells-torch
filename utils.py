"""
Utility functions for the grid cells PyTorch reimplementation.

PyTorch port of the original TensorFlow utility functions from:
  Banino et al., "Vector-based navigation using grid-like representations
  in artificial agents", Nature 2018.
"""

from concurrent.futures import ProcessPoolExecutor
import io
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn.functional as F

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
    # Convert tensors to numpy and add the required seq dimension (dim=1)
    init_pos_np = batch["init_pos"].detach().cpu().to(torch.float32).numpy()[:, np.newaxis, :]  # (B, 1, 2)
    init_hd_np  = batch["init_hd"].detach().cpu().to(torch.float32).numpy()[:, np.newaxis, :]  # (B, 1, 1)

    parts = []
    for ens in pc_ensembles:
        # get_init returns (B, 1, n_cells); squeeze the seq dim
        act = ens.get_init(init_pos_np)  # (B, 1, n_cells)
        parts.append(act[:, 0, :])       # (B, n_cells)

    for ens in hdc_ensembles:
        act = ens.get_init(init_hd_np)   # (B, 1, n_cells)
        parts.append(act[:, 0, :])       # (B, n_cells)

    # Concatenate along cell dimension and convert to Tensor
    concat_np = np.concatenate(parts, axis=-1).astype(np.float32)  # (B, sum_n_cells)
    return torch.from_numpy(concat_np)


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
    target_pos_np = batch["target_pos"].detach().cpu().numpy()  # (B, T, 2)
    target_hd_np  = batch["target_hd"].detach().cpu().numpy()   # (B, T, 1)

    pc_targets = [ens.get_targets(target_pos_np) for ens in pc_ensembles]
    hdc_targets = [ens.get_targets(target_hd_np) for ens in hdc_ensembles]

    return pc_targets, hdc_targets


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
