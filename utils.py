"""
Utility functions for the grid cells PyTorch reimplementation.

PyTorch port of the original TensorFlow utility functions from:
  Banino et al., "Vector-based navigation using grid-like representations
  in artificial agents", Nature 2018.
"""

import os
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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


# ---------------------------------------------------------------------------
# Scoring and plotting
# ---------------------------------------------------------------------------

def get_scores_and_plot(
    scorer,
    data_abs_xy,
    activations,
    directory: str,
    filename: str,
    cm: str = "jet",
    sort_by_score_60: bool = True,
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

    # Flatten the leading (N, T) dimensions
    xy  = data_abs_xy.reshape(-1, data_abs_xy.shape[-1])   # (N*T, 2)
    act = activations.reshape(-1, activations.shape[-1])    # (N*T, n_units)
    n_units = act.shape[1]

    # Compute rate maps
    ratemaps = [
        scorer.calculate_ratemap(xy[:, 0], xy[:, 1], act[:, i])
        for i in range(n_units)
    ]

    # Compute scores
    score_60, score_90, max_60_mask, max_90_mask, sac = zip(
        *[scorer.get_scores(rm) for rm in ratemaps]
    )

    # Determine panel ordering
    if sort_by_score_60:
        ordering = np.argsort(-np.array(score_60))
    else:
        ordering = list(range(n_units))

    # Build figure: top row = rate maps, bottom row = SACs
    cols = 16
    rows = int(np.ceil(n_units / cols))
    fig = plt.figure(figsize=(24, rows * 4))

    for i in range(n_units):
        rf  = plt.subplot(rows * 2, cols, i + 1)
        acr = plt.subplot(rows * 2, cols, rows * cols + i + 1)
        index = ordering[i]
        title = "%d (%.2f)" % (index, score_60[index])
        scorer.plot_ratemap(ratemaps[index], ax=rf,  title=title, cmap=cm)
        scorer.plot_sac(sac[index], mask_params=max_60_mask[index], ax=acr, title=title, cmap=cm)

    # Save to PDF
    os.makedirs(directory, exist_ok=True)
    with PdfPages(os.path.join(directory, filename)) as pdf:
        plt.savefig(pdf, format="pdf")
    plt.close(fig)

    return (
        np.asarray(score_60),
        np.asarray(score_90),
        np.asarray([np.mean(m) for m in max_60_mask]),
        np.asarray([np.mean(m) for m in max_90_mask]),
    )
