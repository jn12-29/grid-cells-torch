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
import shutil
import subprocess
import tempfile
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

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
    pc_acts = np.asarray(pc_acts, dtype=np.float32)
    pc_centers = np.asarray(pc_centers, dtype=np.float32)
    if pc_acts.ndim < 2:
        raise ValueError("pc_acts must include a cell dimension.")
    if pc_centers.ndim != 2 or pc_centers.shape[1] != 2:
        raise ValueError("pc_centers must have shape (n_pc, 2).")
    if pc_acts.shape[-1] != pc_centers.shape[0]:
        raise ValueError(
            "pc_acts and pc_centers must agree on the number of place cells."
        )
    return np.tensordot(pc_acts, pc_centers, axes=([-1], [0])).astype(np.float32)


def prepare_dataset_animation_inputs(
    dataset,
    pc_ensembles: List[PlaceCellEnsemble],
    hdc_ensembles: List[HeadDirectionCellEnsemble],
    max_trajectories: int = 4,
) -> dict:
    """Build eval-style animation inputs directly from a generated dataset."""
    if not pc_ensembles:
        raise ValueError("At least one place-cell ensemble is required for animation.")
    if not hdc_ensembles:
        raise ValueError("At least one head-direction ensemble is required for animation.")

    target_pos_all = dataset._data["target_pos"]
    target_hd_all = dataset._data["target_hd"]
    num_show = min(int(max_trajectories), len(target_pos_all))
    if num_show <= 0:
        raise ValueError("Cannot animate an empty dataset.")

    target_pos = target_pos_all[:num_show]
    target_hd = target_hd_all[:num_show]
    pc_acts = np.concatenate([ens.get_targets(target_pos) for ens in pc_ensembles], axis=-1)
    hdc_acts = np.concatenate([ens.get_targets(target_hd) for ens in hdc_ensembles], axis=-1)
    pc_centers = np.concatenate([ens.means for ens in pc_ensembles], axis=0).astype(np.float32)
    hdc_centers = np.concatenate([ens.means.reshape(-1) for ens in hdc_ensembles], axis=0).astype(
        np.float32
    )
    pred_pos = decode_position_from_pc_activations(pc_acts, pc_centers)

    return {
        "target_pos": target_pos,
        "pred_pos": pred_pos,
        "pc_acts": pc_acts,
        "hdc_acts": hdc_acts,
        "pc_centers": pc_centers,
        "hdc_centers": hdc_centers,
    }


def _sort_hdc_for_animation(hdc_acts: np.ndarray, hdc_centers: np.ndarray):
    """Sort HDC centers around the circle so polar bars render in angular order."""
    hdc_centers = np.asarray(hdc_centers, dtype=np.float32).reshape(-1)
    if hdc_acts.shape[-1] != hdc_centers.shape[0]:
        raise ValueError(
            "hdc_acts and hdc_centers must agree on the number of head-direction cells."
        )
    order = np.argsort(np.mod(hdc_centers, 2 * np.pi))
    return hdc_acts[..., order], hdc_centers[order]


def _make_animation_title(title_prefix: str, traj_idx: int, step_idx: int, total_steps: int) -> str:
    """Build a consistent animation title across writer and chunked render paths."""
    return f"{title_prefix} #{traj_idx}  -  step {step_idx} / {total_steps}"


def _build_frame_chunks(frame_indices, num_workers: int):
    """Split sequential frame indices into evenly sized worker chunks."""
    chunk_size = max(1, (len(frame_indices) + num_workers - 1) // max(num_workers, 1))
    return [
        [(j, frame_indices[j]) for j in range(start, min(start + chunk_size, len(frame_indices)))]
        for start in range(0, len(frame_indices), chunk_size)
    ]


def _save_animation_frame(fig, path: str) -> None:
    """Persist one rendered animation frame, preferring a fast canvas copy path."""
    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None:
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        Image.fromarray(rgba[:, :, :3]).save(path, optimize=False, compress_level=1)
        return

    fig.savefig(path, dpi=110)


def _encode_animation_frames(frames_dir: str, save_path: str, fps: int, n_frames: int) -> None:
    """Encode sequentially-numbered PNG frames into an MP4 via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        save_path,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    bar = (
        _tqdm(total=n_frames, desc=f"encode:{os.path.basename(save_path)}", unit="frame")
        if _tqdm is not None
        else None
    )
    last = 0
    try:
        if proc.stdout:
            for line in proc.stdout:
                if not line.strip().startswith("frame="):
                    continue
                try:
                    cur = min(n_frames, int(line.strip().split("=", 1)[1]))
                except ValueError:
                    continue
                if bar is not None and cur > last:
                    bar.update(cur - last)
                last = cur
        stderr = proc.stderr.read() if proc.stderr else ""
        rc = proc.wait()
        if bar is not None and last < n_frames:
            bar.update(n_frames - last)
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed:\n{stderr}")
    finally:
        if bar is not None:
            bar.close()


def _build_animation_artists(
    target_pos: np.ndarray,
    pred_pos: np.ndarray,
    pc_acts: np.ndarray,
    hdc_acts: np.ndarray,
    pc_centers: np.ndarray,
    hdc_centers: np.ndarray,
    env_size: float,
    pc_vmax: float,
    title_prefix: str,
    pred_label: str,
):
    """Create the shared 3-panel animation figure and artists."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    hdc_acts, hdc_centers = _sort_hdc_for_animation(hdc_acts, hdc_centers)
    total_steps = target_pos.shape[0]
    n_hdc = hdc_acts.shape[1]
    half = env_size / 2.0
    safe_vmax = max(float(pc_vmax), 1e-8)
    hdc_width = (2 * np.pi / max(n_hdc, 1)) * 0.9

    fig = plt.figure(figsize=(13, 4.5), facecolor="#1a1a2e")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.1, wspace=0.35)

    ax_traj = fig.add_subplot(1, 3, 1)
    ax_pc = fig.add_subplot(1, 3, 2)
    ax_hdc = fig.add_subplot(1, 3, 3, projection="polar")

    for ax in (ax_traj, ax_pc):
        ax.set_facecolor("#0d0d1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
    ax_hdc.set_facecolor("#0d0d1a")
    ax_hdc.spines["polar"].set_edgecolor("#444466")

    title_txt = fig.suptitle(
        _make_animation_title(title_prefix, 0, 0, total_steps),
        color="#ccccee",
        fontsize=11,
    )

    ax_traj.add_patch(
        mpatches.FancyBboxPatch(
            (-half, -half),
            env_size,
            env_size,
            boxstyle="square,pad=0",
            linewidth=1.2,
            edgecolor="#5566aa",
            facecolor="none",
        )
    )
    ax_traj.set_xlim(-half * 1.08, half * 1.08)
    ax_traj.set_ylim(-half * 1.08, half * 1.08)
    ax_traj.set_aspect("equal")
    ax_traj.set_title("Trajectory", color="#aaaacc", fontsize=9)
    ax_traj.tick_params(colors="#666688", labelsize=7)

    (line_actual,) = ax_traj.plot([], [], lw=1.0, color="#4488ff", alpha=0.7, label="actual")
    (dot_actual,) = ax_traj.plot([], [], "o", ms=5, color="#ffdd44", zorder=5)
    (line_pred,) = ax_traj.plot(
        [], [], lw=1.0, color="#ff6644", alpha=0.7, linestyle="--", label=pred_label
    )
    (dot_pred,) = ax_traj.plot([], [], "s", ms=4, color="#ffaa44", zorder=5)
    ax_traj.legend(
        fontsize=6,
        loc="upper right",
        labelcolor="#aaaacc",
        facecolor="#1a1a2e",
        edgecolor="#444466",
    )

    ax_pc.set_xlim(-half, half)
    ax_pc.set_ylim(-half, half)
    ax_pc.set_aspect("equal")
    ax_pc.set_title("Place cells", color="#aaaacc", fontsize=9)
    ax_pc.tick_params(colors="#666688", labelsize=7)
    sc_pc = ax_pc.scatter(
        pc_centers[:, 0],
        pc_centers[:, 1],
        c=pc_acts[0],
        cmap="hot",
        s=18,
        vmin=0.0,
        vmax=safe_vmax,
    )
    fig.colorbar(sc_pc, ax=ax_pc, fraction=0.046, pad=0.04).ax.tick_params(
        colors="#aaaacc", labelsize=6
    )

    bars_hdc = ax_hdc.bar(
        hdc_centers,
        hdc_acts[0],
        width=hdc_width,
        color="#4fc3f7",
        alpha=0.85,
        align="center",
    )
    ax_hdc.set_ylim(0, max(float(hdc_acts.max()), 1.0 / max(n_hdc, 1)))
    ax_hdc.set_title("Head direction cells", color="#aaaacc", fontsize=9, pad=8)
    ax_hdc.tick_params(colors="#666688", labelsize=7)
    ax_hdc.yaxis.label.set_color("#666688")

    artists = {
        "title_txt": title_txt,
        "line_actual": line_actual,
        "dot_actual": dot_actual,
        "line_pred": line_pred,
        "dot_pred": dot_pred,
        "sc_pc": sc_pc,
        "bars_hdc": bars_hdc,
    }
    arrays = {
        "target_pos": target_pos,
        "pred_pos": pred_pos,
        "pc_acts": pc_acts,
        "hdc_acts": hdc_acts,
        "total_steps": total_steps,
        "title_prefix": title_prefix,
    }
    return fig, artists, arrays


def _update_animation_frame(artists: dict, arrays: dict, traj_idx: int, t: int) -> None:
    """Update the shared 3-panel animation figure to one timestep."""
    target_pos = arrays["target_pos"]
    pred_pos = arrays["pred_pos"]
    pc_acts = arrays["pc_acts"]
    hdc_acts = arrays["hdc_acts"]
    total_steps = arrays["total_steps"]
    title_prefix = arrays["title_prefix"]

    artists["title_txt"].set_text(_make_animation_title(title_prefix, traj_idx, t, total_steps))
    artists["line_actual"].set_data(target_pos[: t + 1, 0], target_pos[: t + 1, 1])
    artists["dot_actual"].set_data([target_pos[t, 0]], [target_pos[t, 1]])
    artists["line_pred"].set_data(pred_pos[: t + 1, 0], pred_pos[: t + 1, 1])
    artists["dot_pred"].set_data([pred_pos[t, 0]], [pred_pos[t, 1]])
    artists["sc_pc"].set_array(pc_acts[t])
    for bar, h in zip(artists["bars_hdc"], hdc_acts[t]):
        bar.set_height(h)


def _render_trajectory_animation(args) -> str:
    """Render a single 3-panel trajectory animation using a direct writer path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    try:
        from tqdm.auto import tqdm as _local_tqdm
    except ImportError:
        _local_tqdm = None

    (
        traj_idx,
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        env_size,
        pc_vmax,
        fps,
        step,
        out_path,
        title_prefix,
        pred_label,
    ) = args

    frame_indices = list(range(0, target_pos.shape[0], max(1, step)))
    fig, artists, arrays = _build_animation_artists(
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        env_size,
        pc_vmax,
        title_prefix,
        pred_label,
    )

    def _make_writer(path):
        try:
            writer = FFMpegWriter(
                fps=fps,
                metadata={"title": f"{title_prefix} #{traj_idx}"},
            )
            return writer, path
        except Exception:
            gif_path = path.replace(".mp4", ".gif")
            return PillowWriter(fps=fps), gif_path

    writer, out_path = _make_writer(out_path)
    pbar = (
        _local_tqdm(frame_indices, desc=f"render:{os.path.basename(out_path)}", unit="frame")
        if _local_tqdm is not None
        else frame_indices
    )

    with writer.saving(fig, out_path, dpi=110):
        for t in pbar:
            _update_animation_frame(artists, arrays, traj_idx, t)
            writer.grab_frame()

    plt.close(fig)
    return out_path


def _render_trajectory_animation_chunk(task) -> int:
    """Render one worker chunk of a 3-panel trajectory animation to PNG frames."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (
        traj_idx,
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        env_size,
        pc_vmax,
        frames_subset,
        frames_dir,
        title_prefix,
        pred_label,
    ) = task

    fig, artists, arrays = _build_animation_artists(
        target_pos,
        pred_pos,
        pc_acts,
        hdc_acts,
        pc_centers,
        hdc_centers,
        env_size,
        pc_vmax,
        title_prefix,
        pred_label,
    )

    try:
        for out_idx, t in frames_subset:
            _update_animation_frame(artists, arrays, traj_idx, t)
            _save_animation_frame(fig, os.path.join(frames_dir, f"frame_{out_idx:06d}.png"))
    finally:
        plt.close(fig)

    return len(frames_subset)


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
    """Generate a 3-panel trajectory animation MP4 or GIF.

    When ``num_workers == 1``: uses matplotlib ``FFMpegWriter`` — no temporary
    files, with a GIF fallback when ffmpeg is absent.

    When ``num_workers > 1``: renders frames in parallel chunks (PNG → ffmpeg),
    giving a ~N× speed-up.  One file per trajectory; ``_traj0000`` suffix added
    when N > 1.

    Panels:
      1. Ground-truth trajectory (solid) vs decoded/reference trajectory (dashed)
      2. Place-cell softmax probability — spatial scatter with hot colormap
      3. Head-direction-cell softmax probability — polar bar chart

    Args:
        target_pos:  (N, T, 2)    ground-truth positions.
        pred_pos:    (N, T, 2)    decoded or predicted positions.
        pc_acts:     (N, T, n_pc) place-cell softmax probabilities.
        hdc_acts:    (N, T, n_hdc) head-direction-cell softmax probabilities.
        pc_centers:  (n_pc, 2)    2-D centre positions of place cells.
        hdc_centers: (n_hdc,)     preferred HDC directions in radians.
        env_size:    float  side length of the square environment (metres).
        save_path:   destination MP4 path.
        fps:         playback frame rate.
        step:        render every *step*-th timestep (default 4 → 4× speed).
        num_workers: parallel chunk workers (> 1 enables parallel PNG mode).
        title_prefix: per-trajectory title prefix.
        pred_label: label for the second trajectory line.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    N = target_pos.shape[0]
    T = target_pos.shape[1]
    if N == 0 or T == 0:
        raise ValueError("Cannot animate: empty trajectory data.")

    pc_vmax = float(pc_acts.max())
    num_workers = max(1, int(num_workers))
    fps = max(1, int(fps))
    step = max(1, int(step))

    if num_workers > 1 and shutil.which("ffmpeg") is None:
        num_workers = 1

    def _traj_path(i: int) -> str:
        if N == 1:
            return save_path
        base, ext = os.path.splitext(save_path)
        return f"{base}_traj{i:04d}{ext}"

    if num_workers <= 1:
        args_list = [
            (
                i,
                target_pos[i],
                pred_pos[i],
                pc_acts[i],
                hdc_acts[i],
                pc_centers,
                hdc_centers,
                env_size,
                pc_vmax,
                fps,
                step,
                _traj_path(i),
                title_prefix,
                pred_label,
            )
            for i in range(N)
        ]
        for a in args_list:
            saved = _render_trajectory_animation(a)
            print(f"Animation saved to {saved}")
        return

    all_frame_indices = list(range(0, T, step))
    n_frames = len(all_frame_indices)
    frame_chunks = _build_frame_chunks(all_frame_indices, num_workers=num_workers)
    actual_workers = min(num_workers, len(frame_chunks))

    for traj_i in range(N):
        out_path = _traj_path(traj_i)

        tasks = [
            (
                traj_i,
                target_pos[traj_i],
                pred_pos[traj_i],
                pc_acts[traj_i],
                hdc_acts[traj_i],
                pc_centers,
                hdc_centers,
                env_size,
                pc_vmax,
                chunk,
                "",
                title_prefix,
                pred_label,
            )
            for chunk in frame_chunks
        ]

        render_bar = (
            _tqdm(total=n_frames, desc=f"render:{os.path.basename(out_path)}", unit="frame")
            if _tqdm is not None
            else None
        )

        with tempfile.TemporaryDirectory(prefix="eval_anim_") as frames_dir:
            tasks = [t[:-3] + (frames_dir,) + t[-2:] for t in tasks]

            try:
                with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                    futures = [executor.submit(_render_trajectory_animation_chunk, t) for t in tasks]
                    for fut in as_completed(futures):
                        rendered = fut.result()
                        if render_bar is not None:
                            render_bar.update(rendered)
            finally:
                if render_bar is not None:
                    render_bar.close()

            _encode_animation_frames(frames_dir, out_path, fps=fps, n_frames=n_frames)

        print(f"Animation saved to {out_path}")


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
    generate_trajectory_animation(
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
        title_prefix="Eval traj",
        pred_label="predicted",
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
