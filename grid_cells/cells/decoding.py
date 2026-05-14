"""Decode place-cell and head-direction-cell population codes.

This module owns the inverse mapping from ensemble logits or raw scores back to
continuous positions and headings.  PC logit decoding estimates the unknown
positive logit scale, so ``alpha * raw_score + beta`` codes decode to the same
position for any positive alpha.

Usage:
    pos = decode_position_from_pc_logits(pc_logits, pc_ensembles)
    hd = decode_head_direction_from_hdc_logits(hdc_logits, hdc_ensembles)
"""

from typing import List, Sequence, cast

import numpy as np
import torch
import torch.nn.functional as F

from grid_cells.cells.ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble


VALID_DECODE_TYPES = {"analytic", "weighted_mean"}
PC_ANALYTIC_MAX_RELATIVE_RESIDUAL = 0.05


def _as_list(values):
    """Return values as a list without treating arrays/tensors as sequences."""
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def _decode_type(ensemble) -> str:
    """Resolve the ensemble decode type with backward-compatible defaults."""
    decode_type = getattr(ensemble, "decode_type", "analytic")
    if decode_type not in VALID_DECODE_TYPES:
        supported = ", ".join(sorted(VALID_DECODE_TYPES))
        raise ValueError(
            f"Unsupported ensemble decode_type={decode_type!r}; expected one of: {supported}."
        )
    return decode_type


def is_supported_decode_type(decode_type: str) -> bool:
    """Return whether decode_type is one of the supported decoder modes."""
    return decode_type in VALID_DECODE_TYPES


def _pc_variance_vector_np(ensemble: PlaceCellEnsemble) -> np.ndarray | None:
    variances = np.asarray(ensemble.variances, dtype=np.float64)
    if variances.ndim != 2 or variances.shape[1] != 2:
        return None
    if not np.allclose(variances, variances[0], rtol=1e-6, atol=1e-12):
        return None
    return variances[0]


def pc_supports_analytic_decode(ensemble: PlaceCellEnsemble) -> bool:
    """Return whether raw place-cell log-scores can be analytically decoded."""
    if _decode_type(ensemble) != "analytic":
        return False
    if getattr(ensemble, "soft_targets", None) != "softmax":
        return False
    variance = _pc_variance_vector_np(ensemble)
    if variance is None or np.any(variance <= 0.0):
        return False
    means = np.asarray(ensemble.means, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 2 or means.shape[0] < 3:
        return False
    a = means / variance[np.newaxis, :]
    a = a - a.mean(axis=0, keepdims=True)
    return np.linalg.matrix_rank(a) >= 2


def pc_supports_scale_invariant_decode(ensemble: PlaceCellEnsemble) -> bool:
    """Return whether PC logits support scale-invariant analytic decoding."""
    if not pc_supports_analytic_decode(ensemble):
        return False
    a, b = _pc_linear_terms_np(ensemble)
    design = np.concatenate([a, b[:, np.newaxis]], axis=-1)
    return np.linalg.matrix_rank(design) >= 3


def hdc_supports_analytic_decode(ensemble: HeadDirectionCellEnsemble) -> bool:
    """Return whether a head-direction ensemble can be analytically decoded."""
    if _decode_type(ensemble) != "analytic":
        return False
    if getattr(ensemble, "soft_targets", None) != "softmax":
        return False
    means = np.asarray(ensemble.means, dtype=np.float64).reshape(-1)
    kappa = np.asarray(ensemble.kappa, dtype=np.float64).reshape(-1)
    if means.shape != kappa.shape or means.shape[0] < 3 or np.any(kappa <= 0.0):
        return False
    a = np.stack([kappa * np.cos(means), kappa * np.sin(means)], axis=-1)
    a = a - a.mean(axis=0, keepdims=True)
    return np.linalg.matrix_rank(a) >= 2


def _pc_linear_terms_np(ensemble: PlaceCellEnsemble) -> tuple[np.ndarray, np.ndarray]:
    variance = _pc_variance_vector_np(ensemble)
    if variance is None or np.any(variance <= 0.0):
        raise ValueError("Analytic PC decoding requires shared positive variances.")
    means = np.asarray(ensemble.means, dtype=np.float64)
    a = means / variance[np.newaxis, :]
    b = -0.5 * np.sum((means ** 2) / variance[np.newaxis, :], axis=-1)
    return a - a.mean(axis=0, keepdims=True), b - b.mean(axis=0, keepdims=True)


def _pc_linear_terms_torch(
    ensemble: PlaceCellEnsemble,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    variance_np = _pc_variance_vector_np(ensemble)
    if variance_np is None or np.any(variance_np <= 0.0):
        raise ValueError("Analytic PC decoding requires shared positive variances.")
    means = torch.as_tensor(ensemble.means, dtype=dtype, device=device)
    variance = torch.as_tensor(variance_np, dtype=dtype, device=device)
    a = means / variance.unsqueeze(0)
    b = -0.5 * torch.sum((means ** 2) / variance.unsqueeze(0), dim=-1)
    return a - a.mean(dim=0, keepdim=True), b - b.mean(dim=0, keepdim=True)


def _decode_position_scores_one_np(
    scores: np.ndarray,
    ensemble: PlaceCellEnsemble,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[-1] != ensemble.n_cells:
        raise ValueError("PC scores and ensemble must agree on the number of cells.")
    a, b = _pc_linear_terms_np(ensemble)
    rhs = (scores - scores.mean(axis=-1, keepdims=True)) - b
    decoded = rhs @ np.linalg.pinv(a).T
    return decoded.astype(np.float32)


def _decode_position_logits_one_torch(
    logits: torch.Tensor,
    ensemble: PlaceCellEnsemble,
) -> torch.Tensor:
    if logits.shape[-1] != ensemble.n_cells:
        raise ValueError("PC logits and ensemble must agree on the number of cells.")
    a, b = _pc_linear_terms_torch(ensemble, logits.dtype, logits.device)
    design = torch.cat([a, b.unsqueeze(-1)], dim=-1)
    rhs = logits - logits.mean(dim=-1, keepdim=True)
    solution = rhs @ torch.linalg.pinv(design).T
    scaled_pos = solution[..., :2]
    scale = solution[..., 2:3]
    eps = torch.finfo(logits.dtype).eps
    denominator = torch.where(scale != 0.0, scale, torch.ones_like(scale))
    decoded = scaled_pos / denominator
    weighted = _decode_position_logits_weighted_one_torch(logits, ensemble)
    fitted = solution @ design.T
    residual = torch.linalg.vector_norm(rhs - fitted, dim=-1, keepdim=True)
    rhs_norm = torch.linalg.vector_norm(rhs, dim=-1, keepdim=True).clamp_min(eps)
    relative_residual = residual / rhs_norm
    valid = (
        torch.isfinite(decoded).all(dim=-1, keepdim=True)
        & (scale > 0.0)
        & (relative_residual <= PC_ANALYTIC_MAX_RELATIVE_RESIDUAL)
    )
    return torch.where(valid, decoded, weighted)


def _decode_position_logits_weighted_one_torch(
    logits: torch.Tensor,
    ensemble: PlaceCellEnsemble,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    means = torch.as_tensor(ensemble.means, dtype=logits.dtype, device=logits.device)
    return torch.matmul(probs, means)


def decode_position_from_pc_scores(
    pc_scores,
    pc_ensembles: PlaceCellEnsemble | Sequence[PlaceCellEnsemble],
) -> np.ndarray:
    """Decode positions from raw place-cell log-scores via analytic inversion."""
    scores_list = _as_list(pc_scores)
    ensembles = cast(List[PlaceCellEnsemble], _as_list(pc_ensembles))
    if not scores_list or not ensembles:
        raise ValueError("At least one place-cell ensemble is required.")
    if len(scores_list) != len(ensembles):
        raise ValueError("pc_scores and pc_ensembles must have the same length.")
    unsupported = [ensemble for ensemble in ensembles if not pc_supports_analytic_decode(ensemble)]
    if unsupported:
        raise ValueError(
            "decode_position_from_pc_scores requires softmax-compatible place-cell "
            "ensembles with full-rank analytic geometry."
        )

    decoded = [
        _decode_position_scores_one_np(scores, ensemble)
        for scores, ensemble in zip(scores_list, ensembles)
    ]
    return np.stack(decoded, axis=0).mean(axis=0).astype(np.float32)


def decode_position_from_pc_logits(
    pc_logits: List[torch.Tensor],
    pc_ensembles: List[PlaceCellEnsemble],
) -> torch.Tensor:
    """Decode positions from place-cell logits using the configured decoder mode."""
    if not pc_logits or not pc_ensembles:
        raise ValueError("At least one place-cell ensemble is required to decode positions.")
    if len(pc_logits) != len(pc_ensembles):
        raise ValueError("pc_logits and pc_ensembles must have the same length.")

    decoded_positions = []
    for logits, ensemble in zip(pc_logits, pc_ensembles):
        if pc_supports_scale_invariant_decode(ensemble):
            decoded_positions.append(_decode_position_logits_one_torch(logits, ensemble))
        else:
            decoded_positions.append(_decode_position_logits_weighted_one_torch(logits, ensemble))
    return torch.stack(decoded_positions, dim=0).mean(dim=0)


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
        raise ValueError("pc_acts and pc_centers must agree on the number of place cells.")
    return np.tensordot(pc_acts, pc_centers, axes=([-1], [0])).astype(np.float32)


def _hdc_linear_terms_np(
    ensemble: HeadDirectionCellEnsemble,
) -> np.ndarray:
    means = np.asarray(ensemble.means, dtype=np.float64).reshape(-1)
    kappa = np.asarray(ensemble.kappa, dtype=np.float64).reshape(-1)
    if means.shape != kappa.shape or np.any(kappa <= 0.0):
        raise ValueError("Analytic HDC decoding requires positive kappa per cell.")
    a = np.stack([kappa * np.cos(means), kappa * np.sin(means)], axis=-1)
    return a - a.mean(axis=0, keepdims=True)


def _hdc_linear_terms_torch(
    ensemble: HeadDirectionCellEnsemble,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    means = torch.as_tensor(ensemble.means, dtype=dtype, device=device).reshape(-1)
    kappa = torch.as_tensor(ensemble.kappa, dtype=dtype, device=device).reshape(-1)
    a = torch.stack([kappa * torch.cos(means), kappa * torch.sin(means)], dim=-1)
    return a - a.mean(dim=0, keepdim=True)


def _decode_head_direction_scores_one_np(
    scores: np.ndarray,
    ensemble: HeadDirectionCellEnsemble,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[-1] != ensemble.n_cells:
        raise ValueError("HDC scores and ensemble must agree on the number of cells.")
    a = _hdc_linear_terms_np(ensemble)
    rhs = scores - scores.mean(axis=-1, keepdims=True)
    unit = rhs @ np.linalg.pinv(a).T
    angles = np.arctan2(unit[..., 1], unit[..., 0])
    return angles[..., np.newaxis].astype(np.float32)


def _decode_head_direction_logits_one_torch(
    logits: torch.Tensor,
    ensemble: HeadDirectionCellEnsemble,
) -> torch.Tensor:
    if logits.shape[-1] != ensemble.n_cells:
        raise ValueError("HDC logits and ensemble must agree on the number of cells.")
    a = _hdc_linear_terms_torch(ensemble, logits.dtype, logits.device)
    rhs = logits - logits.mean(dim=-1, keepdim=True)
    unit = rhs @ torch.linalg.pinv(a).T
    return torch.atan2(unit[..., 1], unit[..., 0]).unsqueeze(-1)


def _decode_head_direction_logits_weighted_one_torch(
    logits: torch.Tensor,
    ensemble: HeadDirectionCellEnsemble,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    means = torch.as_tensor(ensemble.means, dtype=logits.dtype, device=logits.device)
    sin_mean = torch.matmul(probs, torch.sin(means))
    cos_mean = torch.matmul(probs, torch.cos(means))
    return torch.atan2(sin_mean, cos_mean).unsqueeze(-1)


def _circular_mean_np(angles: List[np.ndarray]) -> np.ndarray:
    stacked = np.stack(angles, axis=0)
    sin_mean = np.mean(np.sin(stacked), axis=0)
    cos_mean = np.mean(np.cos(stacked), axis=0)
    return np.arctan2(sin_mean, cos_mean).astype(np.float32)


def _circular_mean_torch(angles: List[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(angles, dim=0)
    sin_mean = torch.mean(torch.sin(stacked), dim=0)
    cos_mean = torch.mean(torch.cos(stacked), dim=0)
    return torch.atan2(sin_mean, cos_mean)


def decode_head_direction_from_hdc_scores(
    hdc_scores,
    hdc_ensembles: HeadDirectionCellEnsemble | Sequence[HeadDirectionCellEnsemble],
) -> np.ndarray:
    """Decode heading angles from raw HDC log-scores via analytic inversion."""
    scores_list = _as_list(hdc_scores)
    ensembles = cast(List[HeadDirectionCellEnsemble], _as_list(hdc_ensembles))
    if not scores_list or not ensembles:
        raise ValueError("At least one head-direction ensemble is required.")
    if len(scores_list) != len(ensembles):
        raise ValueError("hdc_scores and hdc_ensembles must have the same length.")
    unsupported = [ensemble for ensemble in ensembles if not hdc_supports_analytic_decode(ensemble)]
    if unsupported:
        raise ValueError(
            "decode_head_direction_from_hdc_scores requires softmax-compatible HDC "
            "ensembles with full-rank analytic geometry."
        )
    decoded = [
        _decode_head_direction_scores_one_np(scores, ensemble)
        for scores, ensemble in zip(scores_list, ensembles)
    ]
    return _circular_mean_np(decoded)


def decode_head_direction_from_hdc_logits(
    hdc_logits: List[torch.Tensor],
    hdc_ensembles: List[HeadDirectionCellEnsemble],
) -> torch.Tensor:
    """Decode heading angles from HDC logits using the configured decoder mode."""
    if not hdc_logits or not hdc_ensembles:
        raise ValueError("At least one head-direction ensemble is required.")
    if len(hdc_logits) != len(hdc_ensembles):
        raise ValueError("hdc_logits and hdc_ensembles must have the same length.")

    decoded_angles = []
    for logits, ensemble in zip(hdc_logits, hdc_ensembles):
        if hdc_supports_analytic_decode(ensemble):
            decoded_angles.append(_decode_head_direction_logits_one_torch(logits, ensemble))
        else:
            decoded_angles.append(_decode_head_direction_logits_weighted_one_torch(logits, ensemble))
    return _circular_mean_torch(decoded_angles)


def compute_position_mse(
    pc_logits: List[torch.Tensor],
    target_pos: torch.Tensor,
    pc_ensembles: List[PlaceCellEnsemble],
) -> torch.Tensor:
    """Compute mean-squared error between decoded and ground-truth positions."""
    pred_pos = decode_position_from_pc_logits(pc_logits, pc_ensembles)
    return F.mse_loss(pred_pos, target_pos)


def compute_head_direction_mae_rad(
    hdc_logits: List[torch.Tensor],
    target_hd: torch.Tensor,
    hdc_ensembles: List[HeadDirectionCellEnsemble],
) -> torch.Tensor:
    """Compute circular mean absolute heading error in radians."""
    pred_hd = decode_head_direction_from_hdc_logits(hdc_logits, hdc_ensembles)
    if pred_hd.shape != target_hd.shape:
        raise ValueError(
            f"target_hd must have shape {tuple(pred_hd.shape)}, got {tuple(target_hd.shape)}."
        )
    diff = torch.atan2(torch.sin(pred_hd - target_hd), torch.cos(pred_hd - target_hd))
    return torch.mean(torch.abs(diff))
