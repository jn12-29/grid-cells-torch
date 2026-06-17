"""Factories for place-cell and head-direction-cell ensembles."""

from typing import List

from grid_cells.cells.ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble
from grid_cells.cells.persistence import (
    apply_cell_centers,
    load_cell_centers,
    resolve_existing_cell_centers_path,
)


def _broadcast_ensemble_values(values, count: int, field_name: str, cast):
    """Return one config value per ensemble, accepting a scalar or a singleton list."""
    if isinstance(values, (list, tuple)):
        resolved = list(values)
    else:
        resolved = [values]
    if len(resolved) == 1 and count != 1:
        resolved = resolved * count
    if len(resolved) != count:
        raise ValueError(
            f"{field_name} must have length 1 or match task.n_pc length "
            f"({count}), got {len(resolved)}."
        )
    return [cast(value) for value in resolved]


def _create_place_cell_ensembles(cfg) -> List[PlaceCellEnsemble]:
    """Create place-cell ensembles from config without loading persisted centers."""
    task_cfg = cfg.task
    pc_surround_scale = _broadcast_ensemble_values(
        getattr(task_cfg, "pc_surround_scale", 2.0),
        len(task_cfg.n_pc),
        "task.pc_surround_scale",
        float,
    )
    pc_target_family = getattr(task_cfg, "pc_target_family", "gaussian")
    pc_target_normalization = getattr(task_cfg, "pc_target_normalization", "global")
    return [
        PlaceCellEnsemble(
            n_cells,
            stdev=scale,
            pos_min=-task_cfg.env_size / 2,
            pos_max=task_cfg.env_size / 2,
            seed=task_cfg.neurons_seed,
            soft_targets=task_cfg.targets_type,
            soft_init=task_cfg.lstm_init_type,
            decode_type=getattr(task_cfg, "decode_type", "analytic"),
            decode_top_k=getattr(task_cfg, "decode_top_k", 3),
            pc_target_family=pc_target_family,
            pc_target_normalization=pc_target_normalization,
            surround_scale=surround_scale,
        )
        for n_cells, scale, surround_scale in zip(
            task_cfg.n_pc,
            task_cfg.pc_scale,
            pc_surround_scale,
        )
    ]


def _create_head_direction_ensembles(cfg) -> List[HeadDirectionCellEnsemble]:
    """Create HDC ensembles from config without loading persisted centers."""
    task_cfg = cfg.task
    decode_type = getattr(task_cfg, "decode_type", "analytic")
    hdc_decode_type = "analytic" if decode_type == "top_k" else decode_type
    return [
        HeadDirectionCellEnsemble(
            n_cells,
            concentration=concentration,
            seed=task_cfg.neurons_seed,
            soft_targets=task_cfg.targets_type,
            soft_init=task_cfg.lstm_init_type,
            decode_type=hdc_decode_type,
            decode_top_k=getattr(task_cfg, "decode_top_k", 3),
        )
        for n_cells, concentration in zip(
            task_cfg.n_hdc,
            task_cfg.hdc_concentration,
        )
    ]


def get_cell_ensembles(cfg) -> tuple[List[PlaceCellEnsemble], List[HeadDirectionCellEnsemble]]:
    """Create all cell ensembles and load persisted centers when available."""
    pc_ensembles, hdc_ensembles = create_cell_ensembles_from_config(cfg)
    path = resolve_existing_cell_centers_path(cfg)
    if path is not None:
        pc_centers, hdc_centers = load_cell_centers(path, cfg)
        apply_cell_centers(pc_ensembles, hdc_ensembles, pc_centers, hdc_centers)
    return pc_ensembles, hdc_ensembles


def create_cell_ensembles_from_config(cfg) -> tuple[List[PlaceCellEnsemble], List[HeadDirectionCellEnsemble]]:
    """Create all cell ensembles without loading persisted centers."""
    pc_ensembles = _create_place_cell_ensembles(cfg)
    hdc_ensembles = _create_head_direction_ensembles(cfg)
    return pc_ensembles, hdc_ensembles


def get_place_cell_ensembles(cfg) -> List[PlaceCellEnsemble]:
    """Create place-cell ensembles from the task config."""
    pc_ensembles, _ = get_cell_ensembles(cfg)
    return pc_ensembles


def get_head_direction_ensembles(cfg) -> List[HeadDirectionCellEnsemble]:
    """Create head-direction ensembles from the task config."""
    _, hdc_ensembles = get_cell_ensembles(cfg)
    return hdc_ensembles
