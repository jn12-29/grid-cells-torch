"""Factories for place-cell and head-direction-cell ensembles."""

from typing import List

from grid_cells.cells.ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble


def get_place_cell_ensembles(cfg) -> List[PlaceCellEnsemble]:
    """Create place-cell ensembles from the task config."""
    task_cfg = cfg.task
    return [
        PlaceCellEnsemble(
            n_cells,
            stdev=scale,
            pos_min=-task_cfg.env_size / 2,
            pos_max=task_cfg.env_size / 2,
            seed=task_cfg.neurons_seed,
            soft_targets=task_cfg.targets_type,
            soft_init=task_cfg.lstm_init_type,
        )
        for n_cells, scale in zip(task_cfg.n_pc, task_cfg.pc_scale)
    ]


def get_head_direction_ensembles(cfg) -> List[HeadDirectionCellEnsemble]:
    """Create head-direction ensembles from the task config."""
    task_cfg = cfg.task
    return [
        HeadDirectionCellEnsemble(
            n_cells,
            concentration=concentration,
            seed=task_cfg.neurons_seed,
            soft_targets=task_cfg.targets_type,
            soft_init=task_cfg.lstm_init_type,
        )
        for n_cells, concentration in zip(
            task_cfg.n_hdc,
            task_cfg.hdc_concentration,
        )
    ]
