"""Persist and restore fixed cell preferred locations.

This module stores place-cell centers and head-direction preferred angles in a
small NPZ file so experiments can reuse exactly the same cell layout across
machines and runs.

Usage:
    save_cell_centers("data/latest/cells.npz", pc_ens, hdc_ens, cfg)
    pc_centers, hdc_centers = load_cell_centers("data/latest/cells.npz", cfg)
"""

import json
import os

import numpy as np


def resolve_cell_centers_path(cfg) -> str | None:
    """Resolve the configured cell-center file path, if any."""
    task_cfg = getattr(cfg, "task", None)
    return getattr(task_cfg, "cells_path", None) if task_cfg is not None else None


def default_datadir_cell_centers_path(cfg) -> str | None:
    """Return the conventional cells.npz path under training.datadir."""
    training_cfg = getattr(cfg, "training", None)
    datadir = getattr(training_cfg, "datadir", None) if training_cfg is not None else None
    if not datadir:
        return None
    return os.path.join(datadir, "cells.npz")


def resolve_existing_cell_centers_path(cfg) -> str | None:
    """Find the cell-center file that should be loaded for this config."""
    configured = resolve_cell_centers_path(cfg)
    if configured:
        return configured if os.path.exists(configured) else None

    datadir_path = default_datadir_cell_centers_path(cfg)
    if datadir_path and os.path.exists(datadir_path):
        return datadir_path
    return None


def cell_metadata_from_config(cfg) -> dict:
    """Build persisted metadata used to validate cell-center compatibility."""
    task_cfg = cfg.task
    return {
        "env_size": float(task_cfg.env_size),
        "n_pc": [int(value) for value in task_cfg.n_pc],
        "pc_scale": [float(value) for value in task_cfg.pc_scale],
        "n_hdc": [int(value) for value in task_cfg.n_hdc],
        "hdc_concentration": [float(value) for value in task_cfg.hdc_concentration],
        "neurons_seed": int(task_cfg.neurons_seed),
    }


def _strict_metadata_subset(meta: dict) -> dict:
    """Return fields that must match for safe center reuse."""
    return {
        key: meta[key]
        for key in (
            "env_size",
            "n_pc",
            "n_hdc",
            "hdc_concentration",
        )
    }


def save_cell_centers(path: str | None, pc_ensembles, hdc_ensembles, cfg) -> None:
    """Persist place-cell centers and HDC preferred angles to a compressed NPZ."""
    if path is None:
        return

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = {
        "meta": np.array(json.dumps(cell_metadata_from_config(cfg), sort_keys=True)),
    }
    for idx, ensemble in enumerate(pc_ensembles):
        arrays[f"pc_centers_{idx}"] = np.asarray(ensemble.means, dtype=np.float32)
    for idx, ensemble in enumerate(hdc_ensembles):
        arrays[f"hdc_centers_{idx}"] = np.asarray(ensemble.means, dtype=np.float32).reshape(-1)

    np.savez_compressed(path, **arrays)


def _validate_metadata(saved: dict, expected: dict, path: str) -> None:
    """Validate semantic compatibility without requiring identical provenance."""
    saved_subset = _strict_metadata_subset(saved)
    expected_subset = _strict_metadata_subset(expected)
    for key, expected_value in expected_subset.items():
        saved_value = saved_subset.get(key)
        if saved_value != expected_value:
            raise ValueError(
                f"Cell-center file {path} does not match config for {key}: "
                f"saved={saved_value!r}, expected={expected_value!r}."
            )


def load_cell_centers(path: str, cfg) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load and validate persisted place-cell centers and HDC preferred angles."""
    expected = cell_metadata_from_config(cfg)
    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data:
            raise ValueError(f"Cell-center file {path} is missing meta.")
        saved = json.loads(str(data["meta"]))
        _validate_metadata(saved, expected, path)

        pc_centers = []
        for idx, n_cells in enumerate(expected["n_pc"]):
            key = f"pc_centers_{idx}"
            if key not in data:
                raise ValueError(f"Cell-center file {path} is missing {key}.")
            centers = np.asarray(data[key], dtype=np.float32)
            expected_shape = (n_cells, 2)
            if centers.shape != expected_shape:
                raise ValueError(
                    f"{key} in {path} has shape {centers.shape}, expected {expected_shape}."
                )
            pc_centers.append(centers)

        hdc_centers = []
        for idx, n_cells in enumerate(expected["n_hdc"]):
            key = f"hdc_centers_{idx}"
            if key not in data:
                raise ValueError(f"Cell-center file {path} is missing {key}.")
            centers = np.asarray(data[key], dtype=np.float32).reshape(-1)
            expected_shape = (n_cells,)
            if centers.shape != expected_shape:
                raise ValueError(
                    f"{key} in {path} has shape {centers.shape}, expected {expected_shape}."
                )
            hdc_centers.append(centers)

    return pc_centers, hdc_centers


def apply_cell_centers(pc_ensembles, hdc_ensembles, pc_centers, hdc_centers) -> None:
    """Override ensemble preferred locations with loaded centers."""
    for ensemble, centers in zip(pc_ensembles, pc_centers):
        ensemble.means = centers.astype(np.float32, copy=True)
    for ensemble, centers in zip(hdc_ensembles, hdc_centers):
        ensemble.means = centers.astype(np.float32, copy=True)
