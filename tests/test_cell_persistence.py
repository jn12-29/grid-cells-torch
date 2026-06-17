"""Tests for persisted cell-center files.

Usage:
    pytest tests/test_cell_persistence.py
"""

import os
import sys
import json
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grid_cells.cells.ensemble_utils import get_cell_ensembles
from grid_cells.cells.persistence import load_cell_centers, save_cell_centers


def _cfg(tmp_path, *, cells_path=None, datadir=None, env_size=2.2, seed=5):
    return SimpleNamespace(
        task=SimpleNamespace(
            env_size=env_size,
            cells_path=cells_path,
            neurons_seed=seed,
            targets_type="softmax",
            lstm_init_type="softmax",
            pc_target_family="gaussian",
            pc_target_normalization="global",
            pc_surround_scale=[2.0],
            n_pc=[4],
            pc_scale=[0.35],
            n_hdc=[3],
            hdc_concentration=[20.0],
        ),
        training=SimpleNamespace(
            datadir=str(tmp_path / "data" / "latest") if datadir is None else datadir,
        ),
    )


def test_save_and_load_cell_centers_round_trips(tmp_path):
    """Persisted centers should load back exactly as float32 arrays."""
    cfg = _cfg(tmp_path)
    pc_ens, hdc_ens = get_cell_ensembles(cfg)
    path = tmp_path / "cells.npz"

    save_cell_centers(str(path), pc_ens, hdc_ens, cfg)
    pc_centers, hdc_centers = load_cell_centers(str(path), cfg)

    assert np.array_equal(pc_centers[0], pc_ens[0].means)
    assert np.array_equal(hdc_centers[0], hdc_ens[0].means)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
    assert "pc_target_family" not in meta
    assert "pc_target_normalization" not in meta
    assert "pc_surround_scale" not in meta


def test_load_cell_centers_accepts_legacy_metadata_without_pc_target_fields(tmp_path):
    """Files saved before PC target-family metadata should remain loadable."""
    cfg = _cfg(tmp_path)
    pc_centers = np.full((4, 2), 0.25, dtype=np.float32)
    hdc_centers = np.full((3,), -0.5, dtype=np.float32)
    path = tmp_path / "legacy_cells.npz"
    legacy_meta = {
        "env_size": 2.2,
        "n_pc": [4],
        "pc_scale": [0.35],
        "n_hdc": [3],
        "hdc_concentration": [20.0],
        "neurons_seed": 5,
    }
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(legacy_meta, sort_keys=True)),
        pc_centers_0=pc_centers,
        hdc_centers_0=hdc_centers,
    )

    loaded_pc, loaded_hdc = load_cell_centers(str(path), cfg)

    assert np.array_equal(loaded_pc[0], pc_centers)
    assert np.array_equal(loaded_hdc[0], hdc_centers)


def test_get_cell_ensembles_applies_place_cell_target_family_config(tmp_path):
    """PC-only target fields should be passed through without affecting HDC."""
    cfg = _cfg(tmp_path)
    cfg.task.n_pc = [4, 5]
    cfg.task.pc_scale = [0.35, 0.25]
    cfg.task.pc_target_family = "difference_of_gaussians"
    cfg.task.pc_target_normalization = "global"
    cfg.task.pc_surround_scale = [2.5]

    pc_ens, hdc_ens = get_cell_ensembles(cfg)

    assert [ens.pc_target_family for ens in pc_ens] == [
        "difference_of_gaussians",
        "difference_of_gaussians",
    ]
    assert [ens.pc_target_normalization for ens in pc_ens] == ["global", "global"]
    assert [ens.surround_scale for ens in pc_ens] == [2.5, 2.5]
    assert all(not hasattr(ens, "pc_target_family") for ens in hdc_ens)


def test_get_cell_ensembles_loads_datadir_cells_when_present(tmp_path):
    """The unified factory should prefer training.datadir/cells.npz when available."""
    datadir = tmp_path / "data" / "latest"
    cfg = _cfg(tmp_path, datadir=str(datadir), seed=5)
    pc_ens, hdc_ens = get_cell_ensembles(cfg)
    pc_ens[0].means = np.full((4, 2), 0.25, dtype=np.float32)
    hdc_ens[0].means = np.full((3,), -0.5, dtype=np.float32)
    save_cell_centers(str(datadir / "cells.npz"), pc_ens, hdc_ens, cfg)

    loaded_pc, loaded_hdc = get_cell_ensembles(_cfg(tmp_path, datadir=str(datadir), seed=99))

    assert np.array_equal(loaded_pc[0].means, pc_ens[0].means)
    assert np.array_equal(loaded_hdc[0].means, hdc_ens[0].means)


def test_load_cell_centers_rejects_metadata_mismatch(tmp_path):
    """Semantic metadata mismatches should fail instead of silently reusing centers."""
    cfg = _cfg(tmp_path, env_size=2.2)
    pc_ens, hdc_ens = get_cell_ensembles(cfg)
    path = tmp_path / "cells.npz"
    save_cell_centers(str(path), pc_ens, hdc_ens, cfg)

    with pytest.raises(ValueError, match="does not match config for env_size"):
        load_cell_centers(str(path), _cfg(tmp_path, env_size=3.0))
