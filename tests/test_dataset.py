"""Tests for trajectory generation, encoding, and dataloader behavior.

Usage:
    pytest tests/test_dataset.py
    pytest tests/test_dataset.py -k parallel
"""
import numpy as np
import torch
import pytest
from torch.utils.data import RandomSampler, SequentialSampler
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataset import TrajectoryDataset, get_dataloader
from ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble
from types import SimpleNamespace


def make_ensembles():
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    return pc_ens, hdc_ens


def make_cfg():
    return SimpleNamespace(
        task=SimpleNamespace(
            env_size=2.2, seq_len=10, neurons_seed=0,
            targets_type="softmax", lstm_init_type="softmax",
            velocity_noise=[0.0, 0.0, 0.0],
        ),
        training=SimpleNamespace(
            batch_size=4, steps_per_epoch=3,
        ),
    )


def test_attach_ensembles_adds_init_cond():
    """After attach_ensembles, __getitem__ returns init_cond."""
    ds = TrajectoryDataset(num_samples=5, seq_len=10, env_size=2.2, seed=1)
    pc_ens, hdc_ens = make_ensembles()
    ds.attach_ensembles(pc_ens, hdc_ens)

    item = ds[0]
    assert "init_cond" in item
    assert item["init_cond"].shape == (12,)          # 8 pc + 4 hdc
    assert item["init_cond"].dtype == np.float32


def test_attach_ensembles_adds_targets():
    """After attach_ensembles, __getitem__ returns pc_targets_0 and hdc_targets_0."""
    ds = TrajectoryDataset(num_samples=5, seq_len=10, env_size=2.2, seed=1)
    pc_ens, hdc_ens = make_ensembles()
    ds.attach_ensembles(pc_ens, hdc_ens)

    item = ds[0]
    assert "pc_targets_0" in item
    assert "hdc_targets_0" in item
    assert item["pc_targets_0"].shape == (10, 8)     # (seq_len, n_pc)
    assert item["hdc_targets_0"].shape == (10, 4)    # (seq_len, n_hdc)


def test_without_attach_no_encoded_keys():
    """Without attach_ensembles, __getitem__ returns raw trajectory only."""
    ds = TrajectoryDataset(num_samples=5, seq_len=10, env_size=2.2, seed=1)
    item = ds[0]
    assert "init_cond" not in item
    assert "pc_targets_0" not in item


def test_parallel_generation_matches_single_worker_output():
    """Worker count should not change generated trajectories for a fixed seed."""
    serial_ds = TrajectoryDataset(
        num_samples=12,
        seq_len=8,
        env_size=2.2,
        velocity_noise=(0.0, 0.0, 0.0),
        seed=7,
        num_workers=1,
    )
    parallel_ds = TrajectoryDataset(
        num_samples=12,
        seq_len=8,
        env_size=2.2,
        velocity_noise=(0.0, 0.0, 0.0),
        seed=7,
        num_workers=2,
    )

    for key in ("init_pos", "init_hd", "ego_vel", "target_pos", "target_hd"):
        assert np.allclose(serial_ds._data[key], parallel_ds._data[key])


def test_init_cond_consistent_with_manual_encode():
    """init_cond from dataset matches encode_initial_conditions output."""
    from utils import encode_initial_conditions
    ds = TrajectoryDataset(num_samples=5, seq_len=10, env_size=2.2, seed=1)
    pc_ens, hdc_ens = make_ensembles()
    ds.attach_ensembles(pc_ens, hdc_ens)

    idx = 2
    item = ds[idx]
    # Build a single-item batch to pass to encode_initial_conditions
    batch = {
        "init_pos": torch.from_numpy(ds._data["init_pos"][idx:idx+1]),
        "init_hd":  torch.from_numpy(ds._data["init_hd"][idx:idx+1]),
    }
    ref = encode_initial_conditions(batch, pc_ens, hdc_ens).numpy()[0]  # (init_cond_size,)
    assert np.allclose(item["init_cond"], ref, atol=1e-5)


def test_get_dataloader_attaches_ensembles_and_keeps_workers():
    """DataLoader exposes pre-encoded fields when ensembles are provided."""
    cfg = make_cfg()
    pc_ens, hdc_ens = make_ensembles()

    loader = get_dataloader(cfg, pc_ens=pc_ens, hdc_ens=hdc_ens)
    batch = next(iter(loader))

    assert loader.persistent_workers is True
    assert "init_cond" in batch
    assert "pc_targets_0" in batch
    assert "hdc_targets_0" in batch


def test_get_dataloader_can_build_fixed_size_eval_loader():
    """Eval loaders should support explicit sample counts and sequential order."""
    cfg = make_cfg()

    loader = get_dataloader(cfg, num_samples=5, shuffle=False)

    assert len(loader.dataset) == 5
    assert isinstance(loader.batch_sampler.sampler, SequentialSampler)


def test_get_dataloader_defaults_to_shuffled_training_loader():
    """Training loaders should keep random shuffling by default."""
    cfg = make_cfg()

    loader = get_dataloader(cfg)

    assert isinstance(loader.batch_sampler.sampler, RandomSampler)
