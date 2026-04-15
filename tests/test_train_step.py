"""Smoke tests for a single training step."""
import logging
import os
import numpy as np
import torch
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from types import SimpleNamespace

import train as train_module
from dataset import get_dataloader
from ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble
from model import GridCellsRNN
from train import get_step_log_interval, resolve_save_dir


def make_cfg():
    return SimpleNamespace(
        task=SimpleNamespace(
            env_size=2.2,
            seq_len=10,
            neurons_seed=0,
            targets_type="softmax",
            lstm_init_type="softmax",
            velocity_noise=[0.0, 0.0, 0.0],
            n_pc=[8],
            pc_scale=[0.35],
            n_hdc=[4],
            hdc_concentration=[20.0],
        ),
        model=SimpleNamespace(
            nh_lstm=16,
            nh_bottleneck=32,
            dropout_rate=0.5,
            bottleneck_has_bias=False,
            init_weight_disp=0.0,
        ),
        training=SimpleNamespace(
            batch_size=4,
            steps_per_epoch=2,
            lr=1e-4,
            momentum=0.9,
            weight_decay=1e-5,
            grad_clip=1e-5,
        ),
    )


def test_training_step_with_preencoded_batch():
    """A single forward/backward/step completes with worker-encoded targets."""
    cfg = make_cfg()
    device = torch.device("cpu")

    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model)).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=cfg.training.lr,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay,
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=False)

    loader = get_dataloader(cfg, pc_ens=pc_ens, hdc_ens=hdc_ens)
    batch = next(iter(loader))

    assert "init_cond" in batch
    assert "pc_targets_0" in batch
    assert "hdc_targets_0" in batch

    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    init_cond = batch["init_cond"].float()

    optimizer.zero_grad()
    with torch.autocast(device_type=device.type, enabled=False):
        pc_logits, hdc_logits, _, _ = model(init_cond, batch["ego_vel"], training=True)
        loss = sum(
            ens.loss(logits, batch[f"pc_targets_{i}"])
            for i, (ens, logits) in enumerate(zip(pc_ens, pc_logits))
        )
        loss += sum(
            ens.loss(logits, batch[f"hdc_targets_{i}"])
            for i, (ens, logits) in enumerate(zip(hdc_ens, hdc_logits))
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_value_(model.parameters(), cfg.training.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    assert np.isfinite(loss.item())


def test_resolve_save_dir_appends_timestamp_directory():
    """Timestamped runs should be nested under the configured base directory."""
    save_dir = resolve_save_dir(
        "./results",
        timestamp_save_dir=True,
        run_name=None,
        now=datetime(2026, 4, 15, 15, 30, 0),
    )

    assert os.path.normpath(save_dir) == os.path.normpath("./results/20260415-153000")


def test_resolve_save_dir_can_keep_base_directory():
    """Disabling timestamps should preserve the original save_dir."""
    save_dir = resolve_save_dir(
        "./results",
        timestamp_save_dir=False,
        run_name="debug",
        now=datetime(2026, 4, 15, 15, 30, 0),
    )

    assert os.path.normpath(save_dir) == os.path.normpath("./results")


def test_step_log_interval_caps_step_scalars_per_epoch():
    """Automatic TensorBoard sampling should emit at most ten points per epoch."""
    assert get_step_log_interval(1000) == 100
    assert get_step_log_interval(95) == 10
    assert get_step_log_interval(9) == 1


def test_create_summary_writer_uses_tensorboard_subdir(monkeypatch):
    """TensorBoard logs should be written under <save_dir>/tensorboard."""
    calls = {}

    class DummyWriter:
        def __init__(self, log_dir):
            calls["log_dir"] = log_dir

        def add_text(self, *args):
            calls["tag"] = args[0]

    cfg = SimpleNamespace(
        task=SimpleNamespace(),
        model=SimpleNamespace(),
        training=SimpleNamespace(save_dir="./results/run1", use_tensorboard=True),
    )
    monkeypatch.setattr(train_module, "SummaryWriter", DummyWriter)

    writer = train_module.create_summary_writer(cfg, logging.getLogger("test_writer"))

    assert isinstance(writer, DummyWriter)
    assert os.path.normpath(calls["log_dir"]) == os.path.normpath("./results/run1/tensorboard")
    assert calls["tag"] == "run/config"
