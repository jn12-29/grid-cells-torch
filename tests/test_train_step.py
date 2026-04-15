"""Smoke tests for the training entrypoint and one-step optimization flow.

Usage:
    pytest tests/test_train_step.py
    pytest tests/test_train_step.py -k save_dir
"""
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
from train import (
    _evaluate,
    _build_eval_loader,
    _build_train_loader,
    build_optimizer,
    get_step_log_interval,
    resolve_save_dir,
)


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
            data_path="data/train.npz",
            optimizer="rmsprop",
            lr=1e-4,
            momentum=0.9,
            adamw_betas=[0.9, 0.999],
            adamw_eps=1e-8,
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
    pc_logits, hdc_logits, _, _ = model(init_cond, batch["ego_vel"], training=True)
    loss = sum(
        ens.loss(logits, batch[f"pc_targets_{i}"])
        for i, (ens, logits) in enumerate(zip(pc_ens, pc_logits))
    )
    loss += sum(
        ens.loss(logits, batch[f"hdc_targets_{i}"])
        for i, (ens, logits) in enumerate(zip(hdc_ens, hdc_logits))
    )

    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), cfg.training.grad_clip)
    optimizer.step()

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


def test_build_optimizer_defaults_to_rmsprop():
    """Optimizer builder should preserve the existing RMSprop default."""
    cfg = make_cfg()
    model = torch.nn.Linear(3, 2)

    optimizer = build_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.RMSprop)


def test_build_optimizer_can_switch_to_adamw():
    """Optimizer builder should support AdamW via config."""
    cfg = make_cfg()
    cfg.training.optimizer = "adamw"
    cfg.training.adamw_betas = [0.8, 0.95]
    cfg.training.adamw_eps = 1e-6
    model = torch.nn.Linear(3, 2)

    optimizer = build_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["betas"] == (0.8, 0.95)
    assert optimizer.defaults["eps"] == 1e-6


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


def test_build_eval_loader_prefers_configured_eval_data_path(tmp_path, monkeypatch):
    """Eval loading should reuse the configured eval split when the file exists."""
    cfg = make_cfg()
    cfg.training.eval_batch_size = 7
    cfg.training.eval_data_path = str(tmp_path / "eval.npz")
    cfg.training.batch_size = 4
    calls = {}

    cfg_path = tmp_path / "eval.npz"
    cfg_path.write_bytes(b"placeholder")

    def fake_get_dataloader(cfg_arg, **kwargs):
        calls["kwargs"] = kwargs
        return "eval-loader"

    monkeypatch.setattr(train_module, "get_dataloader", fake_get_dataloader)

    loader = _build_eval_loader(cfg, logging.getLogger("test_eval"))

    assert loader == "eval-loader"
    assert calls["kwargs"] == {"data_path": str(cfg_path), "shuffle": False}


def test_build_eval_loader_falls_back_to_fixed_generated_set(monkeypatch):
    """Missing eval files should fall back to one fixed generated eval dataset."""
    cfg = make_cfg()
    cfg.training.eval_batch_size = 7
    cfg.training.eval_data_path = "data/missing-eval.npz"
    calls = {}

    def fake_get_dataloader(cfg_arg, **kwargs):
        calls["kwargs"] = kwargs
        return "generated-eval-loader"

    monkeypatch.setattr(train_module, "get_dataloader", fake_get_dataloader)

    loader = _build_eval_loader(cfg, logging.getLogger("test_eval_missing"))

    assert loader == "generated-eval-loader"
    assert calls["kwargs"] == {"num_samples": 7, "shuffle": False}


def test_build_train_loader_prefers_configured_data_path(tmp_path, monkeypatch):
    """Training should reuse the configured train split when the file exists."""
    cfg = make_cfg()
    cfg.training.data_path = str(tmp_path / "train.npz")
    calls = {}

    cfg_path = tmp_path / "train.npz"
    cfg_path.write_bytes(b"placeholder")

    def fake_get_dataloader(cfg_arg, **kwargs):
        calls["kwargs"] = kwargs
        return "train-loader"

    monkeypatch.setattr(train_module, "get_dataloader", fake_get_dataloader)

    loader = _build_train_loader(
        cfg,
        logging.getLogger("test_train_loader"),
        pc_ens="pc",
        hdc_ens="hdc",
    )

    assert loader == "train-loader"
    assert calls["kwargs"] == {
        "data_path": str(cfg_path),
        "pc_ens": "pc",
        "hdc_ens": "hdc",
    }


def test_build_train_loader_falls_back_when_data_path_missing(monkeypatch):
    """Missing train files should keep the original on-the-fly training mode."""
    cfg = make_cfg()
    cfg.training.data_path = "data/missing-train.npz"

    def fail_get_dataloader(*args, **kwargs):
        raise AssertionError("train loader should not be built from a missing file")

    monkeypatch.setattr(train_module, "get_dataloader", fail_get_dataloader)

    loader = _build_train_loader(
        cfg,
        logging.getLogger("test_train_loader_missing"),
        pc_ens="pc",
        hdc_ens="hdc",
    )

    assert loader is None


def test_evaluate_reports_position_mse(monkeypatch, tmp_path):
    """Evaluation should log decoded position MSE alongside grid scores."""
    cfg = make_cfg()
    cfg.training.save_dir = str(tmp_path)
    cfg.training.eval_num_workers = 0
    cfg.training.eval_chunk_size = 2
    cfg.training.eval_units_per_page = 8
    pc_ens = [PlaceCellEnsemble(2, stdev=0.35, pos_min=-1.0, pos_max=1.0, seed=0)]
    pc_ens[0].means = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]

    class DummyModel:
        def __init__(self):
            self.training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def __call__(self, init_cond, ego_vel, training=False):
            batch, seq_len = ego_vel.shape[:2]
            pc_logits = [torch.tensor([[[15.0, -15.0]]] * batch, dtype=ego_vel.dtype).repeat(1, seq_len, 1)]
            hdc_logits = [torch.zeros(batch, seq_len, 4, dtype=ego_vel.dtype)]
            bottleneck = torch.zeros(batch, seq_len, cfg.model.nh_bottleneck, dtype=ego_vel.dtype)
            lstm_acts = torch.zeros(batch, seq_len, cfg.model.nh_lstm, dtype=ego_vel.dtype)
            return pc_logits, hdc_logits, bottleneck, lstm_acts

    class DummyScorer:
        def allocate_ratemap_accumulators(self, n_units):
            return np.zeros((n_units, 2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)

        def accumulate_ratemaps(self, positions, activations, ratemap_sums, ratemap_counts):
            ratemap_sums += 1.0
            ratemap_counts += 1.0

        def finalize_ratemaps(self, ratemap_sums, ratemap_counts):
            return ratemap_sums

    class DummyWriter:
        def __init__(self):
            self.scalars = {}

        def add_scalar(self, tag, value, step):
            self.scalars[tag] = (value, step)

    monkeypatch.setattr(
        train_module,
        "get_scores_and_plot_from_ratemaps",
        lambda *args, **kwargs: (np.array([0.5]), np.array([0.25])),
    )

    batch = {
        "init_pos": torch.zeros(1, 2),
        "init_hd": torch.zeros(1, 1),
        "ego_vel": torch.zeros(1, 3, 3),
        "target_pos": torch.zeros(1, 3, 2),
        "target_hd": torch.zeros(1, 3, 1),
    }
    writer = DummyWriter()

    _evaluate(
        DummyModel(),
        pc_ens,
        hdc_ens,
        DummyScorer(),
        [batch],
        cfg,
        torch.device("cpu"),
        epoch=2,
        writer=writer,
    )

    assert "eval/pos_mse" in writer.scalars
    assert writer.scalars["eval/pos_mse"][1] == 2
    assert writer.scalars["eval/pos_mse"][0] < 1e-6
