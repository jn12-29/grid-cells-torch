"""Smoke tests for the training entrypoint and one-step optimization flow.

Usage:
    pytest tests/test_train_step.py
    pytest tests/test_train_step.py -k save_dir
"""
import argparse
import logging
import os
import numpy as np
import pytest
import torch
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from types import SimpleNamespace

import train as train_module
from grid_cells.data.dataset import get_dataloader
from grid_cells.cells.ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble
from grid_cells.cells.model import GridCellsRNN
from train import (
    _evaluate,
    _apply_cli_path_overrides,
    _apply_overrides,
    _build_eval_loader,
    _build_train_loader,
    build_lr_scheduler,
    build_optimizer,
    get_step_log_interval,
    load_config,
    resolve_save_dir,
    save_config,
)
from grid_cells.training.session import TrainingSession, TrainingSessionHooks
from grid_cells.training.cli import parse_train_args
from grid_cells.training.runtime import save_checkpoint


def make_cfg():
    return SimpleNamespace(
        task=SimpleNamespace(
            env_size=2.2,
            seq_len=10,
            cells_path=None,
            neurons_seed=0,
            targets_type="softmax",
            lstm_init_type="softmax",
            decode_type="analytic",
            velocity_noise=[0.0, 0.0, 0.0],
            n_pc=[8],
            pc_scale=[0.35],
            n_hdc=[4],
            hdc_concentration=[20.0],
            motion_dt=0.02,
            motion_b=0.26,
            motion_mv=0.1,
            motion_sv=0.13,
            motion_mw=0.0,
            motion_sw=0.52 * np.pi,
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
            datadir="data/latest",
            data_path="data/latest/train.npz",
            optimizer="rmsprop",
            lr=1e-4,
            lr_scheduler="none",
            lr_min=1e-6,
            lr_step_size=2,
            lr_gamma=0.5,
            momentum=0.9,
            adamw_betas=[0.9, 0.999],
            adamw_eps=1e-8,
            weight_decay=1e-5,
            grad_clip=1e-5,
            checkpoint_every=0,
            save_final_checkpoint=True,
            eval_every=1,
        ),
        visualization=SimpleNamespace(
            spatial_bins=32,
            directional_bins=20,
            anim_num_traj=4,
            anim_fps=20,
            anim_step=4,
            anim_workers=4,
        ),
        analysis=SimpleNamespace(
            enabled=False,
            compute_grid_selectivity=True,
            compute_grid_geometry=True,
            compute_shuffle_significance=True,
            compute_split_half=True,
            compute_hd_selectivity=True,
            num_shuffles=200,
            scale_discreteness_num_shuffles=500,
            scale_discreteness_bins=13,
            scale_discreteness_min_bins=10.0,
            scale_discreteness_max_bins=36.0,
            scale_gmm_max_components=8,
            gridness_threshold=0.37,
            fdr_alpha=0.05,
            min_shift_fraction=0.1,
            split_half_min_corr=0.3,
            max_eval_trajectories=512,
            random_seed=0,
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
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))

    optimizer, decoder_params = build_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.RMSprop)
    assert decoder_params


def test_build_optimizer_can_switch_to_adamw():
    """Optimizer builder should support AdamW via config."""
    cfg = make_cfg()
    cfg.training.optimizer = "adamw"
    cfg.training.adamw_betas = [0.8, 0.95]
    cfg.training.adamw_eps = 1e-6
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))

    optimizer, decoder_params = build_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert decoder_params
    assert optimizer.defaults["betas"] == (0.8, 0.95)
    assert optimizer.defaults["eps"] == 1e-6


def test_build_optimizer_can_switch_to_sgd():
    """Optimizer builder should support SGD via config."""
    cfg = make_cfg()
    cfg.training.optimizer = "sgd"
    cfg.training.momentum = 0.7
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))

    optimizer, decoder_params = build_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.SGD)
    assert decoder_params
    assert optimizer.defaults["momentum"] == 0.7


def test_build_lr_scheduler_defaults_to_cosine_when_field_missing():
    """Scheduler builder should default missing lr_scheduler to cosine."""
    cfg = make_cfg()
    delattr(cfg.training, "lr_scheduler")
    cfg.training.epochs = 4
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))
    optimizer, _ = build_optimizer(model, cfg)

    scheduler = build_lr_scheduler(optimizer, cfg)

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_build_lr_scheduler_supports_cosine():
    """Cosine scheduling should decay the optimizer learning rate by epoch."""
    cfg = make_cfg()
    cfg.training.epochs = 4
    cfg.training.lr_scheduler = "cosine"
    cfg.training.lr = 1e-3
    cfg.training.lr_min = 1e-5
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))
    optimizer, _ = build_optimizer(model, cfg)

    scheduler = build_lr_scheduler(optimizer, cfg)
    optimizer.step()
    scheduler.step()

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert optimizer.param_groups[0]["lr"] < cfg.training.lr
    assert optimizer.param_groups[0]["lr"] > cfg.training.lr_min


def test_build_lr_scheduler_supports_step():
    """Step scheduling should apply the configured gamma after each interval."""
    cfg = make_cfg()
    cfg.training.lr_scheduler = "step"
    cfg.training.lr = 1e-3
    cfg.training.lr_step_size = 1
    cfg.training.lr_gamma = 0.25
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))
    optimizer, _ = build_optimizer(model, cfg)

    scheduler = build_lr_scheduler(optimizer, cfg)
    optimizer.step()
    scheduler.step()

    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert optimizer.param_groups[0]["lr"] == cfg.training.lr * cfg.training.lr_gamma


def test_build_lr_scheduler_rejects_unknown_name():
    """Unsupported scheduler names should fail fast."""
    cfg = make_cfg()
    cfg.training.lr_scheduler = "linear"
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))
    optimizer, _ = build_optimizer(model, cfg)

    try:
        build_lr_scheduler(optimizer, cfg)
    except ValueError as exc:
        assert "Unsupported learning-rate scheduler" in str(exc)
    else:
        raise AssertionError("unknown scheduler should raise ValueError")


def test_load_config_returns_nested_namespace(tmp_path):
    """YAML config loading should preserve nested attribute access."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "task:\n"
        "  env_size: 2.2\n"
        "training:\n"
        "  epochs: 5\n"
        "visualization:\n"
        "  anim_fps: 24\n"
    )

    cfg = load_config(str(config_path))

    assert cfg.task.env_size == 2.2
    assert cfg.training.epochs == 5
    assert cfg.visualization.anim_fps == 24


def test_apply_overrides_updates_nested_sections_and_ignores_cli_paths():
    """CLI overrides should mutate config sections without touching loader paths."""
    cfg = SimpleNamespace(
        task=SimpleNamespace(env_size=2.2),
        model=SimpleNamespace(dropout_rate=0.5),
        training=SimpleNamespace(epochs=10),
        visualization=SimpleNamespace(anim_step=4),
        analysis=SimpleNamespace(enabled=True),
        data_generation=SimpleNamespace(num_workers=8),
    )
    args = SimpleNamespace(
        config="alt.yaml",
        data_path="data/latest/train.npz",
        eval_data_path="data/latest/eval.npz",
        task__env_size=3.0,
        model__dropout_rate=0.25,
        training__epochs=20,
        visualization__anim_step=2,
        analysis__enabled=False,
        data_generation__num_workers=3,
        training__lr=None,
    )

    updated = _apply_overrides(cfg, args)

    assert updated is cfg
    assert cfg.task.env_size == 3.0
    assert cfg.model.dropout_rate == 0.25
    assert cfg.training.epochs == 20
    assert cfg.visualization.anim_step == 2
    assert cfg.analysis.enabled is False
    assert cfg.data_generation.num_workers == 3


def test_apply_cli_path_overrides_updates_effective_training_config():
    """Explicit dataset CLI paths should appear in the saved effective config."""
    cfg = SimpleNamespace(
        training=SimpleNamespace(
            data_path="data/default-train.npz",
            eval_data_path="data/default-eval.npz",
        )
    )
    args = SimpleNamespace(
        data_path="data/cli-train.npz",
        eval_data_path="data/cli-eval.npz",
    )

    updated = _apply_cli_path_overrides(cfg, args)

    assert updated is cfg
    assert cfg.training.data_path == "data/cli-train.npz"
    assert cfg.training.eval_data_path == "data/cli-eval.npz"


def test_save_config_writes_yaml_from_namespace(tmp_path):
    """Saved configs should be plain YAML that includes current namespace values."""
    cfg = SimpleNamespace(
        task=SimpleNamespace(env_size=3.0),
        training=SimpleNamespace(epochs=20, save_dir=str(tmp_path)),
    )
    config_path = tmp_path / "config.yaml"

    save_config(cfg, str(config_path))

    loaded = load_config(str(config_path))
    assert loaded.task.env_size == 3.0
    assert loaded.training.epochs == 20
    assert loaded.training.save_dir == str(tmp_path)


def test_save_checkpoint_writes_cpu_payload_under_checkpoints(tmp_path):
    """Checkpoint helper should save a CPU-loadable payload below the run dir."""
    cfg = SimpleNamespace(
        task=SimpleNamespace(env_size=2.2),
        model=SimpleNamespace(nh_lstm=4),
        training=SimpleNamespace(save_dir=str(tmp_path)),
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()

    class DummyEnsemble:
        def __init__(self, means):
            self.means = means

    pc_ens = [DummyEnsemble(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))]
    hdc_ens = [DummyEnsemble(np.array([[0.0], [1.0]], dtype=np.float32))]

    path = save_checkpoint(
        model,
        optimizer,
        None,
        cfg,
        pc_ens,
        hdc_ens,
        epoch=2,
        global_step=7,
        filename="checkpoint_epoch_0002.pt",
    )

    assert os.path.normpath(os.path.dirname(path)) == os.path.normpath(
        str(tmp_path / "checkpoints")
    )
    payload = torch.load(path, map_location="cpu")
    assert {
        "schema_version",
        "epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "config",
        "cell_centers",
    }.issubset(payload)
    assert payload["schema_version"] == 1
    assert payload["epoch"] == 2
    assert payload["global_step"] == 7
    assert payload["scheduler_state_dict"] is None
    assert payload["model_state_dict"]["weight"].device.type == "cpu"
    assert isinstance(payload["cell_centers"]["pc"][0], torch.Tensor)
    assert isinstance(payload["cell_centers"]["hdc"][0], torch.Tensor)
    assert not isinstance(payload["cell_centers"]["pc"][0], np.ndarray)
    assert payload["cell_centers"]["pc"][0].device.type == "cpu"

    optimizer_tensors = [
        value
        for state in payload["optimizer_state_dict"]["state"].values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    assert optimizer_tensors
    assert all(tensor.device.type == "cpu" for tensor in optimizer_tensors)


def _run_stubbed_training_session(
    cfg,
    tmp_path,
    monkeypatch,
    run_epoch=None,
    save_calls=None,
):
    """Run TrainingSession with injected stubs so scheduling can be tested."""
    cfg.training.save_dir = str(tmp_path)
    cfg.training.timestamp_save_dir = False
    cfg.training.use_tensorboard = False
    cfg.training.run_name = None
    save_calls = save_calls if save_calls is not None else []

    class DummyModel:
        def to(self, device):
            return self

    if run_epoch is None:
        def run_epoch(self, epoch, global_step, **kwargs):
            return global_step + 1

    def capture_checkpoint(*args, **kwargs):
        save_calls.append(
            {
                "epoch": kwargs["epoch"],
                "global_step": kwargs["global_step"],
                "filename": kwargs["filename"],
            }
        )
        return os.path.join(str(tmp_path), "checkpoints", kwargs["filename"])

    monkeypatch.setattr(
        "grid_cells.training.session.get_cell_ensembles",
        lambda cfg_arg: ([], []),
    )
    monkeypatch.setattr(
        "grid_cells.training.session.GridCellsRNN",
        lambda *args, **kwargs: DummyModel(),
    )
    monkeypatch.setattr(TrainingSession, "_build_scorer", lambda self: "scorer")
    monkeypatch.setattr(TrainingSession, "_run_epoch", run_epoch)

    hooks = TrainingSessionHooks(
        resolve_save_dir=resolve_save_dir,
        setup_logger=train_module.setup_logger,
        save_config=save_config,
        create_summary_writer=lambda cfg_arg, logger: None,
        build_optimizer=lambda model, cfg_arg: ("optimizer", []),
        build_lr_scheduler=lambda optimizer, cfg_arg: "scheduler",
        build_train_loader=lambda *args, **kwargs: None,
        build_eval_loader=lambda *args, **kwargs: "eval-loader",
        evaluate=lambda *args, **kwargs: None,
        save_checkpoint=capture_checkpoint,
        get_step_log_interval=get_step_log_interval,
        tqdm=None,
    )

    TrainingSession(cfg, hooks=hooks).run()
    return save_calls


def test_training_session_saves_effective_config_before_training(tmp_path, monkeypatch):
    """Training runs should write config.yaml into the resolved run directory."""
    cfg = make_cfg()
    cfg.training.save_dir = str(tmp_path)
    cfg.training.timestamp_save_dir = False
    cfg.training.epochs = 0
    cfg.training.use_tensorboard = False
    cfg.training.run_name = None
    calls = {}

    class DummyModel:
        def to(self, device):
            return self

    def fake_save_config(cfg_arg, path):
        calls["path"] = path
        save_config(cfg_arg, path)

    monkeypatch.setattr(
        "grid_cells.training.session.get_cell_ensembles",
        lambda cfg_arg: ("pc", "hdc"),
    )
    monkeypatch.setattr(
        "grid_cells.training.session.GridCellsRNN",
        lambda *args, **kwargs: DummyModel(),
    )
    monkeypatch.setattr(
        TrainingSession,
        "_build_scorer",
        lambda self: "scorer",
    )

    hooks = TrainingSessionHooks(
        resolve_save_dir=resolve_save_dir,
        setup_logger=train_module.setup_logger,
        save_config=fake_save_config,
        create_summary_writer=lambda cfg_arg, logger: None,
        build_optimizer=lambda model, cfg_arg: ("optimizer", []),
        build_lr_scheduler=lambda optimizer, cfg_arg: None,
        build_train_loader=lambda *args, **kwargs: None,
        build_eval_loader=lambda *args, **kwargs: "eval-loader",
        evaluate=lambda *args, **kwargs: None,
        save_checkpoint=lambda *args, **kwargs: None,
        get_step_log_interval=get_step_log_interval,
        tqdm=None,
    )

    TrainingSession(cfg, hooks=hooks).run()

    assert os.path.normpath(calls["path"]) == os.path.normpath(
        str(tmp_path / "config.yaml")
    )
    loaded = load_config(calls["path"])
    assert loaded.training.save_dir == str(tmp_path)
    assert loaded.training.epochs == 0


def test_training_session_schedules_periodic_and_final_checkpoints(tmp_path, monkeypatch):
    """Checkpoint scheduling should use completed epoch numbers."""
    cfg = make_cfg()
    cfg.training.epochs = 3
    cfg.training.checkpoint_every = 2
    cfg.training.save_final_checkpoint = True

    def fake_run_epoch(self, epoch, global_step, **kwargs):
        return global_step + 10

    calls = _run_stubbed_training_session(
        cfg,
        tmp_path,
        monkeypatch,
        run_epoch=fake_run_epoch,
    )

    assert calls == [
        {
            "epoch": 2,
            "global_step": 20,
            "filename": "checkpoint_epoch_0002.pt",
        },
        {
            "epoch": 3,
            "global_step": 30,
            "filename": "checkpoint_final.pt",
        },
    ]


def test_training_session_skips_disabled_and_zero_epoch_checkpoints(tmp_path, monkeypatch):
    """Disabled checkpointing and zero-epoch training should write no checkpoints."""
    cfg = make_cfg()
    cfg.training.epochs = 2
    cfg.training.checkpoint_every = 0
    cfg.training.save_final_checkpoint = False

    calls = _run_stubbed_training_session(cfg, tmp_path / "disabled", monkeypatch)

    assert calls == []

    cfg = make_cfg()
    cfg.training.epochs = 0
    cfg.training.checkpoint_every = 1
    cfg.training.save_final_checkpoint = True

    calls = _run_stubbed_training_session(cfg, tmp_path / "zero", monkeypatch)

    assert calls == []


def test_training_session_does_not_save_final_checkpoint_after_exception(
    tmp_path,
    monkeypatch,
):
    """Failed training should not be labeled with a final checkpoint."""
    cfg = make_cfg()
    cfg.training.epochs = 3
    cfg.training.checkpoint_every = 0
    cfg.training.save_final_checkpoint = True
    save_calls = []

    def fail_run_epoch(self, epoch, global_step, **kwargs):
        raise RuntimeError("epoch failed")

    with pytest.raises(RuntimeError, match="epoch failed"):
        _run_stubbed_training_session(
            cfg,
            tmp_path,
            monkeypatch,
            run_epoch=fail_run_epoch,
            save_calls=save_calls,
        )

    assert save_calls == []


def test_register_config_overrides_supports_shared_sections():
    """Shared config registration should expose section.key override args."""
    from grid_cells.common.config_cli import register_config_overrides

    parser = argparse.ArgumentParser()
    register_config_overrides(
        parser,
        sections=(
            "task",
            "model",
            "training",
            "visualization",
            "analysis",
            "data_generation",
        ),
    )

    args = parser.parse_args(
        [
            "--task.env_size",
            "3.5",
            "--task.cells_path",
            "data/cells.npz",
            "--task.targets_type",
            "softmax",
            "--task.lstm_init_type",
            "softmax",
            "--task.decode_type",
            "analytic",
            "--model.nh_lstm",
            "64",
            "--training.batch_size",
            "8",
            "--training.lr_scheduler",
            "cosine",
            "--training.lr_min",
            "1e-5",
            "--training.datadir",
            "data/custom",
            "--training.checkpoint_every",
            "5",
            "--training.save_final_checkpoint",
            "false",
            "--visualization.anim_fps",
            "30",
            "--analysis.enabled",
            "false",
            "--analysis.compute_grid_selectivity",
            "false",
            "--analysis.compute_grid_geometry",
            "false",
            "--analysis.compute_shuffle_significance",
            "false",
            "--analysis.compute_split_half",
            "false",
            "--analysis.num_shuffles",
            "10",
            "--analysis.scale_discreteness_num_shuffles",
            "25",
            "--analysis.scale_gmm_max_components",
            "4",
            "--analysis.gridness_threshold",
            "0.25",
            "--data_generation.num_workers",
            "2",
        ]
    )

    assert args.task__env_size == 3.5
    assert args.task__cells_path == "data/cells.npz"
    assert args.task__targets_type == "softmax"
    assert args.task__lstm_init_type == "softmax"
    assert args.task__decode_type == "analytic"
    assert args.model__nh_lstm == 64
    assert args.training__batch_size == 8
    assert args.training__lr_scheduler == "cosine"
    assert args.training__lr_min == 1e-5
    assert args.training__datadir == "data/custom"
    assert args.training__checkpoint_every == 5
    assert args.training__save_final_checkpoint is False
    assert args.visualization__anim_fps == 30
    assert args.analysis__enabled is False
    assert args.analysis__compute_grid_selectivity is False
    assert args.analysis__compute_grid_geometry is False
    assert args.analysis__compute_shuffle_significance is False
    assert args.analysis__compute_split_half is False
    assert args.analysis__num_shuffles == 10
    assert args.analysis__scale_discreteness_num_shuffles == 25
    assert args.analysis__scale_gmm_max_components == 4
    assert args.analysis__gridness_threshold == 0.25
    assert args.data_generation__num_workers == 2


def test_parse_train_args_supports_task_model_training_and_visualization_overrides(
    monkeypatch,
):
    """Train CLI should reuse the shared override syntax across core sections."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--task.env_size",
            "2.8",
            "--task.cells_path",
            "data/cells.npz",
            "--task.decode_type",
            "weighted_mean",
            "--model.dropout_rate",
            "0.25",
            "--training.batch_size",
            "16",
            "--training.lr_scheduler",
            "step",
            "--training.lr_step_size",
            "10",
            "--training.lr_gamma",
            "0.8",
            "--training.datadir",
            "data/custom",
            "--training.checkpoint_every",
            "5",
            "--training.save_final_checkpoint",
            "false",
            "--visualization.anim_step",
            "2",
            "--analysis.enabled",
            "false",
            "--analysis.compute_hd_selectivity",
            "false",
            "--analysis.max_eval_trajectories",
            "16",
        ],
    )

    args = parse_train_args()

    assert args.task__env_size == 2.8
    assert args.task__cells_path == "data/cells.npz"
    assert args.task__decode_type == "weighted_mean"
    assert args.model__dropout_rate == 0.25
    assert args.training__batch_size == 16
    assert args.training__lr_scheduler == "step"
    assert args.training__lr_step_size == 10
    assert args.training__lr_gamma == 0.8
    assert args.training__datadir == "data/custom"
    assert args.training__checkpoint_every == 5
    assert args.training__save_final_checkpoint is False
    assert args.visualization__anim_step == 2
    assert args.analysis__enabled is False
    assert args.analysis__compute_hd_selectivity is False
    assert args.analysis__max_eval_trajectories == 16


def test_validate_cell_code_contract_warns_once_for_non_softmax():
    """Analytic decode warnings should be emitted once at config validation."""
    from grid_cells.cells import contracts

    contracts._WARNED_KEYS.clear()
    cfg = make_cfg()
    cfg.task.targets_type = "voronoi"
    cfg.task.lstm_init_type = "zeros"
    pc_ens = [
        PlaceCellEnsemble(
            4,
            stdev=0.35,
            pos_min=-1.1,
            pos_max=1.1,
            seed=0,
            soft_targets="voronoi",
            soft_init="zeros",
        )
    ]
    hdc_ens = [
        HeadDirectionCellEnsemble(
            4,
            concentration=20.0,
            seed=0,
            soft_targets="voronoi",
            soft_init="zeros",
        )
    ]
    messages = []

    class DummyLogger:
        def warning(self, message):
            messages.append(message)

    contracts.validate_cell_code_contract(cfg, pc_ens, hdc_ens, DummyLogger())
    contracts.validate_cell_code_contract(cfg, pc_ens, hdc_ens, DummyLogger())

    assert len(messages) == 2
    assert "targets_type='softmax'" in messages[0]
    assert "lstm_init_type='softmax'" in messages[1]


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
    cfg.training.datadir = None
    cfg.training.eval_num_samples = 7
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
    assert calls["kwargs"] == {
        "data_path": str(cfg_path),
        "shuffle": False,
        "batch_size": 4,
    }


def test_build_eval_loader_prefers_datadir_split(tmp_path, monkeypatch):
    """Eval loading should prefer <datadir>/eval.npz over legacy eval_data_path."""
    cfg = make_cfg()
    cfg.training.eval_num_samples = 7
    cfg.training.eval_data_path = "data/legacy-eval.npz"
    cfg.training.datadir = str(tmp_path)
    calls = {}

    datadir_path = tmp_path / "eval.npz"
    datadir_path.write_bytes(b"placeholder")

    def fake_get_dataloader(cfg_arg, **kwargs):
        calls["kwargs"] = kwargs
        return "eval-loader"

    monkeypatch.setattr(train_module, "get_dataloader", fake_get_dataloader)

    loader = _build_eval_loader(cfg, logging.getLogger("test_eval_datadir"))

    assert loader == "eval-loader"
    assert calls["kwargs"] == {
        "data_path": str(datadir_path),
        "shuffle": False,
        "batch_size": 4,
    }


def test_build_eval_loader_explicit_path_overrides_datadir(tmp_path, monkeypatch):
    """Explicit eval_data_path should take precedence over training.datadir."""
    cfg = make_cfg()
    cfg.training.eval_num_samples = 7
    cfg.training.datadir = str(tmp_path)
    explicit_path = tmp_path / "explicit-eval.npz"
    datadir_path = tmp_path / "eval.npz"
    explicit_path.write_bytes(b"placeholder")
    datadir_path.write_bytes(b"placeholder")
    calls = {}

    def fake_get_dataloader(cfg_arg, **kwargs):
        calls["kwargs"] = kwargs
        return "eval-loader"

    monkeypatch.setattr(train_module, "get_dataloader", fake_get_dataloader)

    loader = _build_eval_loader(
        cfg,
        logging.getLogger("test_eval_explicit_datadir"),
        eval_data_path=str(explicit_path),
    )

    assert loader == "eval-loader"
    assert calls["kwargs"]["data_path"] == str(explicit_path)


def test_build_eval_loader_falls_back_to_fixed_generated_set(monkeypatch):
    """Missing eval files should fall back to one fixed generated eval dataset."""
    cfg = make_cfg()
    cfg.training.datadir = None
    cfg.training.eval_num_samples = 7
    cfg.training.eval_data_path = "data/missing-eval.npz"
    calls = {}

    def fake_get_dataloader(cfg_arg, **kwargs):
        calls["kwargs"] = kwargs
        return "generated-eval-loader"

    monkeypatch.setattr(train_module, "get_dataloader", fake_get_dataloader)

    loader = _build_eval_loader(cfg, logging.getLogger("test_eval_missing"))

    assert loader == "generated-eval-loader"
    assert calls["kwargs"] == {"num_samples": 7, "shuffle": False, "batch_size": 4}


def test_build_eval_loader_uses_eval_loader_batch_size(monkeypatch):
    """Eval loader batch size should control forward-pass batch size only."""
    cfg = make_cfg()
    cfg.training.datadir = None
    cfg.training.eval_num_samples = 7
    cfg.training.eval_loader_batch_size = 3
    cfg.training.eval_data_path = "data/missing-eval.npz"
    calls = {}

    def fake_get_dataloader(cfg_arg, **kwargs):
        calls["kwargs"] = kwargs
        return "generated-eval-loader"

    monkeypatch.setattr(train_module, "get_dataloader", fake_get_dataloader)

    loader = _build_eval_loader(cfg, logging.getLogger("test_eval_loader_batch_size"))

    assert loader == "generated-eval-loader"
    assert calls["kwargs"] == {"num_samples": 7, "shuffle": False, "batch_size": 3}


def test_build_train_loader_prefers_configured_data_path(tmp_path, monkeypatch):
    """Training should reuse the configured train split when the file exists."""
    cfg = make_cfg()
    cfg.training.datadir = None
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


def test_build_train_loader_prefers_datadir_split(tmp_path, monkeypatch):
    """Training should prefer <datadir>/train.npz over legacy data_path."""
    cfg = make_cfg()
    cfg.training.data_path = "data/legacy-train.npz"
    cfg.training.datadir = str(tmp_path)
    calls = {}

    datadir_path = tmp_path / "train.npz"
    datadir_path.write_bytes(b"placeholder")

    def fake_get_dataloader(cfg_arg, **kwargs):
        calls["kwargs"] = kwargs
        return "train-loader"

    monkeypatch.setattr(train_module, "get_dataloader", fake_get_dataloader)

    loader = _build_train_loader(
        cfg,
        logging.getLogger("test_train_loader_datadir"),
        pc_ens="pc",
        hdc_ens="hdc",
    )

    assert loader == "train-loader"
    assert calls["kwargs"] == {
        "data_path": str(datadir_path),
        "pc_ens": "pc",
        "hdc_ens": "hdc",
    }


def test_build_train_loader_explicit_path_overrides_datadir(tmp_path, monkeypatch):
    """Explicit data_path should take precedence over training.datadir."""
    cfg = make_cfg()
    cfg.training.datadir = str(tmp_path)
    explicit_path = tmp_path / "explicit-train.npz"
    datadir_path = tmp_path / "train.npz"
    explicit_path.write_bytes(b"placeholder")
    datadir_path.write_bytes(b"placeholder")
    calls = {}

    def fake_get_dataloader(cfg_arg, **kwargs):
        calls["kwargs"] = kwargs
        return "train-loader"

    monkeypatch.setattr(train_module, "get_dataloader", fake_get_dataloader)

    loader = _build_train_loader(
        cfg,
        logging.getLogger("test_train_loader_explicit_datadir"),
        pc_ens="pc",
        hdc_ens="hdc",
        data_path=str(explicit_path),
    )

    assert loader == "train-loader"
    assert calls["kwargs"]["data_path"] == str(explicit_path)


def test_build_train_loader_falls_back_when_data_path_missing(monkeypatch):
    """Missing train files should keep the original on-the-fly training mode."""
    cfg = make_cfg()
    cfg.training.datadir = None
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

    captured = {}

    def fake_get_scores_and_plot_from_ratemaps(*args, **kwargs):
        captured["rates_dir"] = args[2]
        captured["rates_filename"] = args[3]
        return np.array([0.5]), np.array([0.25])

    def fake_plot_hdc_tuning_curves(*args, **kwargs):
        captured["hdc_path"] = kwargs["save_path"]

    def fake_generate_eval_animation(*args, **kwargs):
        captured["anim_path"] = args[7]

    monkeypatch.setattr(
        train_module,
        "get_scores_and_plot_from_ratemaps",
        fake_get_scores_and_plot_from_ratemaps,
    )
    monkeypatch.setattr(train_module, "plot_hdc_tuning_curves", fake_plot_hdc_tuning_curves)
    monkeypatch.setattr(train_module, "generate_eval_animation", fake_generate_eval_animation)

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
    assert "eval/hd_mae_rad" in writer.scalars
    assert writer.scalars["eval/pos_mse"][1] == 2
    assert writer.scalars["eval/hd_mae_rad"][1] == 2
    assert writer.scalars["eval/pos_mse"][0] < 1e-6
    assert writer.scalars["eval/hd_mae_rad"][0] < 1e-6
    assert os.path.normpath(captured["rates_dir"]) == os.path.normpath(
        str(tmp_path / "eval_plots")
    )
    assert captured["rates_filename"] == "rates_and_sac_epoch_0002.pdf"
    assert os.path.normpath(captured["hdc_path"]) == os.path.normpath(
        str(tmp_path / "eval_plots" / "hdc_tuning_epoch_0002.pdf")
    )
    assert os.path.normpath(captured["anim_path"]) == os.path.normpath(
        str(tmp_path / "eval_videos" / "eval_animation_epoch_0002.mp4")
    )


def test_evaluate_analysis_disabled_does_not_write_grid_stats(monkeypatch, tmp_path):
    """Disabled statistical analysis should leave the existing artifact surface unchanged."""
    cfg = make_cfg()
    cfg.training.save_dir = str(tmp_path)
    cfg.training.eval_num_workers = 0
    cfg.training.eval_chunk_size = 2
    cfg.training.eval_units_per_page = 8
    cfg.training.eval_plot_every = 0
    cfg.analysis.enabled = False
    pc_ens = [PlaceCellEnsemble(2, stdev=0.35, pos_min=-1.0, pos_max=1.0, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]

    class DummyModel:
        def __init__(self):
            self.training = False

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def __call__(self, init_cond, ego_vel, training=False):
            batch, seq_len = ego_vel.shape[:2]
            pc_logits = [torch.zeros(batch, seq_len, 2, dtype=ego_vel.dtype)]
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

    monkeypatch.setattr(
        train_module,
        "score_ratemaps",
        lambda *args, **kwargs: (
            np.array([0.5]),
            np.array([0.25]),
            None,
            None,
            None,
        ),
    )

    batch = {
        "init_pos": torch.zeros(1, 2),
        "init_hd": torch.zeros(1, 1),
        "ego_vel": torch.zeros(1, 3, 3),
        "target_pos": torch.zeros(1, 3, 2),
        "target_hd": torch.zeros(1, 3, 1),
    }

    _evaluate(
        DummyModel(),
        pc_ens,
        hdc_ens,
        DummyScorer(),
        [batch],
        cfg,
        torch.device("cpu"),
        epoch=2,
    )

    assert not list((tmp_path / "eval_stats").glob("grid_stats_epoch_0002*"))


def test_evaluate_analysis_enabled_writes_grid_stats(monkeypatch, tmp_path):
    """Enabled statistical analysis should write the configured eval artifacts."""
    cfg = make_cfg()
    cfg.training.save_dir = str(tmp_path)
    cfg.training.eval_num_workers = 0
    cfg.training.eval_chunk_size = 2
    cfg.training.eval_units_per_page = 8
    cfg.training.eval_plot_every = 0
    cfg.analysis.enabled = True
    cfg.analysis.num_shuffles = 1
    cfg.analysis.max_eval_trajectories = 2
    cfg.visualization.directional_bins = 4
    pc_ens = [PlaceCellEnsemble(2, stdev=0.35, pos_min=-1.0, pos_max=1.0, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]

    class DummyModel:
        def __init__(self):
            self.training = False

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def __call__(self, init_cond, ego_vel, training=False):
            batch, seq_len = ego_vel.shape[:2]
            pc_logits = [torch.zeros(batch, seq_len, 2, dtype=ego_vel.dtype)]
            hdc_logits = [torch.zeros(batch, seq_len, 4, dtype=ego_vel.dtype)]
            base = torch.arange(seq_len, dtype=ego_vel.dtype).reshape(1, seq_len, 1)
            unit_scale = torch.linspace(0.2, 1.0, cfg.model.nh_bottleneck).reshape(1, 1, -1)
            bottleneck = torch.cos(base * unit_scale).repeat(batch, 1, 1)
            lstm_acts = torch.zeros(batch, seq_len, cfg.model.nh_lstm, dtype=ego_vel.dtype)
            return pc_logits, hdc_logits, bottleneck, lstm_acts

    monkeypatch.setattr(
        train_module,
        "score_ratemaps",
        lambda *args, **kwargs: (
            np.zeros(cfg.model.nh_bottleneck),
            np.zeros(cfg.model.nh_bottleneck),
            None,
            None,
            None,
        ),
    )

    batch = {
        "init_pos": torch.zeros(2, 2),
        "init_hd": torch.zeros(2, 1),
        "ego_vel": torch.zeros(2, 5, 3),
        "target_pos": torch.tensor(
            [
                [[-0.8, -0.8], [-0.4, -0.4], [0.0, 0.0], [0.4, 0.4], [0.8, 0.8]],
                [[-0.8, 0.8], [-0.4, 0.4], [0.0, 0.0], [0.4, -0.4], [0.8, -0.8]],
            ],
            dtype=torch.float32,
        ),
        "target_hd": torch.linspace(-np.pi, np.pi, 10).reshape(2, 5, 1),
    }

    _evaluate(
        DummyModel(),
        pc_ens,
        hdc_ens,
        train_module.TrainingSession(make_cfg(), hooks=None)._build_scorer(),
        [batch],
        cfg,
        torch.device("cpu"),
        epoch=3,
    )

    stats_dir = tmp_path / "eval_stats"
    assert (stats_dir / "grid_stats_epoch_0003.csv").exists()
    assert (stats_dir / "grid_stats_epoch_0003.npz").exists()
    assert (stats_dir / "grid_stats_summary_epoch_0003.json").exists()
    assert (stats_dir / "grid_scale_histograms_epoch_0003.pdf").exists()
    assert (stats_dir / "grid_scale_histograms_epoch_0003.png").exists()


class _AnalysisDummyModel:
    training = False

    def __init__(self, cfg):
        self.cfg = cfg

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, init_cond, ego_vel, training=False):
        batch, seq_len = ego_vel.shape[:2]
        pc_logits = [torch.zeros(batch, seq_len, 2, dtype=ego_vel.dtype)]
        hdc_logits = [torch.zeros(batch, seq_len, 4, dtype=ego_vel.dtype)]
        bottleneck = torch.zeros(
            batch,
            seq_len,
            self.cfg.model.nh_bottleneck,
            dtype=ego_vel.dtype,
        )
        lstm_acts = torch.zeros(
            batch,
            seq_len,
            self.cfg.model.nh_lstm,
            dtype=ego_vel.dtype,
        )
        return pc_logits, hdc_logits, bottleneck, lstm_acts


class _AnalysisDummyScorer:
    def allocate_ratemap_accumulators(self, n_units):
        return (
            np.zeros((n_units, 2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )

    def accumulate_ratemaps(self, positions, activations, ratemap_sums, ratemap_counts):
        ratemap_sums += 1.0
        ratemap_counts += 1.0

    def finalize_ratemaps(self, ratemap_sums, ratemap_counts):
        return ratemap_sums


def _minimal_analysis_result():
    return {
        "fields": ["unit_id"],
        "rows": [{"unit_id": 0}],
        "arrays": {"unit_id": np.array([0])},
        "summary": {
            "n_grid_fdr": 0,
            "n_grid_threshold": 0,
            "n_reliable_grid": 0,
            "n_hd_selective": 0,
            "n_grid_and_hd": 0,
        },
        "null_grid_score_60": np.empty((0, 1)),
        "null_grid_score_90": np.empty((0, 1)),
        "null_hd_mrl": np.empty((0, 1)),
        "grid_fdr_scale_discreteness_null": np.empty(0),
        "grid_threshold_scale_discreteness_null": np.empty(0),
        "directional_tuning": np.empty((1, 0)),
    }


def _single_eval_batch():
    return {
        "init_pos": torch.zeros(1, 2),
        "init_hd": torch.zeros(1, 1),
        "ego_vel": torch.zeros(1, 3, 3),
        "target_pos": torch.zeros(1, 3, 2),
        "target_hd": torch.zeros(1, 3, 1),
    }


def test_evaluate_passes_analysis_module_flags(monkeypatch, tmp_path):
    """Evaluation should forward selective analysis module flags to spatial stats."""
    cfg = make_cfg()
    cfg.training.save_dir = str(tmp_path)
    cfg.training.eval_num_workers = 0
    cfg.training.eval_chunk_size = 2
    cfg.training.eval_plot_every = 0
    cfg.analysis.enabled = True
    cfg.analysis.compute_grid_selectivity = False
    cfg.analysis.compute_grid_geometry = False
    cfg.analysis.compute_shuffle_significance = False
    cfg.analysis.compute_split_half = True
    cfg.analysis.compute_hd_selectivity = False
    cfg.analysis.scale_discreteness_num_shuffles = 17
    cfg.analysis.scale_discreteness_bins = 9
    cfg.analysis.scale_discreteness_min_bins = 8.0
    cfg.analysis.scale_discreteness_max_bins = 40.0
    cfg.analysis.scale_gmm_max_components = 5
    cfg.analysis.gridness_threshold = 0.29
    cfg.analysis.max_eval_trajectories = 1
    captured = {}
    pc_ens = [PlaceCellEnsemble(2, stdev=0.35, pos_min=-1.0, pos_max=1.0, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]

    def fake_analyze(*args, **kwargs):
        captured.update(kwargs)
        return _minimal_analysis_result()

    monkeypatch.setattr(
        "grid_cells.training.evaluation.analyze_bottleneck_spatial_stats",
        fake_analyze,
    )
    monkeypatch.setattr(
        train_module,
        "score_ratemaps",
        lambda *args, **kwargs: (
            np.zeros(cfg.model.nh_bottleneck),
            np.zeros(cfg.model.nh_bottleneck),
            None,
            None,
            None,
        ),
    )

    _evaluate(
        _AnalysisDummyModel(cfg),
        pc_ens,
        hdc_ens,
        _AnalysisDummyScorer(),
        [_single_eval_batch()],
        cfg,
        torch.device("cpu"),
        epoch=4,
    )

    assert captured["compute_grid_selectivity"] is False
    assert captured["compute_grid_geometry"] is False
    assert captured["compute_shuffle_significance"] is False
    assert captured["compute_split_half"] is True
    assert captured["compute_hd_selectivity"] is False
    assert captured["scale_discreteness_num_shuffles"] == 17
    assert captured["scale_discreteness_bins"] == 9
    assert captured["scale_discreteness_min_bins"] == 8.0
    assert captured["scale_discreteness_max_bins"] == 40.0
    assert captured["scale_gmm_max_components"] == 5
    assert captured["gridness_threshold"] == 0.29


def test_evaluate_accepts_legacy_grid_scores_flag(monkeypatch, tmp_path):
    """Old configs with compute_grid_scores should map to grid selectivity."""
    cfg = make_cfg()
    cfg.training.save_dir = str(tmp_path)
    cfg.training.eval_num_workers = 0
    cfg.training.eval_chunk_size = 2
    cfg.training.eval_plot_every = 0
    cfg.analysis.enabled = True
    delattr(cfg.analysis, "compute_grid_selectivity")
    cfg.analysis.compute_grid_scores = False
    captured = {}
    pc_ens = [PlaceCellEnsemble(2, stdev=0.35, pos_min=-1.0, pos_max=1.0, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]

    def fake_analyze(*args, **kwargs):
        captured.update(kwargs)
        return _minimal_analysis_result()

    monkeypatch.setattr(
        "grid_cells.training.evaluation.analyze_bottleneck_spatial_stats",
        fake_analyze,
    )
    monkeypatch.setattr(
        train_module,
        "score_ratemaps",
        lambda *args, **kwargs: (
            np.zeros(cfg.model.nh_bottleneck),
            np.zeros(cfg.model.nh_bottleneck),
            None,
            None,
            None,
        ),
    )

    _evaluate(
        _AnalysisDummyModel(cfg),
        pc_ens,
        hdc_ens,
        _AnalysisDummyScorer(),
        [_single_eval_batch()],
        cfg,
        torch.device("cpu"),
        epoch=4,
    )

    assert captured["compute_grid_selectivity"] is False
