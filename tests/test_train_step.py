"""Smoke tests for the training entrypoint and one-step optimization flow.

Usage:
    pytest tests/test_train_step.py
    pytest tests/test_train_step.py -k save_dir
"""
import argparse
import copy
import json
import logging
import os
import numpy as np
import pytest
import torch
from pytorch_optimizer import Lion
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from types import SimpleNamespace

import analyze_checkpoint as analyze_checkpoint_module
import train as train_module
from grid_cells.data.dataset import get_dataloader
from grid_cells.cells.encoding import EnsembleEncoder
from grid_cells.cells.ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble
from grid_cells.cells.model import GridCellsRNN
from grid_cells.analysis.linear_spatial_pca import LINEAR_SPATIAL_PCA_FIELDS
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
from grid_cells.training.session import (
    TrainingSession,
    TrainingSessionHooks,
    _collect_grad_group_stats,
    _collect_grad_monitor_change_stats,
    _clip_gradients,
    _merge_grad_monitor_stats,
    _resolve_grad_monitor_norm_type,
)
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
            pc_target_family="gaussian",
            pc_target_normalization="global",
            pc_surround_scale=[2.0],
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
            bottleneck_post_activation="none",
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
            betas=None,
            adamw_eps=1e-8,
            weight_decay=1e-5,
            grad_clip=1e-5,
            grad_clip_mode="norm",
            grad_clip_scope="decoder",
            grad_clip_norm_type="2.0",
            first_pos_loss_multiplier=1.0,
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
            bottleneck_activation="relu",
            compute_grid_selectivity=True,
            compute_grid_geometry=True,
            compute_shuffle_significance=True,
            compute_split_half=True,
            compute_hd_selectivity=True,
            compute_linear_spatial_pca=False,
            num_shuffles=200,
            linear_spatial_pca_top_k=16,
            linear_spatial_pca_fit_fraction=0.5,
            linear_spatial_pca_num_shuffles=0,
            linear_spatial_pca_save_plots=True,
            linear_spatial_pca_plot_top_n=16,
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
    optimizer, decoder_params = build_optimizer(model, cfg)

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
    _clip_gradients(model, decoder_params, cfg)
    optimizer.step()

    assert np.isfinite(loss.item())


def test_run_epoch_logs_first_step_metrics():
    """Training should emit diagnostics for the first predicted timestep."""
    cfg = make_cfg()
    cfg.task.seq_len = 3
    cfg.training.eval_every = 99
    cfg.training.tensorboard_log_every = 1
    cfg.training.use_tqdm = False
    cfg.training.first_pos_loss_multiplier = 2.0
    device = torch.device("cpu")

    pc_ens = [PlaceCellEnsemble(4, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    encoder = EnsembleEncoder(pc_ens, hdc_ens)

    init_pos = torch.zeros(2, 2)
    init_hd = torch.zeros(2, 1)
    ego_vel = torch.zeros(2, 3, 3)
    target_pos = torch.tensor(
        [
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
            [[0.0, 0.0], [0.0, 0.1], [0.0, 0.2]],
        ],
        dtype=torch.float32,
    )
    target_hd = torch.zeros(2, 3, 1)
    encoded_targets = encoder.encode_targets(target_pos, target_hd)

    batch = {
        "init_cond": torch.from_numpy(encoder.encode_initial_conditions(init_pos, init_hd)),
        "ego_vel": ego_vel,
        "target_pos": target_pos,
        "target_hd": target_hd,
        "pc_targets_0": torch.from_numpy(encoded_targets.pc_targets[0]),
        "hdc_targets_0": torch.from_numpy(encoded_targets.hdc_targets[0]),
    }

    class ConstantLogitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, init_cond, ego_vel, training=True):
            batch_size, seq_len = ego_vel.shape[:2]
            pc_logits = [self.logit.expand(batch_size, seq_len, pc_ens[0].n_cells)]
            hdc_logits = [self.logit.expand(batch_size, seq_len, hdc_ens[0].n_cells)]
            return pc_logits, hdc_logits, None, None

    class DummyWriter:
        def __init__(self):
            self.scalars = {}

        def add_scalar(self, tag, value, step):
            self.scalars.setdefault(tag, []).append((value, step))

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass

    hooks = SimpleNamespace(
        tqdm=None,
        get_step_log_interval=lambda num_steps: 1,
        evaluate=lambda *args, **kwargs: None,
    )
    model = ConstantLogitModel()
    writer = DummyWriter()

    TrainingSession(cfg, hooks=hooks)._run_epoch(
        epoch=0,
        global_step=0,
        logger=DummyLogger(),
        writer=writer,
        device=device,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        lr_scheduler=None,
        decoder_params=list(model.parameters()),
        pc_ens=pc_ens,
        hdc_ens=hdc_ens,
        scorer=None,
        fixed_loader=[batch],
        fixed_eval_loader=None,
    )

    for tag in (
        "train/first_pos_mse_step",
        "train/first_pos_mse_mean",
        "train/first_pos_mse_ratio",
        "train/first_hd_mae_rad_step",
        "train/first_hd_mae_rad_mean",
        "train/first_hd_mae_rad_ratio",
        "train/first_pos_loss_multiplier",
    ):
        assert tag in writer.scalars
        assert np.isfinite(writer.scalars[tag][0][0])


def test_grad_monitor_stats_compute_norms_and_clip_ratio():
    """Gradient monitor helpers should report finite group stats without mutation."""
    param = torch.nn.Parameter(torch.zeros(2))
    skipped = torch.nn.Parameter(torch.zeros(2))
    param.grad = torch.tensor([3.0, 4.0])

    stats = _collect_grad_group_stats(
        [param, skipped],
        norm_type=2.0,
        clip_threshold=3.5,
    )
    inf_stats = _collect_grad_group_stats(
        [param],
        norm_type=float("inf"),
        clip_threshold=3.5,
    )
    merged = _merge_grad_monitor_stats(
        {"state_init": stats},
        {
            "state_init": {
                "norm": 2.5,
                "rms": 1.25,
                "mean_abs": 1.0,
                "mean": 0.5,
                "max_abs": 2.0,
                "numel": 2.0,
                "clip_exceed_count": 0.0,
            }
        },
        {
            "state_init": {
                "numel": 2.0,
                "changed_count": 1.0,
                "post_saturated_count": 1.0,
            }
        },
    )

    assert stats["norm"] == 5.0
    assert stats["rms"] == pytest.approx(np.sqrt(12.5))
    assert stats["mean_abs"] == 3.5
    assert stats["mean"] == 3.5
    assert stats["max_abs"] == 4.0
    assert stats["numel"] == 2.0
    assert stats["clip_exceed_count"] == 1.0
    assert inf_stats["norm"] == 4.0
    assert inf_stats["rms"] == pytest.approx(np.sqrt(12.5))
    assert inf_stats["max_abs"] == 4.0
    assert merged["state_init"]["pre_clip_norm"] == 5.0
    assert merged["state_init"]["post_clip_norm"] == 2.5
    assert merged["state_init"]["pre_clip_rms"] == pytest.approx(np.sqrt(12.5))
    assert merged["state_init"]["post_clip_rms"] == 1.25
    assert merged["state_init"]["pre_clip_mean_abs"] == 3.5
    assert merged["state_init"]["post_clip_mean_abs"] == 1.0
    assert merged["state_init"]["pre_clip_mean"] == 3.5
    assert merged["state_init"]["post_clip_mean"] == 0.5
    assert merged["state_init"]["pre_clip_max_abs"] == 4.0
    assert merged["state_init"]["post_clip_max_abs"] == 2.0
    assert merged["state_init"]["pre_clip_exceed_frac"] == 0.5
    assert merged["state_init"]["clip_changed_frac"] == 0.5
    assert merged["state_init"]["post_clip_saturated_frac"] == 0.5
    assert merged["state_init"]["clip_ratio"] == 0.5
    assert param.grad.tolist() == [3.0, 4.0]


def test_grad_monitor_change_stats_counts_changed_and_saturated_elements():
    param = torch.nn.Parameter(torch.zeros(2))
    pre_grad = torch.tensor([3.0, 4.0])
    param.grad = torch.tensor([3.0, 3.5])

    stats = _collect_grad_monitor_change_stats(
        {"state_init": [(param, pre_grad)]},
        clip_threshold=3.5,
    )

    assert stats["state_init"]["numel"] == 2.0
    assert stats["state_init"]["changed_count"] == 1.0
    assert stats["state_init"]["post_saturated_count"] == 1.0


def test_grad_monitor_norm_type_uses_clip_norm_type_for_norm_mode():
    cfg = make_cfg()
    cfg.training.grad_clip_mode = "norm"
    cfg.training.grad_clip_norm_type = "inf"

    assert _resolve_grad_monitor_norm_type(cfg) == float("inf")


@pytest.mark.parametrize(
    ("mode", "norm_type"),
    [
        ("value", None),
        ("valueclip", "unused"),
        ("median", "unused"),
    ],
)
def test_grad_monitor_norm_type_uses_l2_for_non_norm_modes(mode, norm_type):
    cfg = make_cfg()
    cfg.training.grad_clip_mode = mode
    cfg.training.grad_clip_norm_type = norm_type

    assert _resolve_grad_monitor_norm_type(cfg) == 2.0


def test_run_epoch_logs_per_module_grad_monitor_scalars():
    """Training should emit sampled per-module gradient diagnostics."""
    cfg = make_cfg()
    cfg.task.seq_len = 2
    cfg.model.dropout_rate = 0.0
    cfg.training.eval_every = 99
    cfg.training.tensorboard_log_every = 1
    cfg.training.use_tqdm = False
    cfg.training.grad_clip = 1e-6
    cfg.training.grad_clip_mode = "value"
    cfg.training.grad_clip_scope = "all"
    cfg.training.grad_clip_norm_type = "unused"
    device = torch.device("cpu")

    pc_ens = [PlaceCellEnsemble(4, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    encoder = EnsembleEncoder(pc_ens, hdc_ens)

    init_pos = torch.zeros(2, 2)
    init_hd = torch.zeros(2, 1)
    ego_vel = torch.zeros(2, cfg.task.seq_len, 3)
    target_pos = torch.zeros(2, cfg.task.seq_len, 2)
    target_hd = torch.zeros(2, cfg.task.seq_len, 1)
    encoded_targets = encoder.encode_targets(target_pos, target_hd)

    batch = {
        "init_cond": torch.from_numpy(encoder.encode_initial_conditions(init_pos, init_hd)),
        "ego_vel": ego_vel,
        "target_pos": target_pos,
        "target_hd": target_hd,
        "pc_targets_0": torch.from_numpy(encoded_targets.pc_targets[0]),
        "hdc_targets_0": torch.from_numpy(encoded_targets.hdc_targets[0]),
    }

    class DummyWriter:
        def __init__(self):
            self.scalars = {}

        def add_scalar(self, tag, value, step):
            self.scalars.setdefault(tag, []).append((value, step))

    class DummyLogger:
        def __init__(self):
            self.messages = []

        def info(self, message, *args):
            self.messages.append(message % args if args else message)

    hooks = SimpleNamespace(
        tqdm=None,
        get_step_log_interval=lambda num_steps: 1,
        evaluate=lambda *args, **kwargs: None,
    )
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model)).to(device)
    optimizer, decoder_params = build_optimizer(model, cfg)
    writer = DummyWriter()
    logger = DummyLogger()

    TrainingSession(cfg, hooks=hooks)._run_epoch(
        epoch=0,
        global_step=0,
        logger=logger,
        writer=writer,
        device=device,
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        decoder_params=decoder_params,
        pc_ens=pc_ens,
        hdc_ens=hdc_ens,
        scorer=None,
        fixed_loader=[batch],
        fixed_eval_loader=None,
    )

    expected_modules = {
        "state_init",
        "cell_init",
        "lstm",
        "bottleneck",
        "pc_heads",
        "hdc_heads",
    }
    expected_metrics = {
        "pre_clip_norm",
        "post_clip_norm",
        "pre_clip_rms",
        "post_clip_rms",
        "pre_clip_mean_abs",
        "post_clip_mean_abs",
        "pre_clip_mean",
        "post_clip_mean",
        "pre_clip_max_abs",
        "post_clip_max_abs",
        "pre_clip_exceed_frac",
        "clip_changed_frac",
        "post_clip_saturated_frac",
        "clip_ratio",
    }
    clipped_modules = []
    for module_name in expected_modules:
        pre_norm_tag = f"train/grad/{module_name}/pre_clip_norm_step"
        post_norm_tag = f"train/grad/{module_name}/post_clip_norm_step"
        pre_max_abs_tag = f"train/grad/{module_name}/pre_clip_max_abs_step"
        post_max_abs_tag = f"train/grad/{module_name}/post_clip_max_abs_step"
        ratio_tag = f"train/grad/{module_name}/clip_ratio_step"

        for metric_name in expected_metrics:
            step_tag = f"train/grad/{module_name}/{metric_name}_step"
            mean_tag = f"train/grad/{module_name}/{metric_name}_mean"
            assert step_tag in writer.scalars
            assert mean_tag in writer.scalars
            assert np.isfinite(writer.scalars[step_tag][0][0])
            assert np.isfinite(writer.scalars[mean_tag][0][0])

        assert writer.scalars[post_norm_tag][0][0] <= writer.scalars[pre_norm_tag][0][0]
        if writer.scalars[pre_max_abs_tag][0][0] > cfg.training.grad_clip:
            clipped_modules.append(module_name)
        assert writer.scalars[post_max_abs_tag][0][0] <= cfg.training.grad_clip + 1e-12
        assert 0.0 <= writer.scalars[ratio_tag][0][0] <= 1.0
        pre_exceed_frac = writer.scalars[
            f"train/grad/{module_name}/pre_clip_exceed_frac_step"
        ][0][0]
        clip_changed_frac = writer.scalars[
            f"train/grad/{module_name}/clip_changed_frac_step"
        ][0][0]
        post_saturated_frac = writer.scalars[
            f"train/grad/{module_name}/post_clip_saturated_frac_step"
        ][0][0]
        assert 0.0 <= pre_exceed_frac <= 1.0
        assert 0.0 <= clip_changed_frac <= 1.0
        assert 0.0 <= post_saturated_frac <= 1.0
        assert (
            writer.scalars[f"train/grad/{module_name}/post_clip_rms_step"][0][0]
            <= writer.scalars[f"train/grad/{module_name}/pre_clip_rms_step"][0][0]
        )
        assert (
            writer.scalars[f"train/grad/{module_name}/post_clip_mean_abs_step"][0][0]
            <= writer.scalars[f"train/grad/{module_name}/pre_clip_mean_abs_step"][0][0]
        )

    assert clipped_modules
    assert any("pre_exceed" in message for message in logger.messages)


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


class TwoParamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        self.decoder = torch.nn.Parameter(torch.tensor([0.0, 0.0]))


def test_clip_gradients_defaults_to_decoder_norm_clipping():
    """Default gradient clipping should rescale decoder gradients by L2 norm."""
    cfg = make_cfg()
    cfg.training.grad_clip = 5.0
    model = TwoParamModel()
    model.core.grad = torch.tensor([100.0, 1.0])
    model.decoder.grad = torch.tensor([6.0, 8.0])

    _clip_gradients(model, [model.decoder], cfg)

    assert model.core.grad.tolist() == [100.0, 1.0]
    assert model.decoder.grad.tolist() == pytest.approx([3.0, 4.0])


def test_clip_gradients_missing_mode_falls_back_to_norm():
    """Older configs without grad_clip_mode should use the current norm default."""
    cfg = make_cfg()
    delattr(cfg.training, "grad_clip_mode")
    cfg.training.grad_clip = 5.0
    model = TwoParamModel()
    model.core.grad = torch.tensor([100.0, 1.0])
    model.decoder.grad = torch.tensor([6.0, 8.0])

    _clip_gradients(model, [model.decoder], cfg)

    assert model.core.grad.tolist() == [100.0, 1.0]
    assert model.decoder.grad.tolist() == pytest.approx([3.0, 4.0])


def test_clip_gradients_can_use_infinity_norm_across_all_params():
    """Infinity-norm clipping should rescale all selected gradients together."""
    cfg = make_cfg()
    cfg.training.grad_clip = 10.0
    cfg.training.grad_clip_mode = "norm"
    cfg.training.grad_clip_scope = "all"
    cfg.training.grad_clip_norm_type = "inf"
    model = TwoParamModel()
    model.core.grad = torch.tensor([100.0, 1.0])
    model.decoder.grad = torch.tensor([-50.0, 0.5])

    _clip_gradients(model, [model.decoder], cfg)

    assert model.core.grad.tolist() == pytest.approx([10.0, 0.1])
    assert model.decoder.grad.tolist() == pytest.approx([-5.0, 0.05])


def test_clip_gradients_accepts_aliases_and_numeric_norm_type():
    """Gradient clipping should accept documented CLI-friendly aliases."""
    cfg = make_cfg()
    cfg.training.grad_clip = 5.0
    cfg.training.grad_clip_mode = "normclip"
    cfg.training.grad_clip_scope = "model"
    cfg.training.grad_clip_norm_type = "2.0"
    model = TwoParamModel()
    model.core.grad = torch.tensor([3.0, 4.0])
    model.decoder.grad = torch.tensor([0.0, 0.0])

    _clip_gradients(model, [model.decoder], cfg)

    assert model.core.grad.tolist() == pytest.approx([3.0, 4.0])
    assert model.decoder.grad.tolist() == pytest.approx([0.0, 0.0])

    cfg.training.grad_clip_mode = "valueclip"
    cfg.training.grad_clip_scope = "decoder"
    model.core.grad = torch.tensor([100.0, 1.0])
    model.decoder.grad = torch.tensor([100.0, -100.0])

    _clip_gradients(model, [model.decoder], cfg)

    assert model.core.grad.tolist() == [100.0, 1.0]
    assert model.decoder.grad.tolist() == [5.0, -5.0]


def test_clip_gradients_rejects_unknown_mode_and_scope():
    """Unsupported gradient clipping config should fail before optimizer.step."""
    cfg = make_cfg()
    model = TwoParamModel()
    model.core.grad = torch.tensor([1.0, 1.0])
    model.decoder.grad = torch.tensor([1.0, 1.0])

    cfg.training.grad_clip_mode = "median"
    with pytest.raises(ValueError, match="Unsupported gradient clipping mode"):
        _clip_gradients(model, [model.decoder], cfg)

    cfg.training.grad_clip_mode = "value"
    cfg.training.grad_clip_scope = "heads"
    with pytest.raises(ValueError, match="Unsupported gradient clipping scope"):
        _clip_gradients(model, [model.decoder], cfg)


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
    cfg.training.betas = [0.8, 0.95]
    cfg.training.adamw_eps = 1e-6
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))

    optimizer, decoder_params = build_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert decoder_params
    assert optimizer.defaults["betas"] == (0.8, 0.95)
    assert optimizer.defaults["eps"] == 1e-6


def test_build_optimizer_uses_adam_defaults_when_betas_is_null():
    """Adam-family optimizers should keep their defaults when betas is unset."""
    cfg = make_cfg()
    cfg.training.optimizer = "adamw"
    cfg.training.betas = None
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))

    optimizer, decoder_params = build_optimizer(model, cfg)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert decoder_params
    assert optimizer.defaults["betas"] == (0.9, 0.999)


def test_build_optimizer_can_switch_to_adam():
    """Optimizer builder should support Adam via config."""
    cfg = make_cfg()
    cfg.training.optimizer = "adam"
    cfg.training.betas = [0.85, 0.97]
    cfg.training.adamw_eps = 1e-7
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))

    optimizer, decoder_params = build_optimizer(model, cfg)

    assert type(optimizer) is torch.optim.Adam
    assert decoder_params
    assert optimizer.defaults["lr"] == cfg.training.lr
    assert optimizer.defaults["betas"] == (0.85, 0.97)
    assert optimizer.defaults["eps"] == 1e-7
    assert optimizer.defaults["weight_decay"] == 0.0
    assert [group["weight_decay"] for group in optimizer.param_groups] == [
        0.0,
        cfg.training.weight_decay,
    ]


def test_build_optimizer_can_switch_to_lion():
    """Optimizer builder should support Lion via config."""
    cfg = make_cfg()
    cfg.training.optimizer = "lion"
    cfg.training.betas = [0.8, 0.95]
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))

    optimizer, decoder_params = build_optimizer(model, cfg)

    assert isinstance(optimizer, Lion)
    assert decoder_params
    assert optimizer.defaults["lr"] == cfg.training.lr
    assert optimizer.defaults["betas"] == (0.8, 0.95)
    assert optimizer.defaults["weight_decay"] == 0.0
    assert [group["weight_decay"] for group in optimizer.param_groups] == [
        0.0,
        cfg.training.weight_decay,
    ]


def test_build_optimizer_uses_lion_defaults_when_betas_is_null():
    """Lion should use its own beta defaults when betas is unset."""
    cfg = make_cfg()
    cfg.training.optimizer = "lion"
    cfg.training.betas = None
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))

    optimizer, decoder_params = build_optimizer(model, cfg)

    assert isinstance(optimizer, Lion)
    assert decoder_params
    assert optimizer.defaults["betas"] == (0.9, 0.99)


def test_build_optimizer_rejects_old_adamw_betas_field():
    """The old AdamW-specific beta field should not be accepted."""
    cfg = make_cfg()
    cfg.training.optimizer = "adamw"
    cfg.training.adamw_betas = [0.8, 0.95]
    pc_ens = [PlaceCellEnsemble(8, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]
    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))

    with pytest.raises(ValueError, match="training.adamw_betas"):
        build_optimizer(model, cfg)


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


def _save_small_grid_checkpoint(tmp_path):
    """Save a small real GridCellsRNN checkpoint for post-hoc analysis tests."""
    cfg = make_cfg()
    cfg.task.n_pc = [3]
    cfg.task.pc_scale = [0.35]
    cfg.task.n_hdc = [2]
    cfg.task.seq_len = 3
    cfg.model.nh_lstm = 4
    cfg.model.nh_bottleneck = 5
    cfg.training.save_dir = str(tmp_path / "run")
    cfg.training.eval_num_samples = 2
    cfg.training.eval_loader_batch_size = 2
    cfg.training.eval_num_workers = 0
    cfg.training.eval_chunk_size = 2
    cfg.training.eval_units_per_page = 2
    cfg.training.eval_pdf_dpi = 50
    cfg.training.eval_plot_every = 0
    cfg.training.num_workers = 0

    pc_ens = [PlaceCellEnsemble(3, stdev=0.35, pos_min=-1.1, pos_max=1.1, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(2, concentration=20.0, seed=0)]
    pc_centers = np.array(
        [[-0.5, -0.25], [0.0, 0.25], [0.5, 0.75]],
        dtype=np.float32,
    )
    hdc_centers = np.array([0.25, 1.25], dtype=np.float32)
    pc_ens[0].means = pc_centers.copy()
    hdc_ens[0].means = hdc_centers.copy()

    model = GridCellsRNN(pc_ens, hdc_ens, **vars(cfg.model))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        model,
        optimizer,
        None,
        cfg,
        pc_ens,
        hdc_ens,
        epoch=7,
        global_step=11,
        filename="checkpoint_epoch_0007.pt",
    )
    return path, pc_centers, hdc_centers


def test_analyze_checkpoint_parse_args_allows_only_posthoc_overrides():
    """Checkpoint analysis CLI should expose only post-hoc-safe overrides."""
    args = analyze_checkpoint_module.parse_args(
        [
            "--checkpoint",
            "results/run/checkpoints/checkpoint_epoch_0007.pt",
            "--analysis.bottleneck_activation",
            "raw",
            "--analysis.compute_linear_spatial_pca",
            "true",
            "--training.eval_num_workers",
            "0",
            "--visualization.anim_step",
            "2",
        ]
    )

    assert args.checkpoint.endswith("checkpoint_epoch_0007.pt")
    assert args.analysis__bottleneck_activation == "raw"
    assert args.analysis__compute_linear_spatial_pca is True
    assert args.training__eval_num_workers == 0
    assert args.visualization__anim_step == 2

    with pytest.raises(SystemExit):
        analyze_checkpoint_module.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--model.nh_lstm",
                "64",
            ]
        )
    with pytest.raises(SystemExit):
        analyze_checkpoint_module.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--model.bottleneck_post_activation",
                "relu",
            ]
        )


def test_analyze_checkpoint_default_output_dir_uses_ckpt_analysis_folder():
    """Default post-hoc output should not overwrite training eval artifacts."""
    path = "results/run/checkpoints/checkpoint_epoch_0007.pt"

    assert analyze_checkpoint_module.default_output_dir(path) == os.path.join(
        "results",
        "run",
        "ckpt_analysis",
        "checkpoint_epoch_0007",
    )


def test_analyze_checkpoint_rebuilds_from_payload_and_runs_eval(monkeypatch, tmp_path):
    """Post-hoc analysis should load ckpt config, weights, centers, and epoch."""
    checkpoint_path, pc_centers, hdc_centers = _save_small_grid_checkpoint(tmp_path)
    old_payload = analyze_checkpoint_module.load_checkpoint_payload(checkpoint_path)
    old_payload["config"]["model"].pop("bottleneck_post_activation")
    torch.save(old_payload, checkpoint_path)
    output_dir = tmp_path / "posthoc"
    eval_path = tmp_path / "eval_override.npz"
    captured = {}

    def fake_build_eval_loader(cfg, logger, eval_data_path=None):
        captured["loader_eval_data_path"] = eval_data_path
        return "eval-loader"

    def fake_evaluate(model, pc_ens, hdc_ens, scorer, eval_loader, cfg, device, epoch, writer=None):
        captured["model"] = model
        captured["pc_centers"] = pc_ens[0].means.copy()
        captured["hdc_centers"] = hdc_ens[0].means.copy()
        captured["eval_loader"] = eval_loader
        captured["save_dir"] = cfg.training.save_dir
        captured["analysis_enabled"] = cfg.analysis.enabled
        captured["linear_spatial_pca"] = cfg.analysis.compute_linear_spatial_pca
        captured["eval_plot_every"] = cfg.training.eval_plot_every
        captured["anim_step"] = cfg.visualization.anim_step
        captured["epoch"] = epoch
        captured["writer"] = writer
        captured["bottleneck_post_activation"] = model.bottleneck_post_activation

    monkeypatch.setattr(analyze_checkpoint_module, "_build_eval_loader", fake_build_eval_loader)
    monkeypatch.setattr(analyze_checkpoint_module, "_evaluate", fake_evaluate)

    args = analyze_checkpoint_module.parse_args(
        [
            "--checkpoint",
            checkpoint_path,
            "--output_dir",
            str(output_dir),
            "--eval_data_path",
            str(eval_path),
            "--analysis.compute_linear_spatial_pca",
            "true",
            "--visualization.anim_step",
            "3",
        ]
    )

    returned_dir = analyze_checkpoint_module.analyze_checkpoint(
        args.checkpoint,
        args.output_dir,
        args.eval_data_path,
        args=args,
    )

    assert returned_dir == str(output_dir)
    assert captured["loader_eval_data_path"] == str(eval_path)
    assert captured["eval_loader"] == "eval-loader"
    assert captured["save_dir"] == str(output_dir)
    assert captured["analysis_enabled"] is True
    assert captured["linear_spatial_pca"] is True
    assert captured["eval_plot_every"] == 1
    assert captured["anim_step"] == 3
    assert captured["epoch"] == 7
    assert captured["writer"] is None
    assert captured["bottleneck_post_activation"] == "none"
    assert np.array_equal(captured["pc_centers"], pc_centers)
    assert np.array_equal(captured["hdc_centers"], hdc_centers)
    loaded_cfg = load_config(str(output_dir / "config.yaml"))
    assert loaded_cfg.training.save_dir == str(output_dir)
    assert loaded_cfg.training.eval_data_path == str(eval_path)


def test_analyze_checkpoint_programmatic_args_ignore_unsafe_overrides(tmp_path):
    """Programmatic calls should obey the same safe override allow-list as CLI."""
    checkpoint_path, _, _ = _save_small_grid_checkpoint(tmp_path)
    payload = analyze_checkpoint_module.load_checkpoint_payload(checkpoint_path)
    args = SimpleNamespace(
        checkpoint=checkpoint_path,
        output_dir=str(tmp_path / "posthoc"),
        eval_data_path=None,
        model__nh_lstm=99,
        model__bottleneck_post_activation="relu",
        task__env_size=9.9,
        analysis__compute_linear_spatial_pca=True,
    )

    cfg = analyze_checkpoint_module.build_analysis_config(payload, args)

    assert cfg.model.nh_lstm == payload["config"]["model"]["nh_lstm"]
    assert (
        cfg.model.bottleneck_post_activation
        == payload["config"]["model"]["bottleneck_post_activation"]
    )
    assert cfg.task.env_size == payload["config"]["task"]["env_size"]
    assert cfg.analysis.compute_linear_spatial_pca is True


def test_analyze_checkpoint_backfills_old_posthoc_config_fields(tmp_path):
    """Old schema-v1 checkpoints should accept newer safe post-hoc overrides."""
    checkpoint_path, _, _ = _save_small_grid_checkpoint(tmp_path)
    payload = analyze_checkpoint_module.load_checkpoint_payload(checkpoint_path)
    old_payload = copy.deepcopy(payload)
    old_analysis = old_payload["config"]["analysis"]
    missing_analysis_fields = (
        "bottleneck_activation",
        "compute_linear_spatial_pca",
        "linear_spatial_pca_top_k",
        "linear_spatial_pca_fit_fraction",
        "linear_spatial_pca_num_shuffles",
        "linear_spatial_pca_save_plots",
        "linear_spatial_pca_plot_top_n",
    )
    for field in missing_analysis_fields:
        old_analysis.pop(field)
    old_payload["config"]["training"].pop("eval_num_workers")
    old_payload["config"]["visualization"].pop("anim_step")

    args = SimpleNamespace(
        checkpoint=checkpoint_path,
        output_dir=str(tmp_path / "posthoc"),
        eval_data_path=None,
        model__nh_lstm=99,
        task__env_size=9.9,
        analysis__compute_linear_spatial_pca=True,
        analysis__linear_spatial_pca_top_k=7,
        analysis__linear_spatial_pca_save_plots=False,
        training__eval_num_workers=0,
        visualization__anim_step=3,
    )

    cfg = analyze_checkpoint_module.build_analysis_config(old_payload, args)

    assert cfg.analysis.compute_linear_spatial_pca is True
    assert cfg.analysis.bottleneck_activation == "relu"
    assert cfg.analysis.linear_spatial_pca_top_k == 7
    assert cfg.analysis.linear_spatial_pca_fit_fraction == 0.5
    assert cfg.analysis.linear_spatial_pca_num_shuffles == 0
    assert cfg.analysis.linear_spatial_pca_save_plots is False
    assert cfg.analysis.linear_spatial_pca_plot_top_n == 16
    assert cfg.training.eval_num_workers == 0
    assert cfg.visualization.anim_step == 3
    assert cfg.model.nh_lstm == payload["config"]["model"]["nh_lstm"]
    assert cfg.task.env_size == payload["config"]["task"]["env_size"]
    for field in missing_analysis_fields:
        assert field not in old_payload["config"]["analysis"]
    assert "eval_num_workers" not in old_payload["config"]["training"]
    assert "anim_step" not in old_payload["config"]["visualization"]


def test_analyze_checkpoint_rejects_invalid_payload(tmp_path):
    """Checkpoint analysis should fail clearly for unsupported payloads."""
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": 2}, path)

    with pytest.raises(ValueError, match="Unsupported checkpoint schema_version"):
        analyze_checkpoint_module.load_checkpoint_payload(str(path))


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
            "--task.pc_target_family",
            "difference_of_gaussians",
            "--task.pc_target_normalization",
            "global",
            "--task.pc_surround_scale",
            "2.0",
            "--model.nh_lstm",
            "64",
            "--model.bottleneck_post_activation",
            "relu",
            "--training.batch_size",
            "8",
            "--training.lr_scheduler",
            "cosine",
            "--training.lr_min",
            "1e-5",
            "--training.betas",
            "0.8",
            "0.95",
            "--training.first_pos_loss_multiplier",
            "50",
            "--training.grad_clip_mode",
            "norm",
            "--training.grad_clip_scope",
            "all",
            "--training.grad_clip_norm_type",
            "inf",
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
            "--analysis.bottleneck_activation",
            "raw",
            "--analysis.compute_linear_spatial_pca",
            "true",
            "--analysis.num_shuffles",
            "10",
            "--analysis.linear_spatial_pca_top_k",
            "8",
            "--analysis.linear_spatial_pca_fit_fraction",
            "0.6",
            "--analysis.linear_spatial_pca_num_shuffles",
            "3",
            "--analysis.linear_spatial_pca_save_plots",
            "false",
            "--analysis.linear_spatial_pca_plot_top_n",
            "6",
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
    assert args.task__pc_target_family == "difference_of_gaussians"
    assert args.task__pc_target_normalization == "global"
    assert args.task__pc_surround_scale == [2.0]
    assert args.model__nh_lstm == 64
    assert args.model__bottleneck_post_activation == "relu"
    assert args.training__batch_size == 8
    assert args.training__lr_scheduler == "cosine"
    assert args.training__lr_min == 1e-5
    assert args.training__betas == [0.8, 0.95]
    assert args.training__first_pos_loss_multiplier == 50.0
    assert args.training__grad_clip_mode == "norm"
    assert args.training__grad_clip_scope == "all"
    assert args.training__grad_clip_norm_type == "inf"
    assert args.training__datadir == "data/custom"
    assert args.training__checkpoint_every == 5
    assert args.training__save_final_checkpoint is False
    assert args.visualization__anim_fps == 30
    assert args.analysis__enabled is False
    assert args.analysis__compute_grid_selectivity is False
    assert args.analysis__compute_grid_geometry is False
    assert args.analysis__compute_shuffle_significance is False
    assert args.analysis__compute_split_half is False
    assert args.analysis__bottleneck_activation == "raw"
    assert args.analysis__compute_linear_spatial_pca is True
    assert args.analysis__num_shuffles == 10
    assert args.analysis__linear_spatial_pca_top_k == 8
    assert args.analysis__linear_spatial_pca_fit_fraction == 0.6
    assert args.analysis__linear_spatial_pca_num_shuffles == 3
    assert args.analysis__linear_spatial_pca_save_plots is False
    assert args.analysis__linear_spatial_pca_plot_top_n == 6
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
            "--task.pc_target_family",
            "true_difference_of_gaussians",
            "--task.pc_target_normalization",
            "none",
            "--task.pc_surround_scale",
            "2.5",
            "--model.dropout_rate",
            "0.25",
            "--model.bottleneck_post_activation",
            "tanh",
            "--training.batch_size",
            "16",
            "--training.lr_scheduler",
            "step",
            "--training.lr_step_size",
            "10",
            "--training.lr_gamma",
            "0.8",
            "--training.betas",
            "0.85",
            "0.97",
            "--training.first_pos_loss_multiplier",
            "25",
            "--training.grad_clip_mode",
            "value",
            "--training.grad_clip_scope",
            "decoder",
            "--training.grad_clip_norm_type",
            "2.0",
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
            "--analysis.bottleneck_activation",
            "raw",
            "--analysis.compute_linear_spatial_pca",
            "true",
            "--analysis.max_eval_trajectories",
            "16",
        ],
    )

    args = parse_train_args()

    assert args.task__env_size == 2.8
    assert args.task__cells_path == "data/cells.npz"
    assert args.task__decode_type == "weighted_mean"
    assert args.task__pc_target_family == "true_difference_of_gaussians"
    assert args.task__pc_target_normalization == "none"
    assert args.task__pc_surround_scale == [2.5]
    assert args.model__dropout_rate == 0.25
    assert args.model__bottleneck_post_activation == "tanh"
    assert args.training__batch_size == 16
    assert args.training__lr_scheduler == "step"
    assert args.training__lr_step_size == 10
    assert args.training__lr_gamma == 0.8
    assert args.training__betas == [0.85, 0.97]
    assert args.training__first_pos_loss_multiplier == 25.0
    assert args.training__grad_clip_mode == "value"
    assert args.training__grad_clip_scope == "decoder"
    assert args.training__grad_clip_norm_type == "2.0"
    assert args.training__datadir == "data/custom"
    assert args.training__checkpoint_every == 5
    assert args.training__save_final_checkpoint is False
    assert args.visualization__anim_step == 2
    assert args.analysis__enabled is False
    assert args.analysis__compute_hd_selectivity is False
    assert args.analysis__bottleneck_activation == "raw"
    assert args.analysis__compute_linear_spatial_pca is True
    assert args.analysis__max_eval_trajectories == 16


def test_parse_train_args_rejects_old_adamw_betas(monkeypatch):
    """Train CLI should reject the removed AdamW-specific beta override."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--training.adamw_betas",
            "0.8",
            "0.95",
        ],
    )

    with pytest.raises(SystemExit):
        parse_train_args()


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
        "target_pos": torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]],
            dtype=torch.float32,
        ),
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
    assert "eval/first_pos_mse" in writer.scalars
    assert "eval/first_pos_mse_ratio" in writer.scalars
    assert "eval/hd_mae_rad" in writer.scalars
    assert "eval/first_hd_mae_rad" in writer.scalars
    assert "eval/first_hd_mae_rad_ratio" in writer.scalars
    assert writer.scalars["eval/pos_mse"][1] == 2
    assert writer.scalars["eval/first_pos_mse"][1] == 2
    assert writer.scalars["eval/hd_mae_rad"][1] == 2
    assert writer.scalars["eval/first_hd_mae_rad"][1] == 2
    assert writer.scalars["eval/pos_mse"][0] > writer.scalars["eval/first_pos_mse"][0]
    assert writer.scalars["eval/first_pos_mse"][0] < 1e-6
    assert writer.scalars["eval/hd_mae_rad"][0] < 1e-6
    assert writer.scalars["eval/first_hd_mae_rad"][0] < 1e-6
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
    assert not list((tmp_path / "eval_stats").glob("linear_spatial_pca_*_0002*"))


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
    assert not list(stats_dir.glob("linear_spatial_pca_*_0003*"))


def test_evaluate_linear_spatial_pca_enabled_writes_independent_artifacts(
    monkeypatch,
    tmp_path,
):
    """Linear spatial-PCA analysis should use its own artifact contract."""
    cfg = make_cfg()
    cfg.training.save_dir = str(tmp_path)
    cfg.training.eval_num_workers = 0
    cfg.training.eval_chunk_size = 2
    cfg.training.eval_plot_every = 0
    cfg.analysis.enabled = True
    cfg.analysis.compute_linear_spatial_pca = True
    cfg.analysis.linear_spatial_pca_top_k = 2
    cfg.analysis.linear_spatial_pca_fit_fraction = 0.5
    cfg.analysis.linear_spatial_pca_num_shuffles = 1
    cfg.analysis.max_eval_trajectories = 2
    captured = {}
    pc_ens = [PlaceCellEnsemble(2, stdev=0.35, pos_min=-1.0, pos_max=1.0, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]

    def fake_linear_analyze(scorer, positions, activations, **kwargs):
        captured["positions_shape"] = positions.shape
        captured["activations_shape"] = activations.shape
        captured.update(kwargs)
        arrays = {
            "mode_id": np.array([0, 1]),
            "selected_rank": np.array([1, 2]),
            "fit_variance_explained": np.array([0.7, 0.2]),
            "fit_cumulative_variance_explained": np.array([0.7, 0.9]),
            "grid_score_60_fit": np.array([0.4, 0.1]),
            "grid_score_90_fit": np.array([0.2, 0.05]),
            "grid_score_60_test": np.array([0.5, 0.2]),
            "grid_score_90_test": np.array([0.3, 0.1]),
            "grid_scale_bins": np.array([4.0, np.nan]),
            "grid_scale_m": np.array([0.4, np.nan]),
            "grid_scale_valid": np.array([True, False]),
            "loading_participation_ratio": np.array([3.0, 4.0]),
            "mode_order_by_test_grid_score": np.array([0, 1]),
        }
        rows = [
            {field: arrays[field][mode_index].item() for field in LINEAR_SPATIAL_PCA_FIELDS}
            for mode_index in range(2)
        ]
        return {
            "fields": LINEAR_SPATIAL_PCA_FIELDS,
            "rows": rows,
            "arrays": arrays,
            "summary": {
                "version": "v1_spatial_pca_baseline",
                "T_real": 0.5,
                "p_value": 1.0,
                "top_k": 2,
            },
            "W_top": np.zeros((cfg.model.nh_bottleneck, 2)),
            "fit_center": np.zeros(cfg.model.nh_bottleneck),
            "fit_indices": np.array([0]),
            "test_indices": np.array([1]),
            "null_T": np.array([0.25]),
            "null_top_scores": np.array([[0.25, 0.1]]),
            "fit_ratemaps": np.zeros((2, 2, 2)),
            "test_ratemaps": np.zeros((2, 2, 2)),
            "test_sacs": np.zeros((2, 3, 3)),
            "mode_order_by_test_grid_score": np.array([0, 1]),
        }

    monkeypatch.setattr(
        "grid_cells.training.evaluation.analyze_bottleneck_spatial_stats",
        lambda *args, **kwargs: _minimal_analysis_result(),
    )
    monkeypatch.setattr(
        "grid_cells.training.evaluation.analyze_linear_spatial_pca_gridness",
        fake_linear_analyze,
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

    batch = _single_eval_batch()
    batch["ego_vel"] = torch.zeros(2, 3, 3)
    batch["target_pos"] = torch.zeros(2, 3, 2)
    batch["target_hd"] = torch.zeros(2, 3, 1)

    _evaluate(
        _AnalysisDummyModel(cfg),
        pc_ens,
        hdc_ens,
        _AnalysisDummyScorer(),
        [batch],
        cfg,
        torch.device("cpu"),
        epoch=5,
    )

    assert captured["positions_shape"] == (2, 3, 2)
    assert captured["activations_shape"] == (2, 3, cfg.model.nh_bottleneck)
    assert captured["top_k"] == 2
    assert captured["fit_fraction"] == 0.5
    assert captured["num_shuffles"] == 1
    stats_dir = tmp_path / "eval_stats"
    assert (stats_dir / "linear_spatial_pca_modes_epoch_0005.csv").exists()
    assert (stats_dir / "linear_spatial_pca_modes_epoch_0005.npz").exists()
    assert (stats_dir / "linear_spatial_pca_summary_epoch_0005.json").exists()
    assert (stats_dir / "linear_spatial_pca_overview_epoch_0005.pdf").exists()
    assert (stats_dir / "linear_spatial_pca_overview_epoch_0005.png").exists()
    assert (stats_dir / "linear_spatial_pca_mode_maps_epoch_0005.pdf").exists()
    with open(stats_dir / "linear_spatial_pca_modes_epoch_0005.csv") as handle:
        header = handle.readline().strip().split(",")
    assert header == LINEAR_SPATIAL_PCA_FIELDS
    with np.load(stats_dir / "linear_spatial_pca_modes_epoch_0005.npz") as payload:
        assert "grid_score_60_fit" in payload
        assert "loading_participation_ratio" in payload
        assert "mode_order_by_test_grid_score" in payload
        assert payload["fit_ratemaps"].shape == (2, 2, 2)


def test_evaluate_linear_spatial_pca_real_path_writes_summary(monkeypatch, tmp_path):
    """Eval should run the real linear spatial-PCA analyzer on collected tensors."""
    cfg = make_cfg()
    cfg.training.save_dir = str(tmp_path)
    cfg.training.eval_num_workers = 0
    cfg.training.eval_chunk_size = 2
    cfg.training.eval_plot_every = 0
    cfg.visualization.spatial_bins = 5
    cfg.analysis.enabled = True
    cfg.analysis.compute_linear_spatial_pca = True
    cfg.analysis.linear_spatial_pca_top_k = 1
    cfg.analysis.linear_spatial_pca_num_shuffles = 0
    cfg.analysis.max_eval_trajectories = 2
    pc_ens = [PlaceCellEnsemble(2, stdev=0.35, pos_min=-1.0, pos_max=1.0, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]

    monkeypatch.setattr(
        "grid_cells.training.evaluation.analyze_bottleneck_spatial_stats",
        lambda *args, **kwargs: _minimal_analysis_result(),
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

    batch = _single_eval_batch()
    batch["ego_vel"] = torch.zeros(2, 5, 3)
    batch["target_pos"] = torch.tensor(
        [
            [[-0.8, -0.8], [-0.4, -0.4], [0.0, 0.0], [0.4, 0.4], [0.8, 0.8]],
            [[-0.8, 0.8], [-0.4, 0.4], [0.0, 0.0], [0.4, -0.4], [0.8, -0.8]],
        ],
        dtype=torch.float32,
    )
    batch["target_hd"] = torch.zeros(2, 5, 1)

    _evaluate(
        _AnalysisDummyModel(cfg),
        pc_ens,
        hdc_ens,
        train_module.TrainingSession(cfg, hooks=None)._build_scorer(),
        [batch],
        cfg,
        torch.device("cpu"),
        epoch=6,
    )

    stats_dir = tmp_path / "eval_stats"
    assert (stats_dir / "linear_spatial_pca_modes_epoch_0006.csv").exists()
    summary_path = stats_dir / "linear_spatial_pca_summary_epoch_0006.json"
    assert summary_path.exists()
    with open(summary_path) as handle:
        summary = json.load(handle)
    assert summary["version"] == "v1_spatial_pca_baseline"
    assert summary["not_grid_score_optimized"] is True
    with np.load(stats_dir / "linear_spatial_pca_modes_epoch_0006.npz") as payload:
        assert payload["W_top"].shape == (cfg.model.nh_bottleneck, 1)
        assert payload["test_indices"].shape == (1,)
        assert payload["fit_ratemaps"].shape == (1, 5, 5)
        assert payload["test_ratemaps"].shape == (1, 5, 5)
        assert payload["test_sacs"].shape == (1, 9, 9)
    assert (stats_dir / "linear_spatial_pca_overview_epoch_0006.pdf").exists()
    assert (stats_dir / "linear_spatial_pca_overview_epoch_0006.png").exists()
    assert (stats_dir / "linear_spatial_pca_mode_maps_epoch_0006.pdf").exists()


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


class _SignedBottleneckModel(_AnalysisDummyModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.last_bottleneck = None

    def __call__(self, init_cond, ego_vel, training=False):
        batch, seq_len = ego_vel.shape[:2]
        pc_logits = [torch.zeros(batch, seq_len, 2, dtype=ego_vel.dtype)]
        hdc_logits = [torch.zeros(batch, seq_len, 4, dtype=ego_vel.dtype)]
        n_units = self.cfg.model.nh_bottleneck
        bottleneck = torch.linspace(
            -2.0,
            2.0,
            steps=batch * seq_len * n_units,
            dtype=ego_vel.dtype,
            device=ego_vel.device,
        ).reshape(batch, seq_len, n_units)
        lstm_acts = torch.zeros(
            batch,
            seq_len,
            self.cfg.model.nh_lstm,
            dtype=ego_vel.dtype,
            device=ego_vel.device,
        )
        self.last_bottleneck = bottleneck.detach().cpu().numpy()
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


class _CapturingAnalysisScorer(_AnalysisDummyScorer):
    def __init__(self, captured):
        self.captured = captured

    def accumulate_ratemaps(self, positions, activations, ratemap_sums, ratemap_counts):
        self.captured["ratemap_activations"] = activations.copy()
        super().accumulate_ratemaps(positions, activations, ratemap_sums, ratemap_counts)


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


def _run_model_returned_bottleneck_eval(monkeypatch, tmp_path, bottleneck_activation):
    cfg = make_cfg()
    cfg.training.save_dir = str(tmp_path)
    cfg.training.eval_num_workers = 0
    cfg.training.eval_chunk_size = 2
    cfg.training.eval_plot_every = 0
    cfg.analysis.enabled = True
    cfg.analysis.bottleneck_activation = bottleneck_activation
    cfg.analysis.compute_linear_spatial_pca = True
    cfg.analysis.max_eval_trajectories = 2
    captured = {}
    pc_ens = [PlaceCellEnsemble(2, stdev=0.35, pos_min=-1.0, pos_max=1.0, seed=0)]
    hdc_ens = [HeadDirectionCellEnsemble(4, concentration=20.0, seed=0)]

    def fake_spatial_analyze(scorer, positions, head_directions, activations, **kwargs):
        captured["spatial_stats_activations"] = activations.copy()
        return _minimal_analysis_result()

    def fake_linear_analyze(scorer, positions, activations, **kwargs):
        captured["linear_pca_activations"] = activations.copy()
        return {
            "summary": {
                "T_real": 0.0,
                "p_value": 1.0,
                "top_k": kwargs["top_k"],
            }
        }

    monkeypatch.setattr(
        "grid_cells.training.evaluation.analyze_bottleneck_spatial_stats",
        fake_spatial_analyze,
    )
    monkeypatch.setattr(
        "grid_cells.training.evaluation.analyze_linear_spatial_pca_gridness",
        fake_linear_analyze,
    )
    monkeypatch.setattr(
        "grid_cells.training.evaluation.save_spatial_stats_artifacts",
        lambda *args, **kwargs: {"csv": str(tmp_path / "grid_stats.csv")},
    )
    monkeypatch.setattr(
        "grid_cells.training.evaluation.save_linear_spatial_pca_artifacts",
        lambda *args, **kwargs: {"csv": str(tmp_path / "linear_spatial_pca.csv")},
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

    batch = _single_eval_batch()
    batch["init_pos"] = torch.zeros(2, 2)
    batch["init_hd"] = torch.zeros(2, 1)
    batch["ego_vel"] = torch.zeros(2, 3, 3)
    batch["target_pos"] = torch.zeros(2, 3, 2)
    batch["target_hd"] = torch.zeros(2, 3, 1)
    model = _SignedBottleneckModel(cfg)

    _evaluate(
        model,
        pc_ens,
        hdc_ens,
        _CapturingAnalysisScorer(captured),
        [batch],
        cfg,
        torch.device("cpu"),
        epoch=5,
    )

    captured["model_returned_bottleneck"] = model.last_bottleneck
    return captured


def test_evaluate_default_relu_bottleneck_activation_feeds_analysis_paths(
    monkeypatch,
    tmp_path,
):
    """Eval analysis consumers should see ReLU-transformed bottleneck activations."""
    captured = _run_model_returned_bottleneck_eval(monkeypatch, tmp_path, "relu")
    expected = np.maximum(captured["model_returned_bottleneck"], 0.0)

    assert captured["model_returned_bottleneck"].min() < 0.0
    assert captured["ratemap_activations"].min() >= 0.0
    np.testing.assert_allclose(captured["ratemap_activations"], expected)
    np.testing.assert_allclose(captured["spatial_stats_activations"], expected)
    np.testing.assert_allclose(captured["linear_pca_activations"], expected)


def test_evaluate_raw_bottleneck_activation_preserves_model_returned_bottleneck(
    monkeypatch,
    tmp_path,
):
    """The raw analysis mode should preserve model-returned bottleneck values."""
    captured = _run_model_returned_bottleneck_eval(monkeypatch, tmp_path, "raw")
    expected = captured["model_returned_bottleneck"]

    assert expected.min() < 0.0
    assert captured["ratemap_activations"].min() < 0.0
    np.testing.assert_allclose(captured["ratemap_activations"], expected)
    np.testing.assert_allclose(captured["spatial_stats_activations"], expected)
    np.testing.assert_allclose(captured["linear_pca_activations"], expected)


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
