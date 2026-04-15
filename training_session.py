"""Training-session orchestration for grid-cell experiments.

This module encapsulates the experiment lifecycle so ``train.py`` can remain a
thin public entrypoint while preserving the existing helper functions and CLI.

Usage:
    session = TrainingSession(cfg, hooks=hooks)
    session.run()
"""

from dataclasses import dataclass
import os
import time
from typing import Callable

import numpy as np
import torch

from dataset import get_dataloader
from model import GridCellsRNN
from scores import GridScorer
from utils import (
    compute_position_mse,
    encode_initial_conditions,
    encode_targets,
    get_head_direction_ensembles,
    get_place_cell_ensembles,
)


@dataclass
class TrainingSessionHooks:
    """Inject train-module helpers so the public wrappers keep current behavior."""

    resolve_save_dir: Callable
    setup_logger: Callable
    create_summary_writer: Callable
    build_optimizer: Callable
    build_train_loader: Callable
    build_eval_loader: Callable
    evaluate: Callable
    get_step_log_interval: Callable
    tqdm: object


class TrainingSession:
    """Run one end-to-end training session including periodic evaluation."""

    def __init__(self, cfg, hooks: TrainingSessionHooks, data_path: str = None, eval_data_path: str = None):
        self.cfg = cfg
        self.hooks = hooks
        self.data_path = data_path
        self.eval_data_path = eval_data_path

    def run(self) -> None:
        """Execute the full training loop described by the config."""
        self.cfg.training.save_dir = self.hooks.resolve_save_dir(
            self.cfg.training.save_dir,
            timestamp_save_dir=getattr(self.cfg.training, "timestamp_save_dir", True),
            run_name=getattr(self.cfg.training, "run_name", None),
        )
        logger = self.hooks.setup_logger(self.cfg.training.save_dir)
        writer = self.hooks.create_summary_writer(self.cfg, logger)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Using device: %s", device)
        logger.info("Optimizer: %s", getattr(self.cfg.training, "optimizer", "rmsprop"))
        logger.info("Run directory: %s", self.cfg.training.save_dir)
        logger.info(
            "Progress bar enabled: %s",
            getattr(self.cfg.training, "use_tqdm", True),
        )
        logger.info(
            "Eval workers=%s  chunk_size=%s  units_per_page=%s",
            getattr(self.cfg.training, "eval_num_workers", 0),
            getattr(self.cfg.training, "eval_chunk_size", 32),
            getattr(self.cfg.training, "eval_units_per_page", 128),
        )

        pc_ens = get_place_cell_ensembles(self.cfg)
        hdc_ens = get_head_direction_ensembles(self.cfg)
        model = GridCellsRNN(pc_ens, hdc_ens, **vars(self.cfg.model)).to(device)
        optimizer, decoder_params = self.hooks.build_optimizer(model, self.cfg)

        if getattr(self.cfg.training, "use_tqdm", True) and self.hooks.tqdm is None:
            logger.warning("tqdm is not installed; falling back to plain logs")

        scorer = self._build_scorer()
        os.makedirs(self.cfg.training.save_dir, exist_ok=True)

        fixed_loader = self.hooks.build_train_loader(
            self.cfg,
            logger,
            pc_ens,
            hdc_ens,
            data_path=self.data_path,
        )
        fixed_eval_loader = self.hooks.build_eval_loader(
            self.cfg,
            logger,
            eval_data_path=self.eval_data_path,
        )

        global_step = 0
        self.hooks.evaluate(
            model,
            pc_ens,
            hdc_ens,
            scorer,
            fixed_eval_loader,
            self.cfg,
            device,
            epoch=0,
            writer=writer,
        )

        try:
            for epoch in range(self.cfg.training.epochs):
                global_step = self._run_epoch(
                    epoch=epoch,
                    global_step=global_step,
                    logger=logger,
                    writer=writer,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    decoder_params=decoder_params,
                    pc_ens=pc_ens,
                    hdc_ens=hdc_ens,
                    scorer=scorer,
                    fixed_loader=fixed_loader,
                    fixed_eval_loader=fixed_eval_loader,
                )
        finally:
            if writer is not None:
                writer.close()

    def _build_scorer(self) -> GridScorer:
        """Create the grid scorer used during evaluation."""
        nbins = getattr(getattr(self.cfg, "visualization", None), "spatial_bins", 20)
        starts = [0.2] * 10
        ends = list(np.linspace(0.4, 1.0, num=10))
        masks_params = list(zip(starts, ends))
        return GridScorer(
            nbins,
            [
                [-self.cfg.task.env_size / 2, self.cfg.task.env_size / 2],
                [-self.cfg.task.env_size / 2, self.cfg.task.env_size / 2],
            ],
            masks_params,
        )

    def _run_epoch(
        self,
        epoch: int,
        global_step: int,
        logger,
        writer,
        device,
        model,
        optimizer,
        decoder_params,
        pc_ens,
        hdc_ens,
        scorer,
        fixed_loader,
        fixed_eval_loader,
    ) -> int:
        """Run one training epoch and trigger evaluation if configured."""
        dataloader = fixed_loader if fixed_loader is not None else get_dataloader(self.cfg)
        num_steps = len(dataloader)
        step_log_interval = max(
            1,
            getattr(self.cfg.training, "tensorboard_log_every", 0)
            or self.hooks.get_step_log_interval(num_steps),
        )
        model.train()
        loss_acc = []
        pos_mse_acc = []
        epoch_start = time.time()

        use_tqdm = getattr(self.cfg.training, "use_tqdm", True) and self.hooks.tqdm is not None
        progress = (
            self.hooks.tqdm(
                dataloader,
                total=num_steps,
                desc=f"epoch {epoch + 1}/{self.cfg.training.epochs}",
                leave=False,
                dynamic_ncols=True,
            )
            if use_tqdm
            else dataloader
        )

        for step, batch in enumerate(progress):
            optimizer.zero_grad()
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if "init_cond" in batch:
                init_cond = batch["init_cond"].float()
                pc_targets = [batch[f"pc_targets_{i}"] for i in range(len(pc_ens))]
                hdc_targets = [batch[f"hdc_targets_{i}"] for i in range(len(hdc_ens))]
            else:
                init_cond = encode_initial_conditions(batch, pc_ens, hdc_ens).to(device)
                pc_targets, hdc_targets = encode_targets(batch, pc_ens, hdc_ens)

            pc_logits, hdc_logits, _, _ = model(init_cond, batch["ego_vel"], training=True)
            pos_mse = compute_position_mse(pc_logits, batch["target_pos"], pc_ens)

            loss = sum(
                ens.loss(logits, targets)
                for ens, logits, targets in zip(pc_ens, pc_logits, pc_targets)
            )
            loss += sum(
                ens.loss(logits, targets)
                for ens, logits, targets in zip(hdc_ens, hdc_logits, hdc_targets)
            )

            loss.backward()
            torch.nn.utils.clip_grad_value_(decoder_params, self.cfg.training.grad_clip)
            optimizer.step()

            loss_value = loss.item()
            pos_mse_value = float(pos_mse.item())
            loss_acc.append(loss_value)
            pos_mse_acc.append(pos_mse_value)
            global_step += 1

            if use_tqdm:
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    avg=f"{np.mean(loss_acc):.4f}",
                    pos_mse=f"{pos_mse_value:.6f}",
                )

            should_log_step = ((step + 1) % step_log_interval == 0) or (step == num_steps - 1)
            if writer is not None and should_log_step:
                writer.add_scalar("train/loss_step", loss_value, global_step)
                writer.add_scalar("train/pos_mse_step", pos_mse_value, global_step)

        epoch_mean = float(np.mean(loss_acc))
        epoch_std = float(np.std(loss_acc))
        epoch_pos_mse = float(np.mean(pos_mse_acc))
        epoch_time = time.time() - epoch_start

        logger.info(
            "epoch=%4d  loss mean=%.4f  std=%.4f  pos_mse=%.6f  seconds=%.1f",
            epoch,
            epoch_mean,
            epoch_std,
            epoch_pos_mse,
            epoch_time,
        )

        if writer is not None:
            writer.add_scalar("train/loss_mean", epoch_mean, epoch)
            writer.add_scalar("train/loss_std", epoch_std, epoch)
            writer.add_scalar("train/pos_mse_mean", epoch_pos_mse, epoch)
            writer.add_scalar("train/epoch_seconds", epoch_time, epoch)

        if (epoch + 1) % self.cfg.training.eval_every == 0:
            self.hooks.evaluate(
                model,
                pc_ens,
                hdc_ens,
                scorer,
                fixed_eval_loader,
                self.cfg,
                device,
                epoch + 1,
                writer=writer,
            )

        return global_step
