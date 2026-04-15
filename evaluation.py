"""Evaluation orchestration for grid-cell training runs.

This module keeps the validation loop, score export, and media generation in a
single object so the training entrypoint can delegate evaluation without
changing its public function interface.

Usage:
    hooks = EvaluationHooks(...)
    evaluator = Evaluator(model, pc_ens, hdc_ens, scorer, eval_loader, cfg, device, hooks)
    evaluator.run(epoch=10, writer=writer)
"""

from dataclasses import dataclass
import logging
import os
import time
from typing import Callable, Optional

import numpy as np
import torch


@dataclass
class EvaluationHooks:
    """Inject helper functions so the train wrapper keeps monkeypatch compatibility."""

    encode_initial_conditions: Callable
    compute_position_mse: Callable
    decode_position_from_pc_logits: Callable
    get_scores_and_plot_from_ratemaps: Callable
    plot_hdc_tuning_curves: Callable
    generate_eval_animation: Callable
    score_ratemaps: Callable
    get_animation_setting: Callable[[str, object], object]


@dataclass
class EvaluationCollection:
    """Aggregated outputs collected during one evaluation pass."""

    ratemap_sums: np.ndarray
    ratemap_counts: np.ndarray
    eval_pos_mse_sum: float
    eval_pos_batches: int
    infer_seconds: float
    anim_data: Optional[dict]
    hd_list: list
    hdc_list: list


class Evaluator:
    """Run model evaluation, score rate maps, and export optional artifacts."""

    def __init__(
        self,
        model,
        pc_ens,
        hdc_ens,
        scorer,
        eval_loader,
        cfg,
        device,
        hooks: EvaluationHooks,
    ) -> None:
        self.model = model
        self.pc_ens = pc_ens
        self.hdc_ens = hdc_ens
        self.scorer = scorer
        self.eval_loader = eval_loader
        self.cfg = cfg
        self.device = device
        self.hooks = hooks
        self.logger = logging.getLogger("grid_cells")

    def run(self, epoch, writer=None) -> None:
        """Execute one full evaluation pass and write metrics/artifacts."""
        model_was_training = self.model.training
        self.model.eval()

        eval_start = time.time()
        try:
            save_pdf = self._should_save_pdf(epoch)
            collected = self._collect_batches(save_pdf)
            ratemaps = self.scorer.finalize_ratemaps(
                collected.ratemap_sums,
                collected.ratemap_counts,
            )

            score_start = time.time()
            score_60, score_90 = self._score_and_export(
                ratemaps,
                collected,
                save_pdf,
                epoch,
            )
            score_seconds = time.time() - score_start
            eval_seconds = time.time() - eval_start
            eval_pos_mse = collected.eval_pos_mse_sum / max(collected.eval_pos_batches, 1)

            self._log_metrics(
                epoch,
                eval_pos_mse,
                score_60,
                score_90,
                collected.infer_seconds,
                score_seconds,
                eval_seconds,
                save_pdf,
                writer,
            )
        finally:
            if model_was_training:
                self.model.train()

    def _should_save_pdf(self, epoch: int) -> bool:
        """Resolve whether this eval should emit PDFs and animations."""
        plot_every = getattr(self.cfg.training, "eval_plot_every", 1)
        eval_every = self.cfg.training.eval_every
        eval_index = epoch // max(1, eval_every)
        return (plot_every > 0) and (eval_index % plot_every == 0)

    def _collect_batches(self, save_pdf: bool) -> EvaluationCollection:
        """Run the model over the eval loader and gather all derived inputs."""
        ratemap_sums, ratemap_counts = self.scorer.allocate_ratemap_accumulators(
            self.cfg.model.nh_bottleneck
        )
        eval_start = time.time()
        eval_pos_mse_sum = 0.0
        eval_pos_batches = 0
        anim_data = None
        num_anim_traj = int(self.hooks.get_animation_setting("anim_num_traj", 4))
        hd_list = []
        hdc_list = []

        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                init_cond = self.hooks.encode_initial_conditions(
                    batch,
                    self.pc_ens,
                    self.hdc_ens,
                ).to(self.device)
                pc_logits, hdc_logits, bottleneck, _ = self.model(
                    init_cond,
                    batch["ego_vel"],
                    training=False,
                )
                pos_mse = self.hooks.compute_position_mse(
                    pc_logits,
                    batch["target_pos"],
                    self.pc_ens,
                )
                eval_pos_mse_sum += float(pos_mse.item())
                eval_pos_batches += 1
                self.scorer.accumulate_ratemaps(
                    batch["target_pos"].detach().cpu().numpy(),
                    bottleneck.detach().cpu().numpy(),
                    ratemap_sums,
                    ratemap_counts,
                )

                if save_pdf:
                    hd_np = batch["target_hd"].detach().cpu().numpy()
                    hdc_np = torch.softmax(hdc_logits[0], dim=-1).detach().cpu().numpy()
                    hd_list.append(hd_np.reshape(-1))
                    hdc_list.append(hdc_np.reshape(-1, hdc_np.shape[-1]))

                if save_pdf and anim_data is None:
                    anim_data = self._build_animation_batch(
                        batch,
                        pc_logits,
                        hdc_logits,
                        num_anim_traj,
                    )

        return EvaluationCollection(
            ratemap_sums=ratemap_sums,
            ratemap_counts=ratemap_counts,
            eval_pos_mse_sum=eval_pos_mse_sum,
            eval_pos_batches=eval_pos_batches,
            infer_seconds=time.time() - eval_start,
            anim_data=anim_data,
            hd_list=hd_list,
            hdc_list=hdc_list,
        )

    def _build_animation_batch(self, batch, pc_logits, hdc_logits, num_anim_traj: int) -> dict:
        """Prepare the first-batch animation payload used for eval exports."""
        n_anim = min(num_anim_traj, batch["target_pos"].shape[0])
        pred_pos_batch = self.hooks.decode_position_from_pc_logits(pc_logits, self.pc_ens)
        pc_probs = torch.cat([torch.softmax(logits, dim=-1) for logits in pc_logits], dim=-1)
        hdc_probs = torch.cat([torch.softmax(logits, dim=-1) for logits in hdc_logits], dim=-1)
        return {
            "target_pos": batch["target_pos"][:n_anim].detach().cpu().numpy(),
            "pred_pos": pred_pos_batch[:n_anim].detach().cpu().numpy(),
            "pc_acts": pc_probs[:n_anim].detach().cpu().numpy(),
            "hdc_acts": hdc_probs[:n_anim].detach().cpu().numpy(),
        }

    def _score_and_export(self, ratemaps, collected: EvaluationCollection, save_pdf: bool, epoch: int):
        """Compute scores and optionally export PDF/HDC/animation artifacts."""
        num_workers = getattr(self.cfg.training, "eval_num_workers", 0)
        chunk_size = getattr(self.cfg.training, "eval_chunk_size", 32)

        if not save_pdf:
            score_60, score_90, _, _, _ = self.hooks.score_ratemaps(
                self.scorer,
                ratemaps,
                num_workers=num_workers,
                chunk_size=chunk_size,
            )
            return score_60, score_90

        save_dir = self.cfg.training.save_dir
        filename = f"rates_and_sac_epoch_{epoch:04d}.pdf"
        scores = self.hooks.get_scores_and_plot_from_ratemaps(
            self.scorer,
            ratemaps,
            save_dir,
            filename,
            num_workers=num_workers,
            chunk_size=chunk_size,
            units_per_page=getattr(self.cfg.training, "eval_units_per_page", 128),
            pdf_dpi=getattr(self.cfg.training, "eval_pdf_dpi", 72),
        )
        score_60 = scores[0]
        score_90 = scores[1]

        self._export_hdc_tuning_curves(collected, save_dir, epoch)
        self._export_animation(collected.anim_data, save_dir, epoch)
        return score_60, score_90

    def _export_hdc_tuning_curves(
        self,
        collected: EvaluationCollection,
        save_dir: str,
        epoch: int,
    ) -> None:
        """Write directional tuning curves when PDF export is enabled."""
        if not collected.hd_list:
            return

        try:
            hdc_pdf_path = os.path.join(save_dir, f"hdc_tuning_epoch_{epoch:04d}.pdf")
            self.hooks.plot_hdc_tuning_curves(
                np.concatenate(collected.hd_list),
                np.concatenate(collected.hdc_list, axis=0),
                n_bins=getattr(self.cfg.visualization, "directional_bins", 20),
                save_path=hdc_pdf_path,
                pdf_dpi=getattr(self.cfg.training, "eval_pdf_dpi", 100),
            )
            self.logger.info("HDC tuning curves saved to %s", hdc_pdf_path)
        except Exception as exc:
            self.logger.warning("HDC tuning curve plot failed: %s", exc)

    def _export_animation(self, anim_data: Optional[dict], save_dir: str, epoch: int) -> None:
        """Write the eval animation when the first batch payload is available."""
        if anim_data is None:
            return

        anim_path = os.path.join(save_dir, f"eval_animation_epoch_{epoch:04d}.mp4")
        pc_centers = np.concatenate([ens.means for ens in self.pc_ens], axis=0)
        hdc_centers = np.concatenate([ens.means.reshape(-1) for ens in self.hdc_ens], axis=0)
        try:
            self.hooks.generate_eval_animation(
                anim_data["target_pos"],
                anim_data["pred_pos"],
                anim_data["pc_acts"],
                anim_data["hdc_acts"],
                pc_centers,
                hdc_centers,
                self.cfg.task.env_size,
                anim_path,
                fps=int(self.hooks.get_animation_setting("anim_fps", 20)),
                step=int(self.hooks.get_animation_setting("anim_step", 4)),
                num_workers=int(self.hooks.get_animation_setting("anim_workers", 4)),
            )
        except Exception as exc:
            self.logger.warning("eval animation failed: %s", exc)

    def _log_metrics(
        self,
        epoch: int,
        eval_pos_mse: float,
        score_60,
        score_90,
        infer_seconds: float,
        score_seconds: float,
        eval_seconds: float,
        save_pdf: bool,
        writer,
    ) -> None:
        """Emit logs and optional TensorBoard scalars for one eval pass."""
        self.logger.info(
            "eval epoch=%d  pos_mse=%.6f  grid_score_60 max=%.4f  grid_score_90 max=%.4f"
            "  infer=%.1fs  score=%.1fs  total=%.1fs  pdf=%s",
            epoch,
            eval_pos_mse,
            float(score_60.max()),
            float(score_90.max()),
            infer_seconds,
            score_seconds,
            eval_seconds,
            save_pdf,
        )

        if writer is None:
            return

        writer.add_scalar("eval/pos_mse", eval_pos_mse, epoch)
        writer.add_scalar("eval/grid_score_60_max", float(score_60.max()), epoch)
        writer.add_scalar("eval/grid_score_90_max", float(score_90.max()), epoch)
        writer.add_scalar("eval/seconds", eval_seconds, epoch)
        writer.add_scalar("eval/infer_seconds", infer_seconds, epoch)
        writer.add_scalar("eval/score_seconds", score_seconds, epoch)
