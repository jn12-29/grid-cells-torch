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

from grid_cells.analysis.linear_spatial_pca import (
    analyze_linear_spatial_pca_gridness,
    save_linear_spatial_pca_artifacts,
)
from grid_cells.analysis.spatial_stats import (
    analyze_bottleneck_spatial_stats,
    save_spatial_stats_artifacts,
)
from grid_cells.cells.decoding import logits_to_cell_activations


_METRIC_RATIO_EPS = 1e-12
_BOTTLENECK_ACTIVATION_MODES = ("raw", "relu")


def _metric_ratio(numerator: float, denominator: float) -> float:
    """Return a stable diagnostic ratio for nonnegative scalar metrics."""
    return numerator / max(denominator, _METRIC_RATIO_EPS)


def _normalize_bottleneck_activation(value) -> str:
    """Return the configured bottleneck activation mode for analysis artifacts."""
    mode = str(value).lower()
    if mode not in _BOTTLENECK_ACTIVATION_MODES:
        expected = ", ".join(repr(item) for item in _BOTTLENECK_ACTIVATION_MODES)
        raise ValueError(
            f"Unsupported analysis.bottleneck_activation={value!r}; "
            f"expected one of: {expected}."
        )
    return mode


@dataclass
class EvaluationHooks:
    """Inject helper functions so the train wrapper keeps monkeypatch compatibility."""

    encode_initial_conditions: Callable
    compute_position_mse: Callable
    compute_head_direction_mae_rad: Callable
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
    eval_first_pos_mse_sum: float
    eval_first_pos_batches: int
    eval_hd_mae_sum: float
    eval_hd_batches: int
    eval_first_hd_mae_sum: float
    eval_first_hd_batches: int
    infer_seconds: float
    anim_data: Optional[dict]
    hd_list: list
    hdc_list: list
    analysis_pos_list: list
    analysis_hd_list: list
    analysis_bottleneck_list: list


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
            eval_first_pos_mse = collected.eval_first_pos_mse_sum / max(
                collected.eval_first_pos_batches,
                1,
            )
            eval_hd_mae = collected.eval_hd_mae_sum / max(collected.eval_hd_batches, 1)
            eval_first_hd_mae = collected.eval_first_hd_mae_sum / max(
                collected.eval_first_hd_batches,
                1,
            )

            self._log_metrics(
                epoch,
                eval_pos_mse,
                eval_first_pos_mse,
                eval_hd_mae,
                eval_first_hd_mae,
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
        eval_first_pos_mse_sum = 0.0
        eval_first_pos_batches = 0
        eval_hd_mae_sum = 0.0
        eval_hd_batches = 0
        eval_first_hd_mae_sum = 0.0
        eval_first_hd_batches = 0
        anim_data = None
        num_anim_traj = int(self.hooks.get_animation_setting("anim_num_traj", 4))
        hd_list = []
        hdc_list = []
        analysis_pos_list = []
        analysis_hd_list = []
        analysis_bottleneck_list = []
        analysis_enabled = self._analysis_enabled()
        bottleneck_activation = self._bottleneck_activation()
        max_analysis_traj = self._max_analysis_trajectories()
        analysis_traj_count = 0

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
                analysis_bottleneck = self._apply_bottleneck_activation(
                    bottleneck,
                    bottleneck_activation,
                )
                analysis_bottleneck_np = analysis_bottleneck.detach().cpu().numpy()
                pos_mse = self.hooks.compute_position_mse(
                    pc_logits,
                    batch["target_pos"],
                    self.pc_ens,
                )
                first_pos_mse = self.hooks.compute_position_mse(
                    [logits[:, :1, :] for logits in pc_logits],
                    batch["target_pos"][:, :1, :],
                    self.pc_ens,
                )
                hd_mae = self.hooks.compute_head_direction_mae_rad(
                    hdc_logits,
                    batch["target_hd"],
                    self.hdc_ens,
                )
                first_hd_mae = self.hooks.compute_head_direction_mae_rad(
                    [logits[:, :1, :] for logits in hdc_logits],
                    batch["target_hd"][:, :1, :],
                    self.hdc_ens,
                )
                eval_pos_mse_sum += float(pos_mse.item())
                eval_pos_batches += 1
                eval_first_pos_mse_sum += float(first_pos_mse.item())
                eval_first_pos_batches += 1
                eval_hd_mae_sum += float(hd_mae.item())
                eval_hd_batches += 1
                eval_first_hd_mae_sum += float(first_hd_mae.item())
                eval_first_hd_batches += 1
                self.scorer.accumulate_ratemaps(
                    batch["target_pos"].detach().cpu().numpy(),
                    analysis_bottleneck_np,
                    ratemap_sums,
                    ratemap_counts,
                )

                if analysis_enabled and analysis_traj_count < max_analysis_traj:
                    remaining = max_analysis_traj - analysis_traj_count
                    take = min(remaining, batch["target_pos"].shape[0])
                    if take > 0:
                        analysis_pos_list.append(
                            batch["target_pos"][:take].detach().cpu().numpy()
                        )
                        analysis_hd_list.append(
                            batch["target_hd"][:take].detach().cpu().numpy()
                        )
                        analysis_bottleneck_list.append(
                            analysis_bottleneck_np[:take]
                        )
                        analysis_traj_count += take

                if save_pdf:
                    hd_np = batch["target_hd"].detach().cpu().numpy()
                    hdc_np = logits_to_cell_activations(
                        hdc_logits[0],
                        self.hdc_ens[0],
                    ).detach().cpu().numpy()
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
            eval_first_pos_mse_sum=eval_first_pos_mse_sum,
            eval_first_pos_batches=eval_first_pos_batches,
            eval_hd_mae_sum=eval_hd_mae_sum,
            eval_hd_batches=eval_hd_batches,
            eval_first_hd_mae_sum=eval_first_hd_mae_sum,
            eval_first_hd_batches=eval_first_hd_batches,
            infer_seconds=time.time() - eval_start,
            anim_data=anim_data,
            hd_list=hd_list,
            hdc_list=hdc_list,
            analysis_pos_list=analysis_pos_list,
            analysis_hd_list=analysis_hd_list,
            analysis_bottleneck_list=analysis_bottleneck_list,
        )

    def _analysis_enabled(self) -> bool:
        """Resolve whether statistical analysis artifacts should be exported."""
        analysis_cfg = getattr(self.cfg, "analysis", None)
        return bool(getattr(analysis_cfg, "enabled", False))

    def _bottleneck_activation(self) -> str:
        """Resolve the bottleneck transform used by eval analysis artifacts."""
        analysis_cfg = getattr(self.cfg, "analysis", None)
        return _normalize_bottleneck_activation(
            getattr(analysis_cfg, "bottleneck_activation", "relu")
        )

    @staticmethod
    def _apply_bottleneck_activation(bottleneck: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply the configured analysis-only bottleneck transform."""
        if mode == "relu":
            return torch.relu(bottleneck)
        if mode == "raw":
            return bottleneck
        raise ValueError(f"Unsupported bottleneck activation mode: {mode!r}")

    def _max_analysis_trajectories(self) -> int:
        """Return the configured cap on eval trajectories retained for analysis."""
        analysis_cfg = getattr(self.cfg, "analysis", None)
        return max(0, int(getattr(analysis_cfg, "max_eval_trajectories", 0)))

    def _artifact_dir(self, name: str) -> str:
        """Return a shallow eval artifact directory under the current run."""
        return os.path.join(self.cfg.training.save_dir, name)

    def _build_animation_batch(self, batch, pc_logits, hdc_logits, num_anim_traj: int) -> dict:
        """Prepare the first-batch animation payload used for eval exports."""
        n_anim = min(num_anim_traj, batch["target_pos"].shape[0])
        pred_pos_batch = self.hooks.decode_position_from_pc_logits(pc_logits, self.pc_ens)
        pc_acts = torch.cat(
            [
                logits_to_cell_activations(logits, ensemble)
                for logits, ensemble in zip(pc_logits, self.pc_ens)
            ],
            dim=-1,
        )
        hdc_acts = torch.cat(
            [
                logits_to_cell_activations(logits, ensemble)
                for logits, ensemble in zip(hdc_logits, self.hdc_ens)
            ],
            dim=-1,
        )
        return {
            "target_pos": batch["target_pos"][:n_anim].detach().cpu().numpy(),
            "pred_pos": pred_pos_batch[:n_anim].detach().cpu().numpy(),
            "pc_acts": pc_acts[:n_anim].detach().cpu().numpy(),
            "hdc_acts": hdc_acts[:n_anim].detach().cpu().numpy(),
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
            self._export_spatial_stats(collected, epoch)
            return score_60, score_90

        save_dir = self._artifact_dir("eval_plots")
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

        self._export_spatial_stats(collected, epoch)
        self._export_hdc_tuning_curves(collected, save_dir, epoch)
        self._export_animation(collected.anim_data, epoch)
        return score_60, score_90

    def _export_spatial_stats(
        self,
        collected: EvaluationCollection,
        epoch: int,
    ) -> None:
        """Write statistical grid/HD artifacts when analysis is enabled."""
        if not self._analysis_enabled():
            return
        if not collected.analysis_pos_list:
            self.logger.warning("Grid-cell statistical analysis skipped: no eval data collected")
            return

        analysis_cfg = getattr(self.cfg, "analysis", None)
        result = analyze_bottleneck_spatial_stats(
            self.scorer,
            np.concatenate(collected.analysis_pos_list, axis=0),
            np.concatenate(collected.analysis_hd_list, axis=0),
            np.concatenate(collected.analysis_bottleneck_list, axis=0),
            n_direction_bins=getattr(self.cfg.visualization, "directional_bins", 20),
            num_shuffles=int(getattr(analysis_cfg, "num_shuffles", 200)),
            scale_discreteness_num_shuffles=int(
                getattr(analysis_cfg, "scale_discreteness_num_shuffles", 500)
            ),
            scale_discreteness_bins=int(
                getattr(analysis_cfg, "scale_discreteness_bins", 13)
            ),
            scale_discreteness_min_bins=float(
                getattr(analysis_cfg, "scale_discreteness_min_bins", 10.0)
            ),
            scale_discreteness_max_bins=float(
                getattr(analysis_cfg, "scale_discreteness_max_bins", 36.0)
            ),
            scale_gmm_max_components=int(
                getattr(analysis_cfg, "scale_gmm_max_components", 8)
            ),
            gridness_threshold=float(getattr(analysis_cfg, "gridness_threshold", 0.37)),
            fdr_alpha=float(getattr(analysis_cfg, "fdr_alpha", 0.05)),
            min_shift_fraction=float(getattr(analysis_cfg, "min_shift_fraction", 0.1)),
            split_half_min_corr=float(
                getattr(analysis_cfg, "split_half_min_corr", 0.3)
            ),
            random_seed=int(getattr(analysis_cfg, "random_seed", 0)),
            compute_grid_selectivity=bool(
                getattr(
                    analysis_cfg,
                    "compute_grid_selectivity",
                    getattr(analysis_cfg, "compute_grid_scores", True),
                )
            ),
            compute_grid_geometry=bool(getattr(analysis_cfg, "compute_grid_geometry", True)),
            compute_shuffle_significance=bool(
                getattr(analysis_cfg, "compute_shuffle_significance", True)
            ),
            compute_split_half=bool(getattr(analysis_cfg, "compute_split_half", True)),
            compute_hd_selectivity=bool(
                getattr(analysis_cfg, "compute_hd_selectivity", True)
            ),
        )
        paths = save_spatial_stats_artifacts(
            result,
            self._artifact_dir("eval_stats"),
            epoch,
        )
        summary = result["summary"]
        self.logger.info(
            "grid stats saved to %s  grid_fdr=%d  grid_threshold=%d  "
            "reliable_grid=%d  hd_selective=%d  grid_and_hd=%d",
            paths["csv"],
            summary["n_grid_fdr"],
            summary["n_grid_threshold"],
            summary["n_reliable_grid"],
            summary["n_hd_selective"],
            summary["n_grid_and_hd"],
        )
        self._export_linear_spatial_pca(collected, epoch)

    def _linear_spatial_pca_enabled(self) -> bool:
        """Resolve whether linear spatial-PCA mode analysis should run."""
        analysis_cfg = getattr(self.cfg, "analysis", None)
        return bool(getattr(analysis_cfg, "enabled", False)) and bool(
            getattr(analysis_cfg, "compute_linear_spatial_pca", False)
        )

    def _export_linear_spatial_pca(
        self,
        collected: EvaluationCollection,
        epoch: int,
    ) -> None:
        """Write linear spatial-PCA gridness artifacts when enabled."""
        if not self._linear_spatial_pca_enabled():
            return

        analysis_cfg = getattr(self.cfg, "analysis", None)
        result = analyze_linear_spatial_pca_gridness(
            self.scorer,
            np.concatenate(collected.analysis_pos_list, axis=0),
            np.concatenate(collected.analysis_bottleneck_list, axis=0),
            top_k=int(getattr(analysis_cfg, "linear_spatial_pca_top_k", 16)),
            fit_fraction=float(
                getattr(analysis_cfg, "linear_spatial_pca_fit_fraction", 0.5)
            ),
            num_shuffles=int(
                getattr(analysis_cfg, "linear_spatial_pca_num_shuffles", 0)
            ),
            min_shift_fraction=float(getattr(analysis_cfg, "min_shift_fraction", 0.1)),
            random_seed=int(getattr(analysis_cfg, "random_seed", 0)),
        )
        paths = save_linear_spatial_pca_artifacts(
            result,
            self._artifact_dir("eval_stats"),
            epoch,
            save_plots=bool(
                getattr(analysis_cfg, "linear_spatial_pca_save_plots", True)
            ),
            plot_top_n=int(getattr(analysis_cfg, "linear_spatial_pca_plot_top_n", 16)),
            pdf_dpi=getattr(self.cfg.training, "eval_pdf_dpi", 100),
        )
        summary = result["summary"]
        self.logger.info(
            "linear spatial PCA stats saved to %s  T_real=%s  p=%s  top_k=%d",
            paths["csv"],
            summary["T_real"],
            summary["p_value"],
            summary["top_k"],
        )

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

    def _export_animation(self, anim_data: Optional[dict], epoch: int) -> None:
        """Write the eval animation when the first batch payload is available."""
        if anim_data is None:
            return

        anim_path = os.path.join(
            self._artifact_dir("eval_videos"),
            f"eval_animation_epoch_{epoch:04d}.mp4",
        )
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
        eval_first_pos_mse: float,
        eval_hd_mae: float,
        eval_first_hd_mae: float,
        score_60,
        score_90,
        infer_seconds: float,
        score_seconds: float,
        eval_seconds: float,
        save_pdf: bool,
        writer,
    ) -> None:
        """Emit logs and optional TensorBoard scalars for one eval pass."""
        first_pos_ratio = _metric_ratio(eval_first_pos_mse, eval_pos_mse)
        first_hd_ratio = _metric_ratio(eval_first_hd_mae, eval_hd_mae)

        self.logger.info(
            "eval epoch=%d  pos_mse=%.6f  first_pos_mse=%.6f  "
            "first_pos_mse_ratio=%.4f  hd_mae_rad=%.4f  first_hd_mae_rad=%.4f  "
            "first_hd_mae_rad_ratio=%.4f  grid_score_60 max=%.4f  "
            "grid_score_90 max=%.4f  infer=%.1fs  score=%.1fs  total=%.1fs  pdf=%s",
            epoch,
            eval_pos_mse,
            eval_first_pos_mse,
            first_pos_ratio,
            eval_hd_mae,
            eval_first_hd_mae,
            first_hd_ratio,
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
        writer.add_scalar("eval/first_pos_mse", eval_first_pos_mse, epoch)
        writer.add_scalar("eval/first_pos_mse_ratio", first_pos_ratio, epoch)
        writer.add_scalar("eval/hd_mae_rad", eval_hd_mae, epoch)
        writer.add_scalar("eval/first_hd_mae_rad", eval_first_hd_mae, epoch)
        writer.add_scalar("eval/first_hd_mae_rad_ratio", first_hd_ratio, epoch)
        writer.add_scalar("eval/grid_score_60_max", float(score_60.max()), epoch)
        writer.add_scalar("eval/grid_score_90_max", float(score_90.max()), epoch)
        writer.add_scalar("eval/seconds", eval_seconds, epoch)
        writer.add_scalar("eval/infer_seconds", infer_seconds, epoch)
        writer.add_scalar("eval/score_seconds", score_seconds, epoch)
