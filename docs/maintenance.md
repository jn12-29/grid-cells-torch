# Project Maintenance Notes

## Documentation

- `README.md` is the default English landing page.
- `README.zh.md` is the Chinese mirror and must keep the same structure and content as `README.md`, differing only by language.
- README media intended for publication should live under `docs/assets/readme/`, not under `results/`.
- Python modules should keep a current top-of-file docstring describing purpose and basic usage.
- `run_scripts.sh` is the shell entrypoint for common workflows and should keep its built-in help text up to date.

## Entry Points And Package Boundaries

- The root-level Python entrypoints are `train.py`, `generate_data.py`, and `analyze_checkpoint.py`; library code imports from `grid_cells.*`.
- Package boundaries:
  - `grid_cells/common` owns shared config helpers.
  - `grid_cells/cells` owns ensembles, encoding, and model code.
  - `grid_cells/data` owns dataset IO, generation, previews, and trajectory synthesis.
  - `grid_cells/training` owns CLI parsing, runtime wiring, evaluation, and the training session.
  - `grid_cells/analysis` owns scoring and plotting helpers.
  - `grid_cells/viz` owns animation rendering.

## Workflow Contracts

- Shared animation knobs live under `visualization.anim_*`; keep `train.py` eval videos and `generate_data.py --animate` aligned to that interface. Training eval writes one sequential MP4 per plotted epoch under `<run directory>/eval_videos/`.
- `model.bottleneck_post_activation` controls the model-side transform after the bottleneck linear layer and before dropout/output heads. Evaluation statistical-analysis settings live under `analysis.*`; `analysis.bottleneck_activation` controls the eval-only transform applied to the model-returned bottleneck for rate-map/grid-score and statistical-analysis consumers, and `raw` applies no extra eval transform. When enabled, eval writes `grid_stats_epoch_XXXX.csv`, `grid_stats_epoch_XXXX.npz`, `grid_stats_summary_epoch_XXXX.json`, `grid_scale_histograms_epoch_XXXX.pdf`, and `grid_scale_histograms_epoch_XXXX.png` artifacts under `<run directory>/eval_stats/` for bottleneck-unit grid significance, Banino-style grid scale, split-half reliability, and head-direction selectivity. The statistical modules are individually toggled by `analysis.compute_grid_selectivity`, `analysis.compute_grid_geometry`, `analysis.compute_shuffle_significance`, `analysis.compute_split_half`, and `analysis.compute_hd_selectivity`; disabled modules preserve the artifact schema with `NaN` or `false` placeholder values. Grid-scale summary fields include all finite per-unit scales, `grid_fdr_*` variants restricted to FDR-significant grid units, and `grid_threshold_*` variants restricted to `grid_score_60 >= analysis.gridness_threshold` units, including discreteness shuffle significance and GMM/BIC scale-cluster fitting. The scale histogram artifacts plot the selected population scale distributions with the fitted GMM curve and component peak locations. The `grid_fdr_*` population fields require grid selectivity and shuffle significance; `grid_threshold_*` population fields require grid selectivity and use the configured fixed gridness threshold. `analysis.enabled=true` plus `analysis.compute_linear_spatial_pca=true` enables an optional v1 baseline that fits orthogonal spatial-PCA directions from fit trajectories and evaluates projected-mode gridness on held-out trajectories; it writes `linear_spatial_pca_modes_epoch_XXXX.csv`, `linear_spatial_pca_modes_epoch_XXXX.npz`, `linear_spatial_pca_summary_epoch_XXXX.json`, `linear_spatial_pca_overview_epoch_XXXX.pdf`, `linear_spatial_pca_overview_epoch_XXXX.png`, and `linear_spatial_pca_mode_maps_epoch_XXXX.pdf` under `eval_stats/` without changing the `grid_stats_*` schema. `analysis.linear_spatial_pca_save_plots` controls the plot artifacts, and `analysis.linear_spatial_pca_plot_top_n` controls how many held-out-score-ranked modes appear in the mode-map PDF. This baseline is not a grid-score-optimized `W` search; the plots are spatial-PCA baseline diagnostics.
- Evaluation PDFs are exported under `<run directory>/eval_plots/`.
- Training checkpoints are save-only artifacts under `<run directory>/checkpoints/`; `grid_cells/training/runtime.py` owns serialization and `TrainingSession` owns scheduling. `analyze_checkpoint.py` can reload schema-v1 checkpoint weights and rerun full eval/analysis artifacts, defaulting to `<run directory>/ckpt_analysis/<checkpoint_stem>/`; this is post-hoc analysis, not training resume.
- Training runs a baseline evaluation at epoch 0, then evaluates every `training.eval_every` completed epochs; `training.eval_plot_every` counts those evaluation calls, including epoch 0, for PDF/animation export.
- `generate_data.py` defaults to directory mode; `data_generation.*_output` paths apply only to explicit `--output` / `--eval_output` file mode.
- Analytic PC/HDC decoding lives under `grid_cells/cells`; PC logit decoding is scale-invariant for positive scaled raw scores, and raw-score paths should receive log-scores rather than probability activations that may have underflowed.
- `task.pc_target_family` selects the PC-only place-field target family and must not be folded into shared `task.targets_type`, which also controls HDC target/loss mode. `difference_of_gaussians` with `task.pc_target_normalization=global` is the reference-project DoS branch; `true_difference_of_gaussians` subtracts raw Gaussian center/surround responses. DoS/DoG use `task.pc_surround_scale`, shift differences to non-negative values, normalize across PC cells, and fall back to weighted position decoding when `task.decode_type=analytic`. For No-Free-Lunch-style checks and dataset animations, `task.decode_type=top_k` uses single-field PC center voting with `task.decode_top_k` votes; `true_difference_of_gaussians` selects lowest activations, and other PC target families select highest activations.
- `training.first_pos_loss_multiplier` controls timestep-0 place-cell target-loss weighting without adding a decoded `first_pos_mse` auxiliary loss.
- Gradient processing is controlled by `training.grad_processing_mode`; `clip` uses `training.grad_clip_mode`, and `sign_mean_abs` replaces selected gradients with signed per-parameter mean magnitudes while zeroing entries with `abs(grad) <= training.grad_clip`. `training.grad_clip_scope` selects the current processing target (`decoder` or `all`). Training logs sampled per-module gradient diagnostics for `state_init`, `cell_init`, `lstm`, `bottleneck`, `pc_heads`, and `hdc_heads`; TensorBoard step/epoch-mean tags live under `train/grad/<module>/...` for norm, RMS, mean-absolute, signed-mean, max-absolute, `pre_clip_exceed_frac` (`abs(grad) > abs(training.grad_clip)` before processing), `clip_changed_frac`, `post_clip_saturated_frac`, and `clip_ratio`, and `train.log` includes compact epoch sampled means.
- Training and evaluation metric logs include whole-sequence `pos_mse` / `hd_mae_rad` and first-step `first_pos_mse` / `first_hd_mae_rad`. First-step metrics compare decoded timestep-0 outputs with `target_pos[:, 0]` / `target_hd[:, 0]`, which are post-first-velocity-update targets rather than raw initial conditions.
- `task.targets_type=normalized` means independent cell-activation targets with unit peak activations for both PC and HDC ensembles; train with sigmoid/BCE and decode logits through sigmoid-activation weighted means. The default analytic-decoding contract remains `targets_type=softmax`.

## Local Development

- Run project commands in the conda `gce` environment. If the tool shell does not inherit the user's active shell environment, use `conda run -n gce ...` explicitly.
- Pytest can appear to hang under the sandbox for this project. If a test run stalls without output, check for sandbox-related blockage and rerun test outside the sandbox instead of assuming the tests are broken.
