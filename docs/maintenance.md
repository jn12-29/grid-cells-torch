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
- Evaluation statistical-analysis settings live under `analysis.*`; when enabled, eval writes `grid_stats_epoch_XXXX.csv`, `grid_stats_epoch_XXXX.npz`, `grid_stats_summary_epoch_XXXX.json`, `grid_scale_histograms_epoch_XXXX.pdf`, and `grid_scale_histograms_epoch_XXXX.png` artifacts under `<run directory>/eval_stats/` for bottleneck-unit grid significance, Banino-style grid scale, split-half reliability, and head-direction selectivity. The statistical modules are individually toggled by `analysis.compute_grid_selectivity`, `analysis.compute_grid_geometry`, `analysis.compute_shuffle_significance`, `analysis.compute_split_half`, and `analysis.compute_hd_selectivity`; disabled modules preserve the artifact schema with `NaN` or `false` placeholder values. Grid-scale summary fields include all finite per-unit scales, `grid_fdr_*` variants restricted to FDR-significant grid units, and `grid_threshold_*` variants restricted to `grid_score_60 >= analysis.gridness_threshold` units, including discreteness shuffle significance and GMM/BIC scale-cluster fitting. The scale histogram artifacts plot the selected population scale distributions with the fitted GMM curve and component peak locations. The `grid_fdr_*` population fields require grid selectivity and shuffle significance; `grid_threshold_*` population fields require grid selectivity and use the configured fixed gridness threshold. `analysis.enabled=true` plus `analysis.compute_linear_spatial_pca=true` enables an optional v1 baseline that fits orthogonal spatial-PCA directions from fit trajectories and evaluates projected-mode gridness on held-out trajectories; it writes `linear_spatial_pca_modes_epoch_XXXX.csv`, `linear_spatial_pca_modes_epoch_XXXX.npz`, `linear_spatial_pca_summary_epoch_XXXX.json`, `linear_spatial_pca_overview_epoch_XXXX.pdf`, `linear_spatial_pca_overview_epoch_XXXX.png`, and `linear_spatial_pca_mode_maps_epoch_XXXX.pdf` under `eval_stats/` without changing the `grid_stats_*` schema. `analysis.linear_spatial_pca_save_plots` controls the plot artifacts, and `analysis.linear_spatial_pca_plot_top_n` controls how many held-out-score-ranked modes appear in the mode-map PDF. This baseline is not a grid-score-optimized `W` search; the plots are spatial-PCA baseline diagnostics.
- Evaluation PDFs are exported under `<run directory>/eval_plots/`.
- Training checkpoints are save-only artifacts under `<run directory>/checkpoints/`; `grid_cells/training/runtime.py` owns serialization and `TrainingSession` owns scheduling. `analyze_checkpoint.py` can reload schema-v1 checkpoint weights and rerun full eval/analysis artifacts, defaulting to `<run directory>/ckpt_analysis/<checkpoint_stem>/`; this is post-hoc analysis, not training resume.
- Training runs a baseline evaluation at epoch 0, then evaluates every `training.eval_every` completed epochs; `training.eval_plot_every` counts those evaluation calls, including epoch 0, for PDF/animation export.
- `generate_data.py` defaults to directory mode; `data_generation.*_output` paths apply only to explicit `--output` / `--eval_output` file mode.
- Analytic PC/HDC decoding lives under `grid_cells/cells`; PC logit decoding is scale-invariant for positive scaled raw scores, and raw-score paths should receive log-scores rather than probability activations that may have underflowed.
- `training.first_pos_loss_multiplier` controls timestep-0 place-cell target-loss weighting without adding a decoded `first_pos_mse` auxiliary loss.
- Training logs sampled per-module gradient diagnostics for `state_init`, `cell_init`, `lstm`, `bottleneck`, `pc_heads`, and `hdc_heads`; TensorBoard step/epoch-mean tags live under `train/grad/<module>/...`, and `train.log` includes compact epoch sampled means.
- Training and evaluation metric logs include whole-sequence `pos_mse` / `hd_mae_rad` and first-step `first_pos_mse` / `first_hd_mae_rad`. First-step metrics compare decoded timestep-0 outputs with `target_pos[:, 0]` / `target_hd[:, 0]`, which are post-first-velocity-update targets rather than raw initial conditions.
- `task.targets_type=normalized` means independent cell-activation targets with unit peak activations for both PC and HDC ensembles; train with sigmoid/BCE and decode logits through sigmoid-activation weighted means. The default analytic-decoding contract remains `targets_type=softmax`.

## Local Development

- Run project commands in the conda `gce` environment. If the tool shell does not inherit the user's active shell environment, use `conda run -n gce ...` explicitly.
- Pytest can appear to hang under the sandbox for this project. If a test run stalls without output, check for sandbox-related blockage and rerun test outside the sandbox instead of assuming the tests are broken.
