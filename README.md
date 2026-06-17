# grid-cells-torch

[🌐 中文](README.zh.md)

A faithful PyTorch port of [`google-deepmind/grid-cells`](https://github.com/google-deepmind/grid-cells), later expanded into a more practical workflow for data generation, evaluation, visualization, logging, and analysis.

## ✨ Results

Reference run: `results/20260416-040934`, snapshot at `epoch 12`.
README-safe assets are mirrored under `docs/assets/readme/`.

| Metric              |      Value |
| ------------------- | ---------: |
| `pos_mse`           | `0.027598` |
| `grid_score_60 max` |   `1.3284` |
| `grid_score_90 max` |   `1.5351` |

![Epoch 12 evaluation preview](docs/assets/readme/eval_animation_epoch_0012_thumb.png)

[PDF 1](docs/assets/readme/rates_and_sac_epoch_0012.pdf) | [PDF 2](docs/assets/readme/hdc_tuning_epoch_0012.pdf)

![PDF 1 page 1](docs/assets/readme/rates_and_sac_epoch_0012-1.png)
![PDF 1 page 2](docs/assets/readme/rates_and_sac_epoch_0012-2.png)

## 🚀 Beyond The Official Repo

- Pre-generated `train/eval` datasets, with on-the-fly fallback.
- Compact `train.log` and TensorBoard logging, including per-module gradient diagnostics.
- Decoded-position and head-direction metrics for training and evaluation, including whole-sequence `pos_mse` / `hd_mae_rad` and first-step `first_pos_mse` / `first_hd_mae_rad` diagnostics.
- Paginated rate-map PDFs, HDC tuning PDFs, and shared 3-panel MP4s for eval and generated data.
- Optional bottleneck-unit statistical analysis with circular-shift grid-score significance, split-half reliability, head-direction selectivity, and a separate v1 linear spatial-PCA gridness baseline. `model.bottleneck_post_activation` controls the model-side post-linear bottleneck transform, while `analysis.bottleneck_activation` controls the eval/analysis transform applied to the model-returned bottleneck; `raw` applies no extra eval transform.
- CLI-oriented data generation, visualization, and experiment management.

## 📚 References

| Reference                                                                                                                                       | Role                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Banino et al. (2018), [Vector-based navigation using grid-like representations in artificial agents](https://doi.org/10.1038/s41586-018-0102-6) | Original Nature paper                    |
| DeepMind official implementation, [google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells)                                   | Original codebase this repo started from |

## 🧭 Overview

This repository started as a strict PyTorch port of DeepMind's official `grid-cells` codebase. It was later extended with fixed dataset generation, evaluation PDFs, HDC tuning plots, shared 3-panel MP4 animations, TensorBoard logging, whole-sequence and first-step decoded-position/head-direction metrics, and a more complete CLI workflow for reproducible experiments.

The codebase uses a hierarchical Python package rooted at `grid_cells/`. The root-level Python entrypoints are `train.py`, `generate_data.py`, and `analyze_checkpoint.py`; library imports should target `grid_cells.*`.

## ⚡ Quick Start

```bash
pip install torch numpy scipy matplotlib pyyaml tqdm tensorboard

# install ffmpeg if you want MP4 outputs
sudo apt-get update && sudo apt-get install -y ffmpeg

# generate a dataset directory under data/datasets/<dataset-id>/ plus preview artifacts
python generate_data.py --visualize --animate

# speed up preview videos by sampling every other step
python generate_data.py --animate --visualization.anim_step 2

# override data-generation defaults from config.yaml
python generate_data.py --data_generation.num_samples 5000 --task.seq_len 800

# explicit file mode is available with output paths
python generate_data.py --output data/train_small.npz --eval_output data/eval_small.npz

# train with the generated dataset
python train.py

# or choose an optimizer
python train.py --training.optimizer lion

# or save periodic checkpoints every 5 completed epochs
python train.py --training.checkpoint_every 5

# or rerun full eval/analysis artifacts from a saved checkpoint
python analyze_checkpoint.py --checkpoint results/<run>/checkpoints/checkpoint_epoch_0005.pt

# or train with a specific dataset directory containing train.npz and eval.npz
python train.py --training.datadir data/datasets/<dataset-id>

# or switch to a step learning-rate scheduler
python train.py --training.lr_scheduler step --training.lr_step_size 10 --training.lr_gamma 0.8

# or rescale all gradients by the largest absolute gradient value
python train.py --training.grad_clip_mode norm --training.grad_clip_scope all --training.grad_clip_norm_type inf

# or override the shared animation config for eval videos
python train.py --visualization.anim_num_traj 4 --visualization.anim_step 2

# or reduce statistical-analysis cost for a quick run
python train.py --analysis.num_shuffles 20 --analysis.scale_discreteness_num_shuffles 50 --analysis.max_eval_trajectories 64

# or run only per-unit Banino-style grid geometry analysis
python train.py --analysis.compute_grid_selectivity false --analysis.compute_grid_geometry true --analysis.compute_shuffle_significance false --analysis.compute_split_half false --analysis.compute_hd_selectivity false

# or analyze the model-returned bottleneck without an extra eval transform
python train.py --analysis.bottleneck_activation raw

# or enable the v1 linear spatial-PCA gridness baseline diagnostics
python train.py --analysis.compute_linear_spatial_pca true --analysis.linear_spatial_pca_top_k 16 --analysis.linear_spatial_pca_num_shuffles 20 --analysis.linear_spatial_pca_plot_top_n 16

# or make the recommended scale-invariant analytic decoding contract explicit
python train.py --task.targets_type softmax --task.lstm_init_type softmax --task.decode_type analytic

# or print the common command list
bash run_scripts.sh

# inspect metrics
tensorboard --logdir results
```

Default convention:

- Generated dataset directory: `data/datasets/<dataset-id>/`
- Latest train entry for the default training flow: `data/latest/train.npz`
- Latest eval entry for the default training flow: `data/latest/eval.npz`
- Latest fixed cell centers: `data/latest/cells.npz`
- Run directory: `results/<timestamp>/`

Directory mode updates `data/latest/train.npz`, `data/latest/eval.npz`, and `data/latest/cells.npz` after a successful generation run. `train.py` uses `training.datadir` (`data/latest` by default) to find stable split files and fixed cell centers first.

If `data/latest/train.npz` is missing, `train.py` falls back to on-the-fly trajectory generation.

`task.targets_type=normalized` uses independent cell-activation targets instead of a population probability distribution; both PC and HDC normalized targets have unit peak activations, train with sigmoid/BCE loss, and decode with sigmoid-activation weighted means. The default and recommended analytic-decoding contract remains `targets_type=softmax`.

`task.pc_target_family` selects the place-cell target family independently from the shared PC/HDC `task.targets_type` mode. `gaussian` preserves the default Gaussian place-cell targets. `difference_of_gaussians` adds the reference-project Difference of Softmaxes / DoS branch when `task.pc_target_normalization=global`; with `none`, it subtracts raw Gaussian responses. `true_difference_of_gaussians` subtracts raw center and surround Gaussian responses. Both DoS and true DoG use `task.pc_surround_scale` as the surround sigma multiplier, shift the difference target to non-negative values, and normalize it across place cells. Analytic position decoding automatically falls back to weighted decoding for non-Gaussian PC target families.

`training.first_pos_loss_multiplier` controls timestep-0 place-cell target-loss weighting without adding a decoded `first_pos_mse` auxiliary loss.

## 📦 Outputs

- `train.log`: compact training log.
- `config.yaml`: effective training config after CLI overrides, including the resolved run directory.
- `tensorboard/`: scalar metrics and config snapshot, including first-step prediction diagnostics and sampled per-module gradient norms, max-absolute gradients, and clip ratios.
- `checkpoints/checkpoint_epoch_XXXX.pt`: periodic save-only checkpoint after configured completed epochs.
- `checkpoints/checkpoint_final.pt`: final save-only checkpoint after normal training completion.
- `ckpt_analysis/<checkpoint_stem>/`: default output directory for `analyze_checkpoint.py`, which reruns full eval/analysis artifacts from a saved checkpoint without overwriting training-time artifacts.
- `eval_plots/rates_and_sac_epoch_XXXX.pdf`: rate maps and spatial autocorrelograms computed from the configured analysis transform of the model-returned bottleneck.
- `eval_plots/hdc_tuning_epoch_XXXX.pdf`: HDC tuning curves.
- `eval_videos/eval_animation_epoch_XXXX.mp4`: sequential eval trajectory animation; multiple trajectories are joined into one video.
- `eval_stats/grid_stats_epoch_XXXX.csv`, `eval_stats/grid_stats_epoch_XXXX.npz`, `eval_stats/grid_stats_summary_epoch_XXXX.json`, `eval_stats/grid_scale_histograms_epoch_XXXX.pdf`, `eval_stats/grid_scale_histograms_epoch_XXXX.png`: bottleneck-unit grid significance, Banino-style grid scale, split-half reliability, head-direction selectivity, and mixed-selectivity counts when `analysis.enabled=true`, using the configured analysis transform of the model-returned bottleneck. Individual statistical modules can be toggled with `analysis.compute_grid_selectivity`, `analysis.compute_grid_geometry`, `analysis.compute_shuffle_significance`, `analysis.compute_split_half`, and `analysis.compute_hd_selectivity`; disabled modules keep their artifact fields with `NaN` or `false` placeholder values. Grid-scale summary fields include all finite per-unit scales plus `grid_fdr_*` variants for FDR-significant grid units and `grid_threshold_*` variants for `grid_score_60 >= analysis.gridness_threshold` units, including Banino-style discreteness shuffle significance and 1D GMM/BIC scale-cluster fitting. The scale histogram plots show the selected population scale distributions with the fitted GMM curve and component peak locations.
- `eval_stats/linear_spatial_pca_modes_epoch_XXXX.csv`, `eval_stats/linear_spatial_pca_modes_epoch_XXXX.npz`, `eval_stats/linear_spatial_pca_summary_epoch_XXXX.json`, `eval_stats/linear_spatial_pca_overview_epoch_XXXX.pdf`, `eval_stats/linear_spatial_pca_overview_epoch_XXXX.png`, `eval_stats/linear_spatial_pca_mode_maps_epoch_XXXX.pdf`: optional v1 baseline diagnostics when `analysis.enabled=true` and `analysis.compute_linear_spatial_pca=true`, using the configured analysis transform of the model-returned bottleneck. The baseline fits orthogonal spatial-PCA directions on fit trajectories, evaluates projected-mode gridness on held-out trajectories, and records `not_grid_score_optimized=true`; it is not a direct grid-score-optimized `W` search. The overview plots held-out mode grid scores, fit variance explained, shuffle-null `T`, variance-vs-gridness scatter, and `W_top.T` loadings. The mode-map PDF shows the top `analysis.linear_spatial_pca_plot_top_n` modes ranked by held-out grid score, with fit/test rate maps, held-out SACs, and loading weights. Set `analysis.linear_spatial_pca_save_plots=false` to write only the CSV/NPZ/JSON data artifacts.

## 🗂️ Repo Layout

```text
grid-cells-torch/
├── config.yaml
├── analyze_checkpoint.py
├── generate_data.py
├── train.py
├── grid_cells/
│   ├── common/
│   ├── cells/
│   ├── data/
│   ├── training/
│   ├── analysis/
│   └── viz/
├── tests/
├── docs/assets/readme/
└── results/
```

<details>
<summary>🔍 More Details</summary>

- `config.yaml` is the default experiment entry point and supports CLI overrides, for example `python train.py --training.epochs 100 --training.lr 1e-3`.
- Each training run saves the effective config to `<run directory>/config.yaml` after applying CLI overrides.
- Checkpoints are save-only artifacts; training resume is not implemented yet. Use `analyze_checkpoint.py --checkpoint <path>` to rerun full eval/analysis artifacts from checkpoint weights. By default, the post-hoc output is written to `<run directory>/ckpt_analysis/<checkpoint_stem>/`; pass `--output_dir` to choose another directory.
- Optimizers: `training.optimizer` supports `rmsprop`, `adamw`, `adam`, `sgd`, and `lion`; `training.betas` applies to Adam, AdamW, and Lion when set; `training.adamw_eps` applies to Adam and AdamW; `training.momentum` applies to RMSprop and SGD.
- Learning-rate scheduling supports `cosine`, `step`, and `none`; StepLR uses `training.lr_step_size` and `training.lr_gamma`.
- Gradient clipping uses `training.grad_clip` as the threshold. `training.grad_clip_mode` supports `value` elementwise clipping and `norm` proportional rescaling; `training.grad_clip_scope` supports `decoder` and `all`; `training.grad_clip_norm_type: "inf"` uses the largest absolute gradient value for norm clipping. Training logs sampled per-module gradient diagnostics for `state_init`, `cell_init`, `lstm`, `bottleneck`, `pc_heads`, and `hdc_heads`; TensorBoard receives step and epoch-mean scalars under `train/grad/<module>/...` for norm, RMS, mean-absolute, signed-mean, max-absolute, `pre_clip_exceed_frac` (`abs(grad) > abs(training.grad_clip)` before clipping), `clip_changed_frac`, `post_clip_saturated_frac`, and `clip_ratio`, while `train.log` receives compact epoch sampled means.
- `train.py` and `generate_data.py` now share the same explicit `--section.key value` override style for config-backed defaults.
- `generate_data.py` defaults to creating a dataset directory under `data/datasets/<dataset-id>/`, writes dataset metadata plus `cells.npz`, and updates `data/latest/*` for the default training path.
- `train.py --training.datadir data/datasets/<dataset-id>` loads `<datadir>/train.npz`, `<datadir>/eval.npz`, and `<datadir>/cells.npz` before falling back to the legacy `training.data_path` and `training.eval_data_path` fields.
- `task.cells_path` can explicitly point to a fixed `cells.npz`; otherwise training auto-loads `<training.datadir>/cells.npz` when it exists.
- `task.pc_target_family` is PC-only target encoding; keep DoS/DoG there instead of adding them to shared `task.targets_type`, which also configures HDC targets and loss mode.
- Explicit file-mode generation is available when you pass `--output` / `--eval_output` paths.
- Long-lived generation defaults live under `data_generation.*` in `config.yaml`; directory mode owns dataset-local output paths, while `data_generation.*_output` applies to explicit `--output` / `--eval_output` file mode. One-shot run controls such as `--visualize`, `--animate`, `--train_only`, and `--visualize_progress` stay as CLI flags.
- Layered package boundaries:
  `grid_cells/common` owns shared config helpers.
  `grid_cells/cells` owns ensembles, encoding, and model code.
  `grid_cells/data` owns dataset IO, generation, previews, and trajectory synthesis.
  `grid_cells/training` owns CLI parsing, runtime wiring, evaluation, and the training session.
  `grid_cells/analysis` owns scoring and plotting helpers.
  `grid_cells/viz` owns animation rendering.
- Shared animation defaults live under `visualization.anim_*` in `config.yaml`, and both `train.py` and `generate_data.py` can override them from the CLI.
- `model.bottleneck_post_activation` controls the model-side post-linear bottleneck transform before dropout and output heads. Statistical evaluation settings live under `analysis.*`; `analysis.bottleneck_activation` controls eval rate-map/grid-score and statistical-analysis inputs, and `raw` uses the model-returned bottleneck without an extra eval transform. `analysis.max_eval_trajectories` caps the eval subset retained for `spatial_stats` and optional linear spatial-PCA analysis.
- Training always runs a baseline evaluation at epoch 0, then evaluates every `training.eval_every` completed epochs. `training.eval_plot_every` counts these evaluation calls, including the epoch-0 baseline, when deciding whether to export PDFs and animations.
- First-step metrics decode the model output at timestep 0 and compare it with `target_pos[:, 0]` / `target_hd[:, 0]`. This target is the first post-velocity-update state, not the raw `init_pos` / `init_hd` used to seed the LSTM.
- Typical override examples:
  `python train.py --task.env_size 2.4 --training.batch_size 32`
  `python train.py --training.lr_scheduler step --training.lr_step_size 10 --training.lr_gamma 0.8`
  `python generate_data.py --visualization.anim_fps 30 --data_generation.num_workers 4`
- `run_scripts.sh` prints a compact list of common train, generate, checkpoint-analysis, and TensorBoard commands.
- Project maintenance notes and workflow contracts are kept in `docs/maintenance.md`.
- The current default config is tuned for the expanded engineering workflow, not a line-by-line lockstep copy of the original hyperparameters.
- README media is mirrored from selected run outputs into `docs/assets/readme/` so the landing page does not depend on ignored `results/` files.

</details>

## 🙏 Acknowledgements

This project was developed with substantial help from Coding Agents.
