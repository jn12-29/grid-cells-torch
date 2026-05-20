# Project Maintenance Notes

## Documentation

- `README.md` is the default English landing page.
- `README.zh.md` is the Chinese mirror and must keep the same structure and content as `README.md`, differing only by language.
- README media intended for publication should live under `docs/assets/readme/`, not under `results/`.
- Python modules should keep a current top-of-file docstring describing purpose and basic usage.
- `run_scripts.sh` is the shell entrypoint for common workflows and should keep its built-in help text up to date.

## Entry Points And Package Boundaries

- The root-level Python entrypoints are `train.py` and `generate_data.py`; library code imports from `grid_cells.*`.
- Package boundaries:
  - `grid_cells/common` owns shared config helpers.
  - `grid_cells/cells` owns ensembles, encoding, and model code.
  - `grid_cells/data` owns dataset IO, generation, previews, and trajectory synthesis.
  - `grid_cells/training` owns CLI parsing, runtime wiring, evaluation, and the training session.
  - `grid_cells/analysis` owns scoring and plotting helpers.
  - `grid_cells/viz` owns animation rendering.

## Workflow Contracts

- Shared animation knobs live under `visualization.anim_*`; keep `train.py` eval videos and `generate_data.py --animate` aligned to that interface.
- Training checkpoints are save-only artifacts under `<run directory>/checkpoints/`; `grid_cells/training/runtime.py` owns serialization and `TrainingSession` owns scheduling.
- Training runs a baseline evaluation at epoch 0, then evaluates every `training.eval_every` completed epochs; `training.eval_plot_every` counts those evaluation calls, including epoch 0, for PDF/animation export.
- `generate_data.py` defaults to directory mode; `data_generation.*_output` paths apply only to explicit `--output` / `--eval_output` file mode.
- Analytic PC/HDC decoding lives under `grid_cells/cells`; PC logit decoding is scale-invariant for positive scaled raw scores, and raw-score paths should receive log-scores rather than probability activations that may have underflowed.
- `task.targets_type=normalized` means independent cell-activation targets with unit peak activations for both PC and HDC ensembles; train with sigmoid/BCE and decode logits through sigmoid-activation weighted means. The default analytic-decoding contract remains `targets_type=softmax`.

## Local Development

- Run project commands in the conda `gce` environment. If the tool shell does not inherit the user's active shell environment, use `conda run -n gce ...` explicitly.
- Pytest can appear to hang under the sandbox for this project. If a test run stalls without output, check for sandbox-related blockage and rerun the smallest relevant test selection outside the sandbox with approval instead of assuming the tests are broken.
