# Repo Notes

- `README.md` is the default English landing page.
- `README.zh.md` is the Chinese mirror and must keep the same structure and content as `README.md`, differing only by language.
- Any README media intended for publication should live under `docs/assets/readme/`, not under `results/`.
- Python modules should keep a current top-of-file docstring describing purpose and basic usage.
- `run_scripts.sh` is the shell entrypoint for common workflows and should keep its built-in help text up to date.
- Shared animation knobs now live under `visualization.anim_*`; keep `train.py` eval videos and `generate_data.py --animate` aligned to that interface.
- The root-level Python entrypoints are `train.py` and `generate_data.py`; library code imports from `grid_cells.*`.
- Package boundaries:
  `grid_cells/common` owns shared config helpers.
  `grid_cells/cells` owns ensembles, encoding, and model code.
  `grid_cells/data` owns dataset IO, generation, previews, and trajectory synthesis.
  `grid_cells/training` owns CLI parsing, runtime wiring, evaluation, and the training session.
  `grid_cells/analysis` owns scoring and plotting helpers.
  `grid_cells/viz` owns animation rendering.
