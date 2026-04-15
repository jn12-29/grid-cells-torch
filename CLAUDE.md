# Repo Notes

- Recommended dataset layout: `data/train.npz` for training and `data/eval.npz` for the fixed evaluation split.
- `training.data_path` defaults to `data/train.npz`. If the file is missing, `train.py` falls back to on-the-fly trajectory generation each epoch.
- `training.eval_data_path` defaults to `data/eval.npz`. If the file is missing, `train.py` generates one fixed in-memory eval set at startup and reuses it for the whole run.
- `train.py` now logs decoded position MSE for both training and evaluation by mapping place-cell logits back to 2-D coordinates.
- `generate_data.py` now defaults to generating both `training.data_path` (usually `data/train.npz`) and `training.eval_data_path` (`data/eval.npz`); pass `--train_only` to skip the eval split.
- `generate_data.py` supports chunked multiprocessing via `--num_workers`, and can write a continuously refreshed headless-safe preview PNG via `--visualize_progress`.
- `generate_data.py` can also export an MP4 trajectory animation via `--animate` / `--anim_output` (requires `ffmpeg`), using parallel frame rendering with `--anim_workers` and render/encode progress bars.
- Visualisation bins for `generate_data.py` are configurable from `config.yaml` under `visualization.spatial_bins` and `visualization.directional_bins`, and can also be overridden via `--spatial_bins` / `--directional_bins`.
