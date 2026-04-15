# grid-cells-torch

[🌐 中文](README.zh.md)

Faithful PyTorch reimplementation of `google-deepmind/grid-cells`, later extended with richer data generation, evaluation, visualization, logging, and experiment workflow.

## ✨ Results

Reference run: `results/20260416-040934`, snapshot at `epoch 12`.
Publishable README assets are mirrored under `docs/assets/readme/`.

| Metric | Value |
|---|---:|
| `pos_mse` | `0.027598` |
| `grid_score_60 max` | `1.3284` |
| `grid_score_90 max` | `1.5351` |

[![Epoch 12 evaluation animation](docs/assets/readme/eval_animation_epoch_0012_thumb.png)](docs/assets/readme/eval_animation_epoch_0012.mp4)

Video: `docs/assets/readme/eval_animation_epoch_0012.mp4`

[PDF 1](docs/assets/readme/rates_and_sac_epoch_0012.pdf) | [PDF 2](docs/assets/readme/hdc_tuning_epoch_0012.pdf)

![PDF 1 page 1](docs/assets/readme/rates_and_sac_epoch_0012-1.png)
![PDF 1 page 2](docs/assets/readme/rates_and_sac_epoch_0012-2.png)

## 🚀 Extensions Beyond The Official Repo

- Pre-generated `train/eval` datasets with on-the-fly fallback.
- Compact `train.log` plus TensorBoard logging.
- Decoded-position metric `pos_mse` for training and evaluation.
- Paginated rate-map PDFs, HDC tuning PDFs, and eval animation MP4s.
- CLI-driven data generation, visualization, and experiment management.

## 📚 References

| Reference | Role |
|---|---|
| Banino et al. (2018), [Vector-based navigation using grid-like representations in artificial agents](https://doi.org/10.1038/s41586-018-0102-6) | Original Nature paper |
| DeepMind official implementation, [google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells) | Original codebase this repo started from |

## 🧭 Overview

This repository started as a faithful PyTorch port of DeepMind's official `grid-cells` codebase. It was then extended with fixed dataset generation, evaluation PDFs, HDC tuning plots, MP4 animations, TensorBoard logging, a decoded-position metric (`pos_mse`), and a more complete CLI-driven workflow for reproducible experiments.

## ⚡ Quick Start

```bash
pip install torch numpy scipy matplotlib pyyaml tqdm tensorboard

# install ffmpeg if you want MP4 outputs
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# generate train/eval splits plus preview artifacts
python generate_data.py --visualize --animate

# train with the generated dataset
python train.py

# inspect metrics
tensorboard --logdir results
```

Default convention:

- Train split: `data/train.npz`
- Eval split: `data/eval.npz`
- Run directory: `results/<timestamp>/`

If `data/train.npz` is missing, `train.py` falls back to on-the-fly trajectory generation.

## 📦 Outputs

- `train.log`: compact training log.
- `tensorboard/`: scalar metrics and config snapshot.
- `rates_and_sac_epoch_XXXX.pdf`: rate maps and spatial autocorrelograms.
- `hdc_tuning_epoch_XXXX.pdf`: HDC tuning curves.
- `eval_animation_epoch_XXXX.mp4`: evaluation trajectory animation.

## 🗂️ Repo Layout

```text
grid-cells-torch/
├── config.yaml
├── generate_data.py
├── train.py
├── model.py
├── dataset.py
├── ensembles.py
├── scores.py
├── utils.py
└── results/
```

<details>
<summary>🔍 More Details</summary>

- `config.yaml` is the default experiment entry point and supports CLI overrides, for example `python train.py --training.epochs 100 --training.lr 1e-3`.
- `generate_data.py` can export `.npz`, PDF summaries, and MP4 animations in one workflow.
- The current default config is tuned for the expanded engineering workflow, not a line-by-line lockstep copy of the original hyperparameters.

</details>
