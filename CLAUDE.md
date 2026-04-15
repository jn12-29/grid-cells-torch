# Repo Notes

- Recommended dataset layout: `data/train.npz` for training and `data/eval.npz` for the fixed evaluation split.
- `training.eval_data_path` defaults to `data/eval.npz`. If the file is missing, `train.py` generates one fixed in-memory eval set at startup and reuses it for the whole run.
- `generate_data.py` can now emit both splits in one invocation via `--output ... --eval_output ...`.
