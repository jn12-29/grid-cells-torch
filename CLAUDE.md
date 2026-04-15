# Repo Notes

- `README.md` is the default English landing page.
- `README.zh.md` is the Chinese mirror and must keep the same structure and content as `README.md`, differing only by language.
- Any README media intended for publication should live under `docs/assets/readme/`, not under `results/`.
- Python modules should keep a current top-of-file docstring describing purpose and basic usage.
- `run_scripts.sh` is the shell entrypoint for common workflows and should keep its built-in help text up to date.
- Shared animation knobs now live under `visualization.anim_*`; keep `train.py` eval videos and `generate_data.py --animate` aligned to that interface.
- OOP orchestration boundaries: `encoding.py` owns ensemble encoding, `animation.py` owns animation rendering, `evaluation.py` owns eval/export flow, `training_session.py` owns the train loop, and `trajectory_generation.py` owns random-walk synthesis.
