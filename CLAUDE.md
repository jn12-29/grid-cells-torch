# Repo Notes

- Project maintenance notes, workflow contracts, and local development caveats live in `docs/maintenance.md`.
- Statistical-analysis module toggles live under `analysis.compute_*`, including grid selectivity and Banino-style grid geometry; disabled modules preserve `eval_stats` artifact fields with `NaN` or `false` placeholders, and grid-scale summaries expose all-unit, `grid_fdr_*`, and `grid_threshold_*` variants with discreteness shuffle significance, GMM/BIC scale-cluster fitting, and population scale histogram/GMM overlay plots.
- Training optimizers are selected with `training.optimizer`; supported values are `rmsprop`, `adamw`, and `sgd`. `training.momentum` applies to RMSprop and SGD.
- `training.first_pos_loss_multiplier` controls timestep-0 PC target-loss weighting without adding a decoded-MSE auxiliary loss.
- Training and evaluation logs include whole-sequence `pos_mse` / `hd_mae_rad` plus first-step `first_pos_mse` / `first_hd_mae_rad`; the first-step target is `target_pos[:, 0]` / `target_hd[:, 0]`, not raw `init_pos` / `init_hd`.
