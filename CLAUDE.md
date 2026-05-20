# Repo Notes

- Project maintenance notes, workflow contracts, and local development caveats live in `docs/maintenance.md`.
- Statistical-analysis module toggles live under `analysis.compute_*`, including grid selectivity and Banino-style grid geometry; disabled modules preserve `eval_stats` artifact fields with `NaN` or `false` placeholders, and grid-scale summaries expose all-unit, `grid_fdr_*`, and `grid_threshold_*` variants with discreteness shuffle significance and GMM/BIC scale-cluster fitting.
- Training optimizers are selected with `training.optimizer`; supported values are `rmsprop`, `adamw`, and `sgd`. `training.momentum` applies to RMSprop and SGD.
