# Goal: Grid-Cell Statistical Analysis

Evaluation includes a modular statistical analysis layer for bottleneck-unit spatial selectivity, Banino-style grid geometry, reliability, and head-direction mixed selectivity.

## Assumptions

- "Grid cells" means model `bottleneck` units with grid-like spatial representation.
- "Head cells" means head-direction cells / HDC. In this goal, also analyze whether `bottleneck` units are head-direction selective.
- Do not rewrite the existing training flow or grid-score implementation.
- Reuse `grid_cells.analysis.scores.GridScorer` and existing rate-map logic.
- Implement population-level grid-scale distribution analysis for FDR-significant grid units and fixed-threshold grid units, including Banino-style discreteness significance and GMM/BIC scale-cluster fitting.
- Use English for code, comments, fields, and docs. Communicate with the user in Chinese.

## Required Features

### 1. Grid Score Shuffle Significance

For each bottleneck unit, compute observed `grid_score_60` and `grid_score_90`, then generate null distributions.

- Use circular temporal-shift shuffling.
- For each eval trajectory, circularly shift that unit's activation time series.
- Shift distance must be at least `min_shift_fraction * seq_len` to avoid near-identity shuffles.
- Preserve activation distribution, temporal autocorrelation, and trajectory occupancy while breaking activation-position alignment.
- Compute p-values as:

```text
p = (1 + count(null_score >= observed_score)) / (1 + num_shuffles)
```

- Apply Benjamini-Hochberg FDR across units.
- Output q-values.
- Mark FDR-significant units using configured `fdr_alpha`.

### 2. Banino-Style Grid Geometry

For each bottleneck unit, compute a rate map, spatial autocorrelogram, and grid scale.

- Identify local maxima in the spatial autocorrelogram.
- Exclude the central peak.
- Use the six local maxima nearest the origin.
- `grid_scale_bins` is the median distance from those six peaks to the origin in spatial bins.
- `grid_scale_m` converts `grid_scale_bins` using the scorer's spatial bin size.
- Output `grid_scale_peak_count` and `grid_scale_valid`.
- Summary fields should include all finite per-unit scales, `grid_fdr_*` variants restricted to FDR-significant grid units, and `grid_threshold_*` variants restricted to units with `grid_score_60 >= analysis.gridness_threshold`.
- `grid_scale_discreteness` is an all-unit quick histogram summary using the finite scales' natural range; complete Banino-style fixed-range shuffle significance is reported in the `grid_fdr_scale_discreteness*` and `grid_threshold_scale_discreteness*` fields.

### 3. Grid-Scale Population Distribution

For FDR-significant grid units and fixed-threshold grid units with valid `grid_scale_bins`, compute population-level scale-distribution statistics following Banino et al.

- This population analysis requires grid selectivity and shuffle significance; if `compute_grid_selectivity=false` or no unit passes FDR, `grid_fdr_*` population fields are empty or `None`.
- In parallel, `grid_threshold_*` uses the Banino-style fixed gridness threshold from `analysis.gridness_threshold`, default `0.37`; it does not require shuffle significance, but it does require grid selectivity scores.
- Discreteness uses a histogram of `grid_scale_bins` with `analysis.scale_discreteness_bins` bins across `analysis.scale_discreteness_min_bins` to `analysis.scale_discreteness_max_bins`; defaults are 13 bins across 10 to 36 spatial bins.
- Discreteness score is the standard deviation of histogram counts.
- Discreteness significance uses `analysis.scale_discreteness_num_shuffles` shuffles; for each shuffle and each unit, add uniform noise in `[-0.5 * min_scale, +0.5 * min_scale]`, recompute discreteness, and compute a one-sided p-value:

```text
p = (1 + count(null_discreteness >= observed_discreteness)) / (1 + num_shuffles)
```

- Store the discreteness null distributions in the NPZ artifact as `grid_fdr_scale_discreteness_null` and `grid_threshold_scale_discreteness_null`.
- Fit 1-D Gaussian mixture models to valid FDR-grid scales and fixed-threshold grid scales for 1 to `analysis.scale_gmm_max_components` components, default 8.
- Use EM fitting and Bayesian Information Criterion to select the component count with the lowest BIC.
- Report best component count, BIC values, component means in bins and meters, and component weights in summary JSON.

### 4. Split-Half Reliability

Compute split-half reliability across trajectories.

- Support deterministic odd/even split or deterministic random split.
- Compute half A and half B rate maps.
- For each unit, compute Pearson correlation between the two rate maps, ignoring NaN bins.
- Output:
  - `split_half_corr`
  - `grid_score_60_half_a`
  - `grid_score_60_half_b`
  - `is_reliable_grid`

`is_reliable_grid` should require at least:

- FDR-significant grid score.
- `split_half_corr >= split_half_min_corr`.

### 5. Bottleneck Head-Direction Selectivity

Analyze head-direction selectivity for bottleneck units, not only HDC output heads.

- Bin by `target_hd`; default to `visualization.directional_bins`.
- Compute one directional tuning curve per bottleneck unit.
- Compute:
  - `hd_preferred_direction_rad`
  - mean resultant length, `hd_mrl`
  - p-value using circular-shift shuffle, matching the grid shuffle style where practical
  - Benjamini-Hochberg q-value
- Mark significant HD-selective units using configured `fdr_alpha`.

### 6. Output Artifacts

When analysis is enabled, evaluation should additionally write these artifacts to the current run save directory:

```text
grid_stats_epoch_XXXX.csv
grid_stats_epoch_XXXX.npz
grid_stats_summary_epoch_XXXX.json
```

CSV rows should represent bottleneck units and include at least:

```text
unit_id
grid_score_60
grid_score_90
grid_score_60_p
grid_score_60_q
grid_score_90_p
grid_score_90_q
grid_scale_bins
grid_scale_m
grid_scale_peak_count
grid_scale_valid
null_60_mean
null_60_std
null_60_95
split_half_corr
grid_score_60_half_a
grid_score_60_half_b
hd_mrl
hd_preferred_direction_rad
hd_p
hd_q
is_grid_fdr
is_grid_threshold
is_reliable_grid
is_hd_selective
unit_class
```

Allowed `unit_class` values:

```text
grid_only
hd_only
grid_and_hd
non_spatial
```

Summary JSON should include at least:

```text
n_units
n_grid_fdr
n_grid_threshold
n_reliable_grid
n_hd_selective
n_grid_and_hd
median_grid_score_60
n_grid_scale_valid
median_grid_scale_m
mean_grid_scale_m
grid_scale_discreteness
n_grid_fdr_scale_valid
median_grid_fdr_scale_m
mean_grid_fdr_scale_m
grid_fdr_scale_discreteness
grid_fdr_scale_discreteness_p
grid_fdr_scale_discreteness_null_mean
grid_fdr_scale_discreteness_null_std
grid_fdr_scale_discreteness_num_shuffles
grid_fdr_scale_gmm_best_components
grid_fdr_scale_gmm_best_bic
grid_fdr_scale_gmm_component_means_bins
grid_fdr_scale_gmm_component_means_m
grid_fdr_scale_gmm_component_weights
grid_fdr_scale_gmm_bic
n_grid_threshold_scale_valid
median_grid_threshold_scale_m
mean_grid_threshold_scale_m
grid_threshold_scale_discreteness
grid_threshold_scale_discreteness_p
grid_threshold_scale_discreteness_null_mean
grid_threshold_scale_discreteness_null_std
grid_threshold_scale_discreteness_num_shuffles
grid_threshold_scale_gmm_best_components
grid_threshold_scale_gmm_best_bic
grid_threshold_scale_gmm_component_means_bins
grid_threshold_scale_gmm_component_means_m
grid_threshold_scale_gmm_component_weights
grid_threshold_scale_gmm_bic
median_split_half_corr
analysis_num_shuffles
analysis_gridness_threshold
analysis_compute_grid_selectivity
analysis_compute_grid_geometry
analysis_compute_shuffle_significance
analysis_compute_split_half
analysis_compute_hd_selectivity
```

## Config

Add this section to `config.yaml`:

```yaml
analysis:
  enabled: true
  compute_grid_selectivity: true
  compute_grid_geometry: true
  compute_shuffle_significance: true
  compute_split_half: true
  compute_hd_selectivity: true
  num_shuffles: 200
  scale_discreteness_num_shuffles: 500
  scale_discreteness_bins: 13
  scale_discreteness_min_bins: 10.0
  scale_discreteness_max_bins: 36.0
  scale_gmm_max_components: 8
  gridness_threshold: 0.37
  fdr_alpha: 0.05
  min_shift_fraction: 0.1
  split_half_min_corr: 0.3
  max_eval_trajectories: 512
  random_seed: 0
```

`max_eval_trajectories` is important: statistical analysis must not default to full eval-set storage if doing so would create excessive memory or runtime cost.

## Implementation Guidance

- Add the main implementation in:

```text
grid_cells/analysis/spatial_stats.py
```

- Keep functions unit-testable. Do not put all logic into `evaluation.py`.
- In `grid_cells/training/evaluation.py`, keep the integration thin:
  - collect a bounded subset of eval `target_pos`, `target_hd`, and `bottleneck`
  - call the analysis module
  - save CSV, NPZ, and JSON artifacts
- Do not break existing PDF, animation, logging, or TensorBoard behavior.
- If `analysis.enabled=false`, current evaluation behavior should remain effectively unchanged.
- Handle NaN rate-map bins robustly.

## Tests

Add or update tests covering at least:

1. Benjamini-Hochberg q-values behave correctly.
2. Circular shift shuffle preserves activation shape.
3. Split-half reliability returns high correlation for identical or near-identical maps.
4. Analysis disabled does not require new artifacts.
5. A small synthetic input can generate a complete stats table/dict with required fields.
6. Individual analysis modules can be disabled while preserving artifact schema.
7. Grid-scale discreteness shuffle returns a finite score, p-value, and null distribution.
8. GMM/BIC fitting returns sorted component means and a selected component count.

Prefer lightweight tests. Do not run full training.

## Documentation

If implementation changes behavior significantly, update:

- `README.md`
- `README.zh.md`
- `AGENTS.md`
- `run_scripts.sh` if a new example command is needed

Documentation should describe current final behavior only, not old-vs-new migration history.

## Success Criteria

- Relevant `pytest` tests pass.
- Default training/evaluation still runs.
- With `analysis.enabled=true`, evaluation writes statistical artifacts.
- Output can report:
  - number of FDR-significant grid-like units
  - number of reliable grid-like units
  - number of HD-selective units
  - number of mixed grid+HD units
  - Banino-style grid scale for each unit with enough SAC peaks
  - FDR-grid and fixed-threshold grid scale discreteness, shuffle p-values, GMM/BIC component counts, means, and weights
