# Goal: Grid-Cell Statistical Analysis

Current grid-cell analysis in `/home/xh/ai4neuron/gridCells/grid-cells-torch` is too simple: it mainly exports bottleneck-unit rate maps, SACs, and maximum grid scores. Add a v1 statistical analysis layer from a neuroscience research perspective so evaluation reports significance, reliability, and head-direction mixed selectivity.

## Assumptions

- "Grid cells" means model `bottleneck` units with grid-like spatial representation.
- "Head cells" means head-direction cells / HDC. In this goal, also analyze whether `bottleneck` units are head-direction selective.
- Do not rewrite the existing training flow or grid-score implementation.
- Reuse `grid_cells.analysis.scores.GridScorer` and existing rate-map logic.
- Implement the smallest testable statistical version first. Do not implement complex module clustering in v1.
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

### 2. Split-Half Reliability

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

### 3. Bottleneck Head-Direction Selectivity

Analyze head-direction selectivity for bottleneck units, not only HDC output heads.

- Bin by `target_hd`; default to `visualization.directional_bins`.
- Compute one directional tuning curve per bottleneck unit.
- Compute:
  - `hd_preferred_direction_rad`
  - mean resultant length, `hd_mrl`
  - p-value using circular-shift shuffle, matching the grid shuffle style where practical
  - Benjamini-Hochberg q-value
- Mark significant HD-selective units using configured `fdr_alpha`.

### 4. Output Artifacts

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
n_reliable_grid
n_hd_selective
n_grid_and_hd
median_grid_score_60
median_split_half_corr
analysis_num_shuffles
```

## Config

Add this section to `config.yaml`:

```yaml
analysis:
  enabled: true
  num_shuffles: 200
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

Prefer lightweight tests. Do not run full training.

## Documentation

If implementation changes behavior significantly, update:

- `README.md`
- `README.zh.md`
- `CLAUDE.md`
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
