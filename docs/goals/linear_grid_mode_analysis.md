# Linear Grid-Mode Analysis for ANN Hidden States

## 1. Motivation

Current grid-cell analysis in artificial neural networks often measures whether individual hidden units show grid-like spatial firing maps. This corresponds to analyzing each original hidden coordinate:

$$
h_i(t)
$$

where $h_t \in \mathbb{R}^N$ is the hidden state of the network at time $t$.

However, in many ANN architectures, especially when the hidden state is followed by a linear readout, the original hidden-unit axes are not uniquely meaningful. If the readout is

$$
\hat{y}_t = h_t L
$$

then for any invertible matrix $W$,

$$
h_t L = h_t W W^{-1} L
$$

Therefore, the same input-output function can be represented under a different hidden basis:

$$
z_t = h_t W
$$

with a transformed readout

$$
L' = W^{-1}L
$$

This means that axis-aligned hidden-unit gridness is basis-dependent. The absence of grid-like maps in the original hidden units does not necessarily imply that the hidden population lacks grid-like spatial structure.

The goal of this analysis is therefore to test whether the hidden representation contains linearly recoverable grid-like modes after a constrained invertible change of basis.

---

## 2. Core Idea

Instead of only asking:

> Does any original hidden unit look like a grid cell?

we ask:

> Does the hidden representation contain grid-like spatial modes under a constrained invertible linear transformation?

Given hidden states

$$
h_t \in \mathbb{R}^N
$$

we introduce a basis transformation

$$
z_t = h_t W
$$

where

$$
W \in \mathbb{R}^{N \times N}
$$

is invertible.

We then analyze whether some transformed coordinates

$$
z_j(t) = (h_t W)_j
$$

show grid-like spatial maps.

This analysis should not be interpreted as redefining biological grid cells. The transformed coordinates are not necessarily biological neuron analogues. The intended interpretation is representation-level:

> The ANN hidden population may contain grid-like spatial modes that are not aligned with the original hidden-unit axes.

---

## 3. Mathematical Formulation

Let $x_t \in \Omega \subset \mathbb{R}^2$ denote the agent position at time $t$, and let $h_t \in \mathbb{R}^N$ denote the ANN hidden state.

For any linear direction $w \in \mathbb{R}^N$, define the projected activity:

$$
z_w(t) = h_t^\top w
$$

The spatial response map of this projected activity is

$$
R_w(x) = \mathbb{E}[z_w(t) \mid x_t = x]
$$

In practice, $R_w(x)$ is estimated by binning the environment and averaging projected activity within each spatial bin, optionally with occupancy normalization and spatial smoothing.

A standard grid score can then be computed from $R_w(x)$, for example by constructing its spatial autocorrelogram and measuring its rotational symmetry at $60^\circ$ and $120^\circ$ relative to control rotations at $30^\circ$, $90^\circ$, and $150^\circ$.

The representation-level question becomes:

$$
\max_{w \in \mathcal{W}} G(R_w)
$$

where $G(\cdot)$ is a gridness measure and $\mathcal{W}$ is a constrained set of allowable linear directions.

Equivalently, for a full basis transformation $W = [w_1, \dots, w_N]$, we ask whether some coordinates of

$$
h_t W
$$

have significant grid-like spatial maps.

---

## 4. Constraint on the Basis Transform

The transformation $W$ must be constrained. If arbitrary linear transformations are allowed, it may be possible to overfit noise or artificially create grid-like patterns through high-dimensional linear combinations.

The default and preferred constraint is:

$$
W \in O(N)
$$

where $O(N)$ is the orthogonal group:

$$
W^\top W = I
$$

This means:

$$
W^{-1} = W^\top
$$

An orthogonal transformation is preferred because:

1. it is always invertible;
2. it preserves pairwise Euclidean distances in hidden space;
3. it does not arbitrarily amplify or suppress hidden dimensions;
4. it corresponds to a change of basis rather than a distorted reparameterization;
5. it makes the interpretation of transformed hidden coordinates cleaner.

A more general invertible transformation,

$$
W \in GL(N)
$$

may be considered as an extension, but it should not be the default unless additional constraints are imposed, such as a bound on the condition number:

$$
\kappa(W) \leq \kappa_{\max}
$$

Without such constraints, the result may be difficult to interpret.

---

## 5. Fitting the Transform

The transform $W$ should be fitted using only a fitting subset of trajectories or samples.

Let the data be split into two disjoint subsets:

$$
D_{\text{fit}}
$$

and

$$
D_{\text{test}}
$$

The fitting procedure should use only $D_{\text{fit}}$ to identify candidate directions or a candidate basis transformation.

Possible fitting objectives include:

1. maximizing conventional grid score on projected spatial maps;
2. maximizing hexagonal spectral energy in Fourier space;
3. finding orthogonal directions whose spatial maps show strong $60^\circ$ rotational symmetry;
4. using candidate projections or spectral methods to identify grid-like modes;
5. searching for multiple orthogonal grid-like directions and completing them into a full orthogonal basis.

The exact algorithm can be adapted to the existing project structure. The important point is not the specific optimization method, but the separation between fitting and validation.

The transform should be conceptualized as:

$$
W_{\text{fit}} = \operatorname{FitW}(D_{\text{fit}})
$$

No information from $D_{\text{test}}$ should be used when fitting or selecting $W$.

### Initial implementation: spatial-PCA baseline

The first implementation in this repository is a conservative baseline, not a direct optimizer for:

$$
\max_{w \in \mathcal{W}} G(R_w)
$$

It uses spatial-PCA directions fitted on the fitting trajectories:

1. bin fit-set positions into the same spatial bins used for rate maps;
2. compute one mean hidden vector per occupied spatial bin;
3. stack those means into a matrix $M_{\text{fit}} \in \mathbb{R}^{B \times N}$;
4. column-center $M_{\text{fit}}$;
5. compute an SVD:

$$
M_{\text{centered}} = U S V^\top
$$

6. use the top $K$ right singular vectors as:

$$
W_{\text{top}} = V_{1:K}^\top
$$

These directions are orthonormal and are evaluated on held-out trajectories as:

$$
z_t = (h_t - \mu_{\text{fit}}) W_{\text{top}}
$$

This initial baseline does **not** guarantee that $H W_{\text{top}}$ has high grid score, because PCA maximizes spatial-bin mean variance rather than gridness. Its interpretation is therefore narrower:

> The spatially dominant linear modes of the hidden population may or may not show held-out grid-like structure.

When enabled with `analysis.enabled=true` and `analysis.compute_linear_spatial_pca=true`, the current implementation writes independent `linear_spatial_pca_*` artifacts under `eval_stats/`:

```text
linear_spatial_pca_modes_epoch_XXXX.csv
linear_spatial_pca_modes_epoch_XXXX.npz
linear_spatial_pca_summary_epoch_XXXX.json
linear_spatial_pca_overview_epoch_XXXX.pdf
linear_spatial_pca_overview_epoch_XXXX.png
linear_spatial_pca_mode_maps_epoch_XXXX.pdf
```

The CSV/NPZ/JSON artifacts include fit/test grid scores, held-out `T_real`, shuffle-null `T_null` samples when configured, `W_top`, fit/test split indices, fit/test projected rate maps, held-out SACs, held-out-score mode ordering, and loading participation-ratio diagnostics. The overview plot shows held-out `grid_score_60_test` by mode, fit variance explained and cumulative fit variance explained, the shuffle-null `T` panel or a no-shuffle placeholder, fit variance versus held-out gridness, and the `W_top.T` loading heatmap. The mode-map PDF ranks modes by held-out `grid_score_60_test` and shows up to `analysis.linear_spatial_pca_plot_top_n` rows with fit rate map, held-out test rate map, held-out SAC, and loading weights.

These visualizations are diagnostics for the spatial-PCA baseline. They should not be labeled as optimized grid modes or used to imply that `W_top` came from a grid-score-optimized search.

It should not be reported as evidence that the implementation has solved the full grid-score-search problem. Direct grid-score optimization, hexagonal template regression, or spectral grid-mode search should be treated as later extensions with their own shuffle baseline that repeats the full search.

---

## 6. Held-Out Validation

After fitting $W$ on $D_{\text{fit}}$, the transformed hidden states should be evaluated on held-out data:

$$
z_t = h_t W_{\text{fit}}, \qquad t \in D_{\text{test}}
$$

For each transformed coordinate $j$, estimate its held-out spatial map:

$$
R_j^{\text{test}}(x)
=
\mathbb{E}[(h_t W_{\text{fit}})_j \mid x_t = x,\ t \in D_{\text{test}}]
$$

Then compute the standard grid score:

$$
G_j^{\text{test}} = G(R_j^{\text{test}})
$$

The main reported gridness should come from held-out data, not from the fitting data.

A useful summary statistic is:

$$
T_{\text{real}} = \max_j G_j^{\text{test}}
$$

or, if only the top $K$ candidate modes are considered:

$$
T_{\text{real}} = \max_{j \leq K} G_j^{\text{test}}
$$

The result should be interpreted as meaningful only if the grid-like structure generalizes to held-out trajectories.

---

## 7. Shuffle / Null Baseline

A proper null baseline is essential.

The null baseline should test whether the same analysis pipeline can find grid-like modes when the relationship between hidden states and spatial positions is disrupted.

Possible shuffle strategies include:

1. circularly shifting hidden states relative to positions;
2. permuting trajectory segments;
3. shuffling positions while preserving some temporal structure;
4. using any project-appropriate null model that breaks the hidden-state-to-position relationship while preserving relevant marginal statistics.

The key requirement is:

> The full $W$-search procedure must be repeated for each shuffle.

Incorrect baseline:

```text
Fit W on real data.
Shuffle the data.
Evaluate the same W on shuffled data.
```

Correct baseline:

```text
For each shuffle:
    shuffle the relationship between hidden states and positions
    fit W again on shuffled fitting data
    evaluate the shuffled W on shuffled held-out data
```

Formally, for shuffle $b$:

$$
W_{\text{null}}^{(b)}
=
\operatorname{FitW}(D_{\text{fit}}^{(b,\text{shuffle})})
$$

Then evaluate:

$$
T_{\text{null}}^{(b)}
=
\max_j
G\left(
R_{j}^{(b,\text{shuffle-test})}
\right)
$$

where the transformed coordinates are computed using $W_{\text{null}}^{(b)}$.

A p-value can be estimated as:

$$
p
=
\frac{
1 + \sum_b \mathbf{1}[T_{\text{null}}^{(b)} \geq T_{\text{real}}]
}{
1 + B
}
$$

where $B$ is the number of shuffles.

This ensures that the real result is compared against a fair null distribution with the same search flexibility.

---

## 8. Stability Checks

If possible, the analysis should also check whether the discovered grid-like modes are stable.

Useful stability checks include:

1. fitting $W$ on different trajectory splits;
2. bootstrapping trajectories or spatial samples;
3. checking whether the same or similar grid-like subspace is recovered;
4. checking whether grid scale and orientation are stable across splits;
5. checking whether held-out grid scores remain consistently above the shuffle baseline.

If multiple grid-like directions are identified, their span can be treated as a candidate grid-like subspace. Stability of this subspace can be measured using subspace similarity or principal angles between the fitted directions from different splits.

The goal is to distinguish a robust representation-level structure from a one-off projection artifact.

---

## 9. Optional Functional Relevance Analysis

The presence of linearly recoverable grid-like modes does not by itself prove that the network uses those modes for behavior or task performance.

If the project context allows, a stronger analysis is to test functional relevance.

For example, after identifying a candidate grid-like subspace $Q$, one can compare performance using:

$$
h_t
$$

versus hidden states with the candidate subspace removed:

$$
h_t^{\text{ablated}}
=
h_t - h_t Q Q^\top
$$

If removing the grid-like subspace impairs position decoding, path integration, navigation, or another task-relevant readout, this provides stronger evidence that the grid-like modes are functionally relevant.

This step is optional and depends on the project architecture.

---

## 10. Interpretation of Results

If the held-out grid scores are significant relative to the shuffle baseline, the correct interpretation is:

> The hidden representation contains held-out, shuffle-significant, linearly recoverable grid-like spatial modes under a constrained invertible basis transformation.

This is different from claiming:

> The ANN contains biological grid cells.

The transformed coordinates should be described as:

```text
linear grid-like modes
```

or

```text
transformed grid-like coordinates
```

rather than biological grid cells.

The analysis supports a representation-level claim:

> Grid-like spatial structure may exist in the hidden population even when it is not aligned with the original hidden-unit axes.

It does not by itself establish that:

1. the network uses the grid-like modes causally;
2. the modes are biological analogues of MEC grid cells;
3. path integration necessarily produces grid cells;
4. the original hidden units should be interpreted as neurons.

---

## 11. Important Caveats

This analysis is motivated by the basis ambiguity of ANN hidden states.

Important caveats:

1. Original hidden-unit gridness is basis-dependent.
2. A freely chosen $W$ can produce false positives.
3. $W$ should therefore be constrained, preferably orthogonal by default.
4. $W$ must be fitted only on fitting data.
5. Gridness must be reported on held-out data.
6. Shuffle baselines must repeat the full $W$-search procedure.
7. The result should be reported at the population-representation level.
8. Best-looking examples should not replace population-level statistics.
9. The transformed modes should not be overinterpreted as biological neurons.
10. Functional relevance requires additional ablation or decoding analysis.

---

## 12. Summary

This analysis extends standard hidden-unit gridness analysis by accounting for the basis ambiguity of ANN hidden states.

Instead of only measuring whether individual original hidden units are grid-like, it asks whether the hidden population contains grid-like spatial modes that can be recovered by a constrained invertible linear transformation.

The core pipeline is:

```text
1. Collect hidden states and positions from the trained ANN.
2. Split data into fitting and held-out evaluation subsets.
3. Fit a constrained invertible transform W, preferably orthogonal, using only fitting data.
4. Transform hidden states as z = hW.
5. Compute spatial maps and standard grid scores for transformed coordinates on held-out data.
6. Compare against shuffle baselines that repeat the full W-search procedure.
7. Interpret significant results as linearly recoverable grid-like modes in the hidden representation.
```

The central claim tested by this analysis is:

$$
\text{The hidden representation contains grid-like structure in its linear population span,}
$$

not necessarily:

$$
\text{The original hidden units are grid cells.}
$$
