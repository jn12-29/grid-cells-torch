# Supplement: Suggested Free Optimization Strategy for Linear Grid Modes

This document is a supplement to `linear_grid_mode_analysis.md`.

The main document defines the representation-level question:

> Does the ANN hidden representation contain grid-like spatial modes under a constrained invertible linear basis transformation?

This supplement suggests one possible strategy for finding such a transformation. It is not meant to be the only valid implementation. The project may adapt or replace the details depending on existing code, data structure, and analysis utilities.

The key recommendation is:

> Instead of matching predefined grid templates, directly optimize for robust grid-like structure under strong constraints and held-out validation.

---

## 1. Goal of This Suggested Strategy

We want to find a constrained linear transformation

$$
z_t = h_t W
$$

such that some transformed coordinates

$$
z_j(t) = (h_t W)_j
$$

show grid-like spatial maps.

However, the transformation should not be completely free. A highly flexible linear transform can overfit the finite trajectory samples or amplify noisy directions. Therefore, the suggested strategy is to optimize a small number of orthogonal directions

$$
Q = [q_1, \dots, q_K] \in \mathbb{R}^{N \times K}
$$

with

$$
Q^\top Q = I_K
$$

and then optionally complete them into a full orthogonal basis

$$
W = [Q, Q_\perp] \in O(N)
$$

The first \(K\) coordinates of \(h_t W\) are then treated as candidate linear grid modes.

---

## 2. High-Level Optimization Problem

Given a population spatial map

$$
M_{\text{fit}} \in \mathbb{R}^{B \times N}
$$

built only from fitting data, the spatial map of direction \(q_j\) is

$$
r_j = M_{\text{fit}} q_j
$$

The idealized objective is:

$$
\max_Q
\sum_{j=1}^{K}
G(M_{\text{fit}}q_j)
$$

subject to

$$
Q^\top Q = I_K
$$

where \(G(\cdot)\) is the standard grid score.

In practice, the standard grid score is difficult to optimize directly. Therefore, use a differentiable surrogate during fitting, and reserve the conventional grid score for validation.

The practical optimization problem is:

$$
\max_Q
\sum_{j=1}^{K}
\widetilde{G}(M_{\text{fit}}q_j)
-
\text{regularization}(Q)
$$

where \(\widetilde{G}\) is a differentiable gridness surrogate.

---

## 3. Suggested Constraint: Optimize on the Stiefel Manifold

The default constraint should be:

$$
Q^\top Q = I_K
$$

This means the selected directions are orthonormal.

There are several ways to enforce this:

1. parameterize an unconstrained matrix \(A \in \mathbb{R}^{N \times K}\) and compute \(Q = \operatorname{qr}(A)\) at each forward pass;
2. optimize \(Q\) directly with a Stiefel-manifold optimizer;
3. use gradient steps followed by QR retraction;
4. use a strong orthogonality penalty if exact orthogonality is inconvenient.

The preferred interpretation is that \(Q\) defines the first \(K\) axes of an orthogonal change of basis. This keeps the analysis close to a basis change rather than an arbitrary distortion of hidden space.

If a full \(W\) is required, complete \(Q\) to an orthogonal matrix:

$$
W = [Q, Q_\perp]
$$

where

$$
W^\top W = I
$$

---

## 4. Differentiable Gridness Surrogate

The conventional grid score usually involves:

1. constructing a spatial rate map;
2. computing a spatial autocorrelogram;
3. rotating the autocorrelogram;
4. computing correlations at several angles;
5. taking hard min and max operations.

This is not ideal for gradient optimization.

A differentiable surrogate can follow the same structure but replace non-smooth pieces with differentiable approximations.

For a projected spatial map \(r\), reshape it into a 2D map \(R\). Compute a differentiable autocorrelogram \(A\), for example using FFT-based autocorrelation or differentiable convolution.

Let

$$
\rho(\theta)
$$

be the correlation between \(A\) and its rotated version within an annular mask.

The grid angles are:

$$
\Theta_{\text{grid}} = \{60^\circ, 120^\circ\}
$$

The control angles are:

$$
\Theta_{\text{control}} = \{30^\circ, 90^\circ, 150^\circ\}
$$

The conventional structure is:

$$
G(R)
=
\min_{\theta \in \Theta_{\text{grid}}}
\rho(\theta)
-
\max_{\theta \in \Theta_{\text{control}}}
\rho(\theta)
$$

For optimization, use soft approximations:

$$
\operatorname{softmin}_\tau(a_1,\dots,a_m)
=
-\tau \log \sum_i \exp(-a_i/\tau)
$$

$$
\operatorname{softmax}_\tau(a_1,\dots,a_m)
=
\tau \log \sum_i \exp(a_i/\tau)
$$

Then define:

$$
\widetilde{G}(R)
=
\operatorname{softmin}_\tau
\{
\rho(60^\circ),
\rho(120^\circ)
\}
-
\operatorname{softmax}_\tau
\{
\rho(30^\circ),
\rho(90^\circ),
\rho(150^\circ)
\}
$$

This surrogate is only for fitting. The final result should still be evaluated using the project’s standard grid-score implementation.

---

## 5. Robustness Across Internal Fitting Splits

A major risk is overfitting a single estimated spatial map. To reduce this risk, split the fitting data into several internal subsets:

$$
D_{\text{fit}}^{(1)}, \dots, D_{\text{fit}}^{(S)}
$$

Build one population map per subset:

$$
M_{\text{fit}}^{(s)}
$$

For each direction \(q_j\), compute:

$$
\widetilde{G}_{s,j}
=
\widetilde{G}(M_{\text{fit}}^{(s)}q_j)
$$

The objective should reward directions that are grid-like across multiple internal fitting subsets.

A simple robust objective is:

$$
\mathcal{J}_{\text{robust}}
=
\frac{1}{K}
\sum_{j=1}^{K}
\operatorname{softmin}_{s}
\left(
\widetilde{G}_{s,j}
\right)
$$

This encourages each direction to perform well even on its weakest fitting subset.

Another option is:

$$
\mathcal{J}_{\text{robust}}
=
\frac{1}{K}
\sum_{j=1}^{K}
\left[
\operatorname{mean}_s(\widetilde{G}_{s,j})
-
\gamma \operatorname{std}_s(\widetilde{G}_{s,j})
\right]
$$

This rewards high average gridness and penalizes instability.

---

## 6. Suggested Regularization

Because this method directly optimizes gridness, regularization is important.

A suggested objective is:

$$
\mathcal{L}(Q)
=
-\mathcal{J}_{\text{robust}}
+
\lambda_{\text{orth}}\mathcal{L}_{\text{orth}}
+
\lambda_{\text{rough}}\mathcal{L}_{\text{rough}}
+
\lambda_{\text{amp}}\mathcal{L}_{\text{amp}}
+
\lambda_{\text{stability}}\mathcal{L}_{\text{stability}}
$$

where the signs assume minimization.

### Orthogonality

If orthogonality is not enforced exactly, use:

$$
\mathcal{L}_{\text{orth}}
=
\|Q^\top Q - I\|_F^2
$$

If QR or Stiefel optimization is used, this penalty may not be necessary.

### Spatial Roughness

To discourage noisy high-frequency artifacts:

$$
\mathcal{L}_{\text{rough}}
=
\frac{1}{K}
\sum_{j=1}^{K}
\|\nabla R_j\|_2^2
$$

where

$$
R_j = \operatorname{reshape}(M_{\text{fit}}q_j)
$$

This penalty should be used carefully. Too much smoothing may suppress real grid-like periodicity.

### Minimum Variance / Non-Degeneracy

Correlation-based scores can become unstable for near-zero maps. To discourage degenerate projections:

$$
\mathcal{L}_{\text{amp}}
=
\frac{1}{K}
\sum_{j=1}^{K}
\max
\left(
0,
\sigma_{\min}
-
\operatorname{std}(M_{\text{fit}}q_j)
\right)^2
$$

### Split-to-Split Stability

To encourage the same direction to produce consistent maps across internal fitting splits:

$$
\mathcal{L}_{\text{stability}}
=
\frac{1}{K}
\sum_{j=1}^{K}
\operatorname{mean}_{s \neq s'}
\left[
1 -
\operatorname{corr}
\left(
M_{\text{fit}}^{(s)}q_j,
M_{\text{fit}}^{(s')}q_j
\right)
\right]
$$

---

## 7. Data Augmentation During Optimization

The optimization should not rely on one fixed binned map. It is recommended to use perturbations during fitting.

Possible perturbations include:

1. bootstrap resampling trajectories;
2. resampling trajectory segments;
3. dropping a small fraction of time points;
4. dropping a small fraction of spatial bins;
5. adding small noise to hidden states;
6. varying the smoothing bandwidth within a fixed range;
7. recomputing maps from different internal fitting subsets.

The discovered \(Q\) should remain useful under these perturbations.

The purpose is not to create extra evidence, but to prevent the optimizer from exploiting fragile details of one map estimate.

---

## 8. Suggested Optimization Procedure

A simple practical procedure is:

```text
Input:
    hidden states H
    positions X
    number of modes K
    fitting split D_fit
    validation split D_val
    test split D_test

Step 1:
    Build one or more population maps from D_fit.
    Keep D_test completely unused during fitting.

Step 2:
    Initialize A randomly with shape N x K.

Step 3:
    At each optimization step:
        Q = orthonormalize(A), for example using QR.

        For each internal fitting map:
            compute projected maps R_j = M_fit_s @ q_j
            compute differentiable gridness surrogate for each R_j

        Combine scores with a robust objective across fitting splits.

        Add regularization terms:
            orthogonality if not exactly enforced
            roughness penalty
            minimum variance penalty
            split-to-split stability penalty

        Update A with a gradient optimizer.

Step 4:
    Use validation data for early stopping or hyperparameter choice.

Step 5:
    After fitting, freeze Q.

Step 6:
    Evaluate conventional grid score on D_test only.

Step 7:
    If needed, complete Q into a full orthogonal W.
```

The final \(Q\) should be selected without looking at the final test grid scores.

---

## 9. Held-Out Evaluation

After optimization, compute held-out spatial maps:

$$
r_j^{\text{test}}
=
M_{\text{test}}q_j
$$

Then evaluate the conventional grid score:

$$
G_j^{\text{test}}
=
G(r_j^{\text{test}})
$$

The main statistic may be:

$$
T_{\text{real}}
=
\max_{j \leq K}
G_j^{\text{test}}
$$

if the goal is to detect at least one grid-like mode.

Alternatively:

$$
T_{\text{real}}
=
\frac{1}{K}
\sum_{j=1}^{K}
G_j^{\text{test}}
$$

if the goal is to detect a multi-dimensional grid-like subspace.

The choice should be made before looking at test results.

---

## 10. Shuffle Baseline

Because this method uses a flexible optimizer, the shuffle baseline must repeat the full optimization.

For each shuffle:

1. disrupt the relationship between hidden states and positions;
2. rebuild fitting, validation, and test maps under the shuffled relationship;
3. optimize \(Q\_{\text{null}}\) from scratch using the same objective and regularization;
4. use the same early-stopping rule;
5. compute conventional grid scores on shuffled held-out data;
6. record the null statistic

$$
T_{\text{null}}^{(b)}
$$

The p-value is:

$$
p
=
\frac{
1 + \sum_b \mathbf{1}[T_{\text{null}}^{(b)} \geq T_{\text{real}}]
}{
1+B
}
$$

The null baseline should have the same search freedom as the real analysis.

---

## 11. How to Interpret This Suggested Method

This method freely searches for grid-like directions in the hidden population. It is stronger than simply checking the original hidden axes, but it is also easier to overfit.

Therefore, the method should be interpreted only together with:

1. orthogonal or strongly constrained directions;
2. internal robustness during fitting;
3. held-out conventional grid scores;
4. shuffle baselines that repeat the full search;
5. stability across data splits or random initializations.

A positive result should be described as:

> The hidden representation contains robust, held-out, shuffle-significant, linearly recoverable grid-like modes under a constrained invertible basis transformation.

It should not be described as:

> The ANN necessarily contains biological grid cells.

---

## 12. Summary

This supplement suggests a free optimization strategy for finding candidate linear grid modes.

The main idea is:

```text
Optimize orthogonal directions Q in hidden-state space so that projected spatial maps
maximize a differentiable gridness surrogate under strong regularization.
Then evaluate the frozen directions on held-out data using the conventional grid score.
Finally, compare against a shuffle baseline that repeats the full optimization.
```

This strategy avoids predefined grid templates while still controlling overfitting through orthogonality, robustness penalties, validation splits, held-out testing, and fair null baselines.
