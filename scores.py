# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Ported to PyTorch rewrite (numpy/scipy/matplotlib only, no TF dependency).
# Changes from original:
#   - Added missing imports: scipy.stats, scipy.ndimage
#   - Replaced deprecated scipy.ndimage.interpolation.rotate with scipy.ndimage.rotate

"""Grid-score computation and rate-map utilities.

This module adapts the original DeepMind scoring logic to a NumPy/SciPy stack
and is used to build rate maps, spatial autocorrelograms, and grid scores from
model activations.

Usage:
    scorer = GridScorer(nbins=32, coords_range=[[-1.1, 1.1], [-1.1, 1.1]], mask_parameters=[])
    ratemap = scorer.calculate_ratemap(xs, ys, activations)
    score_60, score_90, max_60_mask, max_90_mask, sac = scorer.get_scores(ratemap)
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.sparse
import scipy.stats


def circle_mask(size, radius, in_val=1.0, out_val=0.0):
    sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
    x = np.linspace(-sz[0], sz[1], size[1])
    x = np.expand_dims(x, 0)
    x = x.repeat(size[0], 0)
    y = np.linspace(-sz[0], sz[1], size[1])
    y = np.expand_dims(y, 1)
    y = y.repeat(size[1], 1)
    z = np.sqrt(x**2 + y**2)
    z = np.less_equal(z, radius)
    vfunc = np.vectorize(lambda b: b and in_val or out_val)
    return vfunc(z)


class GridScorer(object):
    def __init__(self, nbins, coords_range, mask_parameters, min_max=False):
        self._nbins = nbins
        self._min_max = min_max
        self._coords_range = coords_range
        self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
        self._angle_to_index = {
            angle: index for index, angle in enumerate(self._corr_angles)
        }
        self._masks = [
            (self._get_ring_mask(mask_min, mask_max), (mask_min, mask_max))
            for mask_min, mask_max in mask_parameters
        ]
        self._mask_stack = np.asarray([mask for mask, _ in self._masks], dtype=np.float64)
        self._mask_params = [params for _, params in self._masks]
        self._mask_ring_areas = np.sum(self._mask_stack, axis=(1, 2))
        self._plotting_sac_mask = circle_mask(
            [self._nbins * 2 - 1, self._nbins * 2 - 1],
            self._nbins,
            in_val=1.0,
            out_val=np.nan,
        )

    def calculate_ratemap(self, xs, ys, activations, statistic="mean"):
        return scipy.stats.binned_statistic_2d(
            xs,
            ys,
            activations,
            bins=self._nbins,
            statistic=statistic,
            range=self._coords_range,
        )[0]

    def allocate_ratemap_accumulators(self, n_units, dtype=np.float64):
        """Allocate accumulators for streaming rate-map construction."""
        counts = np.zeros((self._nbins, self._nbins), dtype=np.int64)
        sums = np.zeros((n_units, self._nbins, self._nbins), dtype=dtype)
        return sums, counts

    def _digitize_positions(self, xs, ys):
        """Map coordinates to bin indices matching binned_statistic_2d semantics."""
        x_edges = np.linspace(
            self._coords_range[0][0], self._coords_range[0][1], self._nbins + 1
        )
        y_edges = np.linspace(
            self._coords_range[1][0], self._coords_range[1][1], self._nbins + 1
        )

        x_idx = np.searchsorted(x_edges, xs, side="right") - 1
        y_idx = np.searchsorted(y_edges, ys, side="right") - 1

        x_idx[xs == x_edges[-1]] = self._nbins - 1
        y_idx[ys == y_edges[-1]] = self._nbins - 1

        valid = (
            (x_idx >= 0)
            & (x_idx < self._nbins)
            & (y_idx >= 0)
            & (y_idx < self._nbins)
        )
        return x_idx[valid], y_idx[valid], valid

    def accumulate_ratemaps(self, positions, activations, sums, counts):
        """Accumulate streaming contributions to per-unit rate maps."""
        flat_pos = positions.reshape(-1, positions.shape[-1])
        flat_act = activations.reshape(-1, activations.shape[-1])

        x_idx, y_idx, valid = self._digitize_positions(flat_pos[:, 0], flat_pos[:, 1])
        if not np.any(valid):
            return

        flat_bins = x_idx * self._nbins + y_idx
        n_valid = len(flat_bins)
        n_bins_flat = self._nbins * self._nbins

        # counts: fast bincount
        counts_flat = counts.reshape(-1)
        counts_flat += np.bincount(flat_bins, minlength=n_bins_flat).astype(counts_flat.dtype)

        # sums: sparse one-hot matmul is ~40x faster than np.add.at for large N
        # one_hot (N_valid, n_bins_flat) @ valid_act (N_valid, n_units)
        # = (n_bins_flat, n_units) via one_hot.T @ valid_act
        valid_act = np.asarray(flat_act[valid], dtype=sums.dtype)
        one_hot = scipy.sparse.csr_matrix(
            (np.ones(n_valid, dtype=sums.dtype), (np.arange(n_valid), flat_bins)),
            shape=(n_valid, n_bins_flat),
        )
        sums_flat = sums.reshape(sums.shape[0], -1)
        sums_flat += (one_hot.T @ valid_act).T

    def finalize_ratemaps(self, sums, counts):
        """Convert accumulated sums/counts into mean rate maps with NaNs for empty bins."""
        ratemaps = np.full(sums.shape, np.nan, dtype=sums.dtype)
        valid = counts > 0
        np.divide(
            sums,
            counts[None, :, :],
            out=ratemaps,
            where=valid[None, :, :],
        )
        return ratemaps

    def _get_ring_mask(self, mask_min, mask_max):
        n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
        return circle_mask(n_points, mask_max * self._nbins) * (
            1 - circle_mask(n_points, mask_min * self._nbins)
        )

    def grid_score_60(self, corr):
        if self._min_max:
            return np.minimum(corr[60], corr[120]) - np.maximum(
                corr[30], np.maximum(corr[90], corr[150])
            )
        else:
            return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

    def grid_score_90(self, corr):
        return corr[90] - (corr[45] + corr[135]) / 2

    def calculate_sac(self, seq1):
        seq2 = seq1

        def filter2(b, x):
            stencil = np.rot90(b, 2)
            return scipy.signal.convolve2d(x, stencil, mode="full")

        seq1 = np.nan_to_num(seq1)
        seq2 = np.nan_to_num(seq2)
        ones_seq1 = np.ones(seq1.shape)
        ones_seq1[np.isnan(seq1)] = 0
        ones_seq2 = np.ones(seq2.shape)
        ones_seq2[np.isnan(seq2)] = 0
        seq1[np.isnan(seq1)] = 0
        seq2[np.isnan(seq2)] = 0
        seq1_sq = np.square(seq1)
        seq2_sq = np.square(seq2)
        seq1_x_seq2 = filter2(seq1, seq2)
        sum_seq1 = filter2(seq1, ones_seq2)
        sum_seq2 = filter2(ones_seq1, seq2)
        sum_seq1_sq = filter2(seq1_sq, ones_seq2)
        sum_seq2_sq = filter2(ones_seq1, seq2_sq)
        n_bins = filter2(ones_seq1, ones_seq2)
        n_bins_sq = np.square(n_bins)
        var_seq1 = np.subtract(
            np.divide(sum_seq1_sq, n_bins),
            np.divide(np.square(sum_seq1), n_bins_sq),
        )
        var_seq2 = np.subtract(
            np.divide(sum_seq2_sq, n_bins),
            np.divide(np.square(sum_seq2), n_bins_sq),
        )
        std_seq1 = np.sqrt(np.maximum(var_seq1, 0.0))
        std_seq2 = np.sqrt(np.maximum(var_seq2, 0.0))
        covar = np.subtract(
            np.divide(seq1_x_seq2, n_bins),
            np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq),
        )
        x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2) + 1e-8)
        x_coef = np.real(x_coef)
        x_coef = np.nan_to_num(x_coef)
        return x_coef

    def rotated_sacs(self, sac, angles):
        return [scipy.ndimage.rotate(sac, angle, reshape=False) for angle in angles]

    def rotated_sacs_batch(self, sacs, angles):
        """Rotate a batch of SACs for all requested angles."""
        return np.stack(
            [
                scipy.ndimage.rotate(sacs, angle, axes=(1, 2), reshape=False)
                for angle in angles
            ],
            axis=1,
        )

    def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
        masked_sac = sac * mask
        ring_area = np.sum(mask)
        masked_sac_mean = np.sum(masked_sac) / ring_area
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.sum(masked_sac_centered**2) / ring_area + 1e-5
        corrs = dict()
        for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
            corrs[angle] = cross_prod / variance
        return self.grid_score_60(corrs), self.grid_score_90(corrs), variance

    def get_scores(self, rate_map):
        sac = self.calculate_sac(rate_map)
        rotated_sacs = self.rotated_sacs(sac, self._corr_angles)
        scores = [
            self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
            for mask, _ in self._masks
        ]
        scores_60, scores_90, _ = map(np.asarray, zip(*scores))
        max_60_ind = np.argmax(scores_60)
        max_90_ind = np.argmax(scores_90)
        return (
            scores_60[max_60_ind],
            scores_90[max_90_ind],
            self._masks[max_60_ind][1],
            self._masks[max_90_ind][1],
            sac,
        )

    def calculate_sac_batch(self, ratemaps):
        """Vectorized SAC for an (N, H, W) batch via numpy rfft2.

        Equivalent to calling calculate_sac() on each ratemap independently,
        but uses batched FFT instead of N×6 serial convolve2d calls.

        The original scalar implementation computes:
            filter2(b, x) = convolve2d(x, rot90(b, 2), 'full')
                          = correlate2d(x, b, 'full')
        Via FFT:
            correlate2d(x, b) = fftshift(irfft2(rfft2(x) * conj(rfft2(b))))
        where fftshift is needed to place zero-lag at the center of the output
        (index [H-1, W-1]) to match scipy's 'full' mode layout.
        """
        N, H, W = ratemaps.shape
        seq = np.nan_to_num(ratemaps)  # NaN → 0, matching calculate_sac
        # In calculate_sac, ones is built AFTER nan_to_num, so np.isnan(seq) is always
        # False and ones is always all-ones — replicate that here.
        ones = np.ones((N, H, W), dtype=seq.dtype)
        seq_sq = seq ** 2

        fft_shape = (2 * H - 1, 2 * W - 1)
        Fs  = np.fft.rfft2(seq,    s=fft_shape, axes=(-2, -1))
        Fo  = np.fft.rfft2(ones,   s=fft_shape, axes=(-2, -1))
        Fss = np.fft.rfft2(seq_sq, s=fft_shape, axes=(-2, -1))

        def corr(Fx, Fb):
            """correlate2d(x, b) = fftshift(irfft2(rfft2(x) * conj(rfft2(b))))."""
            return np.fft.fftshift(
                np.fft.irfft2(Fx * np.conj(Fb), s=fft_shape, axes=(-2, -1)),
                axes=(-2, -1),
            )

        # Map to filter2 calls in calculate_sac (seq1 == seq2 == seq):
        # filter2(seq,     seq)     = correlate2d(seq,    seq)
        # filter2(seq,     ones)    = correlate2d(ones,   seq)
        # filter2(ones,    seq)     = correlate2d(seq,    ones)
        # filter2(seq_sq,  ones)    = correlate2d(ones,   seq_sq)
        # filter2(ones,    seq_sq)  = correlate2d(seq_sq, ones)
        # filter2(ones,    ones)    = correlate2d(ones,   ones)
        seq1_x_seq2 = corr(Fs,  Fs)
        sum_seq1    = corr(Fo,  Fs)
        sum_seq2    = corr(Fs,  Fo)
        sum_seq1_sq = corr(Fo,  Fss)
        sum_seq2_sq = corr(Fss, Fo)
        n_bins      = corr(Fo,  Fo)
        n_bins_sq   = n_bins ** 2

        var1  = sum_seq1_sq / n_bins - sum_seq1 ** 2 / n_bins_sq
        var2  = sum_seq2_sq / n_bins - sum_seq2 ** 2 / n_bins_sq
        covar = seq1_x_seq2 / n_bins - sum_seq1 * sum_seq2 / n_bins_sq
        denom = np.sqrt(np.maximum(var1, 0.0)) * np.sqrt(np.maximum(var2, 0.0)) + 1e-8
        x_coef = covar / denom
        return np.nan_to_num(np.real(x_coef))

    def get_scores_batch(self, ratemaps):
        """Score a chunk of rate maps with batched rotation and mask correlation."""
        ratemaps = np.asarray(ratemaps)
        if ratemaps.ndim == 2:
            ratemaps = ratemaps[np.newaxis, ...]

        sacs = self.calculate_sac_batch(ratemaps)
        rotated_sacs = self.rotated_sacs_batch(sacs, self._corr_angles)

        mask_stack = self._mask_stack[None, :, None, :, :]
        ring_areas = self._mask_ring_areas[None, :, None]

        masked_sacs = sacs[:, None, :, :] * self._mask_stack[None, :, :, :]
        masked_sac_means = np.sum(masked_sacs, axis=(-1, -2)) / self._mask_ring_areas[None, :]
        masked_sac_centered = (
            masked_sacs - masked_sac_means[:, :, None, None]
        )[:, :, None, :, :] * mask_stack
        variance = np.sum(masked_sac_centered ** 2, axis=(-1, -2)) / ring_areas + 1e-5

        masked_rotated_sacs = (
            rotated_sacs[:, None, :, :, :] - masked_sac_means[:, :, None, None, None]
        ) * mask_stack
        cross_prod = np.sum(
            masked_sac_centered * masked_rotated_sacs,
            axis=(-1, -2),
        ) / ring_areas
        corrs = cross_prod / variance

        corr_30 = corrs[:, :, self._angle_to_index[30]]
        corr_45 = corrs[:, :, self._angle_to_index[45]]
        corr_60 = corrs[:, :, self._angle_to_index[60]]
        corr_90 = corrs[:, :, self._angle_to_index[90]]
        corr_120 = corrs[:, :, self._angle_to_index[120]]
        corr_135 = corrs[:, :, self._angle_to_index[135]]
        corr_150 = corrs[:, :, self._angle_to_index[150]]

        if self._min_max:
            scores_60 = np.minimum(corr_60, corr_120) - np.maximum(
                corr_30, np.maximum(corr_90, corr_150)
            )
        else:
            scores_60 = (corr_60 + corr_120) / 2 - (corr_30 + corr_90 + corr_150) / 3
        scores_90 = corr_90 - (corr_45 + corr_135) / 2

        max_60_ind = np.argmax(scores_60, axis=1)
        max_90_ind = np.argmax(scores_90, axis=1)
        unit_indices = np.arange(ratemaps.shape[0])

        return (
            scores_60[unit_indices, max_60_ind],
            scores_90[unit_indices, max_90_ind],
            [self._mask_params[index] for index in max_60_ind],
            [self._mask_params[index] for index in max_90_ind],
            sacs,
        )

    def plot_ratemap(self, ratemap, ax=None, title=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.imshow(ratemap, interpolation="none", *args, **kwargs)
        ax.axis("off")
        if title is not None:
            ax.set_title(title)

    def plot_sac(self, sac, mask_params=None, ax=None, title=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        useful_sac = sac * self._plotting_sac_mask
        ax.imshow(useful_sac, interpolation="none", *args, **kwargs)
        if mask_params is not None:
            center = self._nbins - 1
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[0] * self._nbins,
                    fill=False,
                    edgecolor="k",
                )
            )
            ax.add_artist(
                plt.Circle(
                    (center, center),
                    mask_params[1] * self._nbins,
                    fill=False,
                    edgecolor="k",
                )
            )
        ax.axis("off")
        if title is not None:
            ax.set_title(title)
