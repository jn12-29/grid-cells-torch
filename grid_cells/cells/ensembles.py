"""Cell-ensemble definitions for place and head-direction supervision.

These classes implement probabilistic and independent-activation target
encodings while keeping target generation in NumPy and loss evaluation
compatible with PyTorch tensors.

Usage:
    pc = PlaceCellEnsemble(...)
    hdc = HeadDirectionCellEnsemble(...)
    targets = pc.get_targets(target_pos_np)
    loss = pc.loss(pc_logits, pc_targets)
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.special import logsumexp
import torch
import torch.nn.functional as F


PC_TARGET_FAMILIES = {
    "gaussian",
    "difference_of_gaussians",
    "true_difference_of_gaussians",
}
PC_TARGET_NORMALIZATIONS = {"global", "none"}
PC_DIFFERENCE_TARGET_FAMILIES = {
    "difference_of_gaussians",
    "true_difference_of_gaussians",
}


class CellEnsemble(ABC):
    """Base class for cell ensembles (place cells, head direction cells, etc.)."""

    def __init__(
        self,
        n_cells,
        soft_targets,
        soft_init,
        decode_type="analytic",
        decode_top_k=3,
    ):
        """
        Args:
            n_cells: number of cells in the ensemble.
            soft_targets: one of "softmax", "voronoi", "sample", "normalized".
            soft_init: same options as soft_targets, or "zeros". If None,
                       defaults to soft_targets.
            decode_type: position/heading decoder mode
                ("analytic", "weighted_mean", or PC-only "top_k").
            decode_top_k: place-cell vote count when decode_type is "top_k".
        """
        self.n_cells = n_cells
        self.soft_targets = soft_targets
        self.soft_init = soft_init if soft_init is not None else soft_targets
        self.decode_type = decode_type
        self.decode_top_k = decode_top_k

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def unnor_logpdf(self, x):
        """Un-normalised log probability density.

        Args:
            x: numpy array of shape (batch, seq, input_dim).

        Returns:
            numpy array of shape (batch, seq, n_cells).
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def log_posterior(self, x):
        """Normalised log posterior over cells.

        Args:
            x: numpy array of shape (batch, seq, input_dim).

        Returns:
            numpy array of shape (batch, seq, n_cells).
        """
        logp = self.unnor_logpdf(x)
        # logsumexp over the n_cells axis (last axis)
        lse = logsumexp(logp, axis=-1, keepdims=True)
        return logp - lse

    @staticmethod
    def _softmax(logits):
        """Numerically stable softmax along last axis (numpy)."""
        shifted = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _one_hot_max(logits):
        """One-hot encoding of the argmax along the last axis."""
        idx = np.argmax(logits, axis=-1)          # (batch, seq)
        out = np.zeros_like(logits)
        # advanced indexing — works for arbitrary leading dims
        np.put_along_axis(out, idx[..., np.newaxis], 1.0, axis=-1)
        return out

    def _softmax_sample(self, logits):
        """Sample a one-hot vector from the softmax distribution."""
        probs = CellEnsemble._softmax(logits)
        return self._sample_probs(probs)

    def _sample_probs(self, probs):
        """Sample a one-hot vector from normalized probabilities."""
        batch_shape = probs.shape[:-1]
        n = probs.shape[-1]
        flat_probs = probs.reshape(-1, n)
        # vectorised categorical sample
        cumprobs = np.cumsum(flat_probs, axis=-1)
        u = self._rng.uniform(size=(flat_probs.shape[0], 1))
        samples = (u < cumprobs).argmax(axis=-1)   # (batch*seq,)
        out = np.zeros_like(flat_probs)
        out[np.arange(flat_probs.shape[0]), samples] = 1.0
        return out.reshape(batch_shape + (n,))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _apply_mode(self, x, mode):
        """Compute cell activations according to the given mode.

        Args:
            x: numpy array (batch, seq, input_dim).
            mode: one of "softmax", "voronoi", "sample", "normalized", "zeros".

        Returns:
            numpy array of shape (batch, seq, n_cells).
        """
        if mode == "zeros":
            batch, seq = x.shape[0], x.shape[1]
            return np.zeros((batch, seq, self.n_cells), dtype=np.float32)
        elif mode == "normalized":
            logpdf = self.activation_logpdf(x)
            return np.exp(logpdf)
        elif mode == "softmax":
            return self._softmax(self.log_posterior(x))
        elif mode == "sample":
            return self._softmax_sample(self.log_posterior(x))
        elif mode == "voronoi":
            return self._one_hot_max(self.log_posterior(x))
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def get_targets(self, x):
        """Compute training targets for a trajectory batch.

        Args:
            x: numpy array of shape (batch, seq_len, input_dim).

        Returns:
            numpy array of shape (batch, seq_len, n_cells).
        """
        return self._apply_mode(x, self.soft_targets)

    def activation_logpdf(self, x):
        """Return log activations for independent-cell normalized targets."""
        return self.unnor_logpdf(x)

    def get_init(self, x):
        """Compute initial cell activations for LSTM initialisation.

        Args:
            x: numpy array of shape (batch, 1, input_dim).
               (The seq dimension of 1 is preserved so that the same
               _apply_mode logic works; callers may squeeze as needed.)

        Returns:
            numpy array of shape (batch, 1, n_cells).
        """
        return self._apply_mode(x, self.soft_init)

    def loss(self, predictions, targets, timestep_weights=None):
        """Cross-entropy loss between model predictions and cell targets.

        Args:
            predictions: torch.Tensor of shape (batch, seq_len, n_cells).
                         These are *logits* (pre-softmax / pre-sigmoid).
            targets: numpy array of shape (batch, seq_len, n_cells).
            timestep_weights: optional sequence-length weights for the time
                              reduction. When omitted, all timesteps contribute
                              equally.

        Returns:
            Scalar torch.Tensor (mean over batch and seq dimensions).
        """
        # Convert targets to tensor on the same device as predictions
        targets_t = torch.as_tensor(
            targets, dtype=predictions.dtype, device=predictions.device
        )
        batch, seq, n = predictions.shape

        if self.soft_targets == "normalized":
            # 'normalized' treats each cell as independent binary output (multi-label).
            # Targets are unit-peak activations; labels are smoothed toward 0.5.
            # Sigmoid cross-entropy with label smoothing
            smoothing = 1e-2
            labels = (1.0 - smoothing) * targets_t + smoothing * 0.5
            # binary_cross_entropy_with_logits expects predictions as logits
            loss_per_cell = F.binary_cross_entropy_with_logits(
                predictions, labels, reduction="none"
            )
            loss_per_timestep = loss_per_cell.mean(dim=-1)
        else:
            # Softmax cross-entropy
            # cross_entropy expects (N, C) or (N, C, d1, ...) with class dim = 1
            # Flatten batch and seq into a single dimension
            pred_flat = predictions.reshape(batch * seq, n)       # (B*T, C)
            tgt_flat = targets_t.reshape(batch * seq, n)          # (B*T, C)
            # Use soft labels via log_softmax + sum
            log_probs = F.log_softmax(pred_flat, dim=-1)          # (B*T, C)
            loss_per_timestep = -(tgt_flat * log_probs).sum(dim=-1).reshape(batch, seq)

        if timestep_weights is None:
            return loss_per_timestep.mean()

        weights = torch.as_tensor(
            timestep_weights,
            dtype=predictions.dtype,
            device=predictions.device,
        )
        if weights.ndim != 1 or weights.shape[0] != seq:
            raise ValueError(
                f"timestep_weights must have shape ({seq},), got {tuple(weights.shape)}."
            )
        if not torch.isfinite(weights).all():
            raise ValueError("timestep_weights must be finite.")
        if torch.any(weights < 0.0):
            raise ValueError("timestep_weights must be non-negative.")

        weight_sum = weights.sum()
        if weight_sum <= 0.0:
            raise ValueError("timestep_weights must contain at least one positive value.")

        return (loss_per_timestep * weights.unsqueeze(0)).sum() / (batch * weight_sum)


# ---------------------------------------------------------------------------
# Place Cell Ensemble
# ---------------------------------------------------------------------------

class PlaceCellEnsemble(CellEnsemble):
    """Ensemble of place cells with Gaussian tuning curves."""

    def __init__(
        self,
        n_cells,
        stdev=0.35,
        pos_min=-5.0,
        pos_max=5.0,
        seed=None,
        soft_targets="softmax",
        soft_init=None,
        decode_type="analytic",
        pc_target_family="gaussian",
        pc_target_normalization="global",
        surround_scale=2.0,
        decode_top_k=3,
    ):
        """
        Args:
            n_cells: number of place cells.
            stdev: standard deviation of each cell's Gaussian tuning curve (metres).
            pos_min: lower bound for uniformly sampled cell centres.
            pos_max: upper bound for uniformly sampled cell centres.
            seed: random seed for reproducible cell placement.
            soft_targets: target encoding mode ("softmax"/"voronoi"/"sample"/"normalized").
            soft_init: init encoding mode; defaults to soft_targets if None.
            decode_type: position decoder mode
                ("analytic", "top_k", or "weighted_mean").
            decode_top_k: number of place-cell centers used by top-k decoding.
            pc_target_family: place-cell target family
                ("gaussian"/"difference_of_gaussians"/"true_difference_of_gaussians").
            pc_target_normalization: DoS normalization mode ("global"/"none");
                true DoG records the value but subtracts raw Gaussian responses.
            surround_scale: surround Gaussian sigma multiplier for DoS/DoG targets.
        """
        super().__init__(
            n_cells,
            soft_targets,
            soft_init,
            decode_type=decode_type,
            decode_top_k=decode_top_k,
        )
        if pc_target_family not in PC_TARGET_FAMILIES:
            supported = ", ".join(sorted(PC_TARGET_FAMILIES))
            raise ValueError(
                f"Unsupported pc_target_family={pc_target_family!r}; "
                f"expected one of: {supported}."
            )
        if pc_target_normalization not in PC_TARGET_NORMALIZATIONS:
            supported = ", ".join(sorted(PC_TARGET_NORMALIZATIONS))
            raise ValueError(
                f"Unsupported pc_target_normalization={pc_target_normalization!r}; "
                f"expected one of: {supported}."
            )
        if surround_scale <= 0.0:
            raise ValueError("surround_scale must be positive.")
        if pc_target_family in PC_DIFFERENCE_TARGET_FAMILIES:
            if n_cells < 2:
                raise ValueError("DoS/DoG place-cell targets require at least two cells.")
            if surround_scale <= 1.0:
                raise ValueError("DoS/DoG place-cell targets require surround_scale > 1.")
            if soft_targets == "normalized" or self.soft_init == "normalized":
                raise ValueError(
                    "DoS/DoG place-cell targets are population distributions and "
                    "do not support targets_type/lstm_init_type='normalized'."
                )
        self.pc_target_family = pc_target_family
        self.pc_target_normalization = pc_target_normalization
        self.surround_scale = float(surround_scale)
        self._rng = np.random.RandomState(seed)
        # Cell centres: shape (n_cells, 2)
        self.means = self._rng.uniform(pos_min, pos_max, size=(n_cells, 2)).astype(np.float32)
        # Variances: shape (n_cells, 2)  — isotropic Gaussian
        self.variances = np.full((n_cells, 2), stdev ** 2, dtype=np.float32)

    def unnor_logpdf(self, trajs):
        """Un-normalised log PDF of a 2-D Gaussian for each cell.

        Args:
            trajs: numpy array of shape (batch, seq, 2).

        Returns:
            numpy array of shape (batch, seq, n_cells).
        """
        # (batch, seq, 1, 2) - (1, 1, n_cells, 2) → (batch, seq, n_cells, 2)
        diff = trajs[:, :, np.newaxis, :] - self.means[np.newaxis, np.newaxis, :, :]
        # Sum over the spatial dimension (dim 2 of diff, i.e. x and y)
        return -0.5 * np.sum(diff ** 2 / self.variances[np.newaxis, np.newaxis, :, :], axis=-1)

    def _surround_logpdf(self, trajs):
        """Un-normalised log PDF for the wider surround Gaussian."""
        diff = trajs[:, :, np.newaxis, :] - self.means[np.newaxis, np.newaxis, :, :]
        surround_variances = self.variances * (self.surround_scale ** 2)
        return -0.5 * np.sum(
            diff ** 2 / surround_variances[np.newaxis, np.newaxis, :, :],
            axis=-1,
        )

    @staticmethod
    def _shift_and_normalize(values):
        """Shift difference targets to non-negative population probabilities."""
        if not np.isfinite(values).all():
            raise ValueError("DoS/DoG target values must be finite.")
        shifted = values + np.abs(np.min(values, axis=-1, keepdims=True))
        denom = np.sum(shifted, axis=-1, keepdims=True)
        zero_denom = denom <= 0.0
        if np.any(zero_denom):
            shifted = np.where(zero_denom, np.ones_like(shifted), shifted)
            denom = np.where(zero_denom, float(shifted.shape[-1]), denom)
        return shifted / denom

    def _normalized_center_response(self, logpdf):
        if self.pc_target_normalization == "none":
            return np.exp(logpdf)
        if self.pc_target_normalization == "global":
            return self._softmax(logpdf)
        raise ValueError(
            f"Unsupported pc_target_normalization={self.pc_target_normalization!r}."
        )

    @staticmethod
    def _raw_gaussian_difference(center_logpdf, surround_logpdf):
        """Return a numerically stable raw center-minus-surround response."""
        max_logpdf = np.maximum(center_logpdf, surround_logpdf).max(
            axis=-1,
            keepdims=True,
        )
        center = np.exp(center_logpdf - max_logpdf)
        surround = np.exp(surround_logpdf - max_logpdf)
        return center - surround

    def _difference_targets(self, trajs):
        center_logpdf = self.unnor_logpdf(trajs)
        surround_logpdf = self._surround_logpdf(trajs)

        if self.pc_target_family == "difference_of_gaussians":
            if self.pc_target_normalization == "none":
                return self._shift_and_normalize(
                    self._raw_gaussian_difference(center_logpdf, surround_logpdf)
                )
            elif self.pc_target_normalization == "global":
                center = self._normalized_center_response(center_logpdf)
                surround = self._softmax(surround_logpdf)
            else:
                raise ValueError(
                    f"Unsupported pc_target_normalization={self.pc_target_normalization!r}."
                )
            return self._shift_and_normalize(center - surround)

        if self.pc_target_family == "true_difference_of_gaussians":
            return self._shift_and_normalize(
                self._raw_gaussian_difference(center_logpdf, surround_logpdf)
            )

        raise ValueError(f"Unsupported difference target family: {self.pc_target_family!r}")

    def _apply_mode(self, x, mode):
        """Compute place-cell activations for the configured target family."""
        if self.pc_target_family == "gaussian":
            return super()._apply_mode(x, mode)

        if mode == "zeros":
            batch, seq = x.shape[0], x.shape[1]
            return np.zeros((batch, seq, self.n_cells), dtype=np.float32)
        if mode == "normalized":
            raise ValueError("DoS/DoG place-cell targets do not support normalized mode.")

        targets = self._difference_targets(x)
        if mode == "softmax":
            return targets.astype(np.float32)
        if mode == "sample":
            return self._sample_probs(targets).astype(np.float32)
        if mode == "voronoi":
            return self._one_hot_max(targets).astype(np.float32)
        raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Head Direction Cell Ensemble
# ---------------------------------------------------------------------------

class HeadDirectionCellEnsemble(CellEnsemble):
    """Ensemble of head-direction cells with Von Mises tuning curves."""

    def __init__(
        self,
        n_cells,
        concentration=20.0,
        seed=None,
        soft_targets="softmax",
        soft_init=None,
        decode_type="analytic",
        decode_top_k=3,
    ):
        """
        Args:
            n_cells: number of head direction cells.
            concentration: Von Mises concentration parameter (kappa).
            seed: random seed for reproducible cell placement.
            soft_targets: target encoding mode.
            soft_init: init encoding mode; defaults to soft_targets if None.
            decode_type: heading decoder mode ("analytic" or "weighted_mean").
            decode_top_k: retained for shared config compatibility; unused by HDC decoding.
        """
        super().__init__(
            n_cells,
            soft_targets,
            soft_init,
            decode_type=decode_type,
            decode_top_k=decode_top_k,
        )
        self._rng = np.random.RandomState(seed)
        # Preferred directions uniformly distributed over [-pi, pi)
        self.means = self._rng.uniform(-np.pi, np.pi, size=(n_cells,)).astype(np.float32)
        # Concentration: shape (n_cells,)
        self.kappa = np.full((n_cells,), concentration, dtype=np.float32)

    def unnor_logpdf(self, x):
        """Un-normalised log PDF of a Von Mises distribution for each cell.

        Args:
            x: numpy array of shape (batch, seq, 1) — head direction angles
               in radians.

        Returns:
            numpy array of shape (batch, seq, n_cells).
        """
        # x has shape (batch, seq, 1); means has shape (n_cells,)
        # Broadcasting: (batch, seq, 1) vs (n_cells,) → (batch, seq, n_cells)
        return self.kappa[np.newaxis, np.newaxis, :] * np.cos(
            x - self.means[np.newaxis, np.newaxis, :]
        )

    def activation_logpdf(self, x):
        """Log activation with unit peak for normalized HDC targets."""
        return self.kappa[np.newaxis, np.newaxis, :] * (
            np.cos(x - self.means[np.newaxis, np.newaxis, :]) - 1.0
        )
