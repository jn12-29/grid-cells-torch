"""Cell-ensemble definitions for place and head-direction supervision.

These classes reproduce the probabilistic target encodings used by the original
DeepMind codebase while keeping target generation in NumPy and loss evaluation
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


class CellEnsemble(ABC):
    """Base class for cell ensembles (place cells, head direction cells, etc.)."""

    def __init__(self, n_cells, soft_targets, soft_init):
        """
        Args:
            n_cells: number of cells in the ensemble.
            soft_targets: one of "softmax", "voronoi", "sample", "normalized".
            soft_init: same options as soft_targets, or "zeros". If None,
                       defaults to soft_targets.
        """
        self.n_cells = n_cells
        self.soft_targets = soft_targets
        self.soft_init = soft_init if soft_init is not None else soft_targets

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
            logpdf = self.unnor_logpdf(x)
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

    def loss(self, predictions, targets):
        """Cross-entropy loss between model predictions and cell targets.

        Args:
            predictions: torch.Tensor of shape (batch, seq_len, n_cells).
                         These are *logits* (pre-softmax / pre-sigmoid).
            targets: numpy array of shape (batch, seq_len, n_cells).

        Returns:
            Scalar torch.Tensor (mean over batch and seq dimensions).
        """
        # Convert targets to tensor on the same device as predictions
        targets_t = torch.as_tensor(
            targets, dtype=predictions.dtype, device=predictions.device
        )

        if self.soft_targets == "normalized":
            # 'normalized' treats each cell as independent binary output (multi-label).
            # Targets are exp(unnormalized log-pdf); labels are smoothed toward 0.5.
            # Sigmoid cross-entropy with label smoothing
            smoothing = 1e-2
            labels = (1.0 - smoothing) * targets_t + smoothing * 0.5
            # binary_cross_entropy_with_logits expects predictions as logits
            loss_val = F.binary_cross_entropy_with_logits(
                predictions, labels, reduction="mean"
            )
        else:
            # Softmax cross-entropy
            # cross_entropy expects (N, C) or (N, C, d1, ...) with class dim = 1
            batch, seq, n = predictions.shape
            # Flatten batch and seq into a single dimension
            pred_flat = predictions.reshape(batch * seq, n)       # (B*T, C)
            tgt_flat = targets_t.reshape(batch * seq, n)          # (B*T, C)
            # Use soft labels via log_softmax + sum
            log_probs = F.log_softmax(pred_flat, dim=-1)          # (B*T, C)
            loss_val = -(tgt_flat * log_probs).sum(dim=-1).mean()

        return loss_val


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
        """
        super().__init__(n_cells, soft_targets, soft_init)
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
    ):
        """
        Args:
            n_cells: number of head direction cells.
            concentration: Von Mises concentration parameter (kappa).
            seed: random seed for reproducible cell placement.
            soft_targets: target encoding mode.
            soft_init: init encoding mode; defaults to soft_targets if None.
        """
        super().__init__(n_cells, soft_targets, soft_init)
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
