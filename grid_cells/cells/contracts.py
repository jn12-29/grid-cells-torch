"""Validate cell-code configuration contracts shared across entrypoints.

Analytic decoding is exact for softmax-style PC/HDC codes whose ensemble
geometry has enough rank. PC logit decoding is scale-invariant for positive raw
score scaling. This module centralizes startup warnings so callers do not emit
repeated warnings from per-batch decoding code.

Usage:
    validate_cell_code_contract(cfg, pc_ensembles, hdc_ensembles, logger)
"""

import logging

from grid_cells.cells.decoding import (
    VALID_DECODE_TYPES,
    hdc_supports_analytic_decode,
    is_supported_decode_type,
    pc_supports_scale_invariant_decode,
)

_WARNED_KEYS: set[str] = set()


def _warn_once(logger: logging.Logger, key: str, message: str) -> None:
    """Emit a warning only once per process for a specific contract issue."""
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    logger.warning(message)


def validate_cell_code_contract(
    cfg,
    pc_ensembles=None,
    hdc_ensembles=None,
    logger: logging.Logger | None = None,
) -> None:
    """Validate decode config and warn when codes are not analytically exact."""
    logger = logger or logging.getLogger("grid_cells")
    task_cfg = getattr(cfg, "task", cfg)
    decode_type = getattr(task_cfg, "decode_type", "analytic")
    targets_type = getattr(task_cfg, "targets_type", "softmax")
    lstm_init_type = getattr(task_cfg, "lstm_init_type", targets_type)

    if not is_supported_decode_type(decode_type):
        supported = ", ".join(sorted(VALID_DECODE_TYPES))
        raise ValueError(
            f"Unsupported task.decode_type={decode_type!r}; expected one of: {supported}."
        )

    if decode_type != "analytic":
        return

    if targets_type != "softmax":
        _warn_once(
            logger,
            f"decode-targets-{targets_type}",
            "Analytic decoding is exact only with task.targets_type='softmax'. "
            f"Current task.targets_type={targets_type!r} may discard continuous "
            "position/heading information; decoded metrics will fall back to "
            "approximate weighted decoding where needed.",
        )

    if lstm_init_type != "softmax":
        _warn_once(
            logger,
            f"decode-init-{lstm_init_type}",
            "Recoverable initial position/heading encoding requires "
            "task.lstm_init_type='softmax'. "
            f"Current task.lstm_init_type={lstm_init_type!r} may discard initial "
            "condition information.",
        )

    if pc_ensembles is not None and any(
        getattr(ensemble, "soft_targets", None) == "softmax"
        and not pc_supports_scale_invariant_decode(ensemble)
        for ensemble in pc_ensembles
    ):
        _warn_once(
            logger,
            "decode-pc-rank",
            "Scale-invariant analytic PC decoding requires place-cell centers with "
            "shared positive variances and rank-3 [linear, offset] geometry. Some "
            "PC ensembles do not meet that contract and will use approximate "
            "weighted decoding.",
        )

    if hdc_ensembles is not None and any(
        getattr(ensemble, "soft_targets", None) == "softmax"
        and not hdc_supports_analytic_decode(ensemble)
        for ensemble in hdc_ensembles
    ):
        _warn_once(
            logger,
            "decode-hdc-rank",
            "Analytic HDC decoding requires preferred directions spanning a rank-2 "
            "circular code. Some HDC ensembles do not meet that contract and will "
            "use approximate circular weighted decoding.",
        )
