"""Tests for scoring and plotting helpers."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scores import GridScorer
from utils import get_scores_and_plot_from_ratemaps, score_ratemaps


def make_scorer(nbins=8):
    mask_parameters = [(0.2, 0.4), (0.3, 0.6)]
    return GridScorer(nbins, [[-1.0, 1.0], [-1.0, 1.0]], mask_parameters)


def test_streaming_ratemaps_match_binned_statistic():
    """Streaming accumulation should match per-unit binned_statistic_2d results."""
    rng = np.random.default_rng(0)
    scorer = make_scorer()
    positions = rng.uniform(-1.0, 1.0, size=(4, 6, 2)).astype(np.float32)
    activations = rng.normal(size=(4, 6, 3)).astype(np.float32)

    sums, counts = scorer.allocate_ratemap_accumulators(activations.shape[-1])
    scorer.accumulate_ratemaps(positions, activations, sums, counts)
    ratemaps = scorer.finalize_ratemaps(sums, counts)

    flat_pos = positions.reshape(-1, 2)
    flat_act = activations.reshape(-1, activations.shape[-1])
    expected = np.stack(
        [
            scorer.calculate_ratemap(flat_pos[:, 0], flat_pos[:, 1], flat_act[:, i])
            for i in range(flat_act.shape[-1])
        ],
        axis=0,
    )

    assert np.allclose(ratemaps, expected, equal_nan=True)


def test_score_ratemaps_chunking_matches_direct_scores():
    """Chunked scoring should preserve the exact direct score outputs."""
    rng = np.random.default_rng(1)
    scorer = make_scorer()
    ratemaps = rng.normal(size=(6, scorer._nbins, scorer._nbins)).astype(np.float32)

    direct = [scorer.get_scores(rm) for rm in ratemaps]
    score_60, score_90, max_60_mask, max_90_mask, sacs = score_ratemaps(
        scorer,
        ratemaps,
        num_workers=0,
        chunk_size=2,
    )

    assert np.allclose(score_60, np.asarray([result[0] for result in direct]))
    assert np.allclose(score_90, np.asarray([result[1] for result in direct]))
    assert max_60_mask == [result[2] for result in direct]
    assert max_90_mask == [result[3] for result in direct]
    assert np.allclose(sacs, np.asarray([result[4] for result in direct]))


def test_get_scores_batch_matches_direct_scores():
    """Batched score computation should match the scalar implementation."""
    rng = np.random.default_rng(3)
    scorer = make_scorer()
    ratemaps = rng.normal(size=(5, scorer._nbins, scorer._nbins)).astype(np.float32)

    direct = [scorer.get_scores(rm) for rm in ratemaps]
    score_60, score_90, max_60_mask, max_90_mask, sacs = scorer.get_scores_batch(
        ratemaps
    )

    assert np.allclose(score_60, np.asarray([result[0] for result in direct]))
    assert np.allclose(score_90, np.asarray([result[1] for result in direct]))
    assert max_60_mask == [result[2] for result in direct]
    assert max_90_mask == [result[3] for result in direct]
    assert np.allclose(sacs, np.asarray([result[4] for result in direct]))


def test_get_scores_and_plot_from_ratemaps_writes_paginated_pdf(tmp_path):
    """Paginated plotting should write a PDF while keeping all units."""
    rng = np.random.default_rng(2)
    scorer = make_scorer()
    ratemaps = rng.normal(size=(17, scorer._nbins, scorer._nbins)).astype(np.float32)

    scores = get_scores_and_plot_from_ratemaps(
        scorer,
        ratemaps,
        str(tmp_path),
        "scores.pdf",
        num_workers=0,
        chunk_size=4,
        units_per_page=8,
    )

    assert (tmp_path / "scores.pdf").exists()
    assert len(scores[0]) == ratemaps.shape[0]
    assert len(scores[1]) == ratemaps.shape[0]
