"""Tests for the Reciprocal Rank Fusion helper."""

from app.core.vector_store import _reciprocal_rank_fusion


def test_rrf_rewards_consensus():
    # Item 2 is top of both rankings; it should win even though item 0 and 5
    # each lead one list.
    dense = [0, 2, 1]
    lexical = [5, 2, 3]
    fused = _reciprocal_rank_fusion([dense, lexical], k=60)
    winner = max(fused, key=fused.get)
    assert winner == 2


def test_rrf_single_ranking_is_monotonic():
    fused = _reciprocal_rank_fusion([[10, 11, 12]], k=60)
    assert fused[10] > fused[11] > fused[12]


def test_rrf_accumulates_across_lists():
    fused = _reciprocal_rank_fusion([[1], [1]], k=60)
    # Appearing rank-0 in two lists doubles the single-list score.
    assert fused[1] == 2 * (1.0 / 60)
