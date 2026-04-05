"""Tests for SentimentFilter."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from qts_core.agents.sentiment_filter import (
    SentimentFilter,
    SentimentState,
)


@pytest.fixture
def filt() -> SentimentFilter:
    """Filter with default thresholds."""
    return SentimentFilter(
        block_threshold=-0.70,
        boost_threshold=0.70,
        window_hours=6.0,
        min_samples=3,
    )


NOW = datetime(2026, 3, 29, 12, 0, 0, tzinfo=timezone.utc)


class TestNoData:
    def test_no_scores_returns_no_data(self, filt: SentimentFilter) -> None:
        v = filt.evaluate(now=NOW)
        assert v.state == SentimentState.NO_DATA
        assert not v.block_longs
        assert not v.boost_confidence
        assert v.sample_count == 0

    def test_insufficient_samples_permissive(self, filt: SentimentFilter) -> None:
        filt.add_score(-0.90, timestamp=NOW, source="test")
        filt.add_score(-0.85, timestamp=NOW, source="test")
        # Only 2 < min_samples=3 → permissive
        v = filt.evaluate(now=NOW)
        assert v.state == SentimentState.NO_DATA
        assert not v.block_longs


class TestNegativeSentiment:
    def test_extreme_negative_blocks_longs(self, filt: SentimentFilter) -> None:
        for _ in range(5):
            filt.add_score(-0.80, timestamp=NOW, source="finbert")
        v = filt.evaluate(now=NOW)
        assert v.state == SentimentState.NEGATIVE
        assert v.block_longs
        assert not v.boost_confidence
        assert v.avg_sentiment <= -0.70

    def test_threshold_boundary_blocks(self, filt: SentimentFilter) -> None:
        # Just past threshold (floating point: -0.70*3/3 ≈ -0.6999)
        for _ in range(3):
            filt.add_score(-0.71, timestamp=NOW, source="test")
        v = filt.evaluate(now=NOW)
        assert v.block_longs


class TestPositiveSentiment:
    def test_positive_boosts_confidence(self, filt: SentimentFilter) -> None:
        for _ in range(5):
            filt.add_score(0.85, timestamp=NOW, source="finbert")
        v = filt.evaluate(now=NOW)
        assert v.state == SentimentState.POSITIVE
        assert v.boost_confidence
        assert not v.block_longs
        assert v.avg_sentiment >= 0.70

    def test_threshold_boundary_boosts(self, filt: SentimentFilter) -> None:
        for _ in range(3):
            filt.add_score(0.71, timestamp=NOW, source="test")
        v = filt.evaluate(now=NOW)
        assert v.boost_confidence


class TestNeutralSentiment:
    def test_neutral_no_action(self, filt: SentimentFilter) -> None:
        for _ in range(5):
            filt.add_score(0.10, timestamp=NOW, source="test")
        v = filt.evaluate(now=NOW)
        assert v.state == SentimentState.NEUTRAL
        assert not v.block_longs
        assert not v.boost_confidence


class TestTimeWindow:
    def test_old_scores_excluded(self, filt: SentimentFilter) -> None:
        old = NOW - timedelta(hours=7)
        for _ in range(5):
            filt.add_score(-0.90, timestamp=old, source="old")
        # Recent positive score
        for _ in range(3):
            filt.add_score(0.50, timestamp=NOW, source="new")
        v = filt.evaluate(now=NOW)
        # Old negative scores outside 6h window, only 3 new neutrals count
        assert v.state == SentimentState.NEUTRAL
        assert not v.block_longs

    def test_scores_within_window_counted(self, filt: SentimentFilter) -> None:
        recent = NOW - timedelta(hours=2)
        for _ in range(4):
            filt.add_score(-0.80, timestamp=recent, source="test")
        v = filt.evaluate(now=NOW)
        assert v.block_longs
        assert v.sample_count == 4


class TestBatchAdd:
    def test_add_scores_batch(self, filt: SentimentFilter) -> None:
        filt.add_scores([(-0.80, "a"), (-0.90, "b"), (-0.75, "c")])
        assert filt.score_count == 3


class TestReset:
    def test_reset_clears(self, filt: SentimentFilter) -> None:
        for _ in range(5):
            filt.add_score(0.50, timestamp=NOW, source="test")
        assert filt.score_count == 5
        filt.reset()
        assert filt.score_count == 0


class TestClamping:
    def test_score_clamped_to_bounds(self, filt: SentimentFilter) -> None:
        filt.add_score(2.0, timestamp=NOW, source="test")
        filt.add_score(-2.0, timestamp=NOW, source="test")
        filt.add_score(0.5, timestamp=NOW, source="test")
        v = filt.evaluate(now=NOW)
        # 1.0 + (-1.0) + 0.5 = 0.5 / 3 = 0.167
        assert v.sample_count == 3
        assert -1.0 <= v.avg_sentiment <= 1.0
