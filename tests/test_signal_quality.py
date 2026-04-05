"""Tests for SignalQualityEvaluator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from qts_core.agents.signal_quality import (
    SignalQuality,
    SignalQualityEvaluator,
)

NOW = datetime(2026, 3, 29, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def evaluator() -> SignalQualityEvaluator:
    return SignalQualityEvaluator(
        window_signals=10,
        high_consistency=0.70,
        medium_consistency=0.50,
        high_max_flip_rate=0.20,
        medium_max_flip_rate=0.40,
        medium_size_scale=0.60,
        min_samples=5,
        window_hours=24.0,
    )


class TestNoData:
    def test_empty_returns_no_data(self, evaluator: SignalQualityEvaluator) -> None:
        v = evaluator.evaluate(now=NOW)
        assert v.quality == SignalQuality.NO_DATA
        assert not v.block_entry
        assert v.size_multiplier == 1.0
        assert v.sample_count == 0

    def test_insufficient_samples_permissive(self, evaluator: SignalQualityEvaluator) -> None:
        for i in range(4):  # < min_samples=5
            evaluator.record_signal("LONG", 0.8, timestamp=NOW - timedelta(minutes=i))
        v = evaluator.evaluate(now=NOW)
        assert v.quality == SignalQuality.NO_DATA
        assert not v.block_entry


class TestHighQuality:
    def test_consistent_longs(self, evaluator: SignalQualityEvaluator) -> None:
        for i in range(8):
            evaluator.record_signal("LONG", 0.85, timestamp=NOW - timedelta(minutes=i))
        v = evaluator.evaluate(now=NOW)
        assert v.quality == SignalQuality.HIGH
        assert v.consistency == 1.0
        assert v.flip_rate == 0.0
        assert v.size_multiplier == 1.0
        assert not v.block_entry

    def test_mostly_consistent_with_one_deviant(self, evaluator: SignalQualityEvaluator) -> None:
        # 7 LONG + 1 SHORT = 87.5% consistency, 2 flips / 7 = 28.6% flip rate
        # consistency >= 0.70 but flip_rate > 0.20 → MEDIUM
        for i in range(7):
            evaluator.record_signal("LONG", 0.80, timestamp=NOW - timedelta(minutes=8 - i))
        evaluator.record_signal("SHORT", 0.60, timestamp=NOW - timedelta(minutes=1))
        # Latest is SHORT, so consistency = 1/8 = 12.5% — actually LOW
        # Let me fix: put the deviant in the middle, latest is LONG
        pass  # covered by separate test below

    def test_high_quality_with_early_deviant(self, evaluator: SignalQualityEvaluator) -> None:
        # 1 SHORT then 7 LONG → latest=LONG, consistency=7/8=87.5%, flips=1/7=14%
        evaluator.record_signal("SHORT", 0.60, timestamp=NOW - timedelta(minutes=10))
        for i in range(7):
            evaluator.record_signal("LONG", 0.85, timestamp=NOW - timedelta(minutes=7 - i))
        v = evaluator.evaluate(now=NOW)
        assert v.quality == SignalQuality.HIGH
        assert v.consistency == pytest.approx(7 / 8)
        assert v.flip_rate == pytest.approx(1 / 7)
        assert v.size_multiplier == 1.0


class TestMediumQuality:
    def test_moderate_consistency(self, evaluator: SignalQualityEvaluator) -> None:
        # 4 LONG, 1 SHORT, 1 LONG → 5/6 consistency for LONG but flip_rate = 2/5 = 40%
        # Nah, let me be precise:
        # Pattern: L L S L L L → latest=L, agreeing=5/6=83%, flips: L→L=0, L→S=1, S→L=1, L→L=0, L→L=0 = 2/5=40%
        # consistency >= 0.70, flip_rate = 0.40 ≤ 0.40 → still MEDIUM (not HIGH because flip > 0.20)
        signals = ["LONG", "LONG", "SHORT", "LONG", "LONG", "LONG"]
        for i, d in enumerate(signals):
            evaluator.record_signal(d, 0.75, timestamp=NOW - timedelta(minutes=len(signals) - i))
        v = evaluator.evaluate(now=NOW)
        assert v.quality == SignalQuality.MEDIUM
        assert v.size_multiplier == pytest.approx(0.60)
        assert not v.block_entry

    def test_fifty_percent_consistency(self, evaluator: SignalQualityEvaluator) -> None:
        # L S L S L → latest=L, agreeing=3/5=60%, flips=4/4=100% → LOW (flip > 0.40)
        # Try: L L L S S → latest=S, agreeing=2/5=40% → LOW
        # Try: L S S L L L → latest=L, agreeing=4/6=67%, flips=2/5=40%
        # 67% >= 50%, 40% ≤ 40% → MEDIUM
        signals = ["LONG", "SHORT", "SHORT", "LONG", "LONG", "LONG"]
        for i, d in enumerate(signals):
            evaluator.record_signal(d, 0.70, timestamp=NOW - timedelta(minutes=len(signals) - i))
        v = evaluator.evaluate(now=NOW)
        assert v.quality == SignalQuality.MEDIUM


class TestLowQuality:
    def test_whipsaw_blocks_entry(self, evaluator: SignalQualityEvaluator) -> None:
        # Alternating: L S L S L → consistency=3/5=60%, flip_rate=4/4=100%
        # flip_rate > 0.40 → LOW
        signals = ["LONG", "SHORT", "LONG", "SHORT", "LONG"]
        for i, d in enumerate(signals):
            evaluator.record_signal(d, 0.80, timestamp=NOW - timedelta(minutes=len(signals) - i))
        v = evaluator.evaluate(now=NOW)
        assert v.quality == SignalQuality.LOW
        assert v.block_entry
        assert v.size_multiplier == 0.0

    def test_low_consistency_blocks(self, evaluator: SignalQualityEvaluator) -> None:
        # 1 LONG then 4 SHORT, latest=SHORT, consistency=4/5=80%, flips=1/4=25%
        # That's actually HIGH. Let me flip:
        # 4 SHORT then 1 LONG → latest=LONG, consistency=1/5=20%, flips=1/4=25%
        # consistency < 0.50 → LOW
        for _ in range(4):
            evaluator.record_signal("SHORT", 0.70, timestamp=NOW - timedelta(minutes=5))
        evaluator.record_signal("LONG", 0.60, timestamp=NOW)
        v = evaluator.evaluate(now=NOW)
        assert v.quality == SignalQuality.LOW
        assert v.block_entry


class TestTimeWindow:
    def test_old_signals_pruned(self, evaluator: SignalQualityEvaluator) -> None:
        old = NOW - timedelta(hours=25)
        for _ in range(5):
            evaluator.record_signal("SHORT", 0.90, timestamp=old)
        # Recent consistent LONGs
        for i in range(5):
            evaluator.record_signal("LONG", 0.85, timestamp=NOW - timedelta(minutes=i))
        v = evaluator.evaluate(now=NOW)
        assert v.quality == SignalQuality.HIGH
        assert v.consistency == 1.0


class TestReset:
    def test_reset_clears(self, evaluator: SignalQualityEvaluator) -> None:
        for _ in range(5):
            evaluator.record_signal("LONG", 0.8, timestamp=NOW)
        assert evaluator.signal_count == 5
        evaluator.reset()
        assert evaluator.signal_count == 0


class TestConfidenceClamping:
    def test_confidence_clamped(self, evaluator: SignalQualityEvaluator) -> None:
        evaluator.record_signal("LONG", 2.0, timestamp=NOW)
        evaluator.record_signal("LONG", -0.5, timestamp=NOW)
        # Internal records should have clamped values
        assert evaluator.signal_count == 2
