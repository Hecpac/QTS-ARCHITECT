"""Tests for RegimeDetector."""

from __future__ import annotations

import pytest

from qts_core.models.regime import (
    MarketRegime,
    RegimeDetector,
)


@pytest.fixture
def detector() -> RegimeDetector:
    """Detector with small windows for testing."""
    return RegimeDetector(
        lookback=50,
        atr_period=5,
        low_percentile=0.25,
        high_percentile=0.75,
        crisis_percentile=0.95,
        high_vol_size_scale=0.60,
    )


def _feed_bars(
    detector: RegimeDetector,
    bars: list[tuple[float, float, float]],
) -> None:
    """Feed (high, low, close) bars into detector."""
    for h, l, c in bars:
        detector.update(h, l, c)


class TestWarmup:
    def test_not_ready_during_warmup(self, detector: RegimeDetector) -> None:
        v = detector.update(101.0, 99.0, 100.0)
        assert not detector.ready
        assert v.regime == MarketRegime.NORMAL
        assert "Warming" in v.reason or "Building" in v.reason

    def test_ready_after_enough_bars(self, detector: RegimeDetector) -> None:
        # Feed atr_period * 2 bars to be fully ready
        for i in range(15):
            detector.update(101.0, 99.0, 100.0)
        assert detector.ready


class TestNormalRegime:
    def test_stable_prices_normal_regime(self, detector: RegimeDetector) -> None:
        # Feed consistent bars → all ATRs similar → mid percentile
        for _ in range(30):
            v = detector.update(101.0, 99.0, 100.0)
        assert v.regime == MarketRegime.NORMAL
        assert v.size_multiplier == 1.0


class TestHighVolRegime:
    def test_spike_triggers_high_vol(self, detector: RegimeDetector) -> None:
        # Feed calm bars first
        for _ in range(30):
            detector.update(101.0, 99.0, 100.0)  # TR ≈ 2
        # Now spike volatility
        v = detector.update(115.0, 85.0, 100.0)  # TR = 30
        assert v.regime in (MarketRegime.HIGH_VOL, MarketRegime.CRISIS)
        assert v.size_multiplier < 1.0

    def test_high_vol_size_scaling(self, detector: RegimeDetector) -> None:
        for _ in range(30):
            detector.update(101.0, 99.0, 100.0)
        v = detector.update(110.0, 90.0, 100.0)  # TR = 20 vs normal 2
        if v.regime == MarketRegime.HIGH_VOL:
            assert v.size_multiplier == 0.60


class TestCrisisRegime:
    def test_extreme_spike_triggers_crisis(self, detector: RegimeDetector) -> None:
        for _ in range(40):
            detector.update(101.0, 99.0, 100.0)  # TR ≈ 2
        # Extreme spike over multiple bars to push ATR up
        for _ in range(5):
            v = detector.update(130.0, 70.0, 100.0)  # TR = 60
        assert v.regime == MarketRegime.CRISIS
        assert v.size_multiplier == 0.0


class TestLowVolRegime:
    def test_decreasing_vol_triggers_low_vol(self, detector: RegimeDetector) -> None:
        # Start with normal volatility
        for _ in range(30):
            detector.update(105.0, 95.0, 100.0)  # TR = 10
        # Then very low volatility
        for _ in range(20):
            v = detector.update(100.5, 99.5, 100.0)  # TR = 1
        assert v.regime == MarketRegime.LOW_VOL
        assert v.size_multiplier == 1.0


class TestPercentile:
    def test_percentile_in_range(self, detector: RegimeDetector) -> None:
        for _ in range(30):
            v = detector.update(101.0, 99.0, 100.0)
        assert 0.0 <= v.atr_percentile <= 1.0

    def test_atr_positive(self, detector: RegimeDetector) -> None:
        for _ in range(30):
            v = detector.update(101.0, 99.0, 100.0)
        assert v.current_atr > 0


class TestReset:
    def test_reset_clears_state(self, detector: RegimeDetector) -> None:
        for _ in range(30):
            detector.update(101.0, 99.0, 100.0)
        assert detector.ready
        detector.reset()
        assert not detector.ready
        assert detector.regime == MarketRegime.NORMAL

    def test_regime_property(self, detector: RegimeDetector) -> None:
        assert detector.regime == MarketRegime.NORMAL
