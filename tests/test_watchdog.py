"""Tests for DrawdownWatchdog."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from qts_core.agents.watchdog import (
    DrawdownWatchdog,
    WatchdogState,
)


@pytest.fixture
def watchdog() -> DrawdownWatchdog:
    """Default watchdog: 5% warn, 8% halt, 15% weekly, 50% reduce."""
    wd = DrawdownWatchdog(
        warn_threshold=0.05,
        halt_threshold=0.08,
        weekly_halt_threshold=0.15,
        reduce_factor=0.50,
        cooldown_bars=5,
    )
    wd.reset(100_000.0)
    return wd


class TestNormalOperation:
    def test_no_drawdown_returns_normal(self, watchdog: DrawdownWatchdog) -> None:
        v = watchdog.evaluate(100_000.0)
        assert v.state == WatchdogState.NORMAL
        assert v.size_multiplier == 1.0

    def test_equity_growth_updates_peak(self, watchdog: DrawdownWatchdog) -> None:
        v = watchdog.evaluate(110_000.0)
        assert v.state == WatchdogState.NORMAL
        assert v.peak_equity == 110_000.0

    def test_small_drawdown_stays_normal(self, watchdog: DrawdownWatchdog) -> None:
        watchdog.evaluate(100_000.0)
        v = watchdog.evaluate(96_000.0)  # 4% drawdown
        assert v.state == WatchdogState.NORMAL
        assert v.size_multiplier == 1.0


class TestWarningState:
    def test_warn_threshold_triggers_reduced_size(
        self, watchdog: DrawdownWatchdog
    ) -> None:
        v = watchdog.evaluate(95_000.0)  # 5% drawdown
        assert v.state == WatchdogState.WARNING
        assert v.size_multiplier == 0.50

    def test_warn_at_6_percent(self, watchdog: DrawdownWatchdog) -> None:
        v = watchdog.evaluate(94_000.0)  # 6% drawdown
        assert v.state == WatchdogState.WARNING
        assert v.size_multiplier == 0.50

    def test_recovery_from_warning_to_normal(
        self, watchdog: DrawdownWatchdog
    ) -> None:
        watchdog.evaluate(95_000.0)  # Enter WARNING
        assert watchdog.state == WatchdogState.WARNING
        v = watchdog.evaluate(100_000.0)  # Recover
        assert v.state == WatchdogState.NORMAL
        assert v.size_multiplier == 1.0


class TestHaltState:
    def test_halt_threshold_blocks_entries(
        self, watchdog: DrawdownWatchdog
    ) -> None:
        v = watchdog.evaluate(92_000.0)  # 8% drawdown
        assert v.state == WatchdogState.HALTED
        assert v.size_multiplier == 0.0

    def test_halt_at_10_percent(self, watchdog: DrawdownWatchdog) -> None:
        v = watchdog.evaluate(90_000.0)  # 10% drawdown
        assert v.state == WatchdogState.HALTED
        assert v.size_multiplier == 0.0

    def test_exit_always_allowed_during_halt(
        self, watchdog: DrawdownWatchdog
    ) -> None:
        watchdog.evaluate(90_000.0)  # HALTED
        v = watchdog.evaluate(90_000.0, is_exit=True)
        assert v.size_multiplier == 1.0
        assert "EXIT" in v.reason

    def test_cooldown_after_halt(self, watchdog: DrawdownWatchdog) -> None:
        watchdog.evaluate(90_000.0)  # HALTED
        # Stay halted during cooldown (equity still in warning zone)
        for _ in range(4):
            v = watchdog.evaluate(94_000.0)  # 6% DD — stays in warning zone
            assert v.state == WatchdogState.HALTED
            assert v.size_multiplier == 0.0
        # After cooldown_bars (5), transition to WARNING (6% > 5% warn)
        v = watchdog.evaluate(94_000.0)
        assert v.state == WatchdogState.WARNING
        assert v.size_multiplier == 0.50


class TestWeeklyHalt:
    def test_weekly_halt_triggers(self, watchdog: DrawdownWatchdog) -> None:
        v = watchdog.evaluate(85_000.0)  # 15% weekly drawdown
        assert v.state == WatchdogState.HALTED
        assert v.size_multiplier == 0.0
        assert "Weekly" in v.reason or "HALT" in v.reason

    def test_weekly_reset_after_7_days(self, watchdog: DrawdownWatchdog) -> None:
        now = datetime.now(timezone.utc)
        watchdog.evaluate(90_000.0, timestamp=now)  # 10% DD, HALTED

        # After 7 days, weekly resets
        next_week = now + timedelta(days=7)
        # Reset: new weekly start = current equity
        # Equity recovered to 95k, weekly DD from 90k = +5.5%
        v = watchdog.evaluate(95_000.0, timestamp=next_week)
        # Should not be weekly halted since weekly restarted at 90k
        assert "Weekly" not in v.reason or v.state != WatchdogState.HALTED


class TestPeakTracking:
    def test_peak_updates_on_new_high(self, watchdog: DrawdownWatchdog) -> None:
        watchdog.evaluate(105_000.0)
        v = watchdog.evaluate(100_000.0)  # 4.76% from new peak
        assert v.peak_equity == 105_000.0
        assert v.drawdown_pct == pytest.approx(5000 / 105000, rel=1e-3)

    def test_drawdown_calculated_from_peak(
        self, watchdog: DrawdownWatchdog
    ) -> None:
        watchdog.evaluate(120_000.0)  # New peak
        v = watchdog.evaluate(110_000.0)  # 8.33% from 120k peak
        assert v.state == WatchdogState.HALTED  # > 8%
        assert v.peak_equity == 120_000.0


class TestEdgeCases:
    def test_zero_equity(self, watchdog: DrawdownWatchdog) -> None:
        v = watchdog.evaluate(0.0)
        assert v.state == WatchdogState.HALTED

    def test_exit_during_normal(self, watchdog: DrawdownWatchdog) -> None:
        v = watchdog.evaluate(100_000.0, is_exit=True)
        assert v.size_multiplier == 1.0

    def test_state_property(self, watchdog: DrawdownWatchdog) -> None:
        assert watchdog.state == WatchdogState.NORMAL
        watchdog.evaluate(90_000.0)
        assert watchdog.state == WatchdogState.HALTED
