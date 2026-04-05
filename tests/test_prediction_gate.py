"""Tests for PredictionMarketGate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from qts_core.agents.prediction_gate import (
    GateState,
    ImpliedDirection,
    PredictionMarketGate,
)

NOW = datetime(2026, 3, 29, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def gate() -> PredictionMarketGate:
    return PredictionMarketGate(
        block_threshold=0.80,
        warn_threshold=0.65,
        warn_size_scale=0.50,
        window_hours=12.0,
        min_signals=1,
    )


class TestNoData:
    def test_empty_passes(self, gate: PredictionMarketGate) -> None:
        v = gate.evaluate("LONG", now=NOW)
        assert v.state == GateState.NO_DATA
        assert not v.block_trade
        assert v.size_multiplier == 1.0

    def test_exit_always_passes(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("ev1", ImpliedDirection.BEARISH, 0.95, timestamp=NOW)
        v = gate.evaluate("EXIT", now=NOW)
        assert v.state == GateState.PASS
        assert not v.block_trade

    def test_neutral_always_passes(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("ev1", ImpliedDirection.BEARISH, 0.95, timestamp=NOW)
        v = gate.evaluate("NEUTRAL", now=NOW)
        assert v.state == GateState.PASS


class TestBlocking:
    def test_strong_bearish_blocks_long(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("btc_crash", ImpliedDirection.BEARISH, 0.85, timestamp=NOW)
        v = gate.evaluate("LONG", now=NOW)
        assert v.state == GateState.BLOCKED
        assert v.block_trade
        assert v.size_multiplier == 0.0
        assert v.strongest_contradiction == 0.85

    def test_strong_bullish_blocks_short(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("btc_moon", ImpliedDirection.BULLISH, 0.90, timestamp=NOW)
        v = gate.evaluate("SHORT", now=NOW)
        assert v.state == GateState.BLOCKED
        assert v.block_trade

    def test_threshold_boundary_blocks(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("ev1", ImpliedDirection.BEARISH, 0.80, timestamp=NOW)
        v = gate.evaluate("LONG", now=NOW)
        assert v.state == GateState.BLOCKED


class TestReduced:
    def test_moderate_contradiction_reduces(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("ev1", ImpliedDirection.BEARISH, 0.70, timestamp=NOW)
        v = gate.evaluate("LONG", now=NOW)
        assert v.state == GateState.REDUCED
        assert not v.block_trade
        assert v.size_multiplier == pytest.approx(0.50)

    def test_warn_boundary_reduces(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("ev1", ImpliedDirection.BEARISH, 0.65, timestamp=NOW)
        v = gate.evaluate("LONG", now=NOW)
        assert v.state == GateState.REDUCED


class TestPass:
    def test_low_contradiction_passes(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("ev1", ImpliedDirection.BEARISH, 0.40, timestamp=NOW)
        v = gate.evaluate("LONG", now=NOW)
        assert v.state == GateState.PASS
        assert v.size_multiplier == 1.0

    def test_aligned_signals_pass(self, gate: PredictionMarketGate) -> None:
        # Bullish signal + LONG trade = aligned, not contradicting
        gate.add_signal("btc_up", ImpliedDirection.BULLISH, 0.90, timestamp=NOW)
        v = gate.evaluate("LONG", now=NOW)
        assert v.state == GateState.PASS

    def test_bearish_aligned_with_short(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("ev1", ImpliedDirection.BEARISH, 0.90, timestamp=NOW)
        v = gate.evaluate("SHORT", now=NOW)
        assert v.state == GateState.PASS


class TestTimeWindow:
    def test_old_signals_ignored(self, gate: PredictionMarketGate) -> None:
        old = NOW - timedelta(hours=13)
        gate.add_signal("ev1", ImpliedDirection.BEARISH, 0.95, timestamp=old)
        v = gate.evaluate("LONG", now=NOW)
        assert v.state == GateState.NO_DATA

    def test_recent_signals_used(self, gate: PredictionMarketGate) -> None:
        recent = NOW - timedelta(hours=6)
        gate.add_signal("ev1", ImpliedDirection.BEARISH, 0.85, timestamp=recent)
        v = gate.evaluate("LONG", now=NOW)
        assert v.state == GateState.BLOCKED


class TestMultipleSignals:
    def test_strongest_wins(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("ev1", ImpliedDirection.BEARISH, 0.50, timestamp=NOW)
        gate.add_signal("ev2", ImpliedDirection.BEARISH, 0.85, timestamp=NOW)
        gate.add_signal("ev3", ImpliedDirection.BEARISH, 0.60, timestamp=NOW)
        v = gate.evaluate("LONG", now=NOW)
        assert v.strongest_contradiction == 0.85
        assert v.state == GateState.BLOCKED


class TestReset:
    def test_reset_clears(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("ev1", ImpliedDirection.BEARISH, 0.90, timestamp=NOW)
        assert gate.signal_count == 1
        gate.reset()
        assert gate.signal_count == 0


class TestStringDirection:
    def test_accepts_string_direction(self, gate: PredictionMarketGate) -> None:
        gate.add_signal("ev1", "bearish", 0.85, timestamp=NOW)
        v = gate.evaluate("LONG", now=NOW)
        assert v.state == GateState.BLOCKED
