"""Prediction Market Gate — Capa 5 of QTS risk management.

Ingests probability signals from prediction markets (Polymarket, etc.)
and gates or scales trades when market-implied probabilities contradict
the proposed direction.

Example: If system wants to go LONG BTC but Polymarket shows 75% probability
of a "BTC drops below 50k this week" event, the gate blocks or reduces the trade.

Each prediction signal has:
    - event_id: unique identifier for the event
    - direction_implication: "BULLISH" or "BEARISH" for the traded asset
    - probability: 0-1 (market-implied probability of the event)
    - source: e.g. "polymarket", "kalshi"

Gate logic:
    - If strongest contradicting signal probability >= block_threshold → block
    - If strongest contradicting signal probability >= warn_threshold → scale down
    - Otherwise → pass through
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum


class ImpliedDirection(str, Enum):
    """What the prediction market event implies for the asset."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class GateState(str, Enum):
    """Gate decision state."""

    PASS = "PASS"
    REDUCED = "REDUCED"
    BLOCKED = "BLOCKED"
    NO_DATA = "NO_DATA"


@dataclass(frozen=True)
class PredictionSignal:
    """A single prediction market probability signal."""

    event_id: str
    direction: ImpliedDirection
    probability: float  # 0-1
    source: str
    timestamp: datetime


@dataclass(frozen=True)
class PredictionGateVerdict:
    """Result of prediction market gate evaluation."""

    state: GateState
    block_trade: bool
    size_multiplier: float
    strongest_contradiction: float  # 0-1, highest contradicting probability
    signal_count: int
    reason: str


class PredictionMarketGate:
    """Gates trades when prediction markets contradict trading direction.

    Parameters
    ----------
    block_threshold : float
        Contradicting probability above this blocks the trade. Default 0.80.
    warn_threshold : float
        Contradicting probability above this reduces size. Default 0.65.
    warn_size_scale : float
        Size multiplier when in REDUCED state. Default 0.50.
    window_hours : float
        Time window for relevant signals. Default 12.0.
    min_signals : int
        Minimum contradicting signals to act on. Default 1.
    """

    def __init__(
        self,
        *,
        block_threshold: float = 0.80,
        warn_threshold: float = 0.65,
        warn_size_scale: float = 0.50,
        window_hours: float = 12.0,
        min_signals: int = 1,
    ) -> None:
        self._block_threshold = block_threshold
        self._warn_threshold = warn_threshold
        self._warn_size_scale = warn_size_scale
        self._window_hours = window_hours
        self._min_signals = max(1, min_signals)
        self._signals: deque[PredictionSignal] = deque(maxlen=200)

    def add_signal(
        self,
        event_id: str,
        direction: ImpliedDirection | str,
        probability: float,
        source: str = "polymarket",
        timestamp: datetime | None = None,
    ) -> None:
        """Record a prediction market signal."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        if isinstance(direction, str):
            direction = ImpliedDirection(direction.upper())
        probability = max(0.0, min(1.0, probability))
        self._signals.append(
            PredictionSignal(
                event_id=event_id,
                direction=direction,
                probability=probability,
                source=source,
                timestamp=timestamp,
            )
        )

    def evaluate(
        self,
        trade_direction: str,
        now: datetime | None = None,
    ) -> PredictionGateVerdict:
        """Evaluate whether prediction markets contradict the proposed trade.

        Parameters
        ----------
        trade_direction : str
            "LONG" or "SHORT". EXIT always passes.
        now : datetime, optional
            Reference time for window filtering.

        Returns
        -------
        PredictionGateVerdict
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if trade_direction in ("EXIT", "NEUTRAL"):
            return PredictionGateVerdict(
                state=GateState.PASS,
                block_trade=False,
                size_multiplier=1.0,
                strongest_contradiction=0.0,
                signal_count=0,
                reason="EXIT/NEUTRAL always passes prediction gate",
            )

        cutoff = now - timedelta(hours=self._window_hours)
        recent = [s for s in self._signals if s.timestamp >= cutoff]

        if not recent:
            return PredictionGateVerdict(
                state=GateState.NO_DATA,
                block_trade=False,
                size_multiplier=1.0,
                strongest_contradiction=0.0,
                signal_count=0,
                reason="No prediction signals available",
            )

        # Determine which direction contradicts the trade
        if trade_direction == "LONG":
            contradicting = [s for s in recent if s.direction == ImpliedDirection.BEARISH]
        else:  # SHORT
            contradicting = [s for s in recent if s.direction == ImpliedDirection.BULLISH]

        if len(contradicting) < self._min_signals:
            return PredictionGateVerdict(
                state=GateState.PASS,
                block_trade=False,
                size_multiplier=1.0,
                strongest_contradiction=0.0,
                signal_count=len(recent),
                reason=f"Insufficient contradicting signals ({len(contradicting)}/{self._min_signals})",
            )

        strongest = max(s.probability for s in contradicting)

        if strongest >= self._block_threshold:
            return PredictionGateVerdict(
                state=GateState.BLOCKED,
                block_trade=True,
                size_multiplier=0.0,
                strongest_contradiction=strongest,
                signal_count=len(recent),
                reason=(
                    f"Prediction market contradicts {trade_direction}: "
                    f"{strongest:.0%} probability — BLOCKED"
                ),
            )

        if strongest >= self._warn_threshold:
            return PredictionGateVerdict(
                state=GateState.REDUCED,
                block_trade=False,
                size_multiplier=self._warn_size_scale,
                strongest_contradiction=strongest,
                signal_count=len(recent),
                reason=(
                    f"Prediction market warns against {trade_direction}: "
                    f"{strongest:.0%} probability — size {self._warn_size_scale:.0%}"
                ),
            )

        return PredictionGateVerdict(
            state=GateState.PASS,
            block_trade=False,
            size_multiplier=1.0,
            strongest_contradiction=strongest,
            signal_count=len(recent),
            reason=f"Prediction markets aligned ({strongest:.0%} contradiction)",
        )

    def reset(self) -> None:
        """Clear all signals."""
        self._signals.clear()

    @property
    def signal_count(self) -> int:
        return len(self._signals)
