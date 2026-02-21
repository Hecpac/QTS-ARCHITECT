"""ICT Smart Money Concepts Agent.

Implements Inner Circle Trader (ICT) methodology focusing on:
- Fair Value Gaps (FVG) - price imbalances
- Kill Zones - optimal trading windows
- Market Structure - swing highs/lows

All parameters are configurable via Hydra for strategy optimization.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import structlog

from qts_core.agents.base import BaseStrategyAgent
from qts_core.agents.protocol import AgentPriority, AgentSignal, SignalType
from qts_core.common.types import InstrumentId

log = structlog.get_logger()

# Type alias for OHLCV tuple: (timestamp, open, high, low, close, volume)
OHLCVTuple = tuple[datetime, float, float, float, float, float]

# Indices for OHLCV tuple
IDX_TIMESTAMP: Final[int] = 0
IDX_OPEN: Final[int] = 1
IDX_HIGH: Final[int] = 2
IDX_LOW: Final[int] = 3
IDX_CLOSE: Final[int] = 4
IDX_VOLUME: Final[int] = 5


@dataclass(frozen=True)
class FVGResult:
    """Result of Fair Value Gap detection."""

    detected: bool
    direction: SignalType
    gap_size: float  # Size of the gap as percentage
    pattern_name: str


class ICTSmartMoneyAgent(BaseStrategyAgent):
    """ICT Smart Money Concepts strategy agent.

    Detects Fair Value Gaps (FVG) during kill zone sessions.
    FVG represents institutional order flow imbalances.

    Kill Zones (UTC):
    - London Open: 07:00-10:00
    - NY Open: 13:00-16:00 (default)
    - London Close: 15:00-17:00

    Attributes:
        symbol: Target trading pair.
        session_start: Kill zone start hour (UTC).
        session_end: Kill zone end hour (UTC).
        min_fvg_size: Minimum FVG size as percentage of price.
        base_confidence: Base confidence for FVG signals.
    """

    # Minimum candles required for FVG detection
    MIN_CANDLES: Final[int] = 3

    def __init__(
        self,
        name: str,
        symbol: str,
        session_start: int = 13,
        session_end: int = 16,
        min_fvg_size: float = 0.001,
        base_confidence: float = 0.80,
        priority: AgentPriority = AgentPriority.HIGH,
        min_confidence: float = 0.6,
    ) -> None:
        """Initialize ICT Smart Money Agent.

        Args:
            name: Agent identifier.
            symbol: Trading pair (e.g., "BTC/USDT").
            session_start: Kill zone start hour (0-23, UTC).
            session_end: Kill zone end hour (0-23, UTC).
            min_fvg_size: Minimum FVG size as fraction (0.001 = 0.1%).
            base_confidence: Base confidence level for signals.
            priority: Signal priority.
            min_confidence: Minimum confidence to emit signals.
        """
        super().__init__(name=name, priority=priority, min_confidence=min_confidence)
        self.symbol = symbol
        self.session_start = session_start
        self.session_end = session_end
        self.min_fvg_size = min_fvg_size
        self.base_confidence = base_confidence

        # Internal history buffer for when OHLCV history isn't provided
        self._history: deque[OHLCVTuple] = deque(maxlen=self.MIN_CANDLES)

    def _is_in_kill_zone(self, ts: datetime) -> bool:
        """Check if timestamp falls within the kill zone.

        Args:
            ts: Timestamp to check.

        Returns:
            True if within kill zone hours.
        """
        # Ensure timezone awareness
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        return self.session_start <= ts.hour < self.session_end

    def _detect_fvg(self, candles: list[OHLCVTuple]) -> FVGResult:
        """Detect Fair Value Gap in the last 3 candles.

        FVG occurs when there's a gap between candle 1's high/low
        and candle 3's low/high, with candle 2 being impulsive.

        Bullish FVG: candle_3.low > candle_1.high (gap up)
        Bearish FVG: candle_3.high < candle_1.low (gap down)

        Args:
            candles: List of at least 3 OHLCV tuples.

        Returns:
            FVGResult with detection status and details.
        """
        if len(candles) < self.MIN_CANDLES:
            return FVGResult(
                detected=False,
                direction=SignalType.NEUTRAL,
                gap_size=0.0,
                pattern_name="",
            )

        # Get last 3 candles
        c1 = candles[-3]  # First candle
        c2 = candles[-2]  # Middle (impulse) candle
        c3 = candles[-1]  # Current candle

        c1_high = c1[IDX_HIGH]
        c1_low = c1[IDX_LOW]
        c2_open = c2[IDX_OPEN]
        c2_close = c2[IDX_CLOSE]
        c3_high = c3[IDX_HIGH]
        c3_low = c3[IDX_LOW]

        # Calculate reference price for percentage calculation
        ref_price = c2_close

        # Bullish FVG: gap up, middle candle is bullish
        if c3_low > c1_high and c2_close > c2_open:
            gap_size = (c3_low - c1_high) / ref_price
            if gap_size >= self.min_fvg_size:
                return FVGResult(
                    detected=True,
                    direction=SignalType.LONG,
                    gap_size=gap_size,
                    pattern_name="Bullish FVG",
                )

        # Bearish FVG: gap down, middle candle is bearish
        if c3_high < c1_low and c2_close < c2_open:
            gap_size = (c1_low - c3_high) / ref_price
            if gap_size >= self.min_fvg_size:
                return FVGResult(
                    detected=True,
                    direction=SignalType.SHORT,
                    gap_size=gap_size,
                    pattern_name="Bearish FVG",
                )

        return FVGResult(
            detected=False,
            direction=SignalType.NEUTRAL,
            gap_size=0.0,
            pattern_name="",
        )

    async def _generate_signal(
        self,
        instrument_id: InstrumentId,
        current_price: float,
        timestamp: datetime,
        ohlcv_history: list[OHLCVTuple] | None = None,
    ) -> AgentSignal | None:
        """Generate ICT signal based on FVG detection.

        Args:
            instrument_id: The instrument being analyzed.
            current_price: Latest price.
            timestamp: Current timestamp.
            ohlcv_history: Historical OHLCV data.

        Returns:
            AgentSignal if FVG detected in kill zone, None otherwise.
        """
        instrument_symbol = str(instrument_id)
        if instrument_symbol.upper() != self.symbol.upper():
            log.debug(
                "Instrument mismatch for ICT agent",
                agent=self.name,
                configured_symbol=self.symbol,
                received_symbol=instrument_symbol,
            )
            return None

        # Only trade during kill zones
        if not self._is_in_kill_zone(timestamp):
            log.debug(
                "Outside kill zone",
                agent=self.name,
                hour=timestamp.hour,
                kill_zone=f"{self.session_start}-{self.session_end}",
            )
            return None

        # Use provided history or internal buffer
        candles: list[OHLCVTuple]
        if ohlcv_history is not None and len(ohlcv_history) >= self.MIN_CANDLES:
            candles = ohlcv_history[-self.MIN_CANDLES :]
        else:
            # Update internal history
            if ohlcv_history and len(ohlcv_history) > 0:
                latest = ohlcv_history[-1]
                self._history.append(latest)

            if len(self._history) < self.MIN_CANDLES:
                return None

            candles = list(self._history)

        # Detect FVG
        fvg = self._detect_fvg(candles)

        if not fvg.detected:
            return None

        # Calculate confidence based on gap size
        # Larger gaps = higher confidence (up to a point)
        confidence = min(
            self.base_confidence + fvg.gap_size * 10,
            0.95,
        )

        log.info(
            "FVG detected",
            agent=self.name,
            pattern=fvg.pattern_name,
            gap_size=f"{fvg.gap_size:.4%}",
            confidence=f"{confidence:.2%}",
        )

        return AgentSignal(
            source_agent=self.name,
            signal_type=fvg.direction,
            confidence=confidence,
            priority=self.priority,
            timestamp=timestamp,
            metadata={
                "pattern": fvg.pattern_name,
                "gap_size_pct": fvg.gap_size * 100,
                "kill_zone": True,
                "session": f"{self.session_start:02d}:00-{self.session_end:02d}:00 UTC",
                "symbol": self.symbol,
            },
        )
