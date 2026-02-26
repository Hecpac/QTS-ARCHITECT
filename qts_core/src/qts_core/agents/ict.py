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
from datetime import date, datetime, timezone
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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

    Kill Zones (timezone-aware):
    - Configure hours in local session timezone (default: UTC).
    - Example NY open all-year local time: 08:00-11:00 with
      session_timezone="America/New_York" (DST-adjusted automatically).

    Attributes:
        symbol: Target trading pair.
        session_start: Kill zone start hour in session timezone.
        session_end: Kill zone end hour in session timezone.
        session_timezone: IANA timezone for session window evaluation.
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
        session_timezone: str = "UTC",
        min_fvg_size: float = 0.001,
        base_confidence: float = 0.80,
        enable_session_range_breakout_reversal: bool = False,
        session_range_breakout_only: bool = False,
        enable_high_breakout_short: bool = True,
        enable_low_breakout_long: bool = True,
        exit_on_session_target_hit: bool = True,
        breakout_buffer_bps: float = 0.0,
        hard_block_minutes_ny: tuple[str, ...] = (),
        hard_block_pattern_minutes_ny: tuple[str, ...] = (),
        soft_risk_pattern_minutes_ny: tuple[str, ...] = (),
        soft_risk_multiplier: float = 1.0,
        priority: AgentPriority = AgentPriority.HIGH,
        min_confidence: float = 0.6,
    ) -> None:
        """Initialize ICT Smart Money Agent.

        Args:
            name: Agent identifier.
            symbol: Trading pair (e.g., "BTC/USDT").
            session_start: Kill zone start hour (0-23) in session timezone.
            session_end: Kill zone end hour (0-23) in session timezone.
            session_timezone: IANA timezone name (e.g., "UTC", "America/New_York").
            min_fvg_size: Minimum FVG size as fraction (0.001 = 0.1%).
            base_confidence: Base confidence level for signals.
            enable_session_range_breakout_reversal: If true, detect session
                high/low breakouts and trade reversal-to-range targets.
            session_range_breakout_only: If true with breakout reversal enabled,
                disable FVG fallback and trade only session range breakouts.
            enable_high_breakout_short: Allow short entries on session-high breaks.
            enable_low_breakout_long: Allow long entries on session-low breaks.
            exit_on_session_target_hit: Emit EXIT when reversal target is reached.
            breakout_buffer_bps: Extra breakout buffer in bps to avoid noise.
            hard_block_minutes_ny: Entry embargo HH:MM list in session timezone.
            hard_block_pattern_minutes_ny: Entry embargo by "Pattern|HH:MM".
            soft_risk_pattern_minutes_ny: Size-reduction map by "Pattern|HH:MM".
            soft_risk_multiplier: Position-size multiplier for soft-risk keys.
            priority: Signal priority.
            min_confidence: Minimum confidence to emit signals.
        """
        super().__init__(name=name, priority=priority, min_confidence=min_confidence)
        self.symbol = symbol
        self.session_start = session_start
        self.session_end = session_end
        self.session_timezone = session_timezone
        try:
            self._session_tz = ZoneInfo(session_timezone)
        except ZoneInfoNotFoundError:
            log.warning(
                "Invalid ICT session timezone; falling back to UTC",
                agent=name,
                session_timezone=session_timezone,
            )
            self.session_timezone = "UTC"
            self._session_tz = timezone.utc
        self.min_fvg_size = min_fvg_size
        self.base_confidence = base_confidence
        self.enable_session_range_breakout_reversal = (
            enable_session_range_breakout_reversal
        )
        self.session_range_breakout_only = session_range_breakout_only
        self.enable_high_breakout_short = enable_high_breakout_short
        self.enable_low_breakout_long = enable_low_breakout_long
        self.exit_on_session_target_hit = exit_on_session_target_hit
        self.breakout_buffer_bps = max(0.0, breakout_buffer_bps)
        self.soft_risk_multiplier = min(max(float(soft_risk_multiplier), 0.0), 1.0)

        self._hard_block_minutes_ny = {
            hhmm
            for raw in hard_block_minutes_ny
            if (hhmm := self._normalize_hhmm(str(raw))) is not None
        }
        self._hard_block_pattern_minutes_ny = {
            pair
            for raw in hard_block_pattern_minutes_ny
            if (pair := self._normalize_pattern_minute_key(str(raw))) is not None
        }
        self._soft_risk_pattern_minutes_ny = {
            pair
            for raw in soft_risk_pattern_minutes_ny
            if (pair := self._normalize_pattern_minute_key(str(raw))) is not None
        }

        # Internal history buffer for when OHLCV history isn't provided
        self._history: deque[OHLCVTuple] = deque(maxlen=self.MIN_CANDLES)

        # Session range tracking (local session timezone date-scoped).
        self._session_date: date | None = None
        self._session_high: float | None = None
        self._session_low: float | None = None
        self._active_short_target_low: float | None = None
        self._active_long_target_high: float | None = None

    @staticmethod
    def _normalize_hhmm(raw: str) -> str | None:
        """Normalize HH:MM strings; return None for invalid values."""
        value = raw.strip()
        parts = value.split(":")
        if len(parts) != 2:
            return None

        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError:
            return None

        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None

        return f"{hour:02d}:{minute:02d}"

    def _normalize_pattern_minute_key(self, raw: str) -> tuple[str, str] | None:
        """Normalize "Pattern|HH:MM" keys for pattern-minute gating."""
        if "|" not in raw:
            return None

        pattern, minute = raw.split("|", 1)
        pattern_key = pattern.strip().lower()
        minute_key = self._normalize_hhmm(minute)
        if not pattern_key or minute_key is None:
            return None

        return (pattern_key, minute_key)

    def _entry_policy(self, pattern_name: str, minute_hhmm: str) -> tuple[bool, float]:
        """Return entry allowance and optional size multiplier for pattern/minute."""
        if minute_hhmm in self._hard_block_minutes_ny:
            return (False, 1.0)

        pattern_key = pattern_name.strip().lower()
        pattern_minute_key = (pattern_key, minute_hhmm)

        if pattern_minute_key in self._hard_block_pattern_minutes_ny:
            return (False, 1.0)

        if (
            pattern_minute_key in self._soft_risk_pattern_minutes_ny
            and self.soft_risk_multiplier < 1.0
        ):
            return (True, self.soft_risk_multiplier)

        return (True, 1.0)

    def _is_in_kill_zone(self, ts: datetime) -> bool:
        """Check if timestamp falls within configured kill-zone window."""
        # Ensure timezone awareness first (assume UTC when naive).
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        local_ts = ts.astimezone(self._session_tz)
        hour = local_ts.hour

        # Standard window (e.g., 08-11).
        if self.session_start < self.session_end:
            return self.session_start <= hour < self.session_end

        # Overnight window (e.g., 22-02).
        if self.session_start > self.session_end:
            return hour >= self.session_start or hour < self.session_end

        # Start == end means disabled/full-day ambiguity; keep conservative.
        return False

    def _reset_session_state_if_needed(self, local_date: date) -> None:
        """Reset per-session tracking when local trading date changes."""
        if self._session_date == local_date:
            return

        self._session_date = local_date
        self._session_high = None
        self._session_low = None
        self._active_short_target_low = None
        self._active_long_target_high = None

    def _breakout_threshold(self, reference: float, direction: str) -> float:
        """Return breakout threshold with optional bps buffer."""
        if self.breakout_buffer_bps <= 0 or reference <= 0:
            return reference

        buffer = self.breakout_buffer_bps / 10_000.0
        if direction == "up":
            return reference * (1.0 + buffer)
        return reference * (1.0 - buffer)

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

        local_ts = timestamp
        if local_ts.tzinfo is None:
            local_ts = local_ts.replace(tzinfo=timezone.utc)
        local_ts = local_ts.astimezone(self._session_tz)
        self._reset_session_state_if_needed(local_ts.date())
        minute_hhmm = f"{local_ts.hour:02d}:{local_ts.minute:02d}"

        latest_candle = ohlcv_history[-1] if ohlcv_history else None
        candle_high = float(latest_candle[IDX_HIGH]) if latest_candle else current_price
        candle_low = float(latest_candle[IDX_LOW]) if latest_candle else current_price

        if self.enable_session_range_breakout_reversal and self.exit_on_session_target_hit:
            if (
                self._active_short_target_low is not None
                and current_price <= self._active_short_target_low
            ):
                target_low = self._active_short_target_low
                self._active_short_target_low = None
                return AgentSignal(
                    source_agent=self.name,
                    signal_type=SignalType.EXIT,
                    confidence=max(self.base_confidence, 0.8),
                    priority=self.priority,
                    timestamp=timestamp,
                    metadata={
                        "pattern": "Session short target hit",
                        "target_low": target_low,
                        "session": (
                            f"{self.session_start:02d}:00-{self.session_end:02d}:00 "
                            f"{self.session_timezone}"
                        ),
                        "symbol": self.symbol,
                    },
                )

            if (
                self._active_long_target_high is not None
                and current_price >= self._active_long_target_high
            ):
                target_high = self._active_long_target_high
                self._active_long_target_high = None
                return AgentSignal(
                    source_agent=self.name,
                    signal_type=SignalType.EXIT,
                    confidence=max(self.base_confidence, 0.8),
                    priority=self.priority,
                    timestamp=timestamp,
                    metadata={
                        "pattern": "Session long target hit",
                        "target_high": target_high,
                        "session": (
                            f"{self.session_start:02d}:00-{self.session_end:02d}:00 "
                            f"{self.session_timezone}"
                        ),
                        "symbol": self.symbol,
                    },
                )

        in_kill_zone = self._is_in_kill_zone(timestamp)
        if not in_kill_zone:
            log.debug(
                "Outside kill zone",
                agent=self.name,
                hour_local=local_ts.hour,
                session_timezone=self.session_timezone,
                kill_zone=f"{self.session_start}-{self.session_end}",
            )
            return None

        if self.enable_session_range_breakout_reversal:
            prev_high = self._session_high
            prev_low = self._session_low

            if prev_high is None or prev_low is None:
                self._session_high = candle_high
                self._session_low = candle_low
            else:
                high_break = candle_high > self._breakout_threshold(prev_high, "up")
                low_break = candle_low < self._breakout_threshold(prev_low, "down")

                self._session_high = max(prev_high, candle_high)
                self._session_low = min(prev_low, candle_low)

                if high_break and self.enable_high_breakout_short:
                    pattern_name = "Session high breakout reversal"
                    allow_entry, size_multiplier = self._entry_policy(
                        pattern_name,
                        minute_hhmm,
                    )
                    if not allow_entry:
                        return None

                    self._active_short_target_low = prev_low
                    self._active_long_target_high = None
                    metadata: dict[str, str | float | int | bool] = {
                        "pattern": pattern_name,
                        "session_high": prev_high,
                        "target_low": prev_low,
                        "entry_hhmm_ny": minute_hhmm,
                        "entry_hour_ny": local_ts.hour,
                        "session": (
                            f"{self.session_start:02d}:00-{self.session_end:02d}:00 "
                            f"{self.session_timezone}"
                        ),
                        "symbol": self.symbol,
                    }
                    if size_multiplier < 1.0:
                        metadata["size_multiplier"] = size_multiplier

                    return AgentSignal(
                        source_agent=self.name,
                        signal_type=SignalType.SHORT,
                        confidence=max(self.base_confidence, 0.8),
                        priority=self.priority,
                        timestamp=timestamp,
                        metadata=metadata,
                    )

                if low_break and self.enable_low_breakout_long:
                    pattern_name = "Session low breakout reversal"
                    allow_entry, size_multiplier = self._entry_policy(
                        pattern_name,
                        minute_hhmm,
                    )
                    if not allow_entry:
                        return None

                    self._active_long_target_high = prev_high
                    self._active_short_target_low = None
                    metadata: dict[str, str | float | int | bool] = {
                        "pattern": pattern_name,
                        "session_low": prev_low,
                        "target_high": prev_high,
                        "entry_hhmm_ny": minute_hhmm,
                        "entry_hour_ny": local_ts.hour,
                        "session": (
                            f"{self.session_start:02d}:00-{self.session_end:02d}:00 "
                            f"{self.session_timezone}"
                        ),
                        "symbol": self.symbol,
                    }
                    if size_multiplier < 1.0:
                        metadata["size_multiplier"] = size_multiplier

                    return AgentSignal(
                        source_agent=self.name,
                        signal_type=SignalType.LONG,
                        confidence=max(self.base_confidence, 0.8),
                        priority=self.priority,
                        timestamp=timestamp,
                        metadata=metadata,
                    )

        if (
            self.enable_session_range_breakout_reversal
            and self.session_range_breakout_only
        ):
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

        allow_entry, size_multiplier = self._entry_policy(
            fvg.pattern_name,
            minute_hhmm,
        )
        if not allow_entry:
            return None

        log.info(
            "FVG detected",
            agent=self.name,
            pattern=fvg.pattern_name,
            gap_size=f"{fvg.gap_size:.4%}",
            confidence=f"{confidence:.2%}",
        )

        metadata: dict[str, str | float | int | bool] = {
            "pattern": fvg.pattern_name,
            "gap_size_pct": fvg.gap_size * 100,
            "kill_zone": True,
            "entry_hhmm_ny": minute_hhmm,
            "entry_hour_ny": local_ts.hour,
            "session": (
                f"{self.session_start:02d}:00-{self.session_end:02d}:00 "
                f"{self.session_timezone}"
            ),
            "symbol": self.symbol,
        }
        if size_multiplier < 1.0:
            metadata["size_multiplier"] = size_multiplier

        return AgentSignal(
            source_agent=self.name,
            signal_type=fvg.direction,
            confidence=confidence,
            priority=self.priority,
            timestamp=timestamp,
            metadata=metadata,
        )
