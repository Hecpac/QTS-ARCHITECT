"""EMA crossover + RSI strategy agent.

Optimized for Gold (XAUT/USDT) via grid search over 1,584 parameter
combinations on 6-month 1H data (Oct 2025 – Apr 2026).

Best config: EMA 12/21, RSI entry < 70, RSI exit > 85,
trailing stop 1.5%, hard stop 3%.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import structlog

from qts_core.agents.base import BaseStrategyAgent
from qts_core.agents.protocol import AgentPriority, AgentSignal, SignalType
from qts_core.common.types import InstrumentId

log = structlog.get_logger()


class EmaRsiAgent(BaseStrategyAgent):
    """EMA crossover with RSI confirmation and stop management.

    Attributes:
        ema_fast: Fast EMA period.
        ema_slow: Slow EMA period.
        rsi_period: RSI calculation period.
        rsi_entry_max: Maximum RSI for entry (avoid overbought entries).
        rsi_exit_min: RSI above this triggers exit.
        stop_loss_pct: Hard stop-loss percentage.
        trailing_stop_pct: Trailing stop percentage (from peak).
        symbol: Trading symbol override.
    """

    def __init__(
        self,
        name: str,
        symbol: str = "",
        ema_fast: int = 12,
        ema_slow: int = 21,
        rsi_period: int = 14,
        rsi_entry_max: float = 70.0,
        rsi_exit_min: float = 85.0,
        stop_loss_pct: float = 0.03,
        trailing_stop_pct: float = 0.015,
        priority: AgentPriority = AgentPriority.NORMAL,
        min_confidence: float = 0.5,
    ) -> None:
        super().__init__(name=name, priority=priority, min_confidence=min_confidence)
        self.symbol = symbol
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_entry_max = rsi_entry_max
        self.rsi_exit_min = rsi_exit_min
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct

        # State
        self._entry_price: float = 0.0
        self._highest_since_entry: float = 0.0
        self._in_position: bool = False

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        result = np.empty_like(data)
        k = 2.0 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = data[i] * k + result[i - 1] * (1 - k)
        return result

    @staticmethod
    def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        if len(gains) >= period:
            avg_gain[period] = np.mean(gains[:period])
            avg_loss[period] = np.mean(losses[:period])
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi_vals = 100.0 - (100.0 / (1.0 + rs))
        rsi_vals[:period] = 50.0
        return rsi_vals

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------
    async def _generate_signal(
        self,
        instrument_id: InstrumentId,
        current_price: float,
        timestamp: datetime,
        ohlcv_history: list[tuple[datetime, float, float, float, float, float]] | None = None,
    ) -> AgentSignal | None:
        if ohlcv_history is None or len(ohlcv_history) < self.ema_slow + 2:
            return None

        close = np.array([c[4] for c in ohlcv_history])
        ema_f = self._ema(close, self.ema_fast)
        ema_s = self._ema(close, self.ema_slow)
        rsi_vals = self._rsi(close, self.rsi_period)

        prev_cross = ema_f[-2] - ema_s[-2]
        curr_cross = ema_f[-1] - ema_s[-1]
        rsi_now = rsi_vals[-1]

        # --- Stop management ---
        if self._in_position:
            self._highest_since_entry = max(self._highest_since_entry, current_price)
            pnl_pct = current_price / self._entry_price - 1
            trail_pct = current_price / self._highest_since_entry - 1

            if pnl_pct <= -self.stop_loss_pct:
                log.info("EmaRsi: hard stop hit", pnl_pct=f"{pnl_pct:.2%}", agent=self.name)
                self._in_position = False
                return AgentSignal(
                    source_agent=self.name,
                    signal_type=SignalType.EXIT,
                    confidence=0.95,
                    priority=AgentPriority.HIGH,
                    timestamp=timestamp,
                    metadata={"reason": "hard_stop", "pnl_pct": round(pnl_pct * 100, 2)},
                )

            if trail_pct <= -self.trailing_stop_pct and pnl_pct > 0:
                log.info("EmaRsi: trailing stop hit", trail_pct=f"{trail_pct:.2%}", agent=self.name)
                self._in_position = False
                return AgentSignal(
                    source_agent=self.name,
                    signal_type=SignalType.EXIT,
                    confidence=0.90,
                    priority=AgentPriority.HIGH,
                    timestamp=timestamp,
                    metadata={"reason": "trailing_stop", "pnl_pct": round(pnl_pct * 100, 2)},
                )

        # --- Entry: EMA fast crosses above slow + RSI below threshold ---
        if not self._in_position and prev_cross <= 0 and curr_cross > 0 and rsi_now < self.rsi_entry_max:
            self._in_position = True
            self._entry_price = current_price
            self._highest_since_entry = current_price

            confidence = 0.60 + 0.20 * (1 - rsi_now / 100)
            return AgentSignal(
                source_agent=self.name,
                signal_type=SignalType.LONG,
                confidence=min(confidence, 0.90),
                priority=self.priority,
                timestamp=timestamp,
                metadata={
                    "reason": "ema_crossover_bullish",
                    "ema_fast": round(float(ema_f[-1]), 2),
                    "ema_slow": round(float(ema_s[-1]), 2),
                    "rsi": round(float(rsi_now), 2),
                },
            )

        # --- Exit: EMA fast crosses below slow OR RSI overbought ---
        if self._in_position and ((prev_cross >= 0 and curr_cross < 0) or rsi_now > self.rsi_exit_min):
            self._in_position = False
            reason = "ema_crossover_bearish" if curr_cross < 0 else "rsi_overbought"
            return AgentSignal(
                source_agent=self.name,
                signal_type=SignalType.EXIT,
                confidence=0.75,
                priority=self.priority,
                timestamp=timestamp,
                metadata={"reason": reason, "rsi": round(float(rsi_now), 2)},
            )

        return None
