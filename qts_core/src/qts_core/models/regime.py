"""Regime Detection — ATR-based volatility regime classifier.

Inspired by AutoResearch 2026-03-29:
- Insight #4 (score 82): FinBERT + XGBoost + regime filters → 135% in 24 months
- Insight #3 (score 82): Agent architecture > LLM backbone

Classifies market state into three regimes based on ATR percentile
relative to a rolling lookback window:

    LOW_VOL:  ATR < low_percentile    → full size, trend strategies
    NORMAL:   between percentiles     → normal operation
    HIGH_VOL: ATR > high_percentile   → reduced size, mean-reversion only
    CRISIS:   ATR > crisis_percentile → halt new entries

No external dependencies beyond what QTS already uses (pure math on
OHLCV tuples). Designed to plug into the LiveTrader tick pipeline.
"""

from __future__ import annotations

from collections import deque
from enum import Enum

import structlog
from pydantic import BaseModel, ConfigDict, Field

log = structlog.get_logger()


class MarketRegime(str, Enum):
    """Current volatility regime."""

    LOW_VOL = "LOW_VOL"
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOL"
    CRISIS = "CRISIS"


class RegimeVerdict(BaseModel):
    """Output of regime detection."""

    model_config = ConfigDict(frozen=True)

    regime: MarketRegime
    size_multiplier: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Position size scaling for this regime",
    )
    current_atr: float = Field(default=0.0, description="Current ATR value")
    atr_percentile: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Where current ATR falls in historical distribution",
    )
    reason: str = ""


class RegimeDetector:
    """ATR-based regime classifier with rolling percentile thresholds.

    Uses a rolling window of ATR values to determine where current
    volatility sits in the historical distribution.

    Attributes:
        lookback: Number of bars for ATR history.
        atr_period: ATR smoothing window.
        low_percentile: Below this → LOW_VOL (e.g., 0.25).
        high_percentile: Above this → HIGH_VOL (e.g., 0.75).
        crisis_percentile: Above this → CRISIS (e.g., 0.95).
        high_vol_size_scale: Size multiplier in HIGH_VOL.
    """

    def __init__(
        self,
        lookback: int = 100,
        atr_period: int = 14,
        low_percentile: float = 0.25,
        high_percentile: float = 0.75,
        crisis_percentile: float = 0.95,
        high_vol_size_scale: float = 0.60,
    ) -> None:
        self.lookback = lookback
        self.atr_period = atr_period
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.crisis_percentile = crisis_percentile
        self.high_vol_size_scale = high_vol_size_scale

        self._tr_buffer: deque[float] = deque(maxlen=atr_period)
        self._atr_history: deque[float] = deque(maxlen=lookback)
        self._prev_close: float | None = None
        self._regime: MarketRegime = MarketRegime.NORMAL

    @property
    def regime(self) -> MarketRegime:
        """Current detected regime."""
        return self._regime

    @property
    def ready(self) -> bool:
        """Whether enough data has been collected for regime detection."""
        return len(self._atr_history) >= self.atr_period

    def reset(self) -> None:
        """Reset all internal state."""
        self._tr_buffer.clear()
        self._atr_history.clear()
        self._prev_close = None
        self._regime = MarketRegime.NORMAL

    def update(
        self,
        high: float,
        low: float,
        close: float,
    ) -> RegimeVerdict:
        """Update with new OHLC bar and return regime verdict.

        Args:
            high: Bar high price.
            low: Bar low price.
            close: Bar close price.

        Returns:
            RegimeVerdict with current regime and size multiplier.
        """
        # Calculate True Range
        if self._prev_close is not None:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        else:
            tr = high - low

        self._prev_close = close
        self._tr_buffer.append(tr)

        # Need enough TR values for ATR
        if len(self._tr_buffer) < self.atr_period:
            return RegimeVerdict(
                regime=MarketRegime.NORMAL,
                size_multiplier=1.0,
                current_atr=0.0,
                atr_percentile=0.5,
                reason=f"Warming up ({len(self._tr_buffer)}/{self.atr_period} bars)",
            )

        # Calculate ATR (simple moving average of TR)
        current_atr = sum(self._tr_buffer) / len(self._tr_buffer)
        self._atr_history.append(current_atr)

        # Need enough ATR history for percentile
        if len(self._atr_history) < self.atr_period:
            return RegimeVerdict(
                regime=MarketRegime.NORMAL,
                size_multiplier=1.0,
                current_atr=current_atr,
                atr_percentile=0.5,
                reason=f"Building history ({len(self._atr_history)}/{self.atr_period})",
            )

        # Calculate percentile of current ATR in history
        # Use midpoint ranking: (strictly_below + 0.5 * equal) / n
        # This ensures identical values → 50th percentile, not 100th
        sorted_history = sorted(self._atr_history)
        n = len(sorted_history)
        below = sum(1 for v in sorted_history if v < current_atr)
        equal = sum(1 for v in sorted_history if v == current_atr)
        percentile = (below + 0.5 * equal) / n

        # Classify regime
        if percentile >= self.crisis_percentile:
            self._regime = MarketRegime.CRISIS
            log.warning(
                "REGIME: CRISIS detected",
                atr=f"{current_atr:.6f}",
                percentile=f"{percentile:.0%}",
            )
            return RegimeVerdict(
                regime=MarketRegime.CRISIS,
                size_multiplier=0.0,
                current_atr=current_atr,
                atr_percentile=percentile,
                reason=(
                    f"ATR at {percentile:.0%} percentile "
                    f"(>= {self.crisis_percentile:.0%}) — CRISIS"
                ),
            )

        if percentile >= self.high_percentile:
            self._regime = MarketRegime.HIGH_VOL
            return RegimeVerdict(
                regime=MarketRegime.HIGH_VOL,
                size_multiplier=self.high_vol_size_scale,
                current_atr=current_atr,
                atr_percentile=percentile,
                reason=(
                    f"ATR at {percentile:.0%} percentile — HIGH_VOL, "
                    f"size scaled to {self.high_vol_size_scale:.0%}"
                ),
            )

        if percentile <= self.low_percentile:
            self._regime = MarketRegime.LOW_VOL
            return RegimeVerdict(
                regime=MarketRegime.LOW_VOL,
                size_multiplier=1.0,
                current_atr=current_atr,
                atr_percentile=percentile,
                reason=f"ATR at {percentile:.0%} percentile — LOW_VOL",
            )

        self._regime = MarketRegime.NORMAL
        return RegimeVerdict(
            regime=MarketRegime.NORMAL,
            size_multiplier=1.0,
            current_atr=current_atr,
            atr_percentile=percentile,
            reason=f"ATR at {percentile:.0%} percentile — NORMAL",
        )
