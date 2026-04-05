"""Event-driven backtesting engine for QTS-Architect.

Provides a hybrid event-driven backtest engine that simulates
the passage of time by feeding historical data bar-by-bar to
the agent system.

Design Decisions:
- Event-driven architecture for realistic simulation
- Supports both single-asset and multi-asset backtests
- Configurable execution models (fill at close, VWAP, etc.)
- Full trade history for analysis
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Final

import numpy as np
import structlog
from pydantic import BaseModel, Field

from qts_core.agents.prediction_gate import GateState, PredictionMarketGate
from qts_core.agents.protocol import SignalType, TradingDecision
from qts_core.agents.sentiment_filter import SentimentFilter
from qts_core.agents.signal_quality import SignalQuality, SignalQualityEvaluator
from qts_core.agents.watchdog import DrawdownWatchdog, WatchdogState
from qts_core.backtest.metrics import MetricsCalculator, PerformanceMetrics
from qts_core.common.types import InstrumentId, MarketData
from qts_core.models.regime import MarketRegime, RegimeDetector


if TYPE_CHECKING:
    import polars as pl

    from qts_core.agents.supervisor import Supervisor

log = structlog.get_logger()


# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_INITIAL_CAPITAL: Final[float] = 100_000.0
DEFAULT_TRADE_SIZE: Final[float] = 10_000.0
DEFAULT_SLIPPAGE_BPS: Final[float] = 0.0  # Basis points
DEFAULT_COMMISSION_BPS: Final[float] = 0.0
DEFAULT_HISTORY_WINDOW: Final[int] = 50
DUST_EPS: Final[float] = 1e-10
MIN_EQUITY_POINTS_FOR_METRICS: Final[int] = 2

# OHLCV tuple used by agents: (timestamp, open, high, low, close, volume)
OHLCVTuple = tuple[datetime, float, float, float, float, float]


# ==============================================================================
# Enums
# ==============================================================================
class FillModel(str, Enum):
    """Order fill price model."""

    CLOSE = "CLOSE"  # Fill at bar close
    OPEN = "OPEN"  # Fill at next bar open
    VWAP = "VWAP"  # Fill at VWAP (requires tick data)
    MID = "MID"  # Fill at (high + low) / 2


class TradeSide(str, Enum):
    """Trade direction."""

    BUY = "BUY"
    SELL = "SELL"


# ==============================================================================
# Domain Models
# ==============================================================================
class Trade(BaseModel):
    """Record of an executed trade."""

    model_config = {"frozen": True}

    trade_id: str
    decision_id: str
    timestamp: datetime
    instrument_id: InstrumentId
    side: TradeSide
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    commission: float = Field(default=0.0, ge=0)
    slippage: float = Field(default=0.0)
    rationale: str = ""

    @property
    def value(self) -> float:
        """Trade notional value."""
        return self.quantity * self.price

    @property
    def total_cost(self) -> float:
        """Total cost including commission."""
        return self.value + self.commission + abs(self.slippage)


class BacktestConfig(BaseModel):
    """Configuration for backtest execution."""

    model_config = {"frozen": True}

    initial_capital: float = Field(default=DEFAULT_INITIAL_CAPITAL, gt=0)
    trade_size: float = Field(default=DEFAULT_TRADE_SIZE, gt=0)
    fill_model: FillModel = FillModel.CLOSE
    slippage_bps: float = Field(default=DEFAULT_SLIPPAGE_BPS, ge=0)
    commission_bps: float = Field(default=DEFAULT_COMMISSION_BPS, ge=0)
    allow_shorting: bool = True
    max_position_size: float | None = None  # None = unlimited
    risk_fraction: float = Field(default=0.1, gt=0, le=1)
    short_leverage: float = Field(default=1.0, gt=0)
    short_borrow_rate_bps_per_day: float = Field(default=0.0, ge=0)
    min_short_liquidation_buffer: float = Field(default=0.0, ge=0, le=1)
    high_volatility_hours_utc: tuple[int, ...] = Field(default_factory=tuple)
    high_volatility_hours_size_scale: float = Field(default=1.0, ge=0.0, le=1.0)
    high_volatility_hours_entry_block: bool = False
    volatility_regime_enabled: bool = False
    volatility_regime_window_bars: int = Field(default=48, ge=5)
    volatility_regime_min_observations: int = Field(default=12, ge=5)
    volatility_regime_zscore_threshold: float = Field(default=1.5, ge=0.0)
    high_volatility_hours_normal_regime_utc: tuple[int, ...] = Field(
        default_factory=tuple
    )
    high_volatility_hours_high_regime_utc: tuple[int, ...] = Field(default_factory=tuple)
    stop_loss_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    force_close_positions_at_end: bool = True

    # ── Risk Management Layers ──────────────────────────────────
    risk_layers_enabled: bool = False

    # Watchdog
    watchdog_warn_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    watchdog_halt_threshold: float = Field(default=0.08, ge=0.0, le=1.0)
    watchdog_weekly_halt_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    watchdog_reduce_factor: float = Field(default=0.50, ge=0.0, le=1.0)
    watchdog_cooldown_bars: int = Field(default=30, ge=0)

    # Regime Detector
    regime_lookback: int = Field(default=100, ge=10)
    regime_atr_period: int = Field(default=14, ge=2)
    regime_high_percentile: float = Field(default=0.75, ge=0.0, le=1.0)
    regime_crisis_percentile: float = Field(default=0.95, ge=0.0, le=1.0)
    regime_high_vol_size_scale: float = Field(default=0.60, ge=0.0, le=1.0)

    # Signal Quality
    signal_quality_window: int = Field(default=10, ge=2)
    signal_quality_min_samples: int = Field(default=5, ge=1)
    signal_quality_medium_size_scale: float = Field(default=0.60, ge=0.0, le=1.0)


class BacktestResult(BaseModel):
    """Complete backtest results."""

    model_config = {"frozen": True}

    # Configuration
    config: BacktestConfig

    # Equity
    initial_capital: float
    final_capital: float
    equity_curve: list[float]

    # Trades
    trades: list[Trade]
    total_trades: int

    # Performance
    metrics: PerformanceMetrics | None = None

    # Timing
    start_date: datetime | None = None
    end_date: datetime | None = None
    bars_processed: int = 0


# ==============================================================================
# Portfolio State
# ==============================================================================
@dataclass
class PortfolioState:
    """Mutable portfolio state during backtest."""

    cash: float
    positions: dict[InstrumentId, float] = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    short_opened_at: dict[InstrumentId, datetime] = field(default_factory=dict)
    short_borrow_fees_paid: float = 0.0
    entry_prices: dict[InstrumentId, float] = field(default_factory=dict)

    def get_position(self, instrument_id: InstrumentId) -> float:
        """Get position quantity (0 if not held)."""
        return self.positions.get(instrument_id, 0.0)

    def update_position(self, instrument_id: InstrumentId, delta: float) -> None:
        """Update position by delta quantity."""
        current = self.get_position(instrument_id)
        new_qty = current + delta
        if abs(new_qty) < DUST_EPS:  # Clean up dust
            self.positions.pop(instrument_id, None)
        else:
            self.positions[instrument_id] = new_qty

    def mark_to_market(
        self,
        prices: dict[InstrumentId, float],
        timestamp: datetime,
    ) -> float:
        """Calculate total equity and update curve."""
        position_value = sum(
            qty * prices.get(instrument_id, 0.0)
            for instrument_id, qty in self.positions.items()
        )
        equity = self.cash + position_value
        self.equity_curve.append(equity)
        self.timestamps.append(timestamp)
        return equity


# ==============================================================================
# Backtest Engine
# ==============================================================================
class EventEngine:
    """Event-driven backtest engine.

    Simulates trading by replaying historical data through the agent
    system and executing simulated trades.

    Usage:
        ```python
        from qts_core.backtest import EventEngine, BacktestConfig
        from qts_core.agents.supervisor import Supervisor

        supervisor = Supervisor(...)
        config = BacktestConfig(initial_capital=100_000)
        engine = EventEngine(supervisor, config)

        result = await engine.run(historical_data)
        print(f"Final equity: {result.final_capital}")
        ```

    Attributes:
        supervisor: Agent supervisor for decision making.
        config: Backtest configuration.
        state: Current portfolio state.
        trades: List of executed trades.
    """

    def __init__(
        self,
        supervisor: Supervisor,
        config: BacktestConfig | None = None,
        initial_capital: float | None = None,
    ) -> None:
        """Initialize backtest engine.

        Args:
            supervisor: Agent supervisor instance.
            config: Backtest configuration.
            initial_capital: Override initial capital (backward compat).
        """
        self.supervisor = supervisor
        self.config = config or BacktestConfig()

        # Allow override for backward compatibility
        if initial_capital is not None:
            overrides = {
                k: v
                for k, v in self.config.model_dump().items()
                if k != "initial_capital"
            }
            self.config = BacktestConfig(
                initial_capital=initial_capital,
                **overrides,
            )

        self.state = PortfolioState(cash=self.config.initial_capital)
        self.trades: list[Trade] = []
        self._trade_counter = 0

        self._last_close_by_instrument: dict[InstrumentId, float] = {}
        self._recent_abs_log_returns: dict[InstrumentId, deque[float]] = {}

        # ── Risk management layers ──────────────────────────────
        if self.config.risk_layers_enabled:
            self._watchdog = DrawdownWatchdog(
                warn_threshold=self.config.watchdog_warn_threshold,
                halt_threshold=self.config.watchdog_halt_threshold,
                weekly_halt_threshold=self.config.watchdog_weekly_halt_threshold,
                reduce_factor=self.config.watchdog_reduce_factor,
                cooldown_bars=self.config.watchdog_cooldown_bars,
            )
            self._regime = RegimeDetector(
                lookback=self.config.regime_lookback,
                atr_period=self.config.regime_atr_period,
                high_percentile=self.config.regime_high_percentile,
                crisis_percentile=self.config.regime_crisis_percentile,
                high_vol_size_scale=self.config.regime_high_vol_size_scale,
            )
            self._signal_quality = SignalQualityEvaluator(
                window_signals=self.config.signal_quality_window,
                min_samples=self.config.signal_quality_min_samples,
                medium_size_scale=self.config.signal_quality_medium_size_scale,
            )
        else:
            self._watchdog = None
            self._regime = None
            self._signal_quality = None

    @property
    def history(self) -> list[dict]:
        """Backward compatible trade history."""
        return [
            {
                "timestamp": t.timestamp,
                "side": t.side.value,
                "quantity": t.quantity,
                "price": t.price,
                "id": t.decision_id,
            }
            for t in self.trades
        ]

    def _estimated_short_liquidation_buffer(self) -> float:
        """Approximate short liquidation distance proxy from configured leverage."""
        if self.config.short_leverage <= 0:
            return 0.0
        return 1.0 / self.config.short_leverage

    def _estimate_short_borrow_fee(
        self,
        instrument_id: InstrumentId,
        notional: float,
        asof: datetime,
    ) -> float:
        """Estimate carry fee accrued for an open short position."""
        if self.config.short_borrow_rate_bps_per_day <= 0 or notional <= 0:
            return 0.0

        opened_at = self.state.short_opened_at.get(instrument_id)
        if opened_at is None:
            return 0.0

        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=timezone.utc)

        elapsed_seconds = max(0.0, (asof - opened_at).total_seconds())
        elapsed_days = elapsed_seconds / 86_400.0
        rate_per_day = self.config.short_borrow_rate_bps_per_day / 10_000.0
        return notional * rate_per_day * elapsed_days

    def _track_short_position_window(
        self,
        instrument_id: InstrumentId,
        prior_position: float,
        new_position: float,
        asof: datetime,
    ) -> None:
        """Track short exposure windows for borrow/funding accounting."""
        if new_position < -DUST_EPS:
            started = self.state.short_opened_at.get(instrument_id)
            if prior_position >= -DUST_EPS or started is None:
                self.state.short_opened_at[instrument_id] = asof
            return

        self.state.short_opened_at.pop(instrument_id, None)

    async def run(self, data_feed: pl.DataFrame) -> BacktestResult:
        """Run backtest simulation.

        Args:
            data_feed: Historical market data with columns:
                - timestamp: Bar timestamp
                - instrument_id: Trading pair/asset
                - open, high, low, close, volume: OHLCV data

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        log.info(
            "Starting backtest",
            bars=len(data_feed),
            initial_capital=self.config.initial_capital,
        )

        start_date: datetime | None = None
        end_date: datetime | None = None
        bars_processed = 0

        history_by_instrument: dict[InstrumentId, deque[OHLCVTuple]] = {}
        last_prices: dict[InstrumentId, float] = {}

        for row in data_feed.iter_rows(named=True):
            # Parse timestamp
            ts = row["timestamp"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            if start_date is None:
                start_date = ts
            end_date = ts

            # Create market data
            market_data = MarketData(
                instrument_id=InstrumentId(str(row.get("instrument_id", "UNKNOWN"))),
                timestamp=ts,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            # Mark to market using latest known prices across all instruments.
            last_prices[market_data.instrument_id] = market_data.close
            equity = self.state.mark_to_market(last_prices, ts)

            # ── Risk layer: Watchdog + Regime ─────────────────────
            watchdog_verdict = None
            regime_verdict = None
            if self._watchdog is not None:
                if bars_processed == 0:
                    self._watchdog.reset(equity)
                watchdog_verdict = self._watchdog.evaluate(
                    current_equity=equity, timestamp=ts,
                )
                if watchdog_verdict.state == WatchdogState.HALTED:
                    bars_processed += 1
                    continue

            if self._regime is not None:
                regime_verdict = self._regime.update(
                    high=market_data.high,
                    low=market_data.low,
                    close=market_data.close,
                )
                if regime_verdict.regime == MarketRegime.CRISIS:
                    bars_processed += 1
                    continue

            # Per-position stop-loss check
            if self.config.stop_loss_pct > 0:
                for instr_id in list(self.state.positions):
                    qty = self.state.positions.get(instr_id, 0.0)
                    if abs(qty) < DUST_EPS:
                        continue
                    entry_price = self.state.entry_prices.get(instr_id)
                    if not entry_price or entry_price <= 0:
                        continue
                    current_price = last_prices.get(instr_id, 0.0)
                    if current_price <= 0:
                        continue
                    if qty > 0:
                        loss_pct = (entry_price - current_price) / entry_price
                    else:
                        loss_pct = (current_price - entry_price) / entry_price
                    if loss_pct >= self.config.stop_loss_pct:
                        log.warning(
                            "Stop-loss triggered",
                            instrument=instr_id,
                            entry=entry_price,
                            current=current_price,
                            loss_pct=f"{loss_pct:.2%}",
                        )
                        exit_decision = TradingDecision(
                            instrument_id=instr_id,
                            action=SignalType.EXIT,
                            quantity_modifier=1.0,
                            rationale=f"Stop-loss {self.config.stop_loss_pct:.1%}",
                        )
                        self._execute(exit_decision, market_data)

            # Build rolling OHLCV history for agents
            ohlcv_tuple: OHLCVTuple = (
                market_data.timestamp,
                market_data.open,
                market_data.high,
                market_data.low,
                market_data.close,
                market_data.volume,
            )
            history = history_by_instrument.setdefault(
                market_data.instrument_id,
                deque(maxlen=DEFAULT_HISTORY_WINDOW),
            )
            history.append(ohlcv_tuple)

            # Approximate exposure for risk layer
            position_qty = self.state.get_position(market_data.instrument_id)
            position_value = abs(position_qty * market_data.close)
            short_position_value = abs(min(position_qty, 0.0) * market_data.close)
            portfolio_exposure = (position_value / equity) if equity > 0 else 0.0
            short_exposure_fraction = (
                (short_position_value / equity) if equity > 0 else 0.0
            )
            is_high_volatility_regime = self._is_high_volatility_regime(market_data)

            # Agent decision
            decision = await self.supervisor.run(
                market_data,
                ohlcv_history=list(history),
                portfolio_exposure=portfolio_exposure,
                short_exposure_fraction=short_exposure_fraction,
            )

            # ── Risk layer: post-decision gates + scaling ──────────
            if decision and decision.action != SignalType.EXIT and self.config.risk_layers_enabled:
                # Signal quality gate
                q_mult = 1.0
                if self._signal_quality is not None:
                    self._signal_quality.record_signal(
                        direction=decision.action,
                        confidence=decision.quantity_modifier,
                        timestamp=ts,
                    )
                    qv = self._signal_quality.evaluate(now=ts)
                    if qv.block_entry:
                        decision = None
                    elif qv.quality == SignalQuality.MEDIUM:
                        q_mult = qv.size_multiplier

                # Compound scaling (watchdog × regime × quality)
                if decision is not None:
                    w_mult = watchdog_verdict.size_multiplier if watchdog_verdict else 1.0
                    r_mult = regime_verdict.size_multiplier if regime_verdict else 1.0
                    compound = w_mult * r_mult * q_mult
                    if compound < 1.0:
                        decision = decision.model_copy(
                            update={
                                "quantity_modifier": decision.quantity_modifier * compound,
                                "rationale": f"{decision.rationale} | Risk scale {compound:.0%}",
                            }
                        )

            # Execute
            if decision:
                self._execute(
                    decision,
                    market_data,
                    is_high_volatility_regime=is_high_volatility_regime,
                )

                # Re-mark same bar after execution so equity reflects fills/carry.
                updated_position_value = sum(
                    qty * last_prices.get(instrument_id, 0.0)
                    for instrument_id, qty in self.state.positions.items()
                )
                self.state.equity_curve[-1] = self.state.cash + updated_position_value

            bars_processed += 1

        if (
            self.config.force_close_positions_at_end
            and end_date is not None
            and any(abs(qty) > DUST_EPS for qty in self.state.positions.values())
        ):
            self._force_close_open_positions(last_prices=last_prices, asof=end_date)
            final_position_value = sum(
                qty * last_prices.get(instrument_id, 0.0)
                for instrument_id, qty in self.state.positions.items()
            )
            final_capital = self.state.cash + final_position_value
            if self.state.equity_curve:
                self.state.equity_curve[-1] = final_capital
            else:
                self.state.equity_curve.append(final_capital)
                self.state.timestamps.append(end_date)
        else:
            final_capital = (
                self.state.equity_curve[-1]
                if self.state.equity_curve
                else self.config.initial_capital
            )

        log.info(
            "Backtest complete",
            final_equity=final_capital,
            total_trades=len(self.trades),
            bars=bars_processed,
        )

        # Calculate returns and metrics
        metrics = self._calculate_metrics(start_date, end_date)

        return BacktestResult(
            config=self.config,
            initial_capital=self.config.initial_capital,
            final_capital=final_capital,
            equity_curve=self.state.equity_curve,
            trades=self.trades,
            total_trades=len(self.trades),
            metrics=metrics,
            start_date=start_date,
            end_date=end_date,
            bars_processed=bars_processed,
        )

    def _force_close_open_positions(
        self,
        last_prices: dict[InstrumentId, float],
        asof: datetime,
    ) -> None:
        """Force-close open positions at backtest end for realistic accounting."""
        for instrument_id, position_qty in list(self.state.positions.items()):
            if abs(position_qty) < DUST_EPS:
                continue

            last_price = last_prices.get(instrument_id)
            if last_price is None or last_price <= 0:
                log.warning(
                    "Skipping forced close: missing last price",
                    instrument=instrument_id,
                    quantity=position_qty,
                )
                continue

            exit_decision = TradingDecision(
                instrument_id=instrument_id,
                action=SignalType.EXIT,
                quantity_modifier=1.0,
                rationale="Backtest forced close at end-of-run",
            )
            market_data = MarketData(
                instrument_id=instrument_id,
                timestamp=asof,
                open=last_price,
                high=last_price,
                low=last_price,
                close=last_price,
                volume=0.0,
            )
            self._execute(exit_decision, market_data)

    def _infer_periods_per_year(self) -> int:
        """Infer annualization frequency from observed bar spacing."""
        if len(self.state.timestamps) < 2:
            return 252

        deltas_seconds = [
            (curr - prev).total_seconds()
            for prev, curr in zip(self.state.timestamps, self.state.timestamps[1:])
            if (curr - prev).total_seconds() > 0
        ]
        if not deltas_seconds:
            return 252

        median_seconds = float(np.median(np.asarray(deltas_seconds, dtype=np.float64)))
        if median_seconds <= 0:
            return 252

        seconds_per_year = 365.0 * 24.0 * 3600.0
        periods_per_year = int(round(seconds_per_year / median_seconds))
        return max(1, periods_per_year)

    def _resolve_guardrail_hours(
        self,
        is_high_volatility_regime: bool,
    ) -> tuple[int, ...]:
        """Resolve guardrail hours from static/dynamic regime config."""
        if not self.config.volatility_regime_enabled:
            return self.config.high_volatility_hours_utc

        if (
            is_high_volatility_regime
            and self.config.high_volatility_hours_high_regime_utc
        ):
            return self.config.high_volatility_hours_high_regime_utc

        if (
            not is_high_volatility_regime
            and self.config.high_volatility_hours_normal_regime_utc
        ):
            return self.config.high_volatility_hours_normal_regime_utc

        return self.config.high_volatility_hours_utc

    def _is_high_volatility_regime(
        self,
        market_data: MarketData,
    ) -> bool:
        """Classify current bar as high-vol regime using rolling abs-log returns."""
        if not self.config.volatility_regime_enabled:
            return False

        instrument_id = market_data.instrument_id
        current_close = market_data.close
        if current_close <= 0:
            return False

        previous_close = self._last_close_by_instrument.get(instrument_id)
        self._last_close_by_instrument[instrument_id] = current_close

        if previous_close is None or previous_close <= 0:
            return False

        abs_log_return = abs(float(np.log(current_close / previous_close)))
        history = self._recent_abs_log_returns.setdefault(
            instrument_id,
            deque(maxlen=self.config.volatility_regime_window_bars),
        )

        baseline_values = list(history)
        history.append(abs_log_return)

        required_obs = max(
            5,
            min(
                self.config.volatility_regime_window_bars - 1,
                self.config.volatility_regime_min_observations,
            ),
        )
        if len(baseline_values) < required_obs:
            return False

        baseline = np.asarray(baseline_values, dtype=np.float64)
        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline, ddof=0))

        if baseline_std <= DUST_EPS:
            multiplier = 1.0 + self.config.volatility_regime_zscore_threshold
            return abs_log_return >= baseline_mean * multiplier

        zscore = (abs_log_return - baseline_mean) / baseline_std
        return zscore >= self.config.volatility_regime_zscore_threshold

    def _apply_hour_guardrail(
        self,
        decision: TradingDecision,
        market_data: MarketData,
        *,
        is_high_volatility_regime: bool = False,
    ) -> TradingDecision | None:
        """Apply optional hour-of-day volatility guardrail to entry actions."""
        if decision.action not in (SignalType.LONG, SignalType.SHORT):
            return decision

        guardrail_hours_utc = self._resolve_guardrail_hours(
            is_high_volatility_regime=is_high_volatility_regime
        )
        if not guardrail_hours_utc:
            return decision

        timestamp_utc = market_data.timestamp
        if timestamp_utc.tzinfo is None:
            timestamp_utc = timestamp_utc.replace(tzinfo=timezone.utc)
        else:
            timestamp_utc = timestamp_utc.astimezone(timezone.utc)

        hour_utc = timestamp_utc.hour
        if hour_utc not in guardrail_hours_utc:
            return decision

        regime_label = "high" if is_high_volatility_regime else "normal"

        if self.config.high_volatility_hours_entry_block:
            log.warning(
                "Backtest entry blocked by high-volatility hour guardrail",
                instrument=decision.instrument_id,
                hour_utc=hour_utc,
                action=decision.action,
                volatility_regime=regime_label,
            )
            return None

        if self.config.high_volatility_hours_size_scale >= 1.0:
            return decision

        adjusted_modifier = max(
            0.0,
            min(
                1.0,
                decision.quantity_modifier * self.config.high_volatility_hours_size_scale,
            ),
        )
        if adjusted_modifier < DUST_EPS:
            return None

        return decision.model_copy(
            update={
                "quantity_modifier": adjusted_modifier,
                "rationale": (
                    f"{decision.rationale} | Backtest guardrail: "
                    f"high-vol hour {hour_utc:02d}:00 UTC (regime={regime_label})"
                ),
            }
        )

    def _execute(
        self,
        decision: TradingDecision,
        market_data: MarketData,
        *,
        is_high_volatility_regime: bool = False,
    ) -> None:
        """Execute trade based on decision."""
        if decision.action == SignalType.NEUTRAL:
            return

        decision = self._apply_hour_guardrail(
            decision,
            market_data,
            is_high_volatility_regime=is_high_volatility_regime,
        )
        if decision is None:
            return

        # Calculate fill price based on model
        fill_price = self._get_fill_price(market_data)

        # Calculate trade size
        trade_value = self.config.trade_size * decision.quantity_modifier
        quantity = trade_value / fill_price

        # Apply slippage
        slippage = fill_price * (self.config.slippage_bps / 10000)

        # Commission
        commission = trade_value * (self.config.commission_bps / 10000)

        if decision.action == SignalType.LONG:
            self._execute_buy(
                decision,
                market_data,
                quantity,
                fill_price,
                commission,
                slippage,
            )
        elif decision.action == SignalType.SHORT:
            estimated_buffer = self._estimated_short_liquidation_buffer()
            if (
                self.config.min_short_liquidation_buffer > 0
                and estimated_buffer < self.config.min_short_liquidation_buffer
            ):
                log.warning(
                    "Backtest short rejected: liquidation buffer below minimum",
                    estimated_buffer=estimated_buffer,
                    required_buffer=self.config.min_short_liquidation_buffer,
                    short_leverage=self.config.short_leverage,
                    instrument=decision.instrument_id,
                )
                return

            self._execute_sell(
                decision,
                market_data,
                quantity,
                fill_price,
                commission,
                slippage,
            )
        elif decision.action == SignalType.EXIT:
            current_position = self.state.get_position(decision.instrument_id)
            if current_position > DUST_EPS:
                self._execute_sell(
                    decision,
                    market_data,
                    current_position,
                    fill_price,
                    commission,
                    slippage,
                )
            elif current_position < -DUST_EPS:
                self._execute_buy(
                    decision,
                    market_data,
                    abs(current_position),
                    fill_price,
                    commission,
                    slippage,
                )

    def _execute_buy(
        self,
        decision: TradingDecision,
        market_data: MarketData,
        quantity: float,
        price: float,
        commission: float,
        slippage: float,
    ) -> None:
        """Execute buy order."""
        prior_position = self.state.get_position(decision.instrument_id)
        short_cover_qty = min(quantity, abs(min(prior_position, 0.0)))
        borrow_fee = self._estimate_short_borrow_fee(
            decision.instrument_id,
            short_cover_qty * price,
            market_data.timestamp,
        )

        cost = (quantity * price) + commission + slippage + borrow_fee

        if self.state.cash < cost:
            log.warning(
                "Insufficient funds for buy",
                required=cost,
                available=self.state.cash,
            )
            return

        # Update state
        self.state.cash -= cost
        self.state.update_position(decision.instrument_id, quantity)

        new_position = self.state.get_position(decision.instrument_id)

        # Track entry price for stop-loss
        if new_position > DUST_EPS:
            old_entry = self.state.entry_prices.get(decision.instrument_id, 0.0)
            old_qty = prior_position if prior_position > DUST_EPS else 0.0
            if old_qty > DUST_EPS and old_entry > 0:
                self.state.entry_prices[decision.instrument_id] = (
                    (old_qty * old_entry + quantity * price) / new_position
                )
            else:
                self.state.entry_prices[decision.instrument_id] = price
        elif abs(new_position) < DUST_EPS:
            self.state.entry_prices.pop(decision.instrument_id, None)

        self._track_short_position_window(
            decision.instrument_id,
            prior_position=prior_position,
            new_position=new_position,
            asof=market_data.timestamp,
        )

        if borrow_fee > 0:
            self.state.short_borrow_fees_paid += borrow_fee

        # Record trade
        self._record_trade(
            decision,
            market_data.timestamp,
            TradeSide.BUY,
            quantity,
            price,
            commission + borrow_fee,
            slippage,
        )

    def _execute_sell(
        self,
        decision: TradingDecision,
        market_data: MarketData,
        quantity: float,
        price: float,
        commission: float,
        slippage: float,
    ) -> None:
        """Execute sell order."""
        prior_position = self.state.get_position(decision.instrument_id)

        # Check shorting rules
        if not self.config.allow_shorting and prior_position < quantity:
            log.warning(
                "Shorting not allowed",
                position=prior_position,
                requested=quantity,
            )
            quantity = max(0, prior_position)  # Sell only what we have
            if quantity == 0:
                return

        proceeds = (quantity * price) - commission - slippage

        # Update state
        self.state.cash += proceeds
        self.state.update_position(decision.instrument_id, -quantity)

        new_position = self.state.get_position(decision.instrument_id)

        # Track entry price for stop-loss
        if new_position < -DUST_EPS:
            old_entry = self.state.entry_prices.get(decision.instrument_id, 0.0)
            old_qty = abs(prior_position) if prior_position < -DUST_EPS else 0.0
            new_short_qty = abs(new_position)
            if old_qty > DUST_EPS and old_entry > 0:
                self.state.entry_prices[decision.instrument_id] = (
                    (old_qty * old_entry + quantity * price) / new_short_qty
                )
            else:
                self.state.entry_prices[decision.instrument_id] = price
        elif abs(new_position) < DUST_EPS:
            self.state.entry_prices.pop(decision.instrument_id, None)

        self._track_short_position_window(
            decision.instrument_id,
            prior_position=prior_position,
            new_position=new_position,
            asof=market_data.timestamp,
        )

        # Record trade
        self._record_trade(
            decision,
            market_data.timestamp,
            TradeSide.SELL,
            quantity,
            price,
            commission,
            slippage,
        )

    def _record_trade(
        self,
        decision: TradingDecision,
        timestamp: datetime,
        side: TradeSide,
        quantity: float,
        price: float,
        commission: float,
        slippage: float,
    ) -> None:
        """Record executed trade."""
        self._trade_counter += 1

        trade = Trade(
            trade_id=f"bt-{self._trade_counter}",
            decision_id=str(decision.decision_id),
            timestamp=timestamp,
            instrument_id=decision.instrument_id,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            rationale=decision.rationale,
        )

        self.trades.append(trade)

        log.info(
            "Trade executed",
            side=side.value,
            qty=f"{quantity:.4f}",
            price=f"{price:.2f}",
            rationale=decision.rationale[:50] if decision.rationale else "",
        )

    def _get_fill_price(self, market_data: MarketData) -> float:
        """Get fill price based on fill model."""
        match self.config.fill_model:
            case FillModel.CLOSE:
                return market_data.close
            case FillModel.OPEN:
                return market_data.open
            case FillModel.MID:
                return (market_data.high + market_data.low) / 2
            case FillModel.VWAP:
                # Approximate VWAP as typical price
                return (market_data.high + market_data.low + market_data.close) / 3
            case _:
                return market_data.close

    def _calculate_metrics(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> PerformanceMetrics | None:
        """Calculate performance metrics from equity curve."""
        if len(self.state.equity_curve) < MIN_EQUITY_POINTS_FOR_METRICS:
            return None

        # Calculate returns from equity curve
        equity = np.array(self.state.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        periods_per_year = self._infer_periods_per_year()
        calc = MetricsCalculator(returns, periods_per_year=periods_per_year)
        return calc.calculate(start_date, end_date)

    def _log_trade(
        self,
        side: str,
        qty: float,
        price: float,
        decision: TradingDecision,
    ) -> None:
        """Legacy logging method for backward compatibility."""
        log.info(
            "Trade Executed",
            side=side,
            qty=f"{qty:.4f}",
            price=f"{price:.2f}",
            agent_reason=decision.rationale,
        )
