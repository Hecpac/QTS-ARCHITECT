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

from qts_core.agents.protocol import SignalType, TradingDecision
from qts_core.backtest.metrics import MetricsCalculator, PerformanceMetrics
from qts_core.common.types import InstrumentId, MarketData


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

            # Mark to market
            prices = {market_data.instrument_id: market_data.close}
            equity = self.state.mark_to_market(prices, ts)

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
            portfolio_exposure = (position_value / equity) if equity > 0 else 0.0

            # Agent decision
            decision = await self.supervisor.run(
                market_data,
                ohlcv_history=list(history),
                portfolio_exposure=portfolio_exposure,
            )

            # Execute
            if decision:
                self._execute(decision, market_data)

            bars_processed += 1

        # Calculate final equity
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

    def _execute(self, decision: TradingDecision, market_data: MarketData) -> None:
        """Execute trade based on decision."""
        if decision.action == SignalType.NEUTRAL:
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
        elif decision.action in {SignalType.SHORT, SignalType.EXIT}:
            self._execute_sell(
                decision,
                market_data,
                quantity,
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
        cost = (quantity * price) + commission + slippage

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

        # Record trade
        self._record_trade(
            decision,
            market_data.timestamp,
            TradeSide.BUY,
            quantity,
            price,
            commission,
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
        current_position = self.state.get_position(decision.instrument_id)

        # For EXIT, sell entire position
        if decision.action == SignalType.EXIT:
            if current_position <= 0:
                return
            quantity = current_position

        # Check shorting rules
        if not self.config.allow_shorting and current_position < quantity:
            log.warning(
                "Shorting not allowed",
                position=current_position,
                requested=quantity,
            )
            quantity = max(0, current_position)  # Sell only what we have
            if quantity == 0:
                return

        proceeds = (quantity * price) - commission - slippage

        # Update state
        self.state.cash += proceeds
        self.state.update_position(decision.instrument_id, -quantity)

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

        calc = MetricsCalculator(returns)
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
