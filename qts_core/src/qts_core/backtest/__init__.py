"""Backtesting module for QTS-Architect.

This module provides backtesting infrastructure including:
- Event-driven backtest engine
- Performance metrics calculation
- Triple barrier labeling for ML
- Trade analysis

Example:
    ```python
    from qts_core.backtest import (
        EventEngine,
        BacktestConfig,
        MetricsCalculator,
        triple_barrier_method,
    )

    # Run backtest
    config = BacktestConfig(initial_capital=100_000)
    engine = EventEngine(supervisor, config)
    result = await engine.run(historical_data)

    # Analyze performance
    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio}")
    print(f"Max Drawdown: {result.metrics.max_drawdown}")

    # Calculate custom metrics
    calc = MetricsCalculator(returns)
    metrics = calc.calculate()
    ```
"""

from qts_core.backtest.engine import (
    DEFAULT_COMMISSION_BPS,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_TRADE_SIZE,
    BacktestConfig,
    BacktestResult,
    EventEngine,
    FillModel,
    PortfolioState,
    Trade,
    TradeSide,
)
from qts_core.backtest.labeling import (
    DEFAULT_PT_MULTIPLIER,
    DEFAULT_SL_MULTIPLIER,
    DEFAULT_VOLATILITY_COL,
    BarrierLabel,
    add_volatility,
    calculate_label_distribution,
    filter_valid_labels,
    triple_barrier_method,
)
from qts_core.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    DrawdownInfo,
    MetricsCalculator,
    PerformanceMetrics,
    PerformanceStats,
    TradeMetrics,
    analyze_trades,
)

__all__ = [
    # Engine
    "EventEngine",
    "BacktestConfig",
    "BacktestResult",
    "PortfolioState",
    "Trade",
    "FillModel",
    "TradeSide",
    "DEFAULT_INITIAL_CAPITAL",
    "DEFAULT_TRADE_SIZE",
    "DEFAULT_SLIPPAGE_BPS",
    "DEFAULT_COMMISSION_BPS",
    # Metrics
    "MetricsCalculator",
    "PerformanceMetrics",
    "TradeMetrics",
    "DrawdownInfo",
    "PerformanceStats",
    "analyze_trades",
    "TRADING_DAYS_PER_YEAR",
    # Labeling
    "triple_barrier_method",
    "add_volatility",
    "filter_valid_labels",
    "calculate_label_distribution",
    "BarrierLabel",
    "DEFAULT_PT_MULTIPLIER",
    "DEFAULT_SL_MULTIPLIER",
    "DEFAULT_VOLATILITY_COL",
]
