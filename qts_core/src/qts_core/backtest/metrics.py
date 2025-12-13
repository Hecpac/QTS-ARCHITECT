"""Performance metrics for backtesting.

Provides comprehensive financial performance metrics including:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Trade statistics
- Return distribution analysis

All calculations use vectorized NumPy for performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ==============================================================================
# Constants
# ==============================================================================
TRADING_DAYS_PER_YEAR: Final[int] = 252
MINUTES_PER_DAY: Final[int] = 390  # US market
HOURS_PER_DAY: Final[int] = 24  # Crypto


# ==============================================================================
# Domain Models
# ==============================================================================
class PerformanceMetrics(BaseModel):
    """Complete performance metrics for a backtest.

    All ratios are annualized. Returns are expressed as decimals (0.10 = 10%).
    """

    model_config = {"frozen": True}

    # Returns
    total_return: float = Field(description="Cumulative return")
    annualized_return: float = Field(description="CAGR")
    daily_return_mean: float = Field(description="Average daily return")
    daily_return_std: float = Field(description="Daily return volatility")

    # Risk-Adjusted Returns
    sharpe_ratio: float = Field(description="Annualized Sharpe ratio")
    sortino_ratio: float = Field(description="Annualized Sortino ratio")
    calmar_ratio: float = Field(description="Return / Max Drawdown")
    omega_ratio: float = Field(description="Prob weighted gains vs losses")

    # Drawdowns
    max_drawdown: float = Field(description="Maximum peak-to-trough decline")
    max_drawdown_duration_days: int = Field(description="Longest drawdown in days")
    avg_drawdown: float = Field(description="Average drawdown")
    drawdown_count: int = Field(description="Number of drawdown periods")

    # Volatility
    volatility_annualized: float = Field(description="Annualized volatility")
    downside_deviation: float = Field(description="Downside volatility")
    upside_deviation: float = Field(description="Upside volatility")

    # Distribution
    skewness: float = Field(description="Return distribution skewness")
    kurtosis: float = Field(description="Return distribution kurtosis")
    var_95: float = Field(description="Value at Risk (95%)")
    cvar_95: float = Field(description="Conditional VaR (95%)")

    # Win/Loss
    win_rate: float = Field(description="Percentage of winning periods")
    profit_factor: float = Field(description="Gross profit / gross loss")
    avg_win: float = Field(description="Average winning return")
    avg_loss: float = Field(description="Average losing return")
    win_loss_ratio: float = Field(description="Avg win / avg loss")
    expectancy: float = Field(description="Expected return per trade")

    # Time
    total_periods: int = Field(description="Total number of periods")
    trading_days: int = Field(description="Number of trading days")
    start_date: datetime | None = Field(default=None, description="Backtest start")
    end_date: datetime | None = Field(default=None, description="Backtest end")


class TradeMetrics(BaseModel):
    """Trade-level statistics."""

    model_config = {"frozen": True}

    total_trades: int = Field(description="Total number of trades")
    winning_trades: int = Field(description="Number of winning trades")
    losing_trades: int = Field(description="Number of losing trades")
    breakeven_trades: int = Field(description="Number of breakeven trades")

    win_rate: float = Field(description="Win rate percentage")
    avg_trade_return: float = Field(description="Average return per trade")
    avg_win: float = Field(description="Average winning trade return")
    avg_loss: float = Field(description="Average losing trade return")
    largest_win: float = Field(description="Largest winning trade")
    largest_loss: float = Field(description="Largest losing trade")

    avg_holding_period_bars: float = Field(description="Average bars held")
    max_consecutive_wins: int = Field(description="Max winning streak")
    max_consecutive_losses: int = Field(description="Max losing streak")

    profit_factor: float = Field(description="Gross profit / gross loss")
    expectancy: float = Field(description="Expected value per trade")
    sqn: float = Field(description="System Quality Number")


@dataclass
class DrawdownInfo:
    """Information about a drawdown period."""

    start_idx: int
    end_idx: int
    recovery_idx: int | None
    depth: float  # Maximum depth during this drawdown
    duration_bars: int  # Bars from start to recovery (or end if not recovered)


# ==============================================================================
# Metrics Calculator
# ==============================================================================
class MetricsCalculator:
    """Calculate comprehensive performance metrics from returns.

    Usage:
        ```python
        returns = np.array([0.01, -0.02, 0.015, ...])  # Daily returns
        calc = MetricsCalculator(returns, risk_free_rate=0.02)
        metrics = calc.calculate()
        ```

    Attributes:
        returns: Array of period returns.
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Number of periods per year for annualization.
    """

    def __init__(
        self,
        returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> None:
        """Initialize calculator.

        Args:
            returns: Array of period returns (as decimals).
            risk_free_rate: Annualized risk-free rate.
            periods_per_year: Periods per year for annualization.
        """
        self.returns = np.asarray(returns, dtype=np.float64)
        self.rf = risk_free_rate
        self.periods_per_year = periods_per_year
        self._rf_per_period = risk_free_rate / periods_per_year

        # Pre-compute equity curve
        self._equity_curve = np.cumprod(1 + self.returns)

    def calculate(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> PerformanceMetrics:
        """Calculate all performance metrics.

        Args:
            start_date: Optional backtest start date.
            end_date: Optional backtest end date.

        Returns:
            PerformanceMetrics with all calculated values.
        """
        n = len(self.returns)
        if n == 0:
            return self._empty_metrics(start_date, end_date)

        # Returns
        total_return = self.total_return()
        annualized_return = self.annualized_return()
        mean_ret = float(np.mean(self.returns))
        std_ret = float(np.std(self.returns, ddof=1)) if n > 1 else 0.0

        # Drawdowns
        dd_info = self._analyze_drawdowns()
        max_dd = self.max_drawdown()
        max_dd_duration = max((d.duration_bars for d in dd_info), default=0)
        avg_dd = float(np.mean([d.depth for d in dd_info])) if dd_info else 0.0

        # Volatility
        vol = self.volatility()
        downside_dev = self.downside_deviation()
        upside_dev = self.upside_deviation()

        # Risk-adjusted
        sharpe = self.sharpe_ratio()
        sortino = self.sortino_ratio()
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0
        omega = self.omega_ratio()

        # Distribution
        skew = self.skewness()
        kurt = self.kurtosis()
        var = self.var_95()
        cvar = self.cvar_95()

        # Win/Loss
        wins = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]
        win_rate = len(wins) / n if n > 0 else 0.0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        win_loss = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        profit_factor = self.profit_factor()
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return PerformanceMetrics(
            # Returns
            total_return=total_return,
            annualized_return=annualized_return,
            daily_return_mean=mean_ret,
            daily_return_std=std_ret,
            # Risk-Adjusted
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            # Drawdowns
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration,
            avg_drawdown=avg_dd,
            drawdown_count=len(dd_info),
            # Volatility
            volatility_annualized=vol,
            downside_deviation=downside_dev,
            upside_deviation=upside_dev,
            # Distribution
            skewness=skew,
            kurtosis=kurt,
            var_95=var,
            cvar_95=cvar,
            # Win/Loss
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss,
            expectancy=expectancy,
            # Time
            total_periods=n,
            trading_days=n,
            start_date=start_date,
            end_date=end_date,
        )

    def _empty_metrics(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> PerformanceMetrics:
        """Return empty metrics when no data."""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            daily_return_mean=0.0,
            daily_return_std=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            omega_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration_days=0,
            avg_drawdown=0.0,
            drawdown_count=0,
            volatility_annualized=0.0,
            downside_deviation=0.0,
            upside_deviation=0.0,
            skewness=0.0,
            kurtosis=0.0,
            var_95=0.0,
            cvar_95=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            win_loss_ratio=0.0,
            expectancy=0.0,
            total_periods=0,
            trading_days=0,
            start_date=start_date,
            end_date=end_date,
        )

    # ==========================================================================
    # Return Metrics
    # ==========================================================================
    def total_return(self) -> float:
        """Calculate cumulative return."""
        if len(self.returns) == 0:
            return 0.0
        return float(np.prod(1 + self.returns) - 1)

    def annualized_return(self) -> float:
        """Calculate Compound Annual Growth Rate (CAGR)."""
        n = len(self.returns)
        if n == 0:
            return 0.0

        total = self.total_return()
        years = n / self.periods_per_year

        if years <= 0 or total <= -1:
            return 0.0

        return float((1 + total) ** (1 / years) - 1)

    # ==========================================================================
    # Risk-Adjusted Metrics
    # ==========================================================================
    def sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio.

        Sharpe = sqrt(periods) * (mean(excess_returns) / std(excess_returns))
        """
        if len(self.returns) < 2:
            return 0.0

        excess = self.returns - self._rf_per_period
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)

        if std_excess == 0:
            return 0.0

        return float(np.sqrt(self.periods_per_year) * mean_excess / std_excess)

    def sortino_ratio(self) -> float:
        """Calculate annualized Sortino ratio.

        Like Sharpe but uses downside deviation instead of total volatility.
        """
        if len(self.returns) < 2:
            return 0.0

        excess = self.returns - self._rf_per_period
        mean_excess = np.mean(excess)
        downside = self.downside_deviation()

        if downside == 0:
            return float("inf") if mean_excess > 0 else 0.0

        return float(np.sqrt(self.periods_per_year) * mean_excess / downside)

    def omega_ratio(self, threshold: float = 0.0) -> float:
        """Calculate Omega ratio.

        Probability-weighted ratio of gains vs losses above/below threshold.
        """
        if len(self.returns) == 0:
            return 0.0

        gains = self.returns[self.returns > threshold] - threshold
        losses = threshold - self.returns[self.returns <= threshold]

        sum_gains = np.sum(gains)
        sum_losses = np.sum(losses)

        if sum_losses == 0:
            return float("inf") if sum_gains > 0 else 1.0

        return float(sum_gains / sum_losses)

    # ==========================================================================
    # Drawdown Metrics
    # ==========================================================================
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown (peak to trough).

        Returns negative value (e.g., -0.20 for 20% drawdown).
        """
        if len(self.returns) == 0:
            return 0.0

        running_max = np.maximum.accumulate(self._equity_curve)
        drawdowns = (self._equity_curve - running_max) / running_max

        return float(np.min(drawdowns))

    def _analyze_drawdowns(self) -> list[DrawdownInfo]:
        """Analyze all drawdown periods."""
        if len(self.returns) == 0:
            return []

        running_max = np.maximum.accumulate(self._equity_curve)
        drawdowns = (self._equity_curve - running_max) / running_max

        result: list[DrawdownInfo] = []
        in_drawdown = False
        start_idx = 0
        max_depth = 0.0

        for i, dd in enumerate(drawdowns):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
                max_depth = dd
            elif dd < 0 and in_drawdown:
                # Continuing drawdown
                max_depth = min(max_depth, dd)
            elif dd == 0 and in_drawdown:
                # Recovery
                result.append(
                    DrawdownInfo(
                        start_idx=start_idx,
                        end_idx=i - 1,
                        recovery_idx=i,
                        depth=max_depth,
                        duration_bars=i - start_idx,
                    )
                )
                in_drawdown = False
                max_depth = 0.0

        # Handle ongoing drawdown at end of data
        if in_drawdown:
            result.append(
                DrawdownInfo(
                    start_idx=start_idx,
                    end_idx=len(drawdowns) - 1,
                    recovery_idx=None,
                    depth=max_depth,
                    duration_bars=len(drawdowns) - start_idx,
                )
            )

        return result

    # ==========================================================================
    # Volatility Metrics
    # ==========================================================================
    def volatility(self) -> float:
        """Calculate annualized volatility."""
        if len(self.returns) < 2:
            return 0.0
        return float(np.std(self.returns, ddof=1) * np.sqrt(self.periods_per_year))

    def downside_deviation(self) -> float:
        """Calculate downside deviation (semi-deviation).

        Only considers returns below the threshold (default: risk-free rate).
        """
        if len(self.returns) < 2:
            return 0.0

        excess = self.returns - self._rf_per_period
        downside = excess[excess < 0]

        if len(downside) == 0:
            return 0.0

        return float(np.sqrt(np.mean(downside**2)))

    def upside_deviation(self) -> float:
        """Calculate upside deviation.

        Only considers returns above the threshold.
        """
        if len(self.returns) < 2:
            return 0.0

        excess = self.returns - self._rf_per_period
        upside = excess[excess > 0]

        if len(upside) == 0:
            return 0.0

        return float(np.sqrt(np.mean(upside**2)))

    # ==========================================================================
    # Distribution Metrics
    # ==========================================================================
    def skewness(self) -> float:
        """Calculate return distribution skewness.

        Positive = right skew (fat right tail)
        Negative = left skew (fat left tail)
        """
        if len(self.returns) < 3:
            return 0.0

        mean = np.mean(self.returns)
        std = np.std(self.returns, ddof=1)

        if std == 0:
            return 0.0

        n = len(self.returns)
        return float(
            (n / ((n - 1) * (n - 2)))
            * np.sum(((self.returns - mean) / std) ** 3)
        )

    def kurtosis(self) -> float:
        """Calculate excess kurtosis.

        Positive = fat tails (leptokurtic)
        Negative = thin tails (platykurtic)
        Normal distribution = 0
        """
        if len(self.returns) < 4:
            return 0.0

        mean = np.mean(self.returns)
        std = np.std(self.returns, ddof=1)

        if std == 0:
            return 0.0

        n = len(self.returns)
        m4 = np.mean((self.returns - mean) ** 4)
        return float(m4 / (std**4) - 3)  # Excess kurtosis

    def var_95(self) -> float:
        """Calculate Value at Risk at 95% confidence.

        Returns the 5th percentile of returns (loss threshold).
        """
        if len(self.returns) == 0:
            return 0.0
        return float(np.percentile(self.returns, 5))

    def cvar_95(self) -> float:
        """Calculate Conditional VaR (Expected Shortfall) at 95%.

        Average return in the worst 5% of cases.
        """
        if len(self.returns) == 0:
            return 0.0

        var = self.var_95()
        tail = self.returns[self.returns <= var]

        if len(tail) == 0:
            return var

        return float(np.mean(tail))

    # ==========================================================================
    # Win/Loss Metrics
    # ==========================================================================
    def win_rate(self) -> float:
        """Calculate percentage of winning periods."""
        if len(self.returns) == 0:
            return 0.0
        return float(np.sum(self.returns > 0) / len(self.returns))

    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(self.returns) == 0:
            return 0.0

        gross_profit = np.sum(self.returns[self.returns > 0])
        gross_loss = abs(np.sum(self.returns[self.returns < 0]))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)


# ==============================================================================
# Trade Analyzer
# ==============================================================================
def analyze_trades(
    trade_returns: NDArray[np.float64],
    holding_periods: NDArray[np.int64] | None = None,
) -> TradeMetrics:
    """Analyze trade-level performance.

    Args:
        trade_returns: Array of return per trade.
        holding_periods: Optional array of bars held per trade.

    Returns:
        TradeMetrics with trade statistics.
    """
    if len(trade_returns) == 0:
        return TradeMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            avg_trade_return=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            avg_holding_period_bars=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            profit_factor=0.0,
            expectancy=0.0,
            sqn=0.0,
        )

    n = len(trade_returns)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    breakeven = trade_returns[trade_returns == 0]

    win_rate = len(wins) / n if n > 0 else 0.0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    gross_profit = np.sum(wins)
    gross_loss = abs(np.sum(losses))
    profit_factor = (
        float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    )

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # System Quality Number (Van Tharp)
    avg_trade = float(np.mean(trade_returns))
    std_trade = float(np.std(trade_returns, ddof=1)) if n > 1 else 0.0
    sqn = float(np.sqrt(n) * avg_trade / std_trade) if std_trade > 0 else 0.0

    # Consecutive wins/losses
    max_wins = _max_consecutive(trade_returns > 0)
    max_losses = _max_consecutive(trade_returns < 0)

    # Holding periods
    avg_hold = float(np.mean(holding_periods)) if holding_periods is not None else 0.0

    return TradeMetrics(
        total_trades=n,
        winning_trades=len(wins),
        losing_trades=len(losses),
        breakeven_trades=len(breakeven),
        win_rate=win_rate,
        avg_trade_return=avg_trade,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=float(np.max(wins)) if len(wins) > 0 else 0.0,
        largest_loss=float(np.min(losses)) if len(losses) > 0 else 0.0,
        avg_holding_period_bars=avg_hold,
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses,
        profit_factor=profit_factor,
        expectancy=expectancy,
        sqn=sqn,
    )


def _max_consecutive(mask: NDArray[np.bool_]) -> int:
    """Count maximum consecutive True values."""
    if len(mask) == 0:
        return 0

    max_count = 0
    current = 0

    for val in mask:
        if val:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0

    return max_count


# ==============================================================================
# Backward Compatibility
# ==============================================================================
class PerformanceStats:
    """Legacy interface for backward compatibility.

    Use MetricsCalculator for new code.
    """

    def __init__(
        self,
        returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
    ) -> None:
        self._calc = MetricsCalculator(returns, risk_free_rate)
        self.returns = returns
        self.rf = risk_free_rate

    def calculate_all(self) -> dict[str, float]:
        """Return dictionary of metrics."""
        return {
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "max_drawdown": self.max_drawdown(),
            "win_rate": self.win_rate(),
            "total_return": self.total_return(),
            "volatility": self.volatility(),
        }

    def sharpe_ratio(self, periods_per_year: int = 252) -> float:
        calc = MetricsCalculator(self.returns, self.rf, periods_per_year)
        return calc.sharpe_ratio()

    def sortino_ratio(self, periods_per_year: int = 252) -> float:
        calc = MetricsCalculator(self.returns, self.rf, periods_per_year)
        return calc.sortino_ratio()

    def max_drawdown(self) -> float:
        return self._calc.max_drawdown()

    def win_rate(self) -> float:
        return self._calc.win_rate()

    def total_return(self) -> float:
        return self._calc.total_return()

    def volatility(self, periods_per_year: int = 252) -> float:
        calc = MetricsCalculator(self.returns, self.rf, periods_per_year)
        return calc.volatility()
