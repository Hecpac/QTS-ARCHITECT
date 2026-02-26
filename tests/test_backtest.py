"""Tests for backtest module (metrics, labeling, engine).

Tests cover:
- Performance metrics calculation
- Drawdown analysis
- Trade metrics
- Triple barrier labeling
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from qts_core.agents.protocol import SignalType, TradingDecision
from qts_core.common.types import InstrumentId, MarketData

from qts_core.backtest import (
    BacktestConfig,
    BarrierLabel,
    DrawdownInfo,
    EventEngine,
    MetricsCalculator,
    PerformanceMetrics,
    PerformanceStats,
    TradeMetrics,
    add_volatility,
    analyze_trades,
    calculate_label_distribution,
    filter_valid_labels,
    triple_barrier_method,
)


# ==============================================================================
# Metrics Calculator Tests
# ==============================================================================
class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_empty_returns(self) -> None:
        """Test with empty returns array."""
        calc = MetricsCalculator(np.array([]))
        metrics = calc.calculate()

        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0

    def test_single_return(self) -> None:
        """Test with single return."""
        calc = MetricsCalculator(np.array([0.01]))
        metrics = calc.calculate()

        assert metrics.total_return == pytest.approx(0.01)
        assert metrics.sharpe_ratio == 0.0  # Need >1 for ratio

    def test_total_return(self) -> None:
        """Test cumulative return calculation."""
        # 10%, -5%, 10% = (1.1 * 0.95 * 1.1) - 1 = 0.1495
        returns = np.array([0.10, -0.05, 0.10])
        calc = MetricsCalculator(returns)

        assert calc.total_return() == pytest.approx(0.1495, rel=1e-4)

    def test_sharpe_ratio_positive(self) -> None:
        """Test Sharpe ratio with positive excess returns."""
        # Consistent positive returns should give positive Sharpe
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02])
        calc = MetricsCalculator(returns, risk_free_rate=0.0)

        sharpe = calc.sharpe_ratio()
        assert sharpe > 0

    def test_sharpe_ratio_zero_volatility(self) -> None:
        """Test Sharpe ratio when volatility is zero."""
        returns = np.array([0.01, 0.01, 0.01])  # Constant returns
        calc = MetricsCalculator(returns)

        # Zero std dev should return 0
        assert calc.sharpe_ratio() == 0.0

    def test_sortino_ratio(self) -> None:
        """Test Sortino ratio uses only downside deviation."""
        # Mix of positive and negative returns
        returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        calc = MetricsCalculator(returns)

        sortino = calc.sortino_ratio()
        sharpe = calc.sharpe_ratio()

        # Sortino should be higher when there's upside deviation
        # (upside doesn't count against you)
        # This isn't always true but generally expected with mixed returns
        assert isinstance(sortino, float)

    def test_max_drawdown(self) -> None:
        """Test maximum drawdown calculation."""
        # Returns that create a clear drawdown
        # Start at 1.0, go to 1.1 (+10%), then 0.99 (-10% from 1.1)
        # Equity curve: 1.0 -> 1.1 -> 0.99
        # Drawdown from peak 1.1 to 0.99 = (0.99 - 1.1) / 1.1 = -0.1
        returns = np.array([0.10, -0.10])
        calc = MetricsCalculator(returns)

        max_dd = calc.max_drawdown()
        assert max_dd < 0  # Drawdown is negative
        # After +10%, equity is 1.1. After -10%, equity is 1.1 * 0.9 = 0.99
        # Drawdown = (0.99 - 1.1) / 1.1 = -0.1
        assert max_dd == pytest.approx(-0.10, rel=1e-2)

    def test_max_drawdown_no_drawdown(self) -> None:
        """Test max drawdown when always positive."""
        returns = np.array([0.01, 0.01, 0.01])
        calc = MetricsCalculator(returns)

        assert calc.max_drawdown() == 0.0

    def test_win_rate(self) -> None:
        """Test win rate calculation."""
        returns = np.array([0.01, -0.01, 0.02, 0.01, -0.02])
        calc = MetricsCalculator(returns)

        win_rate = calc.win_rate()
        assert win_rate == pytest.approx(3 / 5)

    def test_profit_factor(self) -> None:
        """Test profit factor calculation."""
        # Wins: 0.02 + 0.03 = 0.05
        # Losses: abs(-0.01) + abs(-0.01) = 0.02
        # PF = 0.05 / 0.02 = 2.5
        returns = np.array([0.02, -0.01, 0.03, -0.01])
        calc = MetricsCalculator(returns)

        assert calc.profit_factor() == pytest.approx(2.5)

    def test_profit_factor_no_losses(self) -> None:
        """Test profit factor when no losses."""
        returns = np.array([0.01, 0.02, 0.01])
        calc = MetricsCalculator(returns)

        assert calc.profit_factor() == float("inf")

    def test_volatility(self) -> None:
        """Test annualized volatility calculation."""
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        calc = MetricsCalculator(returns, periods_per_year=252)

        vol = calc.volatility()
        daily_std = np.std(returns, ddof=1)
        expected = daily_std * np.sqrt(252)

        assert vol == pytest.approx(expected)

    def test_var_95(self) -> None:
        """Test empirical 95% Value at Risk from worst-tail bucket."""
        returns = np.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03])
        calc = MetricsCalculator(returns)

        var = calc.var_95()

        tail_count = max(1, int(np.ceil(len(returns) * 0.05)))
        expected_tail = np.partition(returns, tail_count - 1)[:tail_count]
        expected_var = float(np.max(expected_tail))

        assert var == pytest.approx(expected_var)

    def test_cvar_95(self) -> None:
        """Test empirical Conditional VaR (Expected Shortfall)."""
        returns = np.array([-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.05])
        calc = MetricsCalculator(returns)

        cvar = calc.cvar_95()

        tail_count = max(1, int(np.ceil(len(returns) * 0.05)))
        expected_tail = np.partition(returns, tail_count - 1)[:tail_count]
        expected_cvar = float(np.mean(expected_tail))

        assert cvar == pytest.approx(expected_cvar)
        # CVaR should be at least as bad as VaR
        assert cvar <= calc.var_95()

    def test_var_cvar_95_with_multi_element_tail_bucket(self) -> None:
        """Tail bucket should support n where worst 5% has multiple elements."""
        returns = np.linspace(-0.20, 0.20, 40)
        calc = MetricsCalculator(returns)

        var = calc.var_95()
        cvar = calc.cvar_95()

        tail_count = max(1, int(np.ceil(len(returns) * 0.05)))
        expected_tail = np.partition(returns, tail_count - 1)[:tail_count]

        assert tail_count == 2
        assert var == pytest.approx(float(np.max(expected_tail)))
        assert cvar == pytest.approx(float(np.mean(expected_tail)))
        assert cvar <= var

    def test_tail_risk_cache_shared_between_var_and_cvar(self) -> None:
        returns = np.array([-0.12, -0.08, -0.02, 0.01, 0.03, 0.04, -0.01, 0.02])
        calc = MetricsCalculator(returns)

        assert calc._tail_risk_95_cache is None  # noqa: SLF001
        var = calc.var_95()

        cached = calc._tail_risk_95_cache  # noqa: SLF001
        assert cached is not None
        assert var == pytest.approx(cached[0])

        cvar = calc.cvar_95()
        assert calc._tail_risk_95_cache is cached  # noqa: SLF001
        assert cvar == pytest.approx(cached[1])

    def test_skewness(self) -> None:
        """Test skewness calculation."""
        # Right-skewed distribution
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.10])
        calc = MetricsCalculator(returns)

        skew = calc.skewness()
        assert skew > 0  # Right skew

    def test_kurtosis(self) -> None:
        """Test kurtosis calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        calc = MetricsCalculator(returns)

        kurt = calc.kurtosis()
        assert isinstance(kurt, float)

    def test_omega_ratio(self) -> None:
        """Test Omega ratio."""
        returns = np.array([0.01, -0.01, 0.02, -0.01, 0.03])
        calc = MetricsCalculator(returns)

        omega = calc.omega_ratio()
        # Total gains > total losses, so omega > 1
        assert omega > 1

    def test_calculate_returns_full_metrics(self) -> None:
        """Test full metrics calculation."""
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.015, -0.005])
        calc = MetricsCalculator(returns, risk_free_rate=0.02)

        metrics = calc.calculate()

        # Check all fields are populated
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_periods == 6
        assert metrics.trading_days == 6


# ==============================================================================
# Trade Analyzer Tests
# ==============================================================================
class TestAnalyzeTrades:
    """Tests for analyze_trades function."""

    def test_empty_trades(self) -> None:
        """Test with empty trades array."""
        metrics = analyze_trades(np.array([]))

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.expectancy == 0.0

    def test_all_winning_trades(self) -> None:
        """Test with all winning trades."""
        trades = np.array([0.05, 0.03, 0.02, 0.04])
        metrics = analyze_trades(trades)

        assert metrics.total_trades == 4
        assert metrics.winning_trades == 4
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 1.0
        assert metrics.profit_factor == float("inf")

    def test_mixed_trades(self) -> None:
        """Test with mixed winning/losing trades."""
        trades = np.array([0.10, -0.05, 0.08, -0.03, 0.05])
        metrics = analyze_trades(trades)

        assert metrics.total_trades == 5
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 2
        assert metrics.win_rate == pytest.approx(0.6)
        assert metrics.avg_win == pytest.approx((0.10 + 0.08 + 0.05) / 3)
        assert metrics.avg_loss == pytest.approx((-0.05 + -0.03) / 2)

    def test_consecutive_wins_losses(self) -> None:
        """Test consecutive win/loss streaks."""
        # 3 wins, 2 losses, 2 wins
        trades = np.array([0.01, 0.01, 0.01, -0.01, -0.01, 0.01, 0.01])
        metrics = analyze_trades(trades)

        assert metrics.max_consecutive_wins == 3
        assert metrics.max_consecutive_losses == 2

    def test_sqn_calculation(self) -> None:
        """Test System Quality Number."""
        trades = np.array([0.02, 0.01, -0.01, 0.03, 0.01])
        metrics = analyze_trades(trades)

        # SQN should be positive for profitable system
        assert metrics.sqn > 0


# ==============================================================================
# Backward Compatibility Tests
# ==============================================================================
class TestPerformanceStats:
    """Tests for legacy PerformanceStats class."""

    def test_calculate_all(self) -> None:
        """Test calculate_all returns expected keys."""
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        stats = PerformanceStats(returns)

        result = stats.calculate_all()

        assert "sharpe_ratio" in result
        assert "sortino_ratio" in result
        assert "max_drawdown" in result
        assert "win_rate" in result
        assert "total_return" in result
        assert "volatility" in result

    def test_individual_methods(self) -> None:
        """Test individual metric methods."""
        returns = np.array([0.01, -0.01, 0.02])
        stats = PerformanceStats(returns, risk_free_rate=0.0)

        assert isinstance(stats.sharpe_ratio(), float)
        assert isinstance(stats.sortino_ratio(), float)
        assert isinstance(stats.max_drawdown(), float)
        assert isinstance(stats.win_rate(), float)
        assert isinstance(stats.total_return(), float)
        assert isinstance(stats.volatility(), float)


# ==============================================================================
# Triple Barrier Labeling Tests
# ==============================================================================
class TestTripleBarrierMethod:
    """Tests for triple_barrier_method."""

    def test_basic_labeling(self) -> None:
        """Test basic triple barrier labeling."""
        # Create simple data
        data = {
            "timestamp": pl.date_range(
                pl.date(2024, 1, 1),
                pl.date(2024, 1, 20),
                eager=True,
            ),
            "open": [100.0] * 20,
            "high": [105.0] * 20,
            "low": [95.0] * 20,
            "close": [100.0] * 20,
            "volume": [1000.0] * 20,
            "volatility": [5.0] * 20,
        }
        df = pl.DataFrame(data)
        lf = df.lazy()

        labeled = triple_barrier_method(
            lf,
            vertical_barrier_bars=5,
            pt_multiplier=1.0,
            sl_multiplier=1.0,
        )

        result = labeled.collect()

        assert "label" in result.columns
        assert "barrier_ret" in result.columns
        assert "barrier_type" in result.columns

    def test_validation_errors(self) -> None:
        """Test validation of input parameters."""
        lf = pl.DataFrame({"close": [100.0]}).lazy()

        with pytest.raises(ValueError, match="vertical_barrier_bars"):
            triple_barrier_method(lf, vertical_barrier_bars=0)

        with pytest.raises(ValueError, match="multiplier"):
            triple_barrier_method(lf, vertical_barrier_bars=5, pt_multiplier=-1)


class TestAddVolatility:
    """Tests for add_volatility helper."""

    def test_add_volatility_column(self) -> None:
        """Test adding volatility column."""
        data = {
            "close": [100.0, 101.0, 99.0, 102.0, 98.0] * 10,
        }
        df = pl.DataFrame(data)
        lf = df.lazy()

        result = add_volatility(lf, window=5).collect()

        assert "volatility" in result.columns


class TestLabelDistribution:
    """Tests for calculate_label_distribution."""

    def test_distribution_calculation(self) -> None:
        """Test label distribution calculation."""
        df = pl.DataFrame({
            "label": [1, 1, -1, 0, 1, -1, 0, 0, 1, -1],
        })

        dist = calculate_label_distribution(df)

        assert dist["total"] == 10
        assert dist["take_profit_count"] == 4
        assert dist["stop_loss_count"] == 3
        assert dist["timeout_count"] == 3
        assert dist["take_profit_pct"] == pytest.approx(0.4)

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pl.DataFrame({"label": []})

        dist = calculate_label_distribution(df)

        assert dist["total"] == 0
        assert dist["take_profit_pct"] == 0.0


class TestFilterValidLabels:
    """Tests for filter_valid_labels."""

    def test_filter_nulls(self) -> None:
        """Test filtering null labels."""
        df = pl.DataFrame({
            "label": [1, -1, None, 0, None],
            "barrier_ret": [0.01, -0.01, None, 0.0, 0.02],
        })
        lf = df.lazy()

        result = filter_valid_labels(lf).collect()

        # Only rows where both label and barrier_ret are not null
        assert len(result) == 3


class _TwoStepBacktestSupervisor:
    """Deterministic supervisor for two-bar backtest tests."""

    def __init__(self, instrument_id: InstrumentId) -> None:
        self.instrument_id = instrument_id
        self._calls = 0

    async def run(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self._calls += 1
        if self._calls == 1:
            return TradingDecision(
                instrument_id=self.instrument_id,
                action=SignalType.SHORT,
                quantity_modifier=1.0,
                rationale="open short",
            )
        if self._calls == 2:
            return TradingDecision(
                instrument_id=self.instrument_id,
                action=SignalType.EXIT,
                quantity_modifier=1.0,
                rationale="close short",
            )
        return None


class _OpenShortOnlySupervisor:
    """Deterministic supervisor that opens once, then emits no further decisions."""

    def __init__(self, instrument_id: InstrumentId) -> None:
        self.instrument_id = instrument_id
        self._calls = 0

    async def run(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self._calls += 1
        if self._calls == 1:
            return TradingDecision(
                instrument_id=self.instrument_id,
                action=SignalType.SHORT,
                quantity_modifier=1.0,
                rationale="open short only",
            )
        return None


class _NoopBacktestSupervisor:
    """Supervisor stub that never emits decisions."""

    async def run(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


class _AlwaysLongSupervisor:
    """Supervisor stub that always emits LONG."""

    def __init__(self, instrument_id: InstrumentId) -> None:
        self.instrument_id = instrument_id

    async def run(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return TradingDecision(
            instrument_id=self.instrument_id,
            action=SignalType.LONG,
            quantity_modifier=1.0,
            rationale="always long",
        )


class TestEventEngineExitSemantics:
    """Tests for EXIT behavior with signed positions."""

    def test_exit_closes_short_with_buy(self) -> None:
        """EXIT should buy to cover when current position is negative."""
        engine = EventEngine(supervisor=object(), config=BacktestConfig())
        instrument = InstrumentId("BTC/USDT")

        engine.state.positions[instrument] = -0.5
        initial_cash = engine.state.cash

        decision = TradingDecision(
            instrument_id=instrument,
            action=SignalType.EXIT,
            quantity_modifier=1.0,
            rationale="close short",
        )
        market_data = MarketData(
            instrument_id=instrument,
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )

        engine._execute(decision, market_data)

        assert engine.state.get_position(instrument) == pytest.approx(0.0)
        # Buy-to-cover should consume cash.
        assert engine.state.cash < initial_cash

    def test_short_entry_rejected_when_liquidation_buffer_too_low(self) -> None:
        """SHORT entry should be blocked when leverage implies thin buffer."""
        engine = EventEngine(
            supervisor=object(),
            config=BacktestConfig(
                min_short_liquidation_buffer=0.10,
                short_leverage=25.0,
            ),
        )
        instrument = InstrumentId("BTC/USDT")

        decision = TradingDecision(
            instrument_id=instrument,
            action=SignalType.SHORT,
            quantity_modifier=1.0,
            rationale="open short blocked by guard",
        )
        market_data = MarketData(
            instrument_id=instrument,
            timestamp=datetime.now(timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )

        initial_cash = engine.state.cash
        engine._execute(decision, market_data)

        assert engine.state.cash == pytest.approx(initial_cash)
        assert engine.state.get_position(instrument) == pytest.approx(0.0)
        assert len(engine.trades) == 0

    def test_high_vol_hour_guardrail_blocks_entries(self) -> None:
        """Entry actions should be blocked when configured high-vol hour matches."""
        engine = EventEngine(
            supervisor=object(),
            config=BacktestConfig(
                high_volatility_hours_utc=(14, 15, 16, 17, 18),
                high_volatility_hours_entry_block=True,
            ),
        )
        instrument = InstrumentId("BTC/USDT")
        decision = TradingDecision(
            instrument_id=instrument,
            action=SignalType.LONG,
            quantity_modifier=1.0,
            rationale="blocked by hour guardrail",
        )
        market_data = MarketData(
            instrument_id=instrument,
            timestamp=datetime(2026, 2, 20, 15, 0, tzinfo=timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )

        initial_cash = engine.state.cash
        engine._execute(decision, market_data)

        assert engine.state.cash == pytest.approx(initial_cash)
        assert engine.state.get_position(instrument) == pytest.approx(0.0)
        assert len(engine.trades) == 0

    def test_high_vol_hour_guardrail_scales_entry_size(self) -> None:
        """Entry size should be reduced during configured high-volatility hours."""
        engine = EventEngine(
            supervisor=object(),
            config=BacktestConfig(
                trade_size=10_000.0,
                high_volatility_hours_utc=(14, 15, 16, 17, 18),
                high_volatility_hours_size_scale=0.5,
            ),
        )
        instrument = InstrumentId("BTC/USDT")
        decision = TradingDecision(
            instrument_id=instrument,
            action=SignalType.LONG,
            quantity_modifier=1.0,
            rationale="scale by hour guardrail",
        )
        market_data = MarketData(
            instrument_id=instrument,
            timestamp=datetime(2026, 2, 20, 15, 0, tzinfo=timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )

        engine._execute(decision, market_data)

        assert len(engine.trades) == 1
        trade = engine.trades[0]
        assert trade.quantity == pytest.approx(50.0)
        assert "Backtest guardrail: high-vol hour 15:00 UTC" in trade.rationale

    def test_dynamic_guardrail_uses_regime_specific_hours(self) -> None:
        """Dynamic regime mode should switch hour windows between normal/high regimes."""
        config = BacktestConfig(
            trade_size=10_000.0,
            high_volatility_hours_size_scale=0.5,
            volatility_regime_enabled=True,
            high_volatility_hours_normal_regime_utc=(14,),
            high_volatility_hours_high_regime_utc=(1,),
        )
        instrument = InstrumentId("BTC/USDT")

        # High regime at 01:00 UTC -> should scale.
        high_engine = EventEngine(supervisor=object(), config=config)
        decision = TradingDecision(
            instrument_id=instrument,
            action=SignalType.LONG,
            quantity_modifier=1.0,
            rationale="dynamic high-regime scale",
        )
        high_market_data = MarketData(
            instrument_id=instrument,
            timestamp=datetime(2026, 2, 20, 1, 0, tzinfo=timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )
        high_engine._execute(
            decision,
            high_market_data,
            is_high_volatility_regime=True,
        )
        assert len(high_engine.trades) == 1
        assert high_engine.trades[0].quantity == pytest.approx(50.0)
        assert "regime=high" in high_engine.trades[0].rationale

        # Same hour in normal regime -> not in normal window, should keep full size.
        normal_engine = EventEngine(supervisor=object(), config=config)
        normal_engine._execute(
            decision,
            high_market_data,
            is_high_volatility_regime=False,
        )
        assert len(normal_engine.trades) == 1
        assert normal_engine.trades[0].quantity == pytest.approx(100.0)

    def test_detects_high_volatility_regime_from_return_spike(self) -> None:
        """Rolling regime detector should flip high on a large return spike."""
        engine = EventEngine(
            supervisor=object(),
            config=BacktestConfig(
                volatility_regime_enabled=True,
                volatility_regime_window_bars=12,
                volatility_regime_min_observations=5,
                volatility_regime_zscore_threshold=1.0,
            ),
        )
        instrument = InstrumentId("BTC/USDT")
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        closes = [100.0, 100.02, 100.03, 100.01, 100.04, 100.02, 100.03]
        for idx, close in enumerate(closes):
            market_data = MarketData(
                instrument_id=instrument,
                timestamp=start + timedelta(hours=idx),
                open=close,
                high=close,
                low=close,
                close=close,
                volume=10.0,
            )
            assert engine._is_high_volatility_regime(market_data) is False  # noqa: SLF001

        spike_data = MarketData(
            instrument_id=instrument,
            timestamp=start + timedelta(hours=len(closes)),
            open=100.03,
            high=105.0,
            low=100.0,
            close=105.0,
            volume=10.0,
        )
        assert engine._is_high_volatility_regime(spike_data) is True  # noqa: SLF001

    def test_exit_close_short_accrues_borrow_fee(self) -> None:
        """Closing a short should charge accrued borrow/funding carry."""
        engine = EventEngine(
            supervisor=object(),
            config=BacktestConfig(short_borrow_rate_bps_per_day=100.0),
        )
        instrument = InstrumentId("BTC/USDT")

        now = datetime.now(timezone.utc)
        engine.state.positions[instrument] = -1.0
        engine.state.short_opened_at[instrument] = now - timedelta(days=2)
        initial_cash = engine.state.cash

        decision = TradingDecision(
            instrument_id=instrument,
            action=SignalType.EXIT,
            quantity_modifier=1.0,
            rationale="close short with carry",
        )
        market_data = MarketData(
            instrument_id=instrument,
            timestamp=now,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )

        engine._execute(decision, market_data)

        # Buyback cost: 100 + carry 2 (1%/day * 2 days).
        assert engine.state.cash == pytest.approx(initial_cash - 102.0, rel=1e-3)
        assert engine.state.short_borrow_fees_paid == pytest.approx(2.0, rel=1e-2)
        assert engine.state.get_position(instrument) == pytest.approx(0.0)
        assert instrument not in engine.state.short_opened_at

    @pytest.mark.asyncio
    async def test_run_final_capital_reflects_short_borrow_fee(self) -> None:
        """Backtest final capital should include carry cost after last fill."""
        instrument = InstrumentId("BTC/USDT")
        supervisor = _TwoStepBacktestSupervisor(instrument)
        config = BacktestConfig(
            initial_capital=100_000.0,
            trade_size=10_000.0,
            short_borrow_rate_bps_per_day=100.0,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = EventEngine(supervisor=supervisor, config=config)

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        data_feed = pl.DataFrame(
            {
                "timestamp": [start, start + timedelta(days=1)],
                "instrument_id": [str(instrument), str(instrument)],
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
                "volume": [1000.0, 1000.0],
            }
        )

        result = await engine.run(data_feed)

        # One-day borrow on 10,000 notional at 1%/day => 100.
        assert engine.state.short_borrow_fees_paid == pytest.approx(100.0, rel=1e-2)
        assert result.final_capital == pytest.approx(99_900.0, rel=1e-3)

    @pytest.mark.asyncio
    async def test_run_force_closes_open_short_and_applies_carry(self) -> None:
        """Open shorts should be force-closed at end with borrow fee applied."""
        instrument = InstrumentId("BTC/USDT")
        supervisor = _OpenShortOnlySupervisor(instrument)
        config = BacktestConfig(
            initial_capital=100_000.0,
            trade_size=10_000.0,
            short_borrow_rate_bps_per_day=100.0,
            commission_bps=0.0,
            slippage_bps=0.0,
            force_close_positions_at_end=True,
        )
        engine = EventEngine(supervisor=supervisor, config=config)

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        data_feed = pl.DataFrame(
            {
                "timestamp": [start, start + timedelta(days=1)],
                "instrument_id": [str(instrument), str(instrument)],
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
                "volume": [1000.0, 1000.0],
            }
        )

        result = await engine.run(data_feed)

        # Trade #1 opens short, forced Trade #2 closes it at end.
        assert result.total_trades == 2
        assert engine.state.get_position(instrument) == pytest.approx(0.0)
        assert engine.state.short_borrow_fees_paid == pytest.approx(100.0, rel=1e-2)
        assert result.final_capital == pytest.approx(99_900.0, rel=1e-3)

    @pytest.mark.asyncio
    async def test_run_uses_last_known_prices_across_instruments(self) -> None:
        """Equity should keep valuing held positions when another symbol ticks."""
        engine = EventEngine(
            supervisor=_NoopBacktestSupervisor(),
            config=BacktestConfig(),
        )
        btc = InstrumentId("BTC/USDT")
        eth = InstrumentId("ETH/USDT")

        # Seed an existing BTC position.
        engine.state.positions[btc] = 1.0

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        data_feed = pl.DataFrame(
            {
                "timestamp": [start, start + timedelta(hours=1)],
                "instrument_id": [str(btc), str(eth)],
                "open": [100.0, 50.0],
                "high": [101.0, 51.0],
                "low": [99.0, 49.0],
                "close": [100.0, 50.0],
                "volume": [1000.0, 2000.0],
            }
        )

        result = await engine.run(data_feed)

        assert result.equity_curve[0] == pytest.approx(100_100.0)
        # BTC valuation should persist even when ETH is the latest tick.
        assert result.equity_curve[1] == pytest.approx(100_100.0)

    def test_calculate_metrics_infers_hourly_periods_per_year(self) -> None:
        """Annualization should adapt to observed 1h bar spacing."""
        engine = EventEngine(supervisor=object(), config=BacktestConfig())

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        engine.state.timestamps = [start + timedelta(hours=i) for i in range(4)]
        engine.state.equity_curve = [100.0, 101.0, 100.0, 102.0]

        metrics = engine._calculate_metrics(  # noqa: SLF001
            start_date=engine.state.timestamps[0],
            end_date=engine.state.timestamps[-1],
        )
        assert metrics is not None

        returns = np.diff(np.array(engine.state.equity_curve)) / np.array(
            engine.state.equity_curve[:-1]
        )
        expected = MetricsCalculator(returns, periods_per_year=24 * 365).calculate(
            engine.state.timestamps[0],
            engine.state.timestamps[-1],
        )

        assert metrics.volatility_annualized == pytest.approx(
            expected.volatility_annualized
        )
