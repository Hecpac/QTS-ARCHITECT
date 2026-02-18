import pytest
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

from omegaconf import OmegaConf

# Ensure src/ is on the path for direct pytest runs
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from qts_core.agents.base import StrictRiskAgent, TechnicalAgent
from qts_core.agents.protocol import SignalType, TradingDecision
from qts_core.agents.supervisor import Supervisor
from qts_core.common.types import InstrumentId, MarketData
from qts_core.main_live import LiveTrader, apply_execution_guardrails
from qts_core.execution.ems import ExecutionGateway, FillReport
from qts_core.execution.oms import Order, OrderManagementSystem, OrderStatus, Portfolio
from qts_core.execution.store import MemoryStore


class StubGateway(ExecutionGateway):
    """
    Minimal EMS stub returning an immediate fill at a configured price.
    """

    def __init__(self, fill_price: float):
        self.fill_price = fill_price

    async def start(self):
        return None

    async def stop(self):
        return None

    async def submit_order(self, order):
        await asyncio.sleep(0)  # Yield control to mimic async behavior
        return FillReport(
            oms_order_id=order.oms_order_id,
            price=self.fill_price,
            quantity=order.quantity,
            fee=0.0,
        )


def _build_live_trader() -> LiveTrader:
    cfg = OmegaConf.create(
        {
            "env": "test",
            "symbol": "BTC/USDT",
            "oms": {
                "initial_cash": 100_000.0,
                "risk_fraction": 0.10,
                "account_mode": "spot",
                "short_leverage": 1.0,
            },
            "agents": {
                "strategies": [
                    {
                        "_target_": "qts_core.agents.base.TechnicalAgent",
                        "name": "TrendFollower_A",
                        "min_confidence": 0.0,
                    }
                ],
                "risk": {
                    "_target_": "qts_core.agents.base.StrictRiskAgent",
                    "name": "Risk",
                },
            },
            "store": {
                "_target_": "qts_core.execution.store.MemoryStore",
            },
            "gateway": {
                "_target_": "qts_core.execution.ems.MockGateway",
                "default_price": 100.0,
                "latency_ms": 1.0,
                "failure_rate": 0.0,
                "partial_fill_rate": 0.0,
            },
            "loop": {
                "tick_interval": 0.01,
                "heartbeat_key": "SYSTEM:HEARTBEAT",
                "execution_timeout": 0.1,
            },
            "execution_guardrails": {
                "enabled": False,
            },
            "telemetry": {
                "publish_views": False,
                "metrics_keys": {
                    "total_value": "METRICS:TOTAL_VALUE",
                    "cash": "METRICS:CASH",
                    "pnl_daily": "METRICS:PNL_DAILY",
                },
            },
        }
    )
    return LiveTrader(cfg)


def test_bullish_tick_creates_and_settles_order():
    """
    Full-path check: Supervisor emits LONG, OMS allocates, EMS fills, OMS settles.
    """
    async def run_flow():
        store = MemoryStore()
        oms = OrderManagementSystem(store)

        supervisor = Supervisor(
            strategy_agents=[TechnicalAgent("TrendFollower_A"), TechnicalAgent("MeanReversion_B")],
            risk_agent=StrictRiskAgent("Risk"),
        )

        price_open = 100.0
        price_close = 110.0  # Bullish candle triggers LONG signal
        instrument = InstrumentId("BTC/USDT")
        market_data = MarketData(
            instrument_id=instrument,
            timestamp=datetime.now(timezone.utc),
            open=price_open,
            high=price_close,
            low=price_open,
            close=price_close,
            volume=1.0,
        )

        decision = await supervisor.run(market_data)
        assert decision is not None
        assert decision.action == SignalType.LONG

        order_request = oms.process_decision(decision, current_price=market_data.close)
        assert order_request is not None

        gateway = StubGateway(fill_price=market_data.close)
        await gateway.start()
        fill_report = await gateway.submit_order(order_request)
        await gateway.stop()

        assert fill_report is not None

        oms.confirm_execution(
            oms_order_id=fill_report.oms_order_id,
            fill_price=fill_report.price,
            fill_qty=fill_report.quantity,
            fee=fill_report.fee,
        )

        order = store.load(f"{oms.ORDERS_KEY_PREFIX}{order_request.oms_order_id}", Order)
        assert order is not None
        assert order.status == OrderStatus.FILLED

        portfolio = store.load(oms.PORTFOLIO_KEY, Portfolio)
        assert portfolio is not None

        expected_cash = 100_000.0 - (order_request.quantity * market_data.close)
        assert portfolio.cash == pytest.approx(expected_cash, rel=1e-6)
        assert portfolio.positions[instrument] == pytest.approx(order_request.quantity, rel=1e-6)
        assert portfolio.blocked_cash == pytest.approx(0.0)
        assert portfolio.blocked_positions.get(instrument, 0.0) == pytest.approx(0.0)

    asyncio.run(run_flow())


def test_guardrails_reject_low_volume():
    instrument = InstrumentId("BTC/USDT")
    decision = TradingDecision(
        instrument_id=instrument,
        action=SignalType.LONG,
        quantity_modifier=1.0,
        rationale="test",
    )
    market_data = MarketData(
        instrument_id=instrument,
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=0.05,
    )

    guarded = apply_execution_guardrails(
        decision,
        market_data,
        enabled=True,
        min_volume=0.10,
    )

    assert guarded is None


def test_guardrails_reduce_size_in_high_volatility():
    instrument = InstrumentId("BTC/USDT")
    decision = TradingDecision(
        instrument_id=instrument,
        action=SignalType.LONG,
        quantity_modifier=0.8,
        rationale="test",
    )
    market_data = MarketData(
        instrument_id=instrument,
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=112.0,
        low=98.0,
        close=100.0,
        volume=2.0,
    )

    guarded = apply_execution_guardrails(
        decision,
        market_data,
        enabled=True,
        min_volume=0.1,
        max_intrabar_volatility=0.05,
        high_volatility_size_scale=0.5,
        max_estimated_slippage_bps=1_000.0,
    )

    assert guarded is not None
    assert guarded.quantity_modifier == pytest.approx(0.4)
    assert "Guardrail: high volatility" in guarded.rationale


def test_guardrails_reject_high_estimated_slippage():
    instrument = InstrumentId("BTC/USDT")
    decision = TradingDecision(
        instrument_id=instrument,
        action=SignalType.LONG,
        quantity_modifier=1.0,
        rationale="test",
    )
    market_data = MarketData(
        instrument_id=instrument,
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=110.0,
        low=90.0,
        close=100.0,
        volume=5.0,
    )

    guarded = apply_execution_guardrails(
        decision,
        market_data,
        enabled=True,
        min_volume=0.1,
        max_intrabar_volatility=1.0,
        high_volatility_size_scale=1.0,
        max_estimated_slippage_bps=20.0,
        slippage_volatility_factor=0.25,
    )

    assert guarded is None


@pytest.mark.asyncio
async def test_reconcile_helper_runs_after_ambiguous_submission() -> None:
    trader = _build_live_trader()

    # Simulate stale blocked cash with no open orders; reconcile should clear it.
    trader.oms.portfolio.blocked_cash = 500.0
    trader.store.save(trader.oms.PORTFOLIO_KEY, trader.oms.portfolio)

    await trader._reconcile_after_ambiguous_submission(
        reason="unit_test_timeout",
        order_id="unit-test-order",
    )

    refreshed = trader.store.load(trader.oms.PORTFOLIO_KEY, Portfolio)
    assert refreshed is not None
    assert refreshed.blocked_cash == pytest.approx(0.0)


def test_portfolio_exposure_includes_blocked_and_multi_instrument_positions() -> None:
    trader = _build_live_trader()

    btc = InstrumentId("BTC/USDT")
    eth = InstrumentId("ETH/USDT")

    # BTC long is moved to blocked inventory (pending CLOSE_LONG) and must still count.
    trader.oms.portfolio.positions[btc] = 0.0
    trader.oms.portfolio.blocked_positions[btc] = 1.0
    trader.oms.portfolio.positions[eth] = 2.0

    trader._instrument_marks[btc] = 100.0
    trader._instrument_marks[eth] = 50.0

    total_value = trader._compute_total_value(default_mark=100.0)
    exposure = trader._compute_portfolio_exposure_fraction(
        default_mark=100.0,
        total_value=total_value,
    )

    # Gross notional = 1*100 + 2*50 = 200
    assert exposure == pytest.approx(200.0 / total_value)


def test_session_drawdown_limit_detection() -> None:
    trader = _build_live_trader()
    trader.max_session_drawdown = 0.10

    # Establish peak.
    assert trader._compute_session_drawdown_fraction(100_000.0) == pytest.approx(0.0)

    # 15% drawdown should breach a 10% limit.
    drawdown = trader._compute_session_drawdown_fraction(85_000.0)
    assert drawdown == pytest.approx(0.15)
    assert trader._drawdown_limit_breached(85_000.0) is True


@pytest.mark.asyncio
async def test_halt_liquidation_attempts_to_close_open_positions() -> None:
    trader = _build_live_trader()

    instrument = InstrumentId("BTC/USDT")
    trader.oms.portfolio.positions[instrument] = 1.0
    trader.store.save(trader.oms.PORTFOLIO_KEY, trader.oms.portfolio)
    trader._last_market_data = MarketData(
        instrument_id=instrument,
        timestamp=datetime.now(timezone.utc),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=10.0,
    )

    await trader.ems.start()
    try:
        await trader._liquidate_open_positions()
    finally:
        await trader.ems.stop()

    refreshed = trader.store.load(trader.oms.PORTFOLIO_KEY, Portfolio)
    assert refreshed is not None
    assert refreshed.positions.get(instrument, 0.0) == pytest.approx(0.0)
    assert refreshed.blocked_positions.get(instrument, 0.0) == pytest.approx(0.0)
