import pytest
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure src/ is on the path for direct pytest runs
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from qts_core.agents.base import StrictRiskAgent, TechnicalAgent
from qts_core.agents.protocol import SignalType
from qts_core.agents.supervisor import Supervisor
from qts_core.common.types import InstrumentId, MarketData
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
