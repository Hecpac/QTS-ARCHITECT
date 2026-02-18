
import pytest
import asyncio
from unittest.mock import MagicMock, patch
from qts_core.execution.ems import AlpacaGateway, RouterGateway, CircuitOpenError, GatewayNotStartedError
from qts_core.execution.oms import OrderRequest, OrderType, OrderSide
from qts_core.common.types import InstrumentId

def test_alpaca_gateway_init():
    gw = AlpacaGateway(api_key="test", secret="test")
    assert gw.api_key == "test"
    assert gw.paper_trading is True

@pytest.mark.asyncio
async def test_router_gateway_routing():
    mock_crypto = MagicMock()
    # async mock
    f_crypto = asyncio.Future()
    f_crypto.set_result("crypto_filled")
    mock_crypto.submit_order = MagicMock(return_value=f_crypto)

    mock_stocks = MagicMock()
    f_stocks = asyncio.Future()
    f_stocks.set_result("stock_filled")
    mock_stocks.submit_order = MagicMock(return_value=f_stocks)

    router = RouterGateway(
        gateways={"crypto": mock_crypto, "stocks": mock_stocks},
        routes={"BTC/": "crypto", "*": "stocks"},
        default_gateway="stocks"
    )

    # Test BTC route
    order_crypto = OrderRequest(
        instrument_id=InstrumentId("BTC/USDT"),
        quantity=1.0,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        oms_order_id="1",
        portfolio_id="main"
    )
    res = await router.submit_order(order_crypto)
    assert res == "crypto_filled"

    # Test Stock route (fallback)
    order_stock = OrderRequest(
        instrument_id=InstrumentId("AAPL"),
        quantity=1.0,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        oms_order_id="2",
        portfolio_id="main"
    )
    res = await router.submit_order(order_stock)
    assert res == "stock_filled"
