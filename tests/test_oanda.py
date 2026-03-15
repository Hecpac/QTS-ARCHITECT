"""Tests for OANDAGateway."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qts_core.execution.ems import (
    CircuitOpenError,
    ExecutionStatus,
    GatewayNotStartedError,
    OANDAGateway,
)
from qts_core.execution.oms import InstrumentId, OrderRequest, OrderSide, OrderType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(symbol: str = "EUR/USD", side: OrderSide = OrderSide.BUY) -> OrderRequest:
    return OrderRequest(
        oms_order_id="test-001",
        instrument_id=InstrumentId(symbol),
        side=side,
        quantity=1000.0,
        order_type=OrderType.MARKET,
    )


def _mock_httpx_client(*, account_summary=None, pricing=None, candles=None, order=None):
    """Return a mock httpx.AsyncClient with pre-configured responses."""
    client = AsyncMock()

    async def _get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "summary" in url:
            resp.json.return_value = account_summary or {"account": {"id": "001-001-test"}}
        elif "pricing" in url:
            resp.json.return_value = pricing or {
                "prices": [{"bids": [{"price": "1.0850"}], "asks": [{"price": "1.0852"}]}]
            }
        elif "candles" in url:
            resp.json.return_value = candles or {
                "candles": [
                    {
                        "time": "2026-03-15T10:00:00.000000000Z",
                        "mid": {"o": "1.0840", "h": "1.0860", "l": "1.0830", "c": "1.0851"},
                        "volume": 1500,
                        "complete": True,
                    }
                ]
            }
        return resp

    async def _post(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = order or {
            "orderFillTransaction": {
                "id": "txn-001",
                "orderID": "ord-001",
                "price": "1.0851",
                "units": "1000",
                "commission": {"units": "-0.50"},
            }
        }
        return resp

    client.get = _get
    client.post = _post
    client.aclose = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------

def test_oanda_gateway_defaults():
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    assert gw.paper_trading is True
    assert gw._base_url == OANDAGateway.PRACTICE_URL
    assert not gw._started


def test_oanda_gateway_live_url():
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234", paper_trading=False)
    assert gw._base_url == OANDAGateway.LIVE_URL


def test_oanda_gateway_requires_token():
    gw = OANDAGateway(account_id="001-001-1234")
    with pytest.raises(ValueError, match="API token"):
        asyncio.run(gw.start())


def test_oanda_gateway_requires_account_id():
    gw = OANDAGateway(api_token="tok")
    with pytest.raises(ValueError, match="account ID"):
        asyncio.run(gw.start())


# ---------------------------------------------------------------------------
# Symbol conversion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol,expected", [
    ("EUR/USD", "EUR_USD"),
    ("GBP/USD", "GBP_USD"),
    ("XAU/USD", "XAU_USD"),
    ("BTC/USDT", "BTC_USDT"),
])
def test_to_oanda_symbol(symbol, expected):
    assert OANDAGateway._to_oanda_symbol(symbol) == expected


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_sets_started():
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    mock_client = _mock_httpx_client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await gw.start()

    assert gw._started
    assert gw.health_check()


@pytest.mark.asyncio
async def test_start_idempotent():
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    mock_client = _mock_httpx_client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await gw.start()
        await gw.start()  # should not raise

    assert gw._started


@pytest.mark.asyncio
async def test_stop_clears_state():
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    mock_client = _mock_httpx_client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await gw.start()

    await gw.stop()
    assert not gw._started
    assert not gw.health_check()


# ---------------------------------------------------------------------------
# fetch_ticker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_ticker_returns_mid_price():
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    mock_client = _mock_httpx_client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await gw.start()

    ticker = await gw.fetch_ticker("EUR/USD")
    assert ticker["bid"] == pytest.approx(1.0850)
    assert ticker["ask"] == pytest.approx(1.0852)
    assert ticker["last"] == pytest.approx(1.0851)
    assert "timestamp" in ticker


@pytest.mark.asyncio
async def test_fetch_ticker_not_started_raises():
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    with pytest.raises(GatewayNotStartedError):
        await gw.fetch_ticker("EUR/USD")


# ---------------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_ohlcv_returns_tuples():
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    mock_client = _mock_httpx_client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await gw.start()

    candles = await gw.fetch_ohlcv("EUR/USD", timeframe="1h", limit=1)
    assert len(candles) == 1
    ts, o, h, l, c, v = candles[0]
    assert o == pytest.approx(1.0840)
    assert h == pytest.approx(1.0860)
    assert l == pytest.approx(1.0830)
    assert c == pytest.approx(1.0851)
    assert v == 1500


@pytest.mark.asyncio
async def test_fetch_ohlcv_skips_incomplete_candles():
    incomplete_candles = {
        "candles": [
            {
                "time": "2026-03-15T10:00:00.000000000Z",
                "mid": {"o": "1.08", "h": "1.09", "l": "1.07", "c": "1.085"},
                "volume": 100,
                "complete": False,  # <-- should be skipped
            }
        ]
    }
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    mock_client = _mock_httpx_client(candles=incomplete_candles)

    with patch("httpx.AsyncClient", return_value=mock_client):
        await gw.start()

    result = await gw.fetch_ohlcv("EUR/USD")
    assert result == []


@pytest.mark.asyncio
async def test_fetch_ohlcv_granularity_mapping():
    """Verify all supported timeframes map to valid OANDA granularities."""
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    mock_client = _mock_httpx_client(candles={"candles": []})

    with patch("httpx.AsyncClient", return_value=mock_client):
        await gw.start()

    for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
        result = await gw.fetch_ohlcv("EUR/USD", timeframe=tf)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# submit_order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_order_buy_returns_fill():
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    mock_client = _mock_httpx_client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await gw.start()

    order = _make_order("EUR/USD", OrderSide.BUY)
    fill = await gw.submit_order(order)

    assert fill is not None
    assert fill.oms_order_id == "test-001"
    assert fill.price == pytest.approx(1.0851)
    assert fill.quantity == pytest.approx(1000.0)
    assert fill.status == ExecutionStatus.SUCCESS


@pytest.mark.asyncio
async def test_submit_order_sell_sends_negative_units():
    """Sell orders must send negative units to OANDA."""
    captured_payload = {}

    async def _post(url, json=None, **kwargs):
        captured_payload.update(json or {})
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "orderFillTransaction": {
                "id": "txn-002", "orderID": "ord-002",
                "price": "1.0851", "units": "-1000",
                "commission": {},
            }
        }
        return resp

    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    mock_client = _mock_httpx_client()
    mock_client.post = _post

    with patch("httpx.AsyncClient", return_value=mock_client):
        await gw.start()

    order = _make_order("EUR/USD", OrderSide.SELL)
    await gw.submit_order(order)

    units = float(captured_payload["order"]["units"])
    assert units < 0


@pytest.mark.asyncio
async def test_submit_order_not_started_raises():
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    with pytest.raises(GatewayNotStartedError):
        await gw.submit_order(_make_order())


@pytest.mark.asyncio
async def test_submit_order_circuit_open_raises():
    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    mock_client = _mock_httpx_client()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await gw.start()

    # Force circuit open
    for _ in range(gw._circuit_breaker.failure_threshold + 1):
        gw._circuit_breaker.record_failure()

    with pytest.raises(CircuitOpenError):
        await gw.submit_order(_make_order())


@pytest.mark.asyncio
async def test_submit_order_http_error_returns_none():
    async def _bad_post(url, **kwargs):
        raise Exception("connection refused")

    gw = OANDAGateway(api_token="tok", account_id="001-001-1234")
    mock_client = _mock_httpx_client()
    mock_client.post = _bad_post

    with patch("httpx.AsyncClient", return_value=mock_client):
        await gw.start()

    result = await gw.submit_order(_make_order())
    assert result is None
