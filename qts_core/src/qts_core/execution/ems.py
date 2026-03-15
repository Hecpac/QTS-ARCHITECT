"""Execution Management System (EMS) for QTS-Architect.

The EMS handles order routing and execution against exchanges.
It provides a protocol-based abstraction over different execution venues.

Responsibilities:
- Order submission to exchanges
- Retry logic with exponential backoff
- Circuit breaker for exchange failures
- Rate limiting per exchange
- Fill report generation

Design Decisions:
- Protocol-based interface for flexibility
- Circuit breaker prevents cascading failures
- Configurable rate limits per exchange
- Async-first for high throughput
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field, PositiveFloat
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from qts_core.execution.oms import AccountMode, OrderRequest

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5
DEFAULT_CIRCUIT_BREAKER_TIMEOUT: Final[float] = 60.0  # seconds
DEFAULT_RATE_LIMIT_PER_SECOND: Final[float] = 10.0
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_MIN_WAIT: Final[float] = 1.0
DEFAULT_RETRY_MAX_WAIT: Final[float] = 10.0


# ==============================================================================
# Enums
# ==============================================================================
class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class ExecutionStatus(str, Enum):
    """Execution result status."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"


# ==============================================================================
# Custom Exceptions
# ==============================================================================
class EMSError(Exception):
    """Base exception for EMS errors."""


class GatewayNotStartedError(EMSError):
    """Raised when gateway used before start()."""


class CircuitOpenError(EMSError):
    """Raised when circuit breaker is open."""

    def __init__(self, reset_time: float) -> None:
        self.reset_time = reset_time
        super().__init__(f"Circuit breaker open, resets in {reset_time:.1f}s")


class RateLimitError(EMSError):
    """Raised when rate limit exceeded."""


class ExecutionError(EMSError):
    """Raised when order execution fails."""

    def __init__(self, reason: str, recoverable: bool = False) -> None:
        self.reason = reason
        self.recoverable = recoverable
        super().__init__(reason)


# ==============================================================================
# Domain Models
# ==============================================================================
class FillReport(BaseModel):
    """Execution fill report from exchange.

    Immutable record of an order fill.
    """

    model_config = {"frozen": True}

    oms_order_id: str
    exchange_order_id: str | None = None
    exchange_trade_id: str | None = None
    price: PositiveFloat
    quantity: PositiveFloat
    fee: float = Field(default=0.0, ge=0)
    fee_currency: str | None = None
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_response: dict | None = Field(default=None, exclude=True)


class ExecutionResult(BaseModel):
    """Result of an execution attempt.

    Contains either a fill report or error information.
    """

    model_config = {"frozen": True}

    success: bool
    fill: FillReport | None = None
    error_message: str | None = None
    status: ExecutionStatus = ExecutionStatus.SUCCESS


# ==============================================================================
# Circuit Breaker
# ==============================================================================
@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for exchange failures.

    Prevents cascading failures by temporarily blocking requests
    after repeated failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, all requests rejected immediately
    - HALF_OPEN: Testing recovery with single request

    Attributes:
        failure_threshold: Failures before opening circuit.
        reset_timeout: Seconds before attempting recovery.
    """

    failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD
    reset_timeout: float = DEFAULT_CIRCUIT_BREAKER_TIMEOUT
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    success_count: int = field(default=0)
    half_open_in_flight: bool = field(default=False)

    def record_success(self) -> None:
        """Record successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_in_flight = False
            self.success_count += 1
            # Require 2 sequential probe successes to fully close.
            if self.success_count >= 2:
                self._close()
        else:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_in_flight = False
            self._open()
        elif self.failure_count >= self.failure_threshold:
            self._open()

    def can_execute(self) -> bool:
        """Check if request can proceed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._half_open()
            else:
                return False

        # HALF_OPEN: allow only one in-flight probe request.
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_in_flight:
                return False
            self.half_open_in_flight = True
            return True

        return False

    def time_until_reset(self) -> float:
        """Seconds until circuit might reset."""
        if self.state != CircuitState.OPEN:
            return 0.0
        elapsed = time.monotonic() - self.last_failure_time
        return max(0.0, self.reset_timeout - elapsed)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time passed to try recovery."""
        return (time.monotonic() - self.last_failure_time) >= self.reset_timeout

    def _open(self) -> None:
        """Open the circuit."""
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.half_open_in_flight = False
        log.warning(
            "Circuit breaker OPEN",
            failures=self.failure_count,
            reset_in=self.reset_timeout,
        )

    def _half_open(self) -> None:
        """Transition to half-open for recovery test."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.half_open_in_flight = False
        log.info("Circuit breaker HALF_OPEN, testing recovery")

    def _close(self) -> None:
        """Close the circuit (normal operation)."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_in_flight = False
        log.info("Circuit breaker CLOSED, recovered")


# ==============================================================================
# Rate Limiter
# ==============================================================================
@dataclass
class RateLimiter:
    """Token bucket rate limiter.

    Limits requests per second to prevent exchange rate limit violations.

    Attributes:
        rate: Requests per second allowed.
        tokens: Current token count.
        last_refill: Last token refill timestamp.
    """

    rate: float = DEFAULT_RATE_LIMIT_PER_SECOND
    tokens: float = field(default=0.0)
    last_refill: float = field(default=0.0)
    max_tokens: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.tokens = self.rate
        self.max_tokens = self.rate * 2  # Allow burst
        self.last_refill = time.monotonic()

    async def acquire(self, timeout: float = 5.0) -> bool:
        """Acquire a token, waiting if necessary.

        Args:
            timeout: Max seconds to wait for token.

        Returns:
            True if token acquired, False on timeout.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            # Wait for refill
            await asyncio.sleep(0.1)

        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + (elapsed * self.rate))
        self.last_refill = now


# ==============================================================================
# Execution Gateway Protocol
# ==============================================================================
@runtime_checkable
class ExecutionGateway(Protocol):
    """Protocol for execution gateways.

    Implementations provide connectivity to trading venues.
    """

    async def start(self) -> None:
        """Initialize gateway connection."""
        ...

    async def stop(self) -> None:
        """Shutdown gateway connection."""
        ...

    async def submit_order(self, order: OrderRequest) -> FillReport | None:
        """Submit order for execution.

        Args:
            order: Order request from OMS.

        Returns:
            FillReport if executed, None on failure.
        """
        ...

    def health_check(self) -> bool:
        """Check gateway health."""
        ...


# ==============================================================================
# CCXT Gateway Implementation
# ==============================================================================
class CCXTGateway:
    """CCXT-based execution gateway.

    Provides connectivity to 100+ exchanges via CCXT library.
    Includes circuit breaker, rate limiting, and retry logic.

    Attributes:
        exchange_id: CCXT exchange identifier (e.g., 'binance').
        sandbox: Use exchange sandbox/testnet.
        paper_trading: Simulate fills without real execution.
    """

    def __init__(
        self,
        exchange_id: str | None = None,
        exchange_name: str | None = None,
        api_key: str = "",
        apiKey: str = "",  # noqa: N803 - config compatibility
        secret: str = "",
        password: str = "",
        sandbox: bool = True,
        paper_trading: bool = False,
        account_mode: AccountMode | str = AccountMode.SPOT,
        rate_limit: float = DEFAULT_RATE_LIMIT_PER_SECOND,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        circuit_breaker_timeout: float = DEFAULT_CIRCUIT_BREAKER_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Initialize CCXT gateway.

        Args:
            exchange_id: CCXT exchange ID.
            exchange_name: Alias for exchange_id.
            api_key: API key.
            apiKey: Alias for api_key (config compatibility).
            secret: API secret.
            password: API password (some exchanges require this).
            sandbox: Use sandbox/testnet mode.
            paper_trading: Simulate execution without real orders.
            account_mode: Account mode (spot/margin/perp) for order params.
            rate_limit: Requests per second.
            circuit_breaker_threshold: Failures before opening circuit.
            circuit_breaker_timeout: Seconds before retry after circuit opens.
            max_retries: Maximum retry attempts per order.
        """
        # Resolve exchange ID
        self.exchange_id = exchange_id or exchange_name
        if not self.exchange_id:
            msg = "exchange_id or exchange_name required"
            raise ValueError(msg)

        # Credentials
        self.api_key = api_key or apiKey
        self.secret = secret
        self.password = password
        self.sandbox = sandbox
        self.paper_trading = paper_trading
        self.account_mode = (
            account_mode if isinstance(account_mode, AccountMode)
            else AccountMode(str(account_mode).lower())
        )

        # Configuration
        self.max_retries = max_retries

        # State
        self.exchange = None
        self._started = False

        # Resilience components
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_timeout,
        )
        self._rate_limiter = RateLimiter(rate=rate_limit)

    async def start(self) -> None:
        """Initialize exchange connection."""
        import ccxt.async_support as ccxt

        if self._started:
            return

        exchange_class = getattr(ccxt, self.exchange_id, None)
        if exchange_class is None:
            msg = f"Unknown exchange: {self.exchange_id}"
            raise ValueError(msg)

        config = {
            "apiKey": self.api_key,
            "secret": self.secret,
            "enableRateLimit": True,
        }
        if self.password:
            config["password"] = self.password

        self.exchange = exchange_class(config)

        if self.sandbox:
            self.exchange.set_sandbox_mode(True)

        try:
            await self.exchange.load_markets()
        except Exception as exc:
            if not self.paper_trading:
                raise
            log.warning(
                "Could not load markets in paper mode",
                exchange=self.exchange_id,
                error=str(exc),
            )

        self._started = True
        log.info(
            "CCXT gateway started",
            exchange=self.exchange_id,
            sandbox=self.sandbox,
            paper_trading=self.paper_trading,
        )

    async def stop(self) -> None:
        """Shutdown exchange connection."""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
        self._started = False
        log.info("CCXT gateway stopped", exchange=self.exchange_id)

    def health_check(self) -> bool:
        """Check if gateway is healthy."""
        if not self._started or self.exchange is None:
            return False
        return self._circuit_breaker.state != CircuitState.OPEN

    async def submit_order(self, order: OrderRequest) -> FillReport | None:
        """Submit order for execution.

        Args:
            order: Order request from OMS.

        Returns:
            FillReport on success, None on failure.

        Raises:
            GatewayNotStartedError: If start() not called.
            CircuitOpenError: If circuit breaker is open.
        """
        if not self._started or self.exchange is None:
            raise GatewayNotStartedError("Call start() before submit_order()")

        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            reset_time = self._circuit_breaker.time_until_reset()
            log.warning(
                "Order rejected: circuit breaker open",
                order_id=order.oms_order_id,
                reset_in=reset_time,
            )
            raise CircuitOpenError(reset_time)

        # Acquire rate limit token
        if not await self._rate_limiter.acquire():
            log.warning(
                "Order rejected: rate limit exceeded",
                order_id=order.oms_order_id,
            )
            raise ExecutionError("Rate limit exceeded", recoverable=False)

        # Paper trading: simulate fill
        if self.paper_trading:
            return await self._simulate_fill(order)

        # Real execution
        try:
            fill = await self._execute_with_retry(order)
        except ExecutionError:
            self._circuit_breaker.record_failure()
            raise
        except Exception as e:
            self._circuit_breaker.record_failure()
            log.error(
                "Order execution failed with ambiguous outcome",
                order_id=order.oms_order_id,
                error=str(e),
            )
            raise ExecutionError(str(e), recoverable=True) from e

        if fill is None:
            self._circuit_breaker.record_failure()
            log.warning(
                "Order execution returned no fill report",
                order_id=order.oms_order_id,
            )
            return None

        self._circuit_breaker.record_success()
        return fill

    async def _simulate_fill(self, order: OrderRequest) -> FillReport:
        """Simulate fill for paper trading."""
        try:
            ticker = await self.exchange.fetch_ticker(str(order.instrument_id))
            price = ticker.get("last") or ticker.get("close") or 0.0
        except Exception as exc:
            log.warning(
                "Paper trading: fetch_ticker failed",
                error=str(exc),
            )
            price = 0.0

        log.info(
            "Paper trade executed",
            order_id=order.oms_order_id,
            price=price,
            quantity=order.quantity,
        )

        return FillReport(
            oms_order_id=order.oms_order_id,
            exchange_trade_id=f"paper:{order.oms_order_id}",
            price=max(price, 0.01),  # Ensure positive
            quantity=order.quantity,
            fee=0.0,
            status=ExecutionStatus.SUCCESS,
        )

    def _build_create_order_payload(
        self,
        order: OrderRequest,
    ) -> tuple[str, str, str, float, float | None, dict[str, bool]]:
        """Build normalized create_order payload for CCXT."""
        symbol = str(order.instrument_id)
        order_type = order.order_type.value.lower()
        side = order.side.value.lower()
        amount = float(order.quantity)
        price = float(order.limit_price) if order.limit_price is not None else None

        params: dict[str, bool] = {}
        if order.reduce_only and self.account_mode in {AccountMode.MARGIN, AccountMode.PERP}:
            params["reduceOnly"] = True

        return symbol, order_type, side, amount, price, params

    @staticmethod
    def _extract_exchange_trade_id(response: dict) -> str | None:
        """Extract best-effort unique trade ID from exchange response."""
        candidates = [
            response.get("trade_id"),
            response.get("tradeId"),
            response.get("lastTradeId"),
            response.get("last_trade_id"),
            response.get("fill_id"),
        ]

        info = response.get("info")
        if isinstance(info, dict):
            candidates.extend(
                [
                    info.get("trade_id"),
                    info.get("tradeId"),
                    info.get("lastTradeId"),
                    info.get("last_trade_id"),
                    info.get("fill_id"),
                ]
            )

        for value in candidates:
            if value not in (None, ""):
                return str(value)

        trades = response.get("trades")
        if isinstance(trades, list):
            for trade in reversed(trades):
                if not isinstance(trade, dict):
                    continue
                trade_id = trade.get("id") or trade.get("trade_id") or trade.get("tradeId")
                if trade_id not in (None, ""):
                    return str(trade_id)

        return None

    async def _execute_with_retry(self, order: OrderRequest) -> FillReport | None:
        """Execute order with retry logic."""
        import ccxt.async_support as ccxt

        symbol, order_type, side, amount, price, params = self._build_create_order_payload(order)

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=DEFAULT_RETRY_MIN_WAIT,
                max=DEFAULT_RETRY_MAX_WAIT,
            ),
            retry=retry_if_exception_type(
                (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.ExchangeNotAvailable)
            ),
            before_sleep=lambda rs: log.warning(
                "Retrying order submission",
                attempt=rs.attempt_number,
                order_id=order.oms_order_id,
            ),
        )
        async def _submit() -> dict:
            return await self.exchange.create_order(
                symbol,
                order_type,
                side,
                amount,
                price,
                params,
            )

        log.info(
            "Submitting order to exchange",
            symbol=symbol,
            side=side,
            type=order_type,
            amount=amount,
            order_id=order.oms_order_id,
            reduce_only=bool(params.get("reduceOnly", False)),
        )

        try:
            response = await _submit()
        except RetryError as e:
            log.error(
                "Order submission exhausted retries",
                order_id=order.oms_order_id,
                error=str(e),
            )
            return None
        except ccxt.InvalidOrder as exc:
            raise ExecutionError(str(exc), recoverable=False) from exc
        except (
            ccxt.AuthenticationError,
            ccxt.PermissionDenied,
            ccxt.BadRequest,
        ) as exc:
            raise ExecutionError(str(exc), recoverable=False) from exc
        except (
            ccxt.NetworkError,
            ccxt.RateLimitExceeded,
            ccxt.ExchangeNotAvailable,
            ccxt.RequestTimeout,
        ) as exc:
            raise ExecutionError(str(exc), recoverable=True) from exc
        except ccxt.BaseError as exc:
            raise ExecutionError(str(exc), recoverable=False) from exc

        return await self._parse_response(order, response, symbol)

    async def _parse_response(
        self,
        order: OrderRequest,
        response: dict,
        symbol: str,
    ) -> FillReport | None:
        """Parse exchange response into FillReport."""
        exchange_order_id = response.get("id")
        response_status = str(response.get("status") or "").lower()

        if response_status in {"rejected", "canceled", "cancelled", "expired"}:
            raise ExecutionError(
                f"Exchange rejected order with status={response_status}",
                recoverable=False,
            )

        fill_price = response.get("price") or response.get("average")
        filled = response.get("filled")

        # Fetch order if needed
        if fill_price is None or filled is None:
            order_id = exchange_order_id
            if order_id:
                try:
                    fetched = await self.exchange.fetch_order(order_id, symbol)
                    if isinstance(fetched, dict):
                        response = {**response, **fetched}
                        exchange_order_id = response.get("id") or exchange_order_id
                        response_status = str(response.get("status") or response_status).lower()
                        fill_price = fill_price or response.get("average") or response.get("price")
                        filled = filled if filled is not None else response.get("filled")
                except Exception as e:
                    log.warning("Could not fetch order details", error=str(e))

        if response_status in {"rejected", "canceled", "cancelled", "expired"}:
            raise ExecutionError(
                f"Exchange rejected order with status={response_status}",
                recoverable=False,
            )

        if filled is None:
            log.warning(
                "Exchange did not return filled quantity; treating as ambiguous",
                order_id=order.oms_order_id,
                response=response,
            )
            return None

        try:
            parsed_fill_qty = float(filled)
        except (TypeError, ValueError):
            log.warning(
                "Exchange returned non-numeric filled quantity",
                filled=filled,
                order_id=order.oms_order_id,
            )
            return None

        if parsed_fill_qty <= 0:
            log.info(
                "Order acknowledged with no fills yet",
                order_id=order.oms_order_id,
                exchange_order_id=exchange_order_id,
                status=response_status or "unknown",
            )
            return None

        if fill_price is None:
            log.warning(
                "Exchange did not return fill price; treating as ambiguous",
                order_id=order.oms_order_id,
                response_keys=sorted(response.keys()),
            )
            return None

        try:
            parsed_fill_price = float(fill_price)
        except (TypeError, ValueError):
            log.warning(
                "Exchange returned non-numeric fill price",
                fill_price=fill_price,
                order_id=order.oms_order_id,
            )
            return None

        if parsed_fill_price <= 0:
            log.warning(
                "Exchange returned non-positive fill price",
                fill_price=parsed_fill_price,
                order_id=order.oms_order_id,
            )
            return None

        # Extract fee
        fee_cost = 0.0
        fee_currency = None
        if "fee" in response and response["fee"]:
            fee_cost = response["fee"].get("cost", 0.0) or 0.0
            fee_currency = response["fee"].get("currency")

        trade_id = self._extract_exchange_trade_id(response)
        if trade_id is None and exchange_order_id is not None:
            ts_component = (
                response.get("lastTradeTimestamp")
                or response.get("last_trade_timestamp")
                or response.get("timestamp")
                or int(datetime.now(timezone.utc).timestamp() * 1000)
            )
            trade_id = f"{exchange_order_id}:{ts_component}:{parsed_fill_qty:.12f}"

        return FillReport(
            oms_order_id=order.oms_order_id,
            exchange_order_id=str(exchange_order_id) if exchange_order_id is not None else None,
            exchange_trade_id=trade_id,
            price=parsed_fill_price,
            quantity=parsed_fill_qty,
            fee=fee_cost,
            fee_currency=fee_currency,
            status=(
                ExecutionStatus.SUCCESS
                if parsed_fill_qty >= order.quantity
                else ExecutionStatus.PARTIAL
            ),
            raw_response=response,
        )


# ==============================================================================
# Alpaca Gateway Implementation
# ==============================================================================
class AlpacaExchangeProxy:
    """CCXT-compatible data interface for Alpaca."""

    def __init__(self, api_key: str, secret: str, paper: bool) -> None:
        from alpaca.data.historical import (
            CryptoHistoricalDataClient,
            StockHistoricalDataClient,
        )

        self.stock_client = StockHistoricalDataClient(api_key, secret)
        self.crypto_client = CryptoHistoricalDataClient(api_key, secret)
        self.paper = paper

    async def fetch_ticker(self, symbol: str) -> dict:
        from alpaca.data.requests import (
            CryptoLatestQuoteRequest,
            StockLatestQuoteRequest,
        )

        if "/" in symbol:
            req = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = await asyncio.to_thread(
                self.crypto_client.get_crypto_latest_quote, req
            )
            if symbol in quotes:
                q = quotes[symbol]
                return {
                    "last": q.ask_price,
                    "bid": q.bid_price,
                    "ask": q.ask_price,
                    "timestamp": q.timestamp.timestamp() * 1000,
                }
        else:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = await asyncio.to_thread(
                self.stock_client.get_stock_latest_quote, req
            )
            if symbol in quotes:
                q = quotes[symbol]
                return {
                    "last": q.ask_price,
                    "bid": q.bid_price,
                    "ask": q.ask_price,
                    "timestamp": q.timestamp.timestamp() * 1000,
                }
        return {}

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 50
    ) -> list:
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        tf = TimeFrame.Hour
        if timeframe == "1m":
            tf = TimeFrame.Minute
        elif timeframe == "1d":
            tf = TimeFrame.Day

        if "/" in symbol:
            req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
            bars = await asyncio.to_thread(self.crypto_client.get_crypto_bars, req)
        else:
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, limit=limit)
            bars = await asyncio.to_thread(self.stock_client.get_stock_bars, req)

        data = []
        if symbol in bars:
            for b in bars[symbol]:
                data.append(
                    [
                        b.timestamp.timestamp() * 1000,
                        b.open,
                        b.high,
                        b.low,
                        b.close,
                        b.volume,
                    ]
                )
        return data


class AlpacaGateway:
    """Alpaca Trading API gateway.

    Provides connectivity to Alpaca for stocks and crypto trading.
    Uses `alpaca-py` SDK.

    Attributes:
        api_key: Alpaca API key ID.
        secret: Alpaca API secret key.
        paper_trading: Use paper trading URL if True.
        rate_limit: Requests per second.
    """

    def __init__(
        self,
        api_key: str = "",
        apiKey: str = "",  # noqa: N803
        secret: str = "",
        paper_trading: bool = True,
        rate_limit: float = DEFAULT_RATE_LIMIT_PER_SECOND,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        circuit_breaker_timeout: float = DEFAULT_CIRCUIT_BREAKER_TIMEOUT,
    ) -> None:
        self.api_key = api_key or apiKey
        self.secret = secret
        self.paper_trading = paper_trading
        self._started = False
        self._client = None  # type: TradingClient | None
        self._exchange_proxy = None  # type: AlpacaExchangeProxy | None

        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_timeout,
        )
        self._rate_limiter = RateLimiter(rate=rate_limit)

    async def start(self) -> None:
        """Initialize Alpaca client."""
        from alpaca.trading.client import TradingClient

        if self._started:
            return

        if not self.api_key or not self.secret:
            raise ValueError("Alpaca API key and secret are required")

        self._client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret,
            paper=self.paper_trading,
        )
        self._exchange_proxy = AlpacaExchangeProxy(
            self.api_key, self.secret, self.paper_trading
        )

        try:
            # Verify connectivity
            _ = self._client.get_account()
        except Exception as exc:
            log.error("Alpaca connection failed", error=str(exc))
            raise

        self._started = True
        log.info(
            "Alpaca gateway started",
            paper_trading=self.paper_trading,
        )

    async def stop(self) -> None:
        """Shutdown gateway."""
        self._started = False
        self._client = None
        self._exchange_proxy = None
        log.info("Alpaca gateway stopped")

    def health_check(self) -> bool:
        """Check if gateway is healthy."""
        if not self._started or self._client is None:
            return False
        return self._circuit_breaker.state != CircuitState.OPEN

    @property
    def exchange(self):
        """Return CCXT-compatible exchange interface."""
        return self._exchange_proxy

    @staticmethod
    def _build_alpaca_trade_id(alpaca_order: object, exchange_order_id: str, filled_qty: float) -> str:
        """Build best-effort unique fill identifier for Alpaca."""
        direct_trade_id = getattr(alpaca_order, "trade_id", None)
        if direct_trade_id not in (None, ""):
            return str(direct_trade_id)

        filled_at = getattr(alpaca_order, "filled_at", None)
        if filled_at:
            return f"{exchange_order_id}:{filled_at}:{filled_qty:.12f}"

        updated_at = getattr(alpaca_order, "updated_at", None)
        if updated_at:
            return f"{exchange_order_id}:{updated_at}:{filled_qty:.12f}"

        return f"{exchange_order_id}:{filled_qty:.12f}"

    async def submit_order(self, order: OrderRequest) -> FillReport | None:
        """Submit order to Alpaca."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        if not self._started or self._client is None:
            raise GatewayNotStartedError("Call start() before submit_order()")

        if not self._circuit_breaker.can_execute():
            raise CircuitOpenError(self._circuit_breaker.time_until_reset())

        if not await self._rate_limiter.acquire():
            raise ExecutionError("Rate limit exceeded", recoverable=False)

        # Map side
        side = OrderSide.BUY if order.side.value == "BUY" else OrderSide.SELL

        # Map type and create request
        if order.order_type.value == "MARKET":
            req = MarketOrderRequest(
                symbol=str(order.instrument_id),
                qty=order.quantity,
                side=side,
                time_in_force=TimeInForce.GTC,
                client_order_id=order.oms_order_id,
            )
        elif order.order_type.value == "LIMIT":
            if order.limit_price is None:
                raise ExecutionError("Limit price required for LIMIT order", recoverable=False)
            req = LimitOrderRequest(
                symbol=str(order.instrument_id),
                qty=order.quantity,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=order.limit_price,
                client_order_id=order.oms_order_id,
            )
        else:
            raise ExecutionError(
                f"Unsupported order type: {order.order_type}",
                recoverable=False,
            )

        try:
            # Note: client.submit_order is synchronous in alpaca-py,
            # but usually fast enough. For high throughput, we'd wrap in run_in_executor.
            alpaca_order = await asyncio.to_thread(
                self._client.submit_order,
                order_data=req,
            )

            order_status = str(getattr(alpaca_order, "status", "")).lower()
            if order_status in {"rejected", "canceled", "cancelled", "expired"}:
                raise ExecutionError(
                    f"Alpaca rejected order with status={order_status}",
                    recoverable=False,
                )

            exchange_order_id = str(getattr(alpaca_order, "id"))

            filled_raw = getattr(alpaca_order, "filled_qty", None)
            try:
                filled_qty = float(filled_raw or 0.0)
            except (TypeError, ValueError):
                filled_qty = 0.0

            # Submission acknowledged but no confirmed fills yet.
            if filled_qty <= 0:
                log.warning(
                    "Alpaca order acknowledged without fills; treating as ambiguous",
                    order_id=order.oms_order_id,
                    exchange_order_id=exchange_order_id,
                    status=order_status or "unknown",
                )
                self._circuit_breaker.record_success()
                return None

            price_raw = getattr(alpaca_order, "filled_avg_price", None)
            price = float(price_raw) if price_raw is not None else float(order.limit_price or 0.0)
            if price <= 0:
                log.warning(
                    "Alpaca fill missing valid average price; treating as ambiguous",
                    order_id=order.oms_order_id,
                    exchange_order_id=exchange_order_id,
                    filled_qty=filled_qty,
                )
                self._circuit_breaker.record_success()
                return None

            status = (
                ExecutionStatus.SUCCESS
                if filled_qty >= order.quantity
                else ExecutionStatus.PARTIAL
            )

            self._circuit_breaker.record_success()
            return FillReport(
                oms_order_id=order.oms_order_id,
                exchange_order_id=exchange_order_id,
                exchange_trade_id=self._build_alpaca_trade_id(
                    alpaca_order,
                    exchange_order_id,
                    filled_qty,
                ),
                price=price,
                quantity=filled_qty,
                fee=0.0,
                status=status,
                timestamp=datetime.now(timezone.utc),
            )

        except ExecutionError as e:
            if e.recoverable:
                self._circuit_breaker.record_failure()
            else:
                self._circuit_breaker.record_success()
            raise
        except Exception as e:
            self._circuit_breaker.record_failure()
            log.error("Alpaca order failed", error=str(e))
            raise ExecutionError(str(e), recoverable=True) from e


class RoutingExchangeProxy:
    """Proxies exchange calls to the appropriate gateway's exchange."""

    def __init__(self, router: RouterGateway) -> None:
        self.router = router

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 50
    ) -> list:
        gw = self.router._resolve_gateway(symbol)
        if hasattr(gw, "exchange") and hasattr(gw.exchange, "fetch_ohlcv"):
            return await gw.exchange.fetch_ohlcv(symbol, timeframe, limit)
        return []

    async def fetch_ticker(self, symbol: str) -> dict:
        gw = self.router._resolve_gateway(symbol)
        if hasattr(gw, "exchange") and hasattr(gw.exchange, "fetch_ticker"):
            return await gw.exchange.fetch_ticker(symbol)
        return {}


# ==============================================================================
# Router Gateway (Composite)
# ==============================================================================
class RouterGateway:
    """Routes orders to different gateways based on instrument symbol.

    Attributes:
        gateways: Map of prefix to gateway instance.
        default_gateway: Fallback gateway.
    """

    def __init__(
        self,
        gateways: dict[str, ExecutionGateway],
        routes: dict[str, str],  # symbol_prefix -> gateway_key
        default_gateway: str | None = None,
    ) -> None:
        self.gateways = gateways
        self.routes = routes
        self.default = gateways.get(default_gateway) if default_gateway else None
        self._exchange_proxy = RoutingExchangeProxy(self)

    async def start(self) -> None:
        for gw in self.gateways.values():
            await gw.start()

    async def stop(self) -> None:
        for gw in self.gateways.values():
            await gw.stop()

    def health_check(self) -> bool:
        return all(gw.health_check() for gw in self.gateways.values())

    def _resolve_gateway(self, symbol: str) -> ExecutionGateway:
        for prefix, gw_key in self.routes.items():
            if symbol.startswith(prefix) or prefix == "*":
                return self.gateways[gw_key]
        if self.default:
            return self.default
        raise ValueError(f"No route for symbol {symbol}")

    async def submit_order(self, order: OrderRequest) -> FillReport | None:
        gw = self._resolve_gateway(str(order.instrument_id))
        return await gw.submit_order(order)

    @property
    def exchange(self):
        """Expose a CCXT-compatible exchange interface for data fetching."""
        return self._exchange_proxy


# ==============================================================================
# OANDA Gateway Implementation
# ==============================================================================
class OANDAGateway:
    """OANDA v20 REST API gateway for forex, metals, and CFD trading.

    Provides connectivity to OANDA for EUR/USD, GBP/USD, XAU/USD, and
    other instruments available on the OANDA platform.

    Attributes:
        api_token: OANDA API bearer token.
        account_id: OANDA account ID (e.g. "001-001-1234567-001").
        paper_trading: Use practice (fxpractice) endpoint if True.
    """

    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"

    # Map QTS timeframe strings to OANDA granularity codes
    GRANULARITIES: dict[str, str] = {
        "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
        "1h": "H1", "4h": "H4", "8h": "H8", "1d": "D", "1w": "W",
    }

    def __init__(
        self,
        api_token: str = "",
        account_id: str = "",
        paper_trading: bool = True,
        rate_limit: float = 2.0,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        circuit_breaker_timeout: float = DEFAULT_CIRCUIT_BREAKER_TIMEOUT,
    ) -> None:
        self.api_token = api_token
        self.account_id = account_id
        self.paper_trading = paper_trading
        self._base_url = self.PRACTICE_URL if paper_trading else self.LIVE_URL
        self._started = False
        self._client = None  # type: Any  # httpx.AsyncClient

        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_timeout,
        )
        self._rate_limiter = RateLimiter(rate=rate_limit)

    @staticmethod
    def _to_oanda_symbol(symbol: str) -> str:
        """Convert QTS symbol to OANDA format (EUR/USD → EUR_USD)."""
        return symbol.replace("/", "_").upper()

    async def start(self) -> None:
        """Initialize OANDA HTTP client and verify connectivity."""
        import httpx

        if self._started:
            return
        if not self.api_token:
            msg = "OANDA API token required (set OANDA_API_TOKEN in .env)"
            raise ValueError(msg)
        if not self.account_id:
            msg = "OANDA account ID required (set OANDA_ACCOUNT_ID in .env)"
            raise ValueError(msg)

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        # Verify credentials with a lightweight account summary request
        resp = await self._client.get(f"/v3/accounts/{self.account_id}/summary")
        resp.raise_for_status()
        self._started = True
        log.info(
            "OANDA gateway started",
            account_id=self.account_id,
            paper=self.paper_trading,
            base_url=self._base_url,
        )

    async def stop(self) -> None:
        """Close OANDA HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._started = False
        log.info("OANDA gateway stopped")

    def health_check(self) -> bool:
        """Return True if gateway is started and circuit is closed."""
        return self._started and self._circuit_breaker.can_execute()

    async def fetch_ticker(self, symbol: str) -> dict:
        """Fetch current bid/ask for a symbol.

        Args:
            symbol: QTS symbol (e.g. "EUR/USD").

        Returns:
            Dict with keys: last, bid, ask, timestamp.
        """
        if not self._started:
            raise GatewayNotStartedError("Call start() first")

        instrument = self._to_oanda_symbol(symbol)
        resp = await self._client.get(
            f"/v3/accounts/{self.account_id}/pricing",
            params={"instruments": instrument},
        )
        resp.raise_for_status()
        prices = resp.json().get("prices", [])
        if not prices:
            msg = f"No pricing data for {instrument}"
            raise ExecutionError(msg)

        price = prices[0]
        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])
        return {
            "last": (bid + ask) / 2,
            "bid": bid,
            "ask": ask,
            "timestamp": datetime.now(timezone.utc).timestamp() * 1000,
        }

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        since: int | None = None,
    ) -> list:
        """Fetch OHLCV candles from OANDA Instruments API.

        Args:
            symbol: QTS symbol (e.g. "EUR/USD").
            timeframe: Candle period (1m, 5m, 1h, 4h, 1d ...).
            limit: Number of candles (max 5000 per request).
            since: Start timestamp in milliseconds (Unix epoch).

        Returns:
            List of (datetime, open, high, low, close, volume) tuples.
        """
        if not self._started:
            raise GatewayNotStartedError("Call start() first")

        instrument = self._to_oanda_symbol(symbol)
        granularity = self.GRANULARITIES.get(timeframe, "H1")
        params: dict = {
            "granularity": granularity,
            "count": min(limit, 5000),
            "price": "M",  # mid-point candles
        }
        if since is not None:
            from_dt = datetime.fromtimestamp(since / 1000, tz=timezone.utc)
            params["from"] = from_dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            params.pop("count", None)

        resp = await self._client.get(
            f"/v3/instruments/{instrument}/candles",
            params=params,
        )
        resp.raise_for_status()

        result = []
        for candle in resp.json().get("candles", []):
            if not candle.get("complete", True):
                continue
            mid = candle.get("mid", {})
            ts = datetime.fromisoformat(candle["time"].replace("Z", "+00:00"))
            result.append((
                ts,
                float(mid.get("o", 0)),
                float(mid.get("h", 0)),
                float(mid.get("l", 0)),
                float(mid.get("c", 0)),
                float(candle.get("volume", 0)),
            ))
        return result

    async def submit_order(self, order: OrderRequest) -> FillReport | None:
        """Submit a market order to OANDA.

        Args:
            order: Order request from OMS.

        Returns:
            FillReport on success, None on failure.
        """
        if not self._started:
            raise GatewayNotStartedError("Call start() first")
        if not self._circuit_breaker.can_execute():
            raise CircuitOpenError(self._circuit_breaker.time_until_reset())

        await self._rate_limiter.acquire()

        instrument = self._to_oanda_symbol(order.instrument_id)
        from qts_core.execution.oms import OrderSide
        units = order.quantity if order.side == OrderSide.BUY else -order.quantity

        payload = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(int(units)) if units == int(units) else str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }

        try:
            resp = await self._client.post(
                f"/v3/accounts/{self.account_id}/orders",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            self._circuit_breaker.record_success()
        except Exception as exc:
            self._circuit_breaker.record_failure()
            log.warning("OANDA order failed", error=str(exc), instrument=instrument)
            return None

        fill = data.get("orderFillTransaction", {})
        if not fill:
            log.warning("OANDA: no fill transaction", instrument=instrument, data=data)
            return None

        fill_price = float(fill.get("price", 1.0))
        fill_units = abs(float(fill.get("units", order.quantity)))
        commission = fill.get("commission", {})
        fee = abs(float(commission.get("units", "0"))) if isinstance(commission, dict) else 0.0

        return FillReport(
            oms_order_id=order.oms_order_id,
            exchange_order_id=fill.get("orderID"),
            exchange_trade_id=fill.get("id"),
            price=fill_price,
            quantity=fill_units,
            fee=fee,
            fee_currency="USD",
            status=ExecutionStatus.SUCCESS,
        )


# ==============================================================================
# Mock Gateway (Testing)
# ==============================================================================
class MockGateway:
    """Mock gateway for testing without exchange.

    Simulates order execution with configurable behavior.
    """

    def __init__(
        self,
        default_price: float = 100_000.0,
        latency_ms: float = 100.0,
        failure_rate: float = 0.0,
        partial_fill_rate: float = 0.0,
    ) -> None:
        """Initialize mock gateway.

        Args:
            default_price: Default fill price.
            latency_ms: Simulated network latency.
            failure_rate: Probability of order failure (0.0-1.0).
            partial_fill_rate: Probability of partial fill (0.0-1.0).
        """
        self.default_price = default_price
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.partial_fill_rate = partial_fill_rate
        self._started = False
        self._order_count = 0

    async def start(self) -> None:
        """Start mock gateway."""
        self._started = True
        log.info("Mock gateway started")

    async def stop(self) -> None:
        """Stop mock gateway."""
        self._started = False
        log.info("Mock gateway stopped")

    def health_check(self) -> bool:
        """Check mock gateway health."""
        return self._started

    async def submit_order(self, order: OrderRequest) -> FillReport | None:
        """Simulate order execution."""
        import random

        if not self._started:
            raise GatewayNotStartedError("Call start() first")

        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        self._order_count += 1

        # Simulate failure
        if random.random() < self.failure_rate:
            log.warning("Mock order failed", order_id=order.oms_order_id)
            return None

        # Simulate partial fill
        fill_qty = order.quantity
        status = ExecutionStatus.SUCCESS
        if random.random() < self.partial_fill_rate:
            fill_qty = order.quantity * random.uniform(0.3, 0.9)
            status = ExecutionStatus.PARTIAL

        log.info(
            "Mock order filled",
            order_id=order.oms_order_id,
            price=self.default_price,
            quantity=fill_qty,
        )

        return FillReport(
            oms_order_id=order.oms_order_id,
            exchange_order_id=f"mock-{self._order_count}",
            exchange_trade_id=f"mock-fill-{self._order_count}",
            price=self.default_price,
            quantity=fill_qty,
            fee=0.0,
            status=status,
        )
