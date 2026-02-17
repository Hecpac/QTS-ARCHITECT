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
            return None

        # Paper trading: simulate fill
        if self.paper_trading:
            return await self._simulate_fill(order)

        # Real execution
        try:
            fill = await self._execute_with_retry(order)
            self._circuit_breaker.record_success()
            return fill
        except Exception as e:
            self._circuit_breaker.record_failure()
            log.error(
                "Order execution failed",
                order_id=order.oms_order_id,
                error=str(e),
            )
            return None

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

        return self._parse_response(order, response, symbol)

    async def _parse_response(
        self,
        order: OrderRequest,
        response: dict,
        symbol: str,
    ) -> FillReport | None:
        """Parse exchange response into FillReport."""
        fill_price = response.get("price") or response.get("average")
        filled = response.get("filled")

        # Fetch order if needed
        if fill_price is None or filled is None:
            order_id = response.get("id")
            if order_id:
                try:
                    fetched = await self.exchange.fetch_order(order_id, symbol)
                    fill_price = fill_price or fetched.get("average") or fetched.get("price")
                    filled = filled or fetched.get("filled")
                except Exception as e:
                    log.warning("Could not fetch order details", error=str(e))

        if filled is None:
            log.error(
                "Exchange did not return filled quantity",
                response=response,
            )
            return None

        try:
            parsed_fill_qty = float(filled)
        except (TypeError, ValueError):
            log.error(
                "Exchange returned non-numeric filled quantity",
                filled=filled,
                order_id=order.oms_order_id,
            )
            return None

        if parsed_fill_qty <= 0:
            log.error(
                "Exchange returned non-positive filled quantity",
                filled=parsed_fill_qty,
                order_id=order.oms_order_id,
            )
            return None

        if fill_price is None:
            log.error(
                "Exchange did not return fill price",
                order_id=order.oms_order_id,
                response_keys=sorted(response.keys()),
            )
            return None

        try:
            parsed_fill_price = float(fill_price)
        except (TypeError, ValueError):
            log.error(
                "Exchange returned non-numeric fill price",
                fill_price=fill_price,
                order_id=order.oms_order_id,
            )
            return None

        if parsed_fill_price <= 0:
            log.error(
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

        return FillReport(
            oms_order_id=order.oms_order_id,
            exchange_order_id=response.get("id"),
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
            price=self.default_price,
            quantity=fill_qty,
            fee=0.0,
            status=status,
        )
