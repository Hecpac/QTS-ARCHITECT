"""Tests for execution layer (OMS, EMS, Store).

Tests cover:
- MemoryStore operations and transactions
- OMS order lifecycle and portfolio management
- EMS circuit breaker and rate limiting
- MockGateway behavior
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from qts_core.agents.protocol import SignalType, TradingDecision
from qts_core.execution import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ExecutionStatus,
    FillReport,
    GatewayNotStartedError,
    MemoryStore,
    MockGateway,
    Order,
    OrderManagementSystem,
    OrderNotFoundError,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    RateLimiter,
    TimeInForce,
)


# ==============================================================================
# Store Tests
# ==============================================================================
class TestMemoryStore:
    """Tests for MemoryStore."""

    def test_set_get(self) -> None:
        """Test basic set/get operations."""
        store = MemoryStore()
        store.set("key1", "value1")
        assert store.get("key1") == "value1"

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent key returns None."""
        store = MemoryStore()
        assert store.get("nonexistent") is None

    def test_exists(self) -> None:
        """Test exists check."""
        store = MemoryStore()
        store.set("key1", "value1")
        assert store.exists("key1") is True
        assert store.exists("key2") is False

    def test_delete(self) -> None:
        """Test key deletion."""
        store = MemoryStore()
        store.set("key1", "value1")
        store.delete("key1")
        assert store.exists("key1") is False

    def test_save_load_model(self) -> None:
        """Test Pydantic model serialization."""
        store = MemoryStore()
        portfolio = Portfolio(cash=10_000.0)
        store.save("portfolio", portfolio)

        loaded = store.load("portfolio", Portfolio)
        assert loaded is not None
        assert loaded.cash == 10_000.0

    def test_increment(self) -> None:
        """Test atomic increment."""
        store = MemoryStore()
        assert store.increment("counter") == 1
        assert store.increment("counter") == 2
        assert store.increment("counter", 5) == 7

    def test_get_many(self) -> None:
        """Test multi-key get."""
        store = MemoryStore()
        store.set("a", "1")
        store.set("b", "2")
        result = store.get_many(["a", "b", "c"])
        assert result == ["1", "2", None]

    def test_set_many(self) -> None:
        """Test multi-key set."""
        store = MemoryStore()
        store.set_many({"a": "1", "b": "2"})
        assert store.get("a") == "1"
        assert store.get("b") == "2"

    def test_clear(self) -> None:
        """Test clearing all data."""
        store = MemoryStore()
        store.set("a", "1")
        store.set("b", "2")
        store.clear()
        assert store.exists("a") is False
        assert store.exists("b") is False

    def test_health_check(self) -> None:
        """Test health check always returns True."""
        store = MemoryStore()
        assert store.health_check() is True


# ==============================================================================
# OMS Models Tests
# ==============================================================================
class TestOrder:
    """Tests for Order model."""

    def test_create_market_order(self) -> None:
        """Test creating a market order."""
        order = Order(
            instrument_id="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
        )
        assert order.id is not None
        assert order.status == OrderStatus.PENDING
        assert order.order_type == OrderType.MARKET
        assert order.time_in_force == TimeInForce.GTC

    def test_order_requires_positive_quantity(self) -> None:
        """Test order validation for quantity."""
        with pytest.raises(ValueError):
            Order(
                instrument_id="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.0,
            )

    def test_limit_order_requires_price(self) -> None:
        """Test limit order requires limit_price."""
        with pytest.raises(ValueError):
            Order(
                instrument_id="BTC/USDT",
                side=OrderSide.BUY,
                quantity=0.1,
                order_type=OrderType.LIMIT,
            )

    def test_limit_order_with_price(self) -> None:
        """Test limit order with price succeeds."""
        order = Order(
            instrument_id="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            order_type=OrderType.LIMIT,
            limit_price=50000.0,
        )
        assert order.limit_price == 50000.0

    def test_remaining_quantity(self) -> None:
        """Test remaining quantity calculation."""
        order = Order(
            instrument_id="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            filled_quantity=0.3,
        )
        assert order.remaining_quantity == 0.7

    def test_is_terminal(self) -> None:
        """Test terminal state detection."""
        order = Order(
            instrument_id="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            status=OrderStatus.PENDING,
        )
        assert order.is_terminal is False

        order.status = OrderStatus.FILLED
        assert order.is_terminal is True

    def test_state_transitions(self) -> None:
        """Test valid state transitions."""
        order = Order(
            instrument_id="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            status=OrderStatus.PENDING,
        )
        assert order.can_transition_to(OrderStatus.SUBMITTED) is True
        assert order.can_transition_to(OrderStatus.FILLED) is False

        order.status = OrderStatus.SUBMITTED
        assert order.can_transition_to(OrderStatus.FILLED) is True
        assert order.can_transition_to(OrderStatus.PENDING) is False


class TestPortfolio:
    """Tests for Portfolio model."""

    def test_create_portfolio(self) -> None:
        """Test creating portfolio with initial cash."""
        portfolio = Portfolio(cash=100_000.0)
        assert portfolio.cash == 100_000.0
        assert portfolio.blocked_cash == 0.0
        assert len(portfolio.positions) == 0

    def test_total_cash(self) -> None:
        """Test total cash includes blocked."""
        portfolio = Portfolio(cash=90_000.0, blocked_cash=10_000.0)
        assert portfolio.total_cash == 100_000.0

    def test_get_position(self) -> None:
        """Test getting position."""
        portfolio = Portfolio(
            cash=100_000.0,
            positions={"BTC/USDT": 1.5},
        )
        assert portfolio.get_position("BTC/USDT") == 1.5
        assert portfolio.get_position("ETH/USDT") == 0.0

    def test_get_available_position(self) -> None:
        """Test available position excludes blocked."""
        portfolio = Portfolio(
            cash=100_000.0,
            positions={"BTC/USDT": 1.5},
            blocked_positions={"BTC/USDT": 0.5},
        )
        assert portfolio.get_available_position("BTC/USDT") == 1.0


# ==============================================================================
# OMS Tests
# ==============================================================================
class TestOrderManagementSystem:
    """Tests for OrderManagementSystem."""

    def test_init_with_initial_cash(self) -> None:
        """Test OMS initialization."""
        store = MemoryStore()
        oms = OrderManagementSystem(store, initial_cash=50_000.0)
        assert oms.portfolio.cash == 50_000.0

    def test_init_loads_existing_portfolio(self) -> None:
        """Test OMS loads existing portfolio from store."""
        store = MemoryStore()
        existing = Portfolio(cash=75_000.0)
        store.save("oms:portfolio", existing)

        oms = OrderManagementSystem(store, initial_cash=100_000.0)
        assert oms.portfolio.cash == 75_000.0

    def test_invalid_risk_fraction(self) -> None:
        """Test OMS rejects invalid risk fraction."""
        store = MemoryStore()
        with pytest.raises(ValueError):
            OrderManagementSystem(store, risk_fraction=0.0)
        with pytest.raises(ValueError):
            OrderManagementSystem(store, risk_fraction=1.5)

    def test_process_neutral_decision(self) -> None:
        """Test neutral decision returns None."""
        store = MemoryStore()
        oms = OrderManagementSystem(store)
        decision = TradingDecision(
            decision_id=uuid.uuid4(),
            instrument_id="BTC/USDT",
            action=SignalType.NEUTRAL,
            confidence=0.5,
            quantity_modifier=1.0,
            rationale="Test neutral signal",
        )
        result = oms.process_decision(decision, current_price=50_000.0)
        assert result is None

    def test_process_buy_decision(self) -> None:
        """Test buy decision creates order and reserves cash."""
        store = MemoryStore()
        oms = OrderManagementSystem(store, initial_cash=100_000.0, risk_fraction=0.1)

        decision = TradingDecision(
            decision_id=uuid.uuid4(),
            instrument_id="BTC/USDT",
            action=SignalType.LONG,
            confidence=0.8,
            quantity_modifier=1.0,
            rationale="Test buy signal",
        )

        request = oms.process_decision(decision, current_price=50_000.0)

        assert request is not None
        assert request.side == OrderSide.BUY
        assert request.quantity == 0.2  # 10% of 100k / 50k price
        assert oms.portfolio.cash == 90_000.0  # 100k - 10k reserved
        assert oms.portfolio.blocked_cash == 10_000.0

    def test_process_buy_insufficient_funds(self) -> None:
        """Test buy with insufficient funds returns None."""
        store = MemoryStore()
        # Initial cash is 100, risk_fraction 0.5 = 50 trade value
        # At price 10, quantity = 5, cost = 50
        # But we set cash to only 10, so insufficient
        oms = OrderManagementSystem(store, initial_cash=10.0, risk_fraction=1.0)

        decision = TradingDecision(
            decision_id=uuid.uuid4(),
            instrument_id="BTC/USDT",
            action=SignalType.LONG,
            confidence=0.8,
            quantity_modifier=1.0,
            rationale="Test insufficient funds",
        )

        # Cash is 10, risk 100% = 10 trade value
        # At price 1000, quantity = 0.01, cost = 10
        # First order succeeds leaving 0 cash
        request1 = oms.process_decision(decision, current_price=1000.0)
        assert request1 is not None  # First order works

        # Second order should fail - no cash left
        request2 = oms.process_decision(decision, current_price=1000.0)
        assert request2 is None  # Second order fails

    def test_process_sell_decision(self) -> None:
        """Test sell decision with existing position."""
        store = MemoryStore()
        oms = OrderManagementSystem(store, initial_cash=0.0, risk_fraction=0.1)
        oms.portfolio.positions["BTC/USDT"] = 1.0
        oms.portfolio.cash = 100_000.0  # Need cash to calculate quantity

        decision = TradingDecision(
            decision_id=uuid.uuid4(),
            instrument_id="BTC/USDT",
            action=SignalType.SHORT,
            confidence=0.8,
            quantity_modifier=1.0,
            rationale="Test sell signal",
        )

        request = oms.process_decision(decision, current_price=50_000.0)

        assert request is not None
        assert request.side == OrderSide.SELL

    def test_confirm_buy_execution(self) -> None:
        """Test confirming buy execution settles trade."""
        store = MemoryStore()
        oms = OrderManagementSystem(store, initial_cash=100_000.0, risk_fraction=0.1)

        # Create order
        decision = TradingDecision(
            decision_id=uuid.uuid4(),
            instrument_id="BTC/USDT",
            action=SignalType.LONG,
            confidence=0.8,
            quantity_modifier=1.0,
            rationale="Test buy execution",
        )
        request = oms.process_decision(decision, current_price=50_000.0)
        assert request is not None

        # Confirm execution
        oms.confirm_execution(
            request.oms_order_id,
            fill_price=50_000.0,
            fill_qty=0.2,
            fee=10.0,
        )

        # Check state
        assert oms.portfolio.blocked_cash == 0.0
        assert oms.portfolio.positions["BTC/USDT"] == 0.2
        # Cash = 90k (after reserve) + (10k reserve - 10k actual - 10 fee)
        assert oms.portfolio.cash == pytest.approx(89_990.0)

    def test_revert_allocation(self) -> None:
        """Test reverting allocation restores portfolio state."""
        store = MemoryStore()
        oms = OrderManagementSystem(store, initial_cash=100_000.0, risk_fraction=0.1)

        # Create order
        decision = TradingDecision(
            decision_id=uuid.uuid4(),
            instrument_id="BTC/USDT",
            action=SignalType.LONG,
            confidence=0.8,
            quantity_modifier=1.0,
            rationale="Test revert allocation",
        )
        request = oms.process_decision(decision, current_price=50_000.0)
        assert request is not None
        assert oms.portfolio.cash == 90_000.0

        # Revert
        oms.revert_allocation(request.oms_order_id)

        # Check state restored
        assert oms.portfolio.cash == 100_000.0
        assert oms.portfolio.blocked_cash == 0.0

    def test_revert_unknown_order(self) -> None:
        """Test reverting unknown order raises error."""
        store = MemoryStore()
        oms = OrderManagementSystem(store)

        with pytest.raises(OrderNotFoundError):
            oms.revert_allocation("nonexistent-id")

    def test_cancel_order(self) -> None:
        """Test cancelling order."""
        store = MemoryStore()
        oms = OrderManagementSystem(store, initial_cash=100_000.0, risk_fraction=0.1)

        # Create order
        decision = TradingDecision(
            decision_id=uuid.uuid4(),
            instrument_id="BTC/USDT",
            action=SignalType.LONG,
            confidence=0.8,
            quantity_modifier=1.0,
            rationale="Test cancel order",
        )
        request = oms.process_decision(decision, current_price=50_000.0)
        assert request is not None

        # Cancel
        result = oms.cancel_order(request.oms_order_id)
        assert result is True

        # Check state
        order = oms.get_order(request.oms_order_id)
        assert order is not None
        assert order.status == OrderStatus.CANCELLED
        assert oms.portfolio.cash == 100_000.0


# ==============================================================================
# Circuit Breaker Tests
# ==============================================================================
class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self) -> None:
        """Test circuit starts closed."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_threshold(self) -> None:
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_success_resets_failure_count(self) -> None:
        """Test success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 1

    def test_time_until_reset(self) -> None:
        """Test time calculation until reset."""
        cb = CircuitBreaker(reset_timeout=60.0)

        # Closed circuit has no wait time
        assert cb.time_until_reset() == 0.0

        # Open circuit
        for _ in range(5):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.time_until_reset() > 0.0


# ==============================================================================
# Rate Limiter Tests
# ==============================================================================
class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_immediate(self) -> None:
        """Test acquiring token when available."""
        rl = RateLimiter(rate=10.0)
        result = await rl.acquire(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_multiple(self) -> None:
        """Test acquiring multiple tokens."""
        rl = RateLimiter(rate=10.0)
        for _ in range(5):
            result = await rl.acquire(timeout=1.0)
            assert result is True

    @pytest.mark.asyncio
    async def test_acquire_timeout(self) -> None:
        """Test timeout when tokens exhausted."""
        rl = RateLimiter(rate=1.0)
        rl.tokens = 0  # Exhaust tokens

        result = await rl.acquire(timeout=0.2)
        assert result is False


# ==============================================================================
# Mock Gateway Tests
# ==============================================================================
class TestMockGateway:
    """Tests for MockGateway."""

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test gateway lifecycle."""
        gateway = MockGateway()
        assert gateway.health_check() is False

        await gateway.start()
        assert gateway.health_check() is True

        await gateway.stop()
        assert gateway.health_check() is False

    @pytest.mark.asyncio
    async def test_submit_without_start(self) -> None:
        """Test submitting before start raises error."""
        gateway = MockGateway()
        order = OrderRequest(
            oms_order_id="test-1",
            instrument_id="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            order_type=OrderType.MARKET,
        )

        with pytest.raises(GatewayNotStartedError):
            await gateway.submit_order(order)

    @pytest.mark.asyncio
    async def test_submit_success(self) -> None:
        """Test successful order submission."""
        gateway = MockGateway(default_price=50_000.0)
        await gateway.start()

        order = OrderRequest(
            oms_order_id="test-1",
            instrument_id="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            order_type=OrderType.MARKET,
        )

        fill = await gateway.submit_order(order)

        assert fill is not None
        assert fill.oms_order_id == "test-1"
        assert fill.price == 50_000.0
        assert fill.quantity == 0.1
        assert fill.status == ExecutionStatus.SUCCESS

        await gateway.stop()

    @pytest.mark.asyncio
    async def test_submit_with_failure_rate(self) -> None:
        """Test gateway with 100% failure rate."""
        gateway = MockGateway(failure_rate=1.0)
        await gateway.start()

        order = OrderRequest(
            oms_order_id="test-1",
            instrument_id="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.1,
            order_type=OrderType.MARKET,
        )

        fill = await gateway.submit_order(order)
        assert fill is None

        await gateway.stop()


# ==============================================================================
# Integration Tests
# ==============================================================================
class TestExecutionIntegration:
    """Integration tests for execution layer."""

    @pytest.mark.asyncio
    async def test_full_order_lifecycle(self) -> None:
        """Test complete order flow: decision -> fill -> settlement."""
        # Setup
        store = MemoryStore()
        oms = OrderManagementSystem(store, initial_cash=100_000.0, risk_fraction=0.1)
        gateway = MockGateway(default_price=50_000.0)
        await gateway.start()

        # Create decision
        decision = TradingDecision(
            decision_id=uuid.uuid4(),
            instrument_id="BTC/USDT",
            action=SignalType.LONG,
            confidence=0.9,
            quantity_modifier=1.0,
            rationale="Test full lifecycle",
        )

        # Process decision
        request = oms.process_decision(decision, current_price=50_000.0)
        assert request is not None

        # Submit to exchange
        fill = await gateway.submit_order(request)
        assert fill is not None

        # Confirm execution
        oms.confirm_execution(
            fill.oms_order_id,
            fill.price,
            fill.quantity,
            fill.fee,
        )

        # Verify final state
        assert oms.portfolio.positions["BTC/USDT"] == 0.2
        assert oms.portfolio.blocked_cash == 0.0

        # Verify order status
        order = oms.get_order(request.oms_order_id)
        assert order is not None
        assert order.status == OrderStatus.FILLED

        await gateway.stop()

    @pytest.mark.asyncio
    async def test_order_failure_reversion(self) -> None:
        """Test order failure properly reverts allocation."""
        # Setup
        store = MemoryStore()
        oms = OrderManagementSystem(store, initial_cash=100_000.0, risk_fraction=0.1)
        gateway = MockGateway(failure_rate=1.0)  # Always fail
        await gateway.start()

        initial_cash = oms.portfolio.cash

        # Create decision
        decision = TradingDecision(
            decision_id=uuid.uuid4(),
            instrument_id="BTC/USDT",
            action=SignalType.LONG,
            confidence=0.9,
            quantity_modifier=1.0,
            rationale="Test failure reversion",
        )

        # Process decision
        request = oms.process_decision(decision, current_price=50_000.0)
        assert request is not None
        assert oms.portfolio.cash < initial_cash  # Cash reserved

        # Submit fails
        fill = await gateway.submit_order(request)
        assert fill is None

        # Revert allocation
        oms.revert_allocation(request.oms_order_id)

        # Cash should be restored
        assert oms.portfolio.cash == initial_cash
        assert oms.portfolio.blocked_cash == 0.0

        await gateway.stop()
