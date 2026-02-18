"""Order Management System (OMS) for QTS-Architect.

The OMS is the "accountant" of the trading system - it maintains the
true state of the portfolio and handles order lifecycle management.

Responsibilities:
- Portfolio state management (cash, positions, blocked amounts)
- Order creation with pessimistic locking
- Pre-trade risk validation
- Fill confirmation and settlement
- Crash recovery via atomic persistence

Design Decisions:
- Pessimistic locking: Reserve funds/positions BEFORE sending to EMS
- Atomic state updates: Use store transactions when available
- Immutable order records: Orders never mutate, only status changes
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

import structlog
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)

from qts_core.agents.protocol import SignalType, TradingDecision
from qts_core.common.types import InstrumentId
from qts_core.execution.store import MemoryStore, RedisStore

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_INITIAL_CASH: Final[float] = 100_000.0
DEFAULT_RISK_FRACTION: Final[float] = 0.10
MIN_QUANTITY: Final[float] = 1e-8  # Minimum tradeable quantity


# ==============================================================================
# Enums
# ==============================================================================
class OrderType(str, Enum):
    """Order execution type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    """Order direction."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order lifecycle state.

    State Transitions:
        PENDING -> SUBMITTED -> FILLED
                            -> PARTIALLY_FILLED -> FILLED
                            -> CANCELLED
                            -> FAILED
        PENDING -> FAILED (validation failure)
    """

    PENDING = "PENDING"  # Created in OMS, not yet sent
    SUBMITTED = "SUBMITTED"  # Sent to EMS/Exchange
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Partial execution
    FILLED = "FILLED"  # Fully executed
    CANCELLED = "CANCELLED"  # Cancelled by user or system
    FAILED = "FAILED"  # Failed to execute
    EXPIRED = "EXPIRED"  # Time-based expiration


class TimeInForce(str, Enum):
    """Order time validity."""

    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    DAY = "DAY"  # Day order


class AccountMode(str, Enum):
    """Execution account mode."""

    SPOT = "spot"
    MARGIN = "margin"
    PERP = "perp"


class OrderIntent(str, Enum):
    """Semantic intent behind an order."""

    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


# ==============================================================================
# Custom Exceptions
# ==============================================================================
class OMSError(Exception):
    """Base exception for OMS errors."""


class InsufficientFundsError(OMSError):
    """Raised when insufficient cash for buy order."""

    def __init__(self, required: float, available: float) -> None:
        self.required = required
        self.available = available
        super().__init__(f"Insufficient funds: required={required:.2f}, available={available:.2f}")


class InsufficientPositionError(OMSError):
    """Raised when insufficient position for sell order."""

    def __init__(self, instrument: str, required: float, available: float) -> None:
        self.instrument = instrument
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient position for {instrument}: required={required:.8f}, available={available:.8f}"
        )


class OrderNotFoundError(OMSError):
    """Raised when order not found in store."""

    def __init__(self, order_id: str) -> None:
        self.order_id = order_id
        super().__init__(f"Order not found: {order_id}")


class InvalidOrderStateError(OMSError):
    """Raised when order state transition is invalid."""

    def __init__(self, order_id: str, current: OrderStatus, attempted: OrderStatus) -> None:
        self.order_id = order_id
        self.current = current
        self.attempted = attempted
        super().__init__(
            f"Invalid state transition for order {order_id}: {current.value} -> {attempted.value}"
        )


# ==============================================================================
# Domain Models
# ==============================================================================
class Order(BaseModel):
    """Internal representation of an order.

    Orders are semi-immutable: core fields (instrument, side, quantity)
    cannot change after creation, only status and metadata fields update.

    Attributes:
        id: Unique order identifier (UUID).
        decision_id: Reference to TradingDecision that triggered this order.
        instrument_id: Trading pair/instrument identifier.
        side: Buy or sell.
        intent: Semantic intent (open/close long/short).
        quantity: Order quantity.
        order_type: Execution type (MARKET, LIMIT, etc.).
        status: Current lifecycle state.
        limit_price: Price for limit orders (optional).
        stop_price: Trigger price for stop orders (optional).
        time_in_force: Order validity duration.
        reserved_cash: Cash blocked for buy orders.
        reserved_quantity: Quantity blocked for sell orders.
        reserved_cash_remaining: Remaining cash reservation for this order.
        reserved_quantity_remaining: Remaining quantity reservation for this order.
        applied_fill_ids: Exchange fill IDs already applied (idempotency).
        filled_quantity: Quantity filled so far.
        average_fill_price: Average price of fills.
        created_at: Order creation timestamp.
        updated_at: Last update timestamp.
        reduce_only: Whether execution must only reduce an existing position.
        external_id: Exchange/broker order ID.
    """

    model_config = {"frozen": False, "validate_assignment": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str | None = None
    instrument_id: InstrumentId
    side: OrderSide
    intent: OrderIntent = OrderIntent.OPEN_LONG
    quantity: float = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.PENDING
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    time_in_force: TimeInForce = TimeInForce.GTC
    reserved_cash: float | None = Field(default=None, ge=0)
    reserved_quantity: float | None = Field(default=None, ge=0)
    reserved_cash_remaining: float | None = Field(default=None, ge=0)
    reserved_quantity_remaining: float | None = Field(default=None, ge=0)
    applied_fill_ids: list[str] = Field(default_factory=list)
    filled_quantity: float = Field(default=0.0, ge=0)
    average_fill_price: float | None = Field(default=None, ge=0)
    reduce_only: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    external_id: str | None = None

    @field_validator("quantity", mode="before")
    @classmethod
    def validate_quantity(cls, v: float) -> float:
        """Ensure quantity meets minimum threshold."""
        if v < MIN_QUANTITY:
            msg = f"Quantity {v} below minimum {MIN_QUANTITY}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_limit_price_for_limit_orders(self) -> Order:
        """Ensure limit orders have limit price."""
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT}:
            if self.limit_price is None:
                msg = f"Limit price required for {self.order_type.value} orders"
                raise ValueError(msg)
        return self

    @property
    def remaining_quantity(self) -> float:
        """Quantity still to be filled."""
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_terminal(self) -> bool:
        """Whether order is in a final state."""
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.FAILED,
            OrderStatus.EXPIRED,
        }

    def can_transition_to(self, new_status: OrderStatus) -> bool:
        """Check if state transition is valid."""
        valid_transitions: dict[OrderStatus, set[OrderStatus]] = {
            OrderStatus.PENDING: {OrderStatus.SUBMITTED, OrderStatus.FAILED},
            OrderStatus.SUBMITTED: {
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.FAILED,
                OrderStatus.EXPIRED,
            },
            OrderStatus.PARTIALLY_FILLED: {
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
            },
            # Terminal states
            OrderStatus.FILLED: set(),
            OrderStatus.CANCELLED: set(),
            OrderStatus.FAILED: set(),
            OrderStatus.EXPIRED: set(),
        }
        return new_status in valid_transitions.get(self.status, set())


class Portfolio(BaseModel):
    """Current state of holdings and cash.

    Implements pessimistic locking through blocked_cash and blocked_positions
    to prevent double-spending during order execution.

    Attributes:
        cash: Available free balance (quote currency).
        positions: Mapping of instrument -> signed net quantity held.
            Positive = long inventory, negative = short inventory.
        blocked_cash: Cash locked for pending orders.
        blocked_positions: Long quantity locked while closing long positions.
    """

    model_config = {"frozen": False}

    cash: float = Field(ge=0, description="Available free balance")
    positions: dict[InstrumentId, float] = Field(
        default_factory=dict,
        description="InstrumentId -> Quantity held",
    )
    blocked_cash: float = Field(
        default=0.0,
        ge=0,
        description="Cash locked in pending buy orders",
    )
    blocked_positions: dict[InstrumentId, float] = Field(
        default_factory=dict,
        description="Quantity locked in pending sell orders",
    )

    @property
    def total_cash(self) -> float:
        """Total cash including blocked amounts."""
        return self.cash + self.blocked_cash

    @property
    def total_equity(self) -> float:
        """Total equity (cash only - positions need prices)."""
        return self.total_cash

    def get_position(self, instrument_id: InstrumentId) -> float:
        """Get position for instrument (0 if not held)."""
        return self.positions.get(instrument_id, 0.0)

    def get_available_position(self, instrument_id: InstrumentId) -> float:
        """Get position minus blocked quantity."""
        held = self.get_position(instrument_id)
        blocked = self.blocked_positions.get(instrument_id, 0.0)
        return max(0.0, held - blocked)


class OrderRequest(BaseModel):
    """Request payload sent to EMS.

    Immutable transfer object containing all information needed
    for the execution gateway to submit the order.
    """

    model_config = {"frozen": True}

    oms_order_id: str
    instrument_id: InstrumentId
    side: OrderSide
    intent: OrderIntent = OrderIntent.OPEN_LONG
    quantity: PositiveFloat
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False


class FillEvent(BaseModel):
    """Represents a fill/execution event from EMS.

    Immutable record of an execution that occurred.
    """

    model_config = {"frozen": True}

    oms_order_id: str
    fill_price: PositiveFloat
    fill_quantity: PositiveFloat
    fee: float = Field(default=0.0, ge=0)
    exchange_trade_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Order Management System
# ==============================================================================
class OrderManagementSystem:
    """The Accountant - manages portfolio state and order lifecycle.

    The OMS implements pessimistic locking to ensure portfolio consistency:
    1. Before creating an order, it reserves (blocks) the required funds/positions
    2. On fill confirmation, it settles the trade and releases blocks
    3. On failure/cancellation, it reverts the reservations

    This approach prevents double-spending and ensures the portfolio
    state is always consistent even if the system crashes mid-execution.

    Attributes:
        store: Persistence backend.
        portfolio: Current portfolio state.
        risk_fraction: Max fraction of cash per trade.
    """

    PORTFOLIO_KEY: Final[str] = "oms:portfolio"
    ORDERS_KEY_PREFIX: Final[str] = "oms:order:"

    def __init__(
        self,
        store: RedisStore | MemoryStore,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        risk_fraction: float = DEFAULT_RISK_FRACTION,
        account_mode: AccountMode | str = AccountMode.SPOT,
        short_leverage: float = 1.0,
    ) -> None:
        """Initialize OMS with store and configuration.

        Args:
            store: Persistence backend (Redis or Memory).
            initial_cash: Starting cash balance for new portfolios.
            risk_fraction: Maximum fraction of cash to risk per trade.
            account_mode: Trading account mode (spot/margin/perp).
            short_leverage: Effective leverage used for short collateral checks.

        Raises:
            ValueError: If risk_fraction not in (0, 1].
        """
        if not 0 < risk_fraction <= 1:
            msg = f"risk_fraction must be in (0, 1], got {risk_fraction}"
            raise ValueError(msg)
        if short_leverage <= 0:
            msg = f"short_leverage must be > 0, got {short_leverage}"
            raise ValueError(msg)

        self.store = store
        self.risk_fraction = risk_fraction
        self.account_mode = (
            account_mode if isinstance(account_mode, AccountMode)
            else AccountMode(str(account_mode).lower())
        )
        self.short_leverage = short_leverage
        self.portfolio = self._load_or_init_portfolio(initial_cash)

    def _load_or_init_portfolio(self, initial_cash: float) -> Portfolio:
        """Load portfolio from store or create new one."""
        portfolio = self.store.load(self.PORTFOLIO_KEY, Portfolio)
        if portfolio:
            log.info(
                "Portfolio loaded from persistence",
                cash=portfolio.cash,
                positions=len(portfolio.positions),
            )
            return portfolio

        portfolio = Portfolio(cash=initial_cash)
        self.store.save(self.PORTFOLIO_KEY, portfolio)
        log.info("New portfolio initialized", cash=initial_cash)
        return portfolio

    def _persist_state(self, order: Order) -> None:
        """Persist portfolio and order atomically if possible."""
        if hasattr(self.store, "save_atomic"):
            self.store.save_atomic(
                {
                    self.PORTFOLIO_KEY: self.portfolio,
                    f"{self.ORDERS_KEY_PREFIX}{order.id}": order,
                }
            )
        else:
            self.store.save(self.PORTFOLIO_KEY, self.portfolio)
            self.store.save(f"{self.ORDERS_KEY_PREFIX}{order.id}", order)

    def _order_key(self, order_id: str) -> str:
        """Generate store key for order."""
        return f"{self.ORDERS_KEY_PREFIX}{order_id}"

    def get_order(self, order_id: str) -> Order | None:
        """Retrieve order by ID."""
        return self.store.load(self._order_key(order_id), Order)

    def calculate_order_quantity(
        self,
        current_price: float,
        quantity_modifier: float = 1.0,
    ) -> float:
        """Calculate order quantity based on risk parameters.

        Args:
            current_price: Current market price.
            quantity_modifier: Multiplier from signal (0.0-1.0).

        Returns:
            Order quantity based on risk fraction.
        """
        trade_value = self.portfolio.cash * self.risk_fraction * quantity_modifier
        return trade_value / current_price if current_price > 0 else 0.0

    def _short_collateral_required(self, quantity: float, current_price: float) -> float:
        """Estimate collateral required to open a short position."""
        notional = quantity * current_price
        return notional / self.short_leverage if self.short_leverage > 0 else notional

    def _has_pending_close_order(self, instrument_id: InstrumentId) -> bool:
        """Check whether there is already a pending close order for instrument."""
        for order in self.get_open_orders():
            if order.instrument_id != instrument_id:
                continue
            if order.intent in {OrderIntent.CLOSE_LONG, OrderIntent.CLOSE_SHORT}:
                return True
        return False

    def _build_order_request(self, order: Order) -> OrderRequest:
        """Convert internal order to immutable request payload."""
        return OrderRequest(
            oms_order_id=order.id,
            instrument_id=order.instrument_id,
            side=order.side,
            intent=order.intent,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            reduce_only=order.reduce_only,
        )

    def process_decision(
        self,
        decision: TradingDecision,
        current_price: float,
    ) -> OrderRequest | None:
        """Convert trading decision into executable order request."""
        if decision.action == SignalType.NEUTRAL:
            return None

        if decision.action == SignalType.EXIT:
            return self._process_exit(decision, current_price)

        quantity = self.calculate_order_quantity(current_price, decision.quantity_modifier)
        if quantity < MIN_QUANTITY:
            log.warning("Order rejected: quantity below minimum", quantity=quantity)
            return None

        reserved_cash: float | None = None
        reserved_quantity: float | None = None
        reduce_only = False

        if decision.action == SignalType.LONG:
            side = OrderSide.BUY
            intent = OrderIntent.OPEN_LONG

            cost = quantity * current_price
            if self.portfolio.cash < cost:
                log.warning(
                    "Order rejected: insufficient funds",
                    required=cost,
                    available=self.portfolio.cash,
                )
                return None

            self.portfolio.cash -= cost
            self.portfolio.blocked_cash += cost
            reserved_cash = cost

        elif decision.action == SignalType.SHORT:
            if self.account_mode == AccountMode.SPOT:
                log.warning(
                    "Order rejected: short opening not allowed in spot mode",
                    instrument=decision.instrument_id,
                )
                return None

            side = OrderSide.SELL
            intent = OrderIntent.OPEN_SHORT

            collateral = self._short_collateral_required(quantity, current_price)
            if self.portfolio.cash < collateral:
                log.warning(
                    "Order rejected: insufficient collateral for short",
                    required=collateral,
                    available=self.portfolio.cash,
                )
                return None

            self.portfolio.cash -= collateral
            self.portfolio.blocked_cash += collateral
            reserved_cash = collateral

        else:
            return None

        order = Order(
            decision_id=str(decision.decision_id),
            instrument_id=decision.instrument_id,
            side=side,
            intent=intent,
            quantity=quantity,
            order_type=OrderType.MARKET,
            status=OrderStatus.SUBMITTED,
            reserved_cash=reserved_cash,
            reserved_quantity=reserved_quantity,
            reserved_cash_remaining=reserved_cash,
            reserved_quantity_remaining=reserved_quantity,
            reduce_only=reduce_only,
        )

        self._persist_state(order)

        log.info(
            "Order created",
            order_id=order.id,
            side=side.value,
            intent=intent.value,
            quantity=quantity,
            instrument=decision.instrument_id,
        )

        return self._build_order_request(order)

    def _process_exit(
        self,
        decision: TradingDecision,
        current_price: float,
    ) -> OrderRequest | None:
        """Process EXIT signal by closing entire net position."""
        if self._has_pending_close_order(decision.instrument_id):
            log.info("Exit skipped: close order already pending", instrument=decision.instrument_id)
            return None

        position = self.portfolio.get_position(decision.instrument_id)
        if abs(position) < MIN_QUANTITY:
            log.info("No position to exit", instrument=decision.instrument_id)
            return None

        reserved_cash: float | None = None
        reserved_quantity: float | None = None

        if position > 0:
            # Close long: reserve inventory and sell with reduce_only.
            quantity = self.portfolio.get_available_position(decision.instrument_id)
            if quantity < MIN_QUANTITY:
                log.info("No available long inventory to exit", instrument=decision.instrument_id)
                return None

            self.portfolio.positions[decision.instrument_id] = max(0.0, position - quantity)
            current_blocked = self.portfolio.blocked_positions.get(decision.instrument_id, 0.0)
            self.portfolio.blocked_positions[decision.instrument_id] = current_blocked + quantity
            side = OrderSide.SELL
            intent = OrderIntent.CLOSE_LONG
            reserved_quantity = quantity
        else:
            # Close short: reserve buyback cash and buy with reduce_only.
            quantity = abs(position)
            buyback_estimate = quantity * current_price
            if self.portfolio.cash < buyback_estimate:
                log.warning(
                    "Exit rejected: insufficient cash to buy back short",
                    required=buyback_estimate,
                    available=self.portfolio.cash,
                )
                return None

            self.portfolio.cash -= buyback_estimate
            self.portfolio.blocked_cash += buyback_estimate
            side = OrderSide.BUY
            intent = OrderIntent.CLOSE_SHORT
            reserved_cash = buyback_estimate

        order = Order(
            decision_id=str(decision.decision_id),
            instrument_id=decision.instrument_id,
            side=side,
            intent=intent,
            quantity=quantity,
            order_type=OrderType.MARKET,
            status=OrderStatus.SUBMITTED,
            reserved_cash=reserved_cash,
            reserved_quantity=reserved_quantity,
            reserved_cash_remaining=reserved_cash,
            reserved_quantity_remaining=reserved_quantity,
            reduce_only=True,
        )

        self._persist_state(order)

        log.info(
            "Exit order created",
            order_id=order.id,
            side=side.value,
            intent=intent.value,
            quantity=quantity,
            instrument=decision.instrument_id,
        )

        return self._build_order_request(order)

    def confirm_execution(
        self,
        oms_order_id: str,
        fill_price: float,
        fill_qty: float,
        fee: float = 0.0,
        exchange_trade_id: str | None = None,
        exchange_order_id: str | None = None,
    ) -> None:
        """Confirm order execution and settle the trade.

        Releases pessimistic reservations and updates portfolio
        with actual execution results.

        Args:
            oms_order_id: OMS order ID.
            fill_price: Execution price.
            fill_qty: Filled quantity.
            fee: Transaction fee (deducted from proceeds/added to cost).
            exchange_trade_id: Optional exchange trade ID for idempotency.
            exchange_order_id: Optional exchange order identifier for reconciliation.

        Raises:
            OrderNotFoundError: If order not found.
            InvalidOrderStateError: If order in terminal state.
        """
        order = self.get_order(oms_order_id)
        if not order:
            log.error("Fill confirmation for unknown order", order_id=oms_order_id)
            raise OrderNotFoundError(oms_order_id)

        if order.is_terminal:
            log.warning(
                "Ignoring fill for terminal order",
                order_id=oms_order_id,
                status=order.status.value,
            )
            return

        if fill_price <= 0 or fill_qty <= 0:
            log.warning(
                "Ignoring invalid fill payload",
                order_id=oms_order_id,
                fill_price=fill_price,
                fill_qty=fill_qty,
            )
            return

        if exchange_trade_id and exchange_trade_id in order.applied_fill_ids:
            log.warning(
                "Ignoring duplicate fill",
                order_id=oms_order_id,
                exchange_trade_id=exchange_trade_id,
            )
            return

        if exchange_order_id and order.external_id is None:
            order.external_id = exchange_order_id

        remaining_before = order.remaining_quantity
        if remaining_before <= MIN_QUANTITY:
            log.warning(
                "Ignoring fill with no remaining quantity",
                order_id=oms_order_id,
            )
            return

        effective_fill_qty = min(fill_qty, remaining_before)
        if fill_qty > remaining_before + MIN_QUANTITY:
            log.warning(
                "Fill quantity exceeded remaining quantity; clamping",
                order_id=oms_order_id,
                requested=fill_qty,
                remaining=remaining_before,
                clamped=effective_fill_qty,
            )
        fill_qty = effective_fill_qty

        # Update order state
        prior_filled = order.filled_quantity
        order.filled_quantity += fill_qty
        order.average_fill_price = self._calculate_average_price(
            order.average_fill_price,
            prior_filled,
            fill_price,
            fill_qty,
        )
        order.updated_at = datetime.now(timezone.utc)

        if order.filled_quantity >= order.quantity - MIN_QUANTITY:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        # Settle based on intent
        if order.intent in {OrderIntent.OPEN_LONG, OrderIntent.CLOSE_SHORT}:
            self._settle_buy(order, fill_price, fill_qty, fee)
        else:
            self._settle_sell(order, fill_price, fill_qty, fee)

        if exchange_trade_id:
            order.applied_fill_ids.append(exchange_trade_id)

        self._persist_state(order)

        log.info(
            "Execution confirmed",
            order_id=oms_order_id,
            fill_price=fill_price,
            fill_qty=fill_qty,
            status=order.status.value,
            exchange_trade_id=exchange_trade_id,
        )

    def _settle_buy(
        self,
        order: Order,
        fill_price: float,
        fill_qty: float,
        fee: float,
    ) -> None:
        """Settle buy order execution."""
        reserved_total = order.reserved_cash or 0.0
        reserved_remaining = order.reserved_cash_remaining
        if reserved_remaining is None:
            reserved_remaining = reserved_total

        reserved_per_unit = (
            reserved_total / order.quantity
            if order.quantity > 0 and reserved_total > 0
            else fill_price
        )
        reserved_to_release = min(reserved_remaining, reserved_per_unit * fill_qty)
        actual_cost = (fill_qty * fill_price) + fee

        # Release only the reservation associated with this fill.
        self.portfolio.blocked_cash = max(
            0.0, self.portfolio.blocked_cash - reserved_to_release
        )

        # Adjust free cash by (released reservation - actual cost).
        self.portfolio.cash += reserved_to_release - actual_cost
        order.reserved_cash_remaining = max(0.0, reserved_remaining - reserved_to_release)

        # On final fill, release any tiny residual reservation (rounding).
        if order.status == OrderStatus.FILLED and (order.reserved_cash_remaining or 0.0) > 0.0:
            residual = order.reserved_cash_remaining or 0.0
            self.portfolio.blocked_cash = max(0.0, self.portfolio.blocked_cash - residual)
            self.portfolio.cash += residual
            order.reserved_cash_remaining = 0.0

        # Add to position
        current_pos = self.portfolio.positions.get(order.instrument_id, 0.0)
        self.portfolio.positions[order.instrument_id] = current_pos + fill_qty

    def _settle_sell(
        self,
        order: Order,
        fill_price: float,
        fill_qty: float,
        fee: float,
    ) -> None:
        """Settle sell order execution."""
        proceeds = (fill_qty * fill_price) - fee

        if order.intent == OrderIntent.OPEN_SHORT:
            reserved_total = order.reserved_cash or 0.0
            reserved_remaining = order.reserved_cash_remaining
            if reserved_remaining is None:
                reserved_remaining = reserved_total

            reserved_per_unit = (
                reserved_total / order.quantity
                if order.quantity > 0 and reserved_total > 0
                else 0.0
            )
            reserved_to_release = min(reserved_remaining, reserved_per_unit * fill_qty)

            self.portfolio.blocked_cash = max(
                0.0, self.portfolio.blocked_cash - reserved_to_release
            )
            self.portfolio.cash += reserved_to_release
            order.reserved_cash_remaining = max(0.0, reserved_remaining - reserved_to_release)

            if order.status == OrderStatus.FILLED and (order.reserved_cash_remaining or 0.0) > 0.0:
                residual = order.reserved_cash_remaining or 0.0
                self.portfolio.blocked_cash = max(0.0, self.portfolio.blocked_cash - residual)
                self.portfolio.cash += residual
                order.reserved_cash_remaining = 0.0

            current_pos = self.portfolio.positions.get(order.instrument_id, 0.0)
            self.portfolio.positions[order.instrument_id] = current_pos - fill_qty
            self.portfolio.cash += proceeds
            return

        # CLOSE_LONG path: release blocked inventory and add sale proceeds.
        reserved_remaining = order.reserved_quantity_remaining
        if reserved_remaining is None:
            reserved_remaining = order.reserved_quantity or order.quantity

        released_quantity = min(reserved_remaining, fill_qty)

        blocked = self.portfolio.blocked_positions.get(order.instrument_id, 0.0)
        self.portfolio.blocked_positions[order.instrument_id] = max(
            0.0, blocked - released_quantity
        )
        order.reserved_quantity_remaining = max(0.0, reserved_remaining - released_quantity)

        if (
            order.status == OrderStatus.FILLED
            and (order.reserved_quantity_remaining or 0.0) > 0.0
        ):
            residual = order.reserved_quantity_remaining or 0.0
            blocked_now = self.portfolio.blocked_positions.get(order.instrument_id, 0.0)
            self.portfolio.blocked_positions[order.instrument_id] = max(0.0, blocked_now - residual)
            order.reserved_quantity_remaining = 0.0

        self.portfolio.cash += proceeds

    def _calculate_average_price(
        self,
        current_avg: float | None,
        current_qty: float,
        new_price: float,
        new_qty: float,
    ) -> float:
        """Calculate volume-weighted average price."""
        if current_avg is None or current_qty == 0:
            return new_price

        total_value = (current_avg * current_qty) + (new_price * new_qty)
        total_qty = current_qty + new_qty
        return total_value / total_qty if total_qty > 0 else new_price

    def revert_allocation(self, oms_order_id: str) -> None:
        """Revert pessimistic reservations on order failure."""
        order = self.get_order(oms_order_id)
        if not order:
            log.error("Cannot revert unknown order", order_id=oms_order_id)
            raise OrderNotFoundError(oms_order_id)

        if order.is_terminal:
            log.warning(
                "Cannot revert terminal order",
                order_id=oms_order_id,
                status=order.status.value,
            )
            return

        if order.intent in {OrderIntent.OPEN_LONG, OrderIntent.OPEN_SHORT, OrderIntent.CLOSE_SHORT}:
            reserved = order.reserved_cash_remaining
            if reserved is None:
                reserved = order.reserved_cash or 0.0
            self.portfolio.cash += reserved
            self.portfolio.blocked_cash = max(0.0, self.portfolio.blocked_cash - reserved)
            order.reserved_cash_remaining = 0.0
        else:
            reserved = order.reserved_quantity_remaining
            if reserved is None:
                reserved = order.reserved_quantity or order.quantity
            blocked = self.portfolio.blocked_positions.get(order.instrument_id, 0.0)
            self.portfolio.blocked_positions[order.instrument_id] = max(0.0, blocked - reserved)
            current_pos = self.portfolio.positions.get(order.instrument_id, 0.0)
            self.portfolio.positions[order.instrument_id] = current_pos + reserved
            order.reserved_quantity_remaining = 0.0

        order.status = OrderStatus.FAILED
        order.updated_at = datetime.now(timezone.utc)

        self._persist_state(order)

        log.warning(
            "Order allocation reverted",
            order_id=oms_order_id,
            side=order.side.value,
            intent=order.intent.value,
        )

    def cancel_order(self, oms_order_id: str) -> bool:
        """Cancel an order and revert its reservations."""
        order = self.get_order(oms_order_id)
        if not order:
            raise OrderNotFoundError(oms_order_id)

        if order.is_terminal:
            return False

        if order.intent in {OrderIntent.OPEN_LONG, OrderIntent.OPEN_SHORT, OrderIntent.CLOSE_SHORT}:
            reserved = order.reserved_cash_remaining
            if reserved is None:
                reserved = order.reserved_cash or 0.0
            self.portfolio.cash += reserved
            self.portfolio.blocked_cash = max(0.0, self.portfolio.blocked_cash - reserved)
            order.reserved_cash_remaining = 0.0
        else:
            reserved = order.reserved_quantity_remaining
            if reserved is None:
                reserved = order.reserved_quantity or order.quantity
            blocked = self.portfolio.blocked_positions.get(order.instrument_id, 0.0)
            self.portfolio.blocked_positions[order.instrument_id] = max(0.0, blocked - reserved)
            current_pos = self.portfolio.positions.get(order.instrument_id, 0.0)
            self.portfolio.positions[order.instrument_id] = current_pos + reserved
            order.reserved_quantity_remaining = 0.0

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now(timezone.utc)

        self._persist_state(order)

        log.info("Order cancelled", order_id=oms_order_id, intent=order.intent.value)
        return True

    def _iter_order_keys(self) -> list[str]:
        """List persisted order keys across supported stores."""
        prefix = self.ORDERS_KEY_PREFIX

        # RedisStore path
        if hasattr(self.store, "client"):
            client = getattr(self.store, "client")
            try:
                return [key for key in client.scan_iter(match=f"{prefix}*")]
            except Exception as exc:
                log.warning("Order key scan failed on redis store", error=str(exc))
                return []

        # MemoryStore path
        if hasattr(self.store, "_data"):
            data = getattr(self.store, "_data", {})
            if isinstance(data, dict):
                return [key for key in data if key.startswith(prefix)]

        return []

    def get_open_orders(self) -> list[Order]:
        """Get all non-terminal orders.

        Note: This is an expensive operation requiring key scanning.
        Consider caching or maintaining an index for production use.
        """
        open_orders: list[Order] = []
        for key in self._iter_order_keys():
            order = self.store.load(key, Order)
            if order and not order.is_terminal:
                open_orders.append(order)
        return open_orders

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        """Best-effort float conversion for exchange payload fields."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def reconcile_with_exchange(
        self,
        exchange: Any,
    ) -> None:
        """Reconcile open orders against exchange state.

        This method complements local reconciliation by querying exchange order
        status for persisted non-terminal orders with known external IDs.
        """
        log.info("Exchange reconciliation started")

        open_orders = self.get_open_orders()
        terminal_statuses = {"closed", "filled", "canceled", "cancelled", "rejected", "expired"}

        for order in open_orders:
            if order.is_terminal or not order.external_id:
                continue

            symbol = str(order.instrument_id)
            try:
                remote_order = await exchange.fetch_order(order.external_id, symbol)
            except Exception as exc:
                log.warning(
                    "Exchange order fetch failed during reconciliation",
                    order_id=order.id,
                    external_id=order.external_id,
                    instrument=symbol,
                    error=str(exc),
                )
                continue

            if not isinstance(remote_order, dict):
                continue

            remote_status = str(remote_order.get("status") or "").lower()
            remote_filled = self._coerce_float(remote_order.get("filled")) or 0.0
            local_filled = order.filled_quantity

            # Apply missing fills using cumulative fill delta.
            if remote_filled > local_filled + MIN_QUANTITY:
                delta_fill = remote_filled - local_filled
                price_candidate = remote_order.get("average") or remote_order.get("price")
                fill_price = self._coerce_float(price_candidate)
                if fill_price is not None and fill_price > 0:
                    synthetic_fill_id = f"{order.external_id}:{remote_filled:.12f}"
                    self.confirm_execution(
                        oms_order_id=order.id,
                        fill_price=fill_price,
                        fill_qty=delta_fill,
                        fee=0.0,
                        exchange_trade_id=synthetic_fill_id,
                        exchange_order_id=order.external_id,
                    )
                else:
                    log.warning(
                        "Skipping remote fill application due to invalid price",
                        order_id=order.id,
                        external_id=order.external_id,
                        remote_price=price_candidate,
                        remote_filled=remote_filled,
                    )

            # If exchange reports order terminal, finalize local state.
            if remote_status in terminal_statuses:
                refreshed = self.get_order(order.id)
                if refreshed is None or refreshed.is_terminal:
                    continue

                if remote_status in {"closed", "filled"} and refreshed.remaining_quantity <= MIN_QUANTITY:
                    refreshed.status = OrderStatus.FILLED
                    refreshed.updated_at = datetime.now(timezone.utc)
                    self._persist_state(refreshed)
                else:
                    self.cancel_order(refreshed.id)

        # Always rebuild blocked reservations from resulting local open orders.
        self.reconcile()
        log.info("Exchange reconciliation completed")

    def reconcile(self) -> None:
        """Reconcile blocked reservations from persisted open orders.

        This recovery step rebuilds blocked cash/position amounts using
        open orders currently persisted in the store.
        """
        log.info("Portfolio reconciliation started")

        open_orders = self.get_open_orders()
        expected_blocked_cash = 0.0
        expected_blocked_positions: dict[InstrumentId, float] = {}

        for order in open_orders:
            if order.intent in {OrderIntent.OPEN_LONG, OrderIntent.OPEN_SHORT, OrderIntent.CLOSE_SHORT}:
                remaining_cash = order.reserved_cash_remaining
                if remaining_cash is None:
                    remaining_cash = order.reserved_cash or 0.0
                expected_blocked_cash += remaining_cash
            else:
                remaining_qty = order.reserved_quantity_remaining
                if remaining_qty is None:
                    remaining_qty = order.reserved_quantity or max(
                        0.0,
                        order.quantity - order.filled_quantity,
                    )
                if remaining_qty > MIN_QUANTITY:
                    expected_blocked_positions[order.instrument_id] = (
                        expected_blocked_positions.get(order.instrument_id, 0.0)
                        + remaining_qty
                    )

        self.portfolio.blocked_cash = max(0.0, expected_blocked_cash)
        self.portfolio.blocked_positions = {
            instrument: qty
            for instrument, qty in expected_blocked_positions.items()
            if qty > MIN_QUANTITY
        }

        self.store.save(self.PORTFOLIO_KEY, self.portfolio)

        log.info(
            "Portfolio reconciliation complete",
            open_orders=len(open_orders),
            blocked_cash=self.portfolio.blocked_cash,
            blocked_positions=len(self.portfolio.blocked_positions),
        )
