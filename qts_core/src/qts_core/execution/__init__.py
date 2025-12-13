"""Execution Layer for QTS-Architect.

This module provides the execution infrastructure for the trading system:
- Order Management System (OMS): Portfolio state and order lifecycle
- Execution Management System (EMS): Exchange connectivity and order routing
- State Store: Persistence layer for portfolio and order state

Architecture:
    TradingDecision -> OMS (process_decision) -> OrderRequest
    OrderRequest -> EMS (submit_order) -> FillReport
    FillReport -> OMS (confirm_execution) -> Portfolio Update

Example:
    ```python
    from qts_core.execution import (
        OrderManagementSystem,
        MockGateway,
        MemoryStore,
    )

    # Initialize components
    store = MemoryStore()
    oms = OrderManagementSystem(store, initial_cash=100_000)
    gateway = MockGateway()

    # Process decision
    order_request = oms.process_decision(decision, current_price=50_000)
    if order_request:
        fill = await gateway.submit_order(order_request)
        if fill:
            oms.confirm_execution(
                fill.oms_order_id,
                fill.price,
                fill.quantity,
                fill.fee,
            )
    ```
"""

from qts_core.execution.ems import (
    CCXTGateway,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    EMSError,
    ExecutionError,
    ExecutionGateway,
    ExecutionResult,
    ExecutionStatus,
    FillReport,
    GatewayNotStartedError,
    MockGateway,
    RateLimiter,
    RateLimitError,
)
from qts_core.execution.oms import (
    DEFAULT_INITIAL_CASH,
    DEFAULT_RISK_FRACTION,
    FillEvent,
    InsufficientFundsError,
    InsufficientPositionError,
    InvalidOrderStateError,
    MIN_QUANTITY,
    OMSError,
    Order,
    OrderManagementSystem,
    OrderNotFoundError,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Portfolio,
    TimeInForce,
)
from qts_core.execution.store import (
    MemoryStore,
    RedisStore,
    StateStore,
    StoreConnectionError,
    TransactionError,
    create_store,
    get_store,
)

__all__ = [
    # Store
    "StateStore",
    "RedisStore",
    "MemoryStore",
    "create_store",
    "get_store",
    "StoreConnectionError",
    "TransactionError",
    # OMS Enums
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",
    # OMS Models
    "Order",
    "Portfolio",
    "OrderRequest",
    "FillEvent",
    # OMS
    "OrderManagementSystem",
    "OMSError",
    "InsufficientFundsError",
    "InsufficientPositionError",
    "OrderNotFoundError",
    "InvalidOrderStateError",
    "DEFAULT_INITIAL_CASH",
    "DEFAULT_RISK_FRACTION",
    "MIN_QUANTITY",
    # EMS Enums
    "CircuitState",
    "ExecutionStatus",
    # EMS Models
    "FillReport",
    "ExecutionResult",
    # EMS Components
    "CircuitBreaker",
    "RateLimiter",
    # EMS Protocol
    "ExecutionGateway",
    # EMS Implementations
    "CCXTGateway",
    "MockGateway",
    # EMS Exceptions
    "EMSError",
    "GatewayNotStartedError",
    "CircuitOpenError",
    "RateLimitError",
    "ExecutionError",
]
