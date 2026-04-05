"""Polymarket Order Management System for QTS-Architect.

Handles order lifecycle for Polymarket CLOB (Central Limit Order Book).
Supports limit orders (the only type Polymarket's CLOB accepts).

Requires:
    - Polymarket wallet setup (`polymarket setup`)
    - API credentials configured

Safety:
    - All orders go through QTS risk checks before submission
    - Paper mode by default — set `live=True` explicitly
    - Order size capped by max_position_usd
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

import structlog

log = structlog.get_logger()

# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_CLI_PATH: Final[str] = "/usr/local/bin/polymarket"
FALLBACK_CLI_PATH: Final[str] = str(
    Path.home() / "claw_workspace" / "polymarket"
)
DEFAULT_MAX_POSITION_USD: Final[float] = 100.0  # Conservative default


# ==============================================================================
# Enums
# ==============================================================================
class PolymarketSide(str, Enum):
    """Order side on Polymarket CLOB."""

    BUY = "BUY"
    SELL = "SELL"


class PolymarketOrderStatus(str, Enum):
    """Order lifecycle states."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PAPER = "PAPER"  # Simulated order (not sent to exchange)


# ==============================================================================
# Order Model
# ==============================================================================
@dataclass
class PolymarketOrder:
    """Represents an order on Polymarket CLOB."""

    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_id: str = ""
    side: PolymarketSide = PolymarketSide.BUY
    price: float = 0.0  # Limit price (0.01 - 0.99)
    size: float = 0.0  # Size in USDC
    status: PolymarketOrderStatus = PolymarketOrderStatus.PENDING
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    filled_at: datetime | None = None
    exchange_order_id: str | None = None
    market_question: str = ""  # For logging/display
    error: str | None = None

    @property
    def notional(self) -> float:
        """Total notional value of the order."""
        return self.size

    @property
    def potential_payout(self) -> float:
        """Maximum payout if the market resolves YES."""
        if self.side == PolymarketSide.BUY:
            return self.size / self.price if self.price > 0 else 0.0
        return self.size

    @property
    def max_loss(self) -> float:
        """Maximum possible loss."""
        if self.side == PolymarketSide.BUY:
            return self.size  # Can lose entire stake
        return self.potential_payout - self.size  # Short: lose if resolves YES


# ==============================================================================
# Order Management System
# ==============================================================================
class PolymarketOMS:
    """Order Management System for Polymarket.

    Manages order creation, submission, and tracking for the
    Polymarket CLOB. Paper mode by default for safety.

    Args:
        cli_path: Path to polymarket CLI binary.
        live: If False, orders are simulated (paper mode).
        max_position_usd: Maximum position size in USD.
        timeout: CLI command timeout in seconds.
    """

    def __init__(
        self,
        cli_path: str = DEFAULT_CLI_PATH,
        live: bool = False,
        max_position_usd: float = DEFAULT_MAX_POSITION_USD,
        timeout: float = 30.0,
    ) -> None:
        self.cli_path = cli_path
        self.live = live
        self.max_position_usd = max_position_usd
        self.timeout = timeout
        self._orders: dict[str, PolymarketOrder] = {}

        # Resolve CLI path
        if not Path(self.cli_path).exists():
            if Path(FALLBACK_CLI_PATH).exists():
                self.cli_path = FALLBACK_CLI_PATH

        mode = "LIVE" if live else "PAPER"
        log.info(
            "polymarket_oms_init",
            mode=mode,
            max_position_usd=max_position_usd,
        )

    # ------------------------------------------------------------------
    # Order Creation
    # ------------------------------------------------------------------
    def create_order(
        self,
        token_id: str,
        side: PolymarketSide,
        price: float,
        size: float,
        market_question: str = "",
    ) -> PolymarketOrder:
        """Create a new order (does not submit it).

        Args:
            token_id: The YES/NO token to trade.
            side: BUY or SELL.
            price: Limit price (0.01 to 0.99).
            size: Order size in USDC.
            market_question: Human-readable market name.

        Returns:
            Created order.

        Raises:
            ValueError: If order parameters are invalid.
        """
        # Validate price
        if not 0.01 <= price <= 0.99:
            raise ValueError(
                f"Price must be between 0.01 and 0.99, got {price}"
            )

        # Validate size
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        if size > self.max_position_usd:
            raise ValueError(
                f"Size ${size} exceeds max position ${self.max_position_usd}"
            )

        order = PolymarketOrder(
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            market_question=market_question,
        )

        self._orders[order.order_id] = order
        log.info(
            "polymarket_order_created",
            order_id=order.order_id[:8],
            side=side.value,
            price=price,
            size=size,
            market=market_question[:50] if market_question else "unknown",
            mode="PAPER" if not self.live else "LIVE",
        )
        return order

    # ------------------------------------------------------------------
    # Order Submission
    # ------------------------------------------------------------------
    async def submit_order(self, order_id: str) -> PolymarketOrder:
        """Submit an order to the exchange.

        In paper mode, immediately marks as PAPER (simulated fill).
        In live mode, sends to Polymarket CLOB via CLI.

        Args:
            order_id: The order to submit.

        Returns:
            Updated order with new status.

        Raises:
            KeyError: If order_id not found.
            RuntimeError: If submission fails.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")

        if not self.live:
            # Paper mode: simulate fill
            order.status = PolymarketOrderStatus.PAPER
            order.filled_at = datetime.now(timezone.utc)
            log.info(
                "polymarket_paper_fill",
                order_id=order.order_id[:8],
                side=order.side.value,
                price=order.price,
                size=order.size,
            )
            return order

        # Live mode: submit via CLI
        try:
            side_str = order.side.value
            cmd = [
                self.cli_path,
                "clob", "create-order",
                order.token_id,
                side_str,
                str(order.price),
                str(order.size),
                "--output", "json",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            if proc.returncode != 0:
                err = stderr.decode().strip()
                order.status = PolymarketOrderStatus.REJECTED
                order.error = err
                log.error(
                    "polymarket_order_rejected",
                    order_id=order.order_id[:8],
                    error=err,
                )
                return order

            result = json.loads(stdout.decode())
            order.status = PolymarketOrderStatus.SUBMITTED
            order.exchange_order_id = result.get("orderID", result.get("id"))
            log.info(
                "polymarket_order_submitted",
                order_id=order.order_id[:8],
                exchange_id=order.exchange_order_id,
            )
            return order

        except asyncio.TimeoutError:
            order.status = PolymarketOrderStatus.REJECTED
            order.error = "CLI timeout"
            return order

    # ------------------------------------------------------------------
    # Order Management
    # ------------------------------------------------------------------
    async def cancel_order(self, order_id: str) -> PolymarketOrder:
        """Cancel a pending/submitted order.

        Args:
            order_id: The order to cancel.

        Returns:
            Updated order.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found")

        if order.status in (
            PolymarketOrderStatus.FILLED,
            PolymarketOrderStatus.CANCELLED,
            PolymarketOrderStatus.PAPER,
        ):
            log.warning(
                "polymarket_cancel_noop",
                order_id=order_id[:8],
                status=order.status.value,
            )
            return order

        if self.live and order.exchange_order_id:
            try:
                cmd = [
                    self.cli_path,
                    "clob", "cancel-order",
                    order.exchange_order_id,
                    "--output", "json",
                ]
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
                if proc.returncode != 0:
                    err = stderr.decode().strip()
                    log.error(
                        "polymarket_cancel_failed",
                        order_id=order_id[:8],
                        exchange_id=order.exchange_order_id,
                        error=err,
                    )
                    order.error = f"Cancel failed: {err}"
                    return order
            except asyncio.TimeoutError:
                log.error(
                    "polymarket_cancel_timeout",
                    order_id=order_id[:8],
                    exchange_id=order.exchange_order_id,
                )
                order.error = "Cancel CLI timeout"
                return order

        order.status = PolymarketOrderStatus.CANCELLED
        log.info("polymarket_order_cancelled", order_id=order_id[:8])
        return order

    def get_order(self, order_id: str) -> PolymarketOrder | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self) -> list[PolymarketOrder]:
        """Get all open (non-terminal) orders."""
        return [
            o
            for o in self._orders.values()
            if o.status
            in (
                PolymarketOrderStatus.PENDING,
                PolymarketOrderStatus.SUBMITTED,
                PolymarketOrderStatus.PARTIALLY_FILLED,
            )
        ]

    def get_all_orders(self) -> list[PolymarketOrder]:
        """Get all orders."""
        return list(self._orders.values())

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------
    def total_exposure(self) -> float:
        """Total USD exposure across all active orders/positions."""
        return sum(
            o.size
            for o in self._orders.values()
            if o.status
            in (
                PolymarketOrderStatus.FILLED,
                PolymarketOrderStatus.PAPER,
                PolymarketOrderStatus.SUBMITTED,
            )
        )

    def remaining_capacity(self) -> float:
        """How much more USD can be deployed."""
        return max(0.0, self.max_position_usd - self.total_exposure())
