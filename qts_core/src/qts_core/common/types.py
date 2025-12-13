"""Domain primitives and value objects for QTS-Architect.

This module defines the immutable core types used throughout the system.
All domain objects follow DDD principles:
- Immutability (frozen=True)
- Strict validation at construction time
- No external dependencies (pure domain layer)

Architectural Decision:
    We deliberately exclude pandas/polars from domain types to maintain
    a clean boundary. Data transformation happens at the infrastructure layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Final, NewType, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from pydantic import ValidationInfo

# ==============================================================================
# Domain Primitives (NewTypes for type safety without runtime overhead)
# ==============================================================================
InstrumentId = NewType("InstrumentId", str)
"""Unique identifier for a tradable instrument. Format: EXCHANGE:SYMBOL (e.g., BINANCE:BTC-USDT)"""

# Price constraints (could be externalized to config, but these are market invariants)
MIN_TICK_SIZE: Final[float] = 1e-8  # Smallest price increment (satoshi-level for crypto)
MAX_REASONABLE_PRICE: Final[float] = 1e12  # Sanity check upper bound


# ==============================================================================
# Enumerations
# ==============================================================================
class SecurityType(str, Enum):
    """Classification of financial instruments.

    Using str mixin for JSON serialization compatibility.
    """

    CRYPTO = "CRYPTO"
    FOREX = "FOREX"
    EQUITY = "EQUITY"
    FUTURE = "FUTURE"
    OPTION = "OPTION"


class OrderSide(str, Enum):
    """Direction of the trade/signal."""

    BUY = "BUY"
    SELL = "SELL"


class Timeframe(str, Enum):
    """Standardized OHLCV timeframes.

    Follows CCXT conventions for exchange compatibility.
    """

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


# ==============================================================================
# Base Domain Model
# ==============================================================================
class DomainModel(BaseModel):
    """Base model for all domain objects.

    Design Decisions:
    - frozen=True: Immutability prevents accidental state mutation
    - strict=True: No implicit type coercion (fail fast)
    - extra="forbid": Catch typos and schema drift early
    - validate_assignment: Even updates go through validation
    """

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True,  # Serialize enums as values, not names
    )


# ==============================================================================
# Domain Entities
# ==============================================================================
class Instrument(DomainModel):
    """Represents a tradable asset.

    Attributes:
        id: Unique identifier (auto-generated if not provided).
        symbol: Ticker symbol normalized to uppercase.
        exchange: Exchange name normalized to uppercase.
        security_type: Asset classification.
    """

    id: InstrumentId = Field(
        default_factory=lambda: InstrumentId(str(uuid4())),
        description="Unique instrument identifier",
    )
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Ticker symbol (e.g., BTC-USDT)",
    )
    exchange: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Exchange name (e.g., BINANCE)",
    )
    security_type: SecurityType

    @field_validator("symbol", "exchange")
    @classmethod
    def normalize_to_uppercase(cls, v: str) -> str:
        """Normalize identifiers to uppercase for consistency."""
        return v.strip().upper()


class MarketData(DomainModel):
    """Represents a discrete market event (OHLCV bar).

    Invariants:
    - All prices must be positive
    - high >= max(open, close) >= min(open, close) >= low
    - Timestamp must be timezone-aware (UTC)

    Attributes:
        instrument_id: Reference to the traded instrument.
        timestamp: Bar timestamp in UTC.
        open: Opening price.
        high: Highest price in the period.
        low: Lowest price in the period.
        close: Closing price.
        volume: Traded volume (base currency units).
    """

    instrument_id: InstrumentId
    timestamp: datetime
    open: float = Field(..., gt=0, lt=MAX_REASONABLE_PRICE)  # noqa: A003
    high: float = Field(..., gt=0, lt=MAX_REASONABLE_PRICE)
    low: float = Field(..., gt=0, lt=MAX_REASONABLE_PRICE)
    close: float = Field(..., gt=0, lt=MAX_REASONABLE_PRICE)
    volume: float = Field(..., ge=0, description="Volume in base currency")

    @field_validator("timestamp")
    @classmethod
    def ensure_utc_timezone(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware and normalized to UTC.

        Raises:
            ValueError: If timestamp is naive (no timezone info).
        """
        if v.tzinfo is None:
            msg = "Timestamp must be timezone-aware. Received naive datetime."
            raise ValueError(msg)
        # Normalize to UTC for consistent comparisons
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_ohlc_consistency(self) -> Self:
        """Validate OHLC bar integrity.

        Market data invariant: high >= max(open, close) and low <= min(open, close)
        This catches data quality issues early in the pipeline.
        """
        if self.high < self.low:
            msg = f"Invalid OHLC: high ({self.high}) < low ({self.low})"
            raise ValueError(msg)

        if self.high < max(self.open, self.close):
            msg = f"Invalid OHLC: high ({self.high}) < max(open, close)"
            raise ValueError(msg)

        if self.low > min(self.open, self.close):
            msg = f"Invalid OHLC: low ({self.low}) > min(open, close)"
            raise ValueError(msg)

        return self


class Signal(DomainModel):
    """Represents a trading signal generated by an agent.

    Signals are the output of strategy agents and input to the risk layer.
    They are intentionally decoupled from orders (which are execution artifacts).

    Attributes:
        id: Unique signal identifier.
        instrument_id: Target instrument.
        timestamp: Signal generation time (UTC).
        side: Directional bias.
        strength: Confidence score normalized to [0, 1].
        metadata: Extensible key-value store for agent-specific data.
    """

    id: UUID = Field(default_factory=uuid4)  # noqa: A003
    instrument_id: InstrumentId
    timestamp: datetime
    side: OrderSide
    strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Signal confidence in [0, 1] range",
    )
    source_agent: str = Field(
        ...,
        min_length=1,
        description="Name of the agent that generated this signal",
    )
    metadata: dict[str, str | float | int | bool] = Field(
        default_factory=dict,
        description="Agent-specific contextual data",
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_utc_timezone(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            msg = "Signal timestamp must be timezone-aware."
            raise ValueError(msg)
        return v.astimezone(timezone.utc)
