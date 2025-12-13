"""Data layer protocols and interfaces.

This module defines the contracts for data ingestion, transformation, and persistence.
We use Protocol (structural subtyping) over ABC for flexibility - any class that
implements the required methods is considered compatible, regardless of inheritance.

Architectural Decision:
    Protocol > ABC because:
    1. Supports duck typing (Pythonic)
    2. Works with third-party classes we don't control
    3. Better IDE/mypy inference
    4. No runtime overhead from inheritance checks
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import polars as pl

if TYPE_CHECKING:
    from qts_core.common.types import InstrumentId, Timeframe


# ==============================================================================
# OHLCV Schema Definition
# ==============================================================================
# Canonical schema for market data. All loaders must conform to this.
OHLCV_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("ms", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


# ==============================================================================
# Data Loader Protocol
# ==============================================================================
@runtime_checkable
class DataLoader(Protocol):
    """Protocol for data ingestion components.

    Implementations must provide a `load()` method that returns a Polars LazyFrame.
    The LazyFrame enables query optimization before execution.

    Implementations:
        - CCXTDataLoader: Real-time exchange data via CCXT
        - DuckDBDataLoader: Historical data from local DuckDB
        - ParquetLoader: Static Parquet files
    """

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.LazyFrame:
        """Load OHLCV data for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            timeframe: Candle timeframe (e.g., "1h", "4h", "1d").
            start_date: Start of data range (inclusive). None = earliest available.
            end_date: End of data range (inclusive). None = latest available.

        Returns:
            LazyFrame with columns: timestamp, open, high, low, close, volume.
            Schema must match OHLCV_SCHEMA.

        Raises:
            DataLoadError: If data cannot be fetched or parsed.
        """
        ...


@runtime_checkable
class AsyncDataLoader(Protocol):
    """Async variant of DataLoader for non-blocking I/O.

    Use this for live trading where we need to fetch data without
    blocking the event loop.
    """

    async def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.LazyFrame:
        """Async version of DataLoader.load()."""
        ...


# ==============================================================================
# Data Sink Protocol
# ==============================================================================
@runtime_checkable
class DataSink(Protocol):
    """Protocol for data persistence.

    Implementations:
        - ParquetSink: Write to Parquet files
        - DuckDBSink: Write to DuckDB tables
    """

    def write(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        destination: str,
    ) -> None:
        """Persist data to storage.

        Args:
            df: Data to persist.
            destination: Target path or table name.
        """
        ...


# ==============================================================================
# Data Transformer Protocol
# ==============================================================================
@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for data transformation pipelines.

    Transformers operate on LazyFrames to maintain query optimization.
    They should be composable (output of one can be input to another).
    """

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply transformation to a LazyFrame.

        Args:
            lf: Input LazyFrame.

        Returns:
            Transformed LazyFrame (new reference, not mutated).
        """
        ...


# ==============================================================================
# Custom Exceptions
# ==============================================================================
class DataLoadError(Exception):
    """Raised when data loading fails."""

    def __init__(self, message: str, source: str | None = None) -> None:
        """Initialize DataLoadError.

        Args:
            message: Error description.
            source: Data source that failed (optional).
        """
        self.source = source
        super().__init__(f"{message}" + (f" [source={source}]" if source else ""))


class SchemaValidationError(Exception):
    """Raised when data schema doesn't match expected format."""

    def __init__(
        self,
        message: str,
        missing_columns: list[str] | None = None,
        type_mismatches: list[str] | None = None,
    ) -> None:
        """Initialize SchemaValidationError.

        Args:
            message: Error description.
            missing_columns: Columns that were expected but not found.
            type_mismatches: Columns with incorrect data types.
        """
        self.missing_columns = missing_columns or []
        self.type_mismatches = type_mismatches or []
        super().__init__(message)
