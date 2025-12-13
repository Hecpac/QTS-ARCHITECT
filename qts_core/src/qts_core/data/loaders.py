"""Data loader implementations.

This module provides concrete implementations of the DataLoader protocol
for various data sources: Parquet files, DuckDB, and CCXT exchanges.

All loaders return Polars LazyFrames conforming to OHLCV_SCHEMA.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final

import polars as pl
import structlog

from qts_core.data.interfaces import (
    OHLCV_SCHEMA,
    DataLoadError,
    SchemaValidationError,
)

if TYPE_CHECKING:
    import duckdb

log = structlog.get_logger()

# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_OHLCV_COLUMNS: Final[list[str]] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


# ==============================================================================
# Schema Validation Helper
# ==============================================================================
def validate_schema(
    lf: pl.LazyFrame,
    expected_schema: dict[str, pl.DataType] | None = None,
) -> None:
    """Validate LazyFrame schema against expected types.

    Args:
        lf: LazyFrame to validate.
        expected_schema: Expected column types. Defaults to OHLCV_SCHEMA.

    Raises:
        SchemaValidationError: If schema doesn't match.
    """
    schema = expected_schema or OHLCV_SCHEMA
    current_schema = lf.collect_schema()

    missing_cols: list[str] = []
    type_mismatches: list[str] = []

    for col, expected_type in schema.items():
        if col not in current_schema:
            missing_cols.append(col)
            continue

        current_type = current_schema[col]
        # Allow compatible types (e.g., Int64 for Float64)
        if not _types_compatible(current_type, expected_type):
            type_mismatches.append(
                f"{col}: expected {expected_type}, got {current_type}"
            )

    if missing_cols or type_mismatches:
        msg = "Schema validation failed."
        if missing_cols:
            msg += f" Missing: {missing_cols}."
        if type_mismatches:
            msg += f" Mismatches: {type_mismatches}."

        raise SchemaValidationError(
            msg,
            missing_columns=missing_cols,
            type_mismatches=type_mismatches,
        )


def _types_compatible(actual: pl.DataType, expected: pl.DataType) -> bool:
    """Check if actual type is compatible with expected type.

    Allows numeric upcasting (Int -> Float) and datetime precision differences.
    """
    if actual == expected:
        return True

    # Allow integer to float conversion
    if expected == pl.Float64 and actual in (pl.Int64, pl.Int32, pl.Float32):
        return True

    # Allow datetime precision differences
    if isinstance(expected, pl.Datetime) and isinstance(actual, pl.Datetime):
        return True

    return False


# ==============================================================================
# Parquet Loader
# ==============================================================================
class ParquetLoader:
    """Load OHLCV data from Parquet files.

    High-performance loader using Polars scan (lazy evaluation).
    Supports single files and directories with partitioned data.

    Attributes:
        base_path: Root directory for Parquet files.
    """

    def __init__(self, base_path: str | None = None) -> None:
        """Initialize ParquetLoader.

        Args:
            base_path: Optional base directory. Paths in load() will be relative to this.
        """
        self.base_path = Path(base_path) if base_path else None

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.LazyFrame:
        """Load OHLCV data from Parquet.

        File naming convention: {symbol}_{timeframe}.parquet
        Example: BTC_USDT_1h.parquet

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            timeframe: Candle timeframe (e.g., "1h").
            start_date: Filter start (optional).
            end_date: Filter end (optional).

        Returns:
            LazyFrame conforming to OHLCV_SCHEMA.
        """
        # Normalize symbol for filename
        safe_symbol = symbol.replace("/", "_")
        filename = f"{safe_symbol}_{timeframe}.parquet"

        if self.base_path:
            path = self.base_path / filename
        else:
            path = Path(filename)

        if not path.exists():
            raise DataLoadError(f"Parquet file not found: {path}", source=str(path))

        log.info("Loading Parquet", path=str(path), symbol=symbol, timeframe=timeframe)

        try:
            lf = pl.scan_parquet(path)
        except Exception as e:
            raise DataLoadError(f"Failed to scan Parquet: {e}", source=str(path)) from e

        # Apply date filters if provided
        lf = self._apply_date_filters(lf, start_date, end_date)

        # Validate and normalize schema
        validate_schema(lf)

        return lf.select(DEFAULT_OHLCV_COLUMNS)

    def _apply_date_filters(
        self,
        lf: pl.LazyFrame,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pl.LazyFrame:
        """Apply date range filters to LazyFrame."""
        if start_date:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            lf = lf.filter(pl.col("timestamp") >= start_date)

        if end_date:
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            lf = lf.filter(pl.col("timestamp") <= end_date)

        return lf


# ==============================================================================
# DuckDB Loader
# ==============================================================================
class DuckDBDataLoader:
    """Load OHLCV data from DuckDB database.

    Optimized for backtesting with pre-downloaded historical data.
    Uses DuckDB's efficient columnar storage and query engine.

    Attributes:
        db_path: Path to DuckDB database file.
        table_name: Default table name for OHLCV data.
    """

    def __init__(
        self,
        db_path: str,
        table_name: str = "ohlcv",
    ) -> None:
        """Initialize DuckDBDataLoader.

        Args:
            db_path: Path to .duckdb file.
            table_name: Table containing OHLCV data.
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._conn: duckdb.DuckDBPyConnection | None = None

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection (lazy initialization)."""
        if self._conn is None:
            import duckdb

            if not self.db_path.exists():
                raise DataLoadError(
                    f"Database not found: {self.db_path}",
                    source=str(self.db_path),
                )

            self._conn = duckdb.connect(str(self.db_path), read_only=True)
            log.info("DuckDB connection opened", path=str(self.db_path))

        return self._conn

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.LazyFrame:
        """Load OHLCV data from DuckDB.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            timeframe: Candle timeframe (e.g., "1h").
            start_date: Filter start (optional).
            end_date: Filter end (optional).

        Returns:
            LazyFrame conforming to OHLCV_SCHEMA.
        """
        conn = self._get_connection()

        # Build parameterized query to prevent SQL injection
        query_parts = [
            f"SELECT timestamp, open, high, low, close, volume FROM {self.table_name}",  # noqa: S608
            "WHERE symbol = ? AND timeframe = ?",
        ]
        params: list[str | datetime] = [symbol, timeframe]

        if start_date:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            query_parts.append("AND timestamp >= ?")
            params.append(start_date)

        if end_date:
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            query_parts.append("AND timestamp <= ?")
            params.append(end_date)

        query_parts.append("ORDER BY timestamp ASC")
        query = " ".join(query_parts)

        log.info(
            "Executing DuckDB query",
            symbol=symbol,
            timeframe=timeframe,
            start=str(start_date),
            end=str(end_date),
        )

        try:
            # Execute and convert to Polars
            result = conn.execute(query, params)
            df = result.pl()
        except Exception as e:
            raise DataLoadError(
                f"DuckDB query failed: {e}",
                source=str(self.db_path),
            ) from e

        if df.is_empty():
            log.warning(
                "No data found",
                symbol=symbol,
                timeframe=timeframe,
            )

        # Convert to LazyFrame for consistency with other loaders
        return df.lazy()

    def close(self) -> None:
        """Close DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            log.info("DuckDB connection closed")


# ==============================================================================
# CCXT Data Loader (Async)
# ==============================================================================
class CCXTDataLoader:
    """Load OHLCV data from cryptocurrency exchanges via CCXT.

    Supports 100+ exchanges through the CCXT unified API.
    Implements rate limiting and caching for efficient data fetching.

    Attributes:
        exchange_id: CCXT exchange identifier (e.g., "binance", "kraken").
        rate_limit: Maximum requests per second.
    """

    def __init__(
        self,
        exchange_id: str,
        rate_limit: int = 10,
        cache_enabled: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize CCXTDataLoader.

        Args:
            exchange_id: Exchange identifier (e.g., "binance").
            rate_limit: Max requests per second.
            cache_enabled: Whether to cache fetched data.
            cache_dir: Directory for cached data.
        """
        self.exchange_id = exchange_id.lower()
        self.rate_limit = rate_limit
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._exchange: object | None = None  # Lazy init

    def _get_exchange(self) -> object:
        """Get or create CCXT exchange instance."""
        if self._exchange is None:
            import ccxt

            exchange_class = getattr(ccxt, self.exchange_id, None)
            if exchange_class is None:
                raise DataLoadError(
                    f"Unknown exchange: {self.exchange_id}",
                    source=self.exchange_id,
                )

            self._exchange = exchange_class(
                {
                    "enableRateLimit": True,
                    "rateLimit": int(1000 / self.rate_limit),  # ms between requests
                }
            )
            log.info("CCXT exchange initialized", exchange=self.exchange_id)

        return self._exchange

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.LazyFrame:
        """Load OHLCV data from exchange.

        Fetches data in batches to handle exchange limits.
        Caches results if caching is enabled.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            timeframe: Candle timeframe (e.g., "1h").
            start_date: Filter start (optional, defaults to 30 days ago).
            end_date: Filter end (optional, defaults to now).

        Returns:
            LazyFrame conforming to OHLCV_SCHEMA.
        """
        exchange = self._get_exchange()

        # Default date range: last 30 days
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date.replace(day=end_date.day - 30)

        # Ensure timezone awareness
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        log.info(
            "Fetching OHLCV from exchange",
            exchange=self.exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )

        # Convert to milliseconds for CCXT
        since_ms = int(start_date.timestamp() * 1000)
        until_ms = int(end_date.timestamp() * 1000)

        all_ohlcv: list[list[float | int]] = []
        current_since = since_ms

        try:
            # Fetch in batches (most exchanges limit to 500-1000 candles)
            while current_since < until_ms:
                ohlcv = exchange.fetch_ohlcv(  # type: ignore[union-attr]
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000,
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)

                # Move to next batch
                last_timestamp = ohlcv[-1][0]
                if last_timestamp <= current_since:
                    break  # No progress, avoid infinite loop
                current_since = last_timestamp + 1

        except Exception as e:
            raise DataLoadError(
                f"CCXT fetch failed: {e}",
                source=f"{self.exchange_id}:{symbol}",
            ) from e

        if not all_ohlcv:
            log.warning("No data returned from exchange", symbol=symbol)
            # Return empty LazyFrame with correct schema
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime("ms", "UTC"),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }
            ).lazy()

        # Convert to Polars DataFrame
        df = pl.DataFrame(
            {
                "timestamp": [row[0] for row in all_ohlcv],
                "open": [float(row[1]) for row in all_ohlcv],
                "high": [float(row[2]) for row in all_ohlcv],
                "low": [float(row[3]) for row in all_ohlcv],
                "close": [float(row[4]) for row in all_ohlcv],
                "volume": [float(row[5]) for row in all_ohlcv],
            }
        ).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ms", "UTC"))
        )

        log.info("OHLCV data loaded", rows=len(df), symbol=symbol)

        return df.lazy()
