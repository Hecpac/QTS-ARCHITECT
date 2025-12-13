"""SQL query engine for advanced data analysis.

Provides a zero-copy interface between Polars and DuckDB for complex
analytical queries that benefit from SQL's expressive power.

Architectural Decision:
    We use DuckDB for queries that are:
    1. Complex aggregations across multiple dimensions
    2. Window functions with custom frames
    3. Queries that are more natural to express in SQL

    For simple transformations, prefer Polars expressions (transform.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import polars as pl
import structlog

if TYPE_CHECKING:
    import pyarrow as pa

log = structlog.get_logger()


class QueryExecutionError(Exception):
    """Raised when SQL query execution fails."""

    def __init__(self, message: str, query: str | None = None) -> None:
        """Initialize QueryExecutionError.

        Args:
            message: Error description.
            query: The failed SQL query.
        """
        self.query = query
        super().__init__(message)


class DataQueryEngine:
    """Zero-copy SQL query interface between Polars and DuckDB.

    Uses Arrow as the interchange format to enable efficient data transfer
    without serialization overhead.

    Example:
        ```python
        engine = DataQueryEngine()
        result = engine.query(
            lf,
            "SELECT symbol, AVG(close) as avg_close FROM data GROUP BY symbol",
            table_name="data"
        )
        ```

    Attributes:
        conn: DuckDB connection (in-memory by default).
    """

    def __init__(self, database: str = ":memory:") -> None:
        """Initialize query engine.

        Args:
            database: DuckDB database path. Defaults to in-memory.
        """
        self.conn: duckdb.DuckDBPyConnection = duckdb.connect(database=database)
        self._registered_tables: set[str] = set()

    def query(
        self,
        lf: pl.LazyFrame,
        sql_query: str,
        table_name: str = "dataset",
    ) -> pl.DataFrame:
        """Execute SQL query against a Polars LazyFrame.

        The LazyFrame is materialized to an Arrow table, which DuckDB
        can query directly (zero-copy). Results are returned as a
        Polars DataFrame.

        Args:
            lf: Source data as LazyFrame.
            sql_query: SQL query string. Reference data using `table_name`.
            table_name: Table alias in the SQL query.

        Returns:
            Query results as Polars DataFrame.

        Raises:
            QueryExecutionError: If query execution fails.
        """
        log.debug("Materializing LazyFrame for SQL", table=table_name)

        # Collect and convert to Arrow (zero-copy from Polars memory)
        try:
            arrow_table: pa.Table = lf.collect().to_arrow()
        except Exception as e:
            raise QueryExecutionError(
                f"Failed to materialize LazyFrame: {e}",
                query=sql_query,
            ) from e

        # Register Arrow table in DuckDB (zero-copy view)
        self.conn.register(table_name, arrow_table)
        self._registered_tables.add(table_name)

        try:
            log.debug(
                "Executing SQL query",
                query_preview=sql_query[:100] + "..." if len(sql_query) > 100 else sql_query,
            )

            result = self.conn.execute(sql_query).pl()

            log.debug("Query completed", rows=len(result))
            return result

        except duckdb.Error as e:
            log.error("SQL query failed", error=str(e), query=sql_query)
            raise QueryExecutionError(f"DuckDB error: {e}", query=sql_query) from e

        finally:
            # Clean up to avoid memory leaks and state pollution
            self._unregister_table(table_name)

    def query_multiple(
        self,
        tables: dict[str, pl.LazyFrame],
        sql_query: str,
    ) -> pl.DataFrame:
        """Execute SQL query against multiple LazyFrames.

        Useful for JOINs between different datasets.

        Args:
            tables: Dictionary mapping table names to LazyFrames.
            sql_query: SQL query referencing the table names.

        Returns:
            Query results as Polars DataFrame.

        Example:
            ```python
            result = engine.query_multiple(
                {"prices": prices_lf, "volumes": volumes_lf},
                "SELECT p.*, v.volume FROM prices p JOIN volumes v ON p.ts = v.ts"
            )
            ```
        """
        # Register all tables
        for name, lf in tables.items():
            arrow_table = lf.collect().to_arrow()
            self.conn.register(name, arrow_table)
            self._registered_tables.add(name)

        try:
            result = self.conn.execute(sql_query).pl()
            return result
        finally:
            # Clean up all registered tables
            for name in list(self._registered_tables):
                self._unregister_table(name)

    def _unregister_table(self, table_name: str) -> None:
        """Safely unregister a table from DuckDB."""
        try:
            self.conn.unregister(table_name)
            self._registered_tables.discard(table_name)
        except duckdb.Error:
            # Table might not exist, ignore
            pass

    def explain(self, lf: pl.LazyFrame) -> str:
        """Get Polars query execution plan.

        Useful for debugging and optimization.

        Args:
            lf: LazyFrame to analyze.

        Returns:
            String representation of the query plan.
        """
        return lf.explain()

    def close(self) -> None:
        """Close DuckDB connection and clean up resources."""
        for table_name in list(self._registered_tables):
            self._unregister_table(table_name)
        self.conn.close()
        log.debug("Query engine closed")

    def __enter__(self) -> DataQueryEngine:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
