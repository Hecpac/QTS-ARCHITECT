"""Data layer for QTS-Architect.

This module provides components for:
- Loading market data from various sources (exchanges, files, databases)
- Transforming raw OHLCV data into ML-ready features
- Executing complex analytical queries

Quick Start:
    ```python
    from qts_core.data import CCXTDataLoader, OHLCVFeatureTransformer

    # Load data
    loader = CCXTDataLoader(exchange_id="binance")
    lf = loader.load("BTC/USDT", "1h")

    # Add features
    transformer = OHLCVFeatureTransformer()
    features = transformer.transform(lf)

    # Execute
    df = features.collect()
    ```
"""

from qts_core.data.interfaces import (
    OHLCV_SCHEMA,
    AsyncDataLoader,
    DataLoadError,
    DataLoader,
    DataSink,
    DataTransformer,
    SchemaValidationError,
)
from qts_core.data.loaders import (
    CCXTDataLoader,
    DuckDBDataLoader,
    ParquetLoader,
    validate_schema,
)
from qts_core.data.query import DataQueryEngine, QueryExecutionError
from qts_core.data.transform import (
    DropNullTransformer,
    NormalizeTransformer,
    OHLCVFeatureTransformer,
    atr,
    bollinger_bands,
    compose_transformers,
    log_returns,
    rolling_volatility,
    rsi,
    simple_returns,
    true_range,
    volume_ma_ratio,
)

__all__ = [
    # Protocols
    "DataLoader",
    "AsyncDataLoader",
    "DataSink",
    "DataTransformer",
    # Schema
    "OHLCV_SCHEMA",
    # Loaders
    "CCXTDataLoader",
    "DuckDBDataLoader",
    "ParquetLoader",
    "validate_schema",
    # Query
    "DataQueryEngine",
    # Transformers
    "OHLCVFeatureTransformer",
    "DropNullTransformer",
    "NormalizeTransformer",
    "compose_transformers",
    # Feature expressions
    "log_returns",
    "simple_returns",
    "rolling_volatility",
    "volume_ma_ratio",
    "true_range",
    "atr",
    "rsi",
    "bollinger_bands",
    # Exceptions
    "DataLoadError",
    "SchemaValidationError",
    "QueryExecutionError",
]
