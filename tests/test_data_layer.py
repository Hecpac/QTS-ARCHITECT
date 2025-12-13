"""Tests for the data layer components.

Tests cover:
- Schema validation
- Feature transformers
- Query engine
"""

from datetime import datetime, timezone

import polars as pl
import pytest

from qts_core.data import (
    OHLCV_SCHEMA,
    DataQueryEngine,
    DropNullTransformer,
    OHLCVFeatureTransformer,
    SchemaValidationError,
    atr,
    log_returns,
    rolling_volatility,
    rsi,
    validate_schema,
)


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def sample_ohlcv_df() -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 23, tzinfo=timezone.utc),
                interval="1h",
                eager=True,
            ),
            "open": [100.0 + i for i in range(24)],
            "high": [101.0 + i for i in range(24)],
            "low": [99.0 + i for i in range(24)],
            "close": [100.5 + i for i in range(24)],
            "volume": [1000.0 + i * 10 for i in range(24)],
        }
    )


@pytest.fixture
def sample_ohlcv_lf(sample_ohlcv_df: pl.DataFrame) -> pl.LazyFrame:
    """Create sample OHLCV LazyFrame."""
    return sample_ohlcv_df.lazy()


# ==============================================================================
# Schema Validation Tests
# ==============================================================================
class TestSchemaValidation:
    """Tests for schema validation."""

    def test_valid_schema_passes(self, sample_ohlcv_lf: pl.LazyFrame) -> None:
        """Valid OHLCV schema should pass validation."""
        # Should not raise
        validate_schema(sample_ohlcv_lf)

    def test_missing_column_fails(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Missing required column should fail validation."""
        lf = sample_ohlcv_df.drop("volume").lazy()

        with pytest.raises(SchemaValidationError) as exc_info:
            validate_schema(lf)

        assert "volume" in exc_info.value.missing_columns

    def test_ohlcv_schema_has_required_columns(self) -> None:
        """OHLCV schema should have all required columns."""
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        assert set(OHLCV_SCHEMA.keys()) == required


# ==============================================================================
# Feature Expression Tests
# ==============================================================================
class TestFeatureExpressions:
    """Tests for individual feature expressions."""

    def test_log_returns(self, sample_ohlcv_lf: pl.LazyFrame) -> None:
        """Log returns calculation."""
        result = sample_ohlcv_lf.select(log_returns()).collect()

        assert "returns" in result.columns
        assert result["returns"].null_count() == 1  # First value is null
        assert result["returns"].drop_nulls().len() == 23

    def test_rolling_volatility(self, sample_ohlcv_lf: pl.LazyFrame) -> None:
        """Rolling volatility calculation."""
        result = sample_ohlcv_lf.select(rolling_volatility(window=5)).collect()

        assert "volatility" in result.columns
        # First 5 values should be null (window size)
        assert result["volatility"].null_count() >= 4

    def test_rsi_bounded(self, sample_ohlcv_lf: pl.LazyFrame) -> None:
        """RSI should be bounded between 0 and 100."""
        result = sample_ohlcv_lf.select(rsi(window=5)).collect()

        non_null_rsi = result["rsi"].drop_nulls()
        assert non_null_rsi.min() >= 0
        assert non_null_rsi.max() <= 100

    def test_atr_positive(self, sample_ohlcv_lf: pl.LazyFrame) -> None:
        """ATR should always be positive."""
        result = sample_ohlcv_lf.select(
            [pl.col("high"), pl.col("low"), pl.col("close"), atr(window=5)]
        ).collect()

        non_null_atr = result["atr"].drop_nulls()
        assert (non_null_atr >= 0).all()


# ==============================================================================
# Transformer Tests
# ==============================================================================
class TestTransformers:
    """Tests for transformer classes."""

    def test_ohlcv_feature_transformer(self, sample_ohlcv_lf: pl.LazyFrame) -> None:
        """OHLCVFeatureTransformer should add expected columns."""
        transformer = OHLCVFeatureTransformer(
            volatility_window=5,
            volume_window=5,
            include_rsi=True,
            include_atr=True,
        )

        result = transformer.transform(sample_ohlcv_lf).collect()

        # Original columns preserved
        assert "timestamp" in result.columns
        assert "close" in result.columns

        # New features added
        assert "returns" in result.columns
        assert "volatility" in result.columns
        assert "volume_ma_ratio" in result.columns
        assert "rsi" in result.columns
        assert "atr" in result.columns

    def test_drop_null_transformer(self, sample_ohlcv_lf: pl.LazyFrame) -> None:
        """DropNullTransformer should remove rows with nulls."""
        # Add a feature that creates nulls
        lf_with_nulls = sample_ohlcv_lf.select([pl.all(), log_returns()])

        transformer = DropNullTransformer(subset=["returns"])
        result = transformer.transform(lf_with_nulls).collect()

        # No nulls in returns column
        assert result["returns"].null_count() == 0
        # One row removed (first row with null return)
        assert len(result) == 23


# ==============================================================================
# Query Engine Tests
# ==============================================================================
class TestQueryEngine:
    """Tests for DataQueryEngine."""

    def test_simple_query(self, sample_ohlcv_lf: pl.LazyFrame) -> None:
        """Simple SQL query should work."""
        with DataQueryEngine() as engine:
            result = engine.query(
                sample_ohlcv_lf,
                "SELECT AVG(close) as avg_close FROM data",
                table_name="data",
            )

        assert "avg_close" in result.columns
        assert len(result) == 1
        assert result["avg_close"][0] > 0

    def test_query_with_groupby(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Query with GROUP BY should work."""
        # Add a category column for grouping
        df = sample_ohlcv_df.with_columns(
            pl.when(pl.col("close") < 110)
            .then(pl.lit("low"))
            .otherwise(pl.lit("high"))
            .alias("category")
        )

        with DataQueryEngine() as engine:
            result = engine.query(
                df.lazy(),
                "SELECT category, COUNT(*) as cnt FROM data GROUP BY category",
                table_name="data",
            )

        assert len(result) == 2
        assert set(result["category"].to_list()) == {"low", "high"}

    def test_explain_returns_string(self, sample_ohlcv_lf: pl.LazyFrame) -> None:
        """explain() should return query plan as string."""
        with DataQueryEngine() as engine:
            plan = engine.explain(sample_ohlcv_lf)

        assert isinstance(plan, str)
        assert len(plan) > 0
