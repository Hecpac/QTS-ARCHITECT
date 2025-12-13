"""Data transformation utilities for feature engineering.

This module provides composable transformers that operate on Polars LazyFrames.
All transformers maintain lazy evaluation for query optimization.

Design Principles:
- Pure functions where possible (no side effects)
- LazyFrame in, LazyFrame out (composable pipeline)
- Vectorized operations (no Python loops in hot paths)
"""

from __future__ import annotations

from typing import Final

import polars as pl
import structlog

log = structlog.get_logger()

# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_RETURNS_COLUMN: Final[str] = "returns"
DEFAULT_VOLATILITY_COLUMN: Final[str] = "volatility"


# ==============================================================================
# Feature Expressions
# ==============================================================================
def log_returns(column: str = "close") -> pl.Expr:
    """Calculate log returns.

    Log returns are preferred for:
    - Time additivity (sum of log returns = total log return)
    - Better statistical properties (closer to normal distribution)
    - Numerical stability for small changes

    Args:
        column: Price column name.

    Returns:
        Polars expression for log returns.
    """
    return (pl.col(column) / pl.col(column).shift(1)).log().alias(DEFAULT_RETURNS_COLUMN)


def simple_returns(column: str = "close") -> pl.Expr:
    """Calculate simple percentage returns.

    Args:
        column: Price column name.

    Returns:
        Polars expression for simple returns.
    """
    return (pl.col(column).pct_change()).alias("simple_returns")


def rolling_volatility(column: str = "close", window: int = 20) -> pl.Expr:
    """Calculate rolling volatility (standard deviation of returns).

    Args:
        column: Price column name.
        window: Rolling window size.

    Returns:
        Polars expression for rolling volatility.
    """
    return (
        (pl.col(column) / pl.col(column).shift(1))
        .log()
        .rolling_std(window_size=window)
        .alias(DEFAULT_VOLATILITY_COLUMN)
    )


def volume_ma_ratio(window: int = 20) -> pl.Expr:
    """Calculate volume relative to moving average.

    Useful for detecting unusual volume spikes.

    Args:
        window: Moving average window.

    Returns:
        Polars expression for volume/MA(volume) ratio.
    """
    return (pl.col("volume") / pl.col("volume").rolling_mean(window_size=window)).alias(
        "volume_ma_ratio"
    )


def true_range() -> pl.Expr:
    """Calculate True Range (TR) for volatility measurement.

    TR = max(high - low, |high - prev_close|, |low - prev_close|)

    Returns:
        Polars expression for True Range.
    """
    prev_close = pl.col("close").shift(1)
    return (
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        )
    ).alias("true_range")


def atr(window: int = 14) -> pl.Expr:
    """Calculate Average True Range (ATR).

    ATR is used for:
    - Position sizing
    - Stop-loss placement
    - Volatility filtering

    Args:
        window: Smoothing window (typically 14).

    Returns:
        Polars expression for ATR.
    """
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return tr.rolling_mean(window_size=window).alias("atr")


def rsi(column: str = "close", window: int = 14) -> pl.Expr:
    """Calculate Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    where RS = avg_gain / avg_loss

    Args:
        column: Price column.
        window: Lookback period (typically 14).

    Returns:
        Polars expression for RSI.
    """
    delta = pl.col(column).diff()
    gain = delta.clip(lower_bound=0).rolling_mean(window_size=window)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(window_size=window)

    rs = gain / loss
    return (100 - (100 / (1 + rs))).alias("rsi")


def bollinger_bands(
    column: str = "close",
    window: int = 20,
    num_std: float = 2.0,
) -> list[pl.Expr]:
    """Calculate Bollinger Bands.

    Args:
        column: Price column.
        window: Moving average window.
        num_std: Number of standard deviations for bands.

    Returns:
        List of expressions: [middle_band, upper_band, lower_band, band_width].
    """
    ma = pl.col(column).rolling_mean(window_size=window)
    std = pl.col(column).rolling_std(window_size=window)

    return [
        ma.alias("bb_middle"),
        (ma + num_std * std).alias("bb_upper"),
        (ma - num_std * std).alias("bb_lower"),
        ((num_std * 2 * std) / ma).alias("bb_width"),
    ]


# ==============================================================================
# Transformer Classes (Protocol-compatible)
# ==============================================================================
class OHLCVFeatureTransformer:
    """Transform raw OHLCV data into ML-ready features.

    Adds common technical indicators and derived features.
    Maintains lazy evaluation throughout.

    Attributes:
        volatility_window: Window for volatility calculation.
        volume_window: Window for volume MA.
        include_rsi: Whether to include RSI.
        include_atr: Whether to include ATR.
    """

    def __init__(
        self,
        volatility_window: int = 20,
        volume_window: int = 20,
        include_rsi: bool = True,
        include_atr: bool = True,
    ) -> None:
        """Initialize transformer.

        Args:
            volatility_window: Window for volatility calculation.
            volume_window: Window for volume MA ratio.
            include_rsi: Whether to add RSI indicator.
            include_atr: Whether to add ATR indicator.
        """
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.include_rsi = include_rsi
        self.include_atr = include_atr

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply feature engineering to OHLCV data.

        Args:
            lf: Input LazyFrame with OHLCV columns.

        Returns:
            LazyFrame with additional feature columns.
        """
        expressions: list[pl.Expr] = [
            pl.all(),  # Keep original columns
            log_returns(),
            rolling_volatility(window=self.volatility_window),
            volume_ma_ratio(window=self.volume_window),
        ]

        if self.include_rsi:
            expressions.append(rsi())

        if self.include_atr:
            expressions.append(atr())

        return lf.select(expressions)


class DropNullTransformer:
    """Remove rows with null values.

    Useful after applying lagged indicators that create nulls
    at the beginning of the series.
    """

    def __init__(self, subset: list[str] | None = None) -> None:
        """Initialize transformer.

        Args:
            subset: Columns to check for nulls. None = all columns.
        """
        self.subset = subset

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Remove rows with nulls.

        Args:
            lf: Input LazyFrame.

        Returns:
            LazyFrame with null rows removed.
        """
        if self.subset:
            return lf.drop_nulls(subset=self.subset)
        return lf.drop_nulls()


class NormalizeTransformer:
    """Z-score normalization for numeric columns.

    Transforms data to zero mean and unit variance.
    Essential for many ML algorithms.
    """

    def __init__(self, columns: list[str]) -> None:
        """Initialize transformer.

        Args:
            columns: Columns to normalize.
        """
        self.columns = columns

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply z-score normalization.

        Args:
            lf: Input LazyFrame.

        Returns:
            LazyFrame with normalized columns (suffixed with '_norm').
        """
        norm_exprs = [
            ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(f"{col}_norm")
            for col in self.columns
        ]

        return lf.with_columns(norm_exprs)


# ==============================================================================
# Pipeline Composition
# ==============================================================================
def compose_transformers(
    transformers: list[object],
) -> object:
    """Compose multiple transformers into a single pipeline.

    Args:
        transformers: List of transformer objects with transform() method.

    Returns:
        A transformer that applies all transforms in sequence.
    """

    class ComposedTransformer:
        """Pipeline of composed transformers."""

        def __init__(self, steps: list[object]) -> None:
            self.steps = steps

        def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
            """Apply all transformers in sequence."""
            result = lf
            for step in self.steps:
                result = step.transform(result)  # type: ignore[union-attr]
            return result

    return ComposedTransformer(transformers)
