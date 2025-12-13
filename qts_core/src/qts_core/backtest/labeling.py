"""Triple Barrier Labeling for ML-based trading strategies.

Implements the Triple Barrier Method from "Advances in Financial Machine
Learning" by Marcos López de Prado. This technique labels price data
for supervised learning based on:

1. **Upper Barrier (Take Profit)**: Price rises by a threshold
2. **Lower Barrier (Stop Loss)**: Price falls by a threshold
3. **Vertical Barrier (Time Limit)**: Time horizon expires

Labels:
- +1: Upper barrier hit first (profitable trade)
- -1: Lower barrier hit first (losing trade)
-  0: Vertical barrier hit (neutral/timeout)

Design Notes:
- Uses Polars for vectorized, lazy evaluation
- Barriers are typically set as multiples of volatility
- Supports asymmetric risk/reward via separate multipliers
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import polars as pl


# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_PT_MULTIPLIER: Final[float] = 1.0
DEFAULT_SL_MULTIPLIER: Final[float] = 1.0
DEFAULT_VOLATILITY_COL: Final[str] = "volatility"


# ==============================================================================
# Enums
# ==============================================================================
class BarrierLabel(IntEnum):
    """Labels for triple barrier method."""

    STOP_LOSS = -1  # Lower barrier hit
    TIMEOUT = 0  # Vertical barrier hit
    TAKE_PROFIT = 1  # Upper barrier hit


# ==============================================================================
# Triple Barrier Method
# ==============================================================================
def triple_barrier_method(
    lf: pl.LazyFrame,
    vertical_barrier_bars: int,
    pt_multiplier: float = DEFAULT_PT_MULTIPLIER,
    sl_multiplier: float = DEFAULT_SL_MULTIPLIER,
    volatility_col: str = DEFAULT_VOLATILITY_COL,
) -> pl.LazyFrame:
    """Apply the Triple Barrier Method to a LazyFrame.

    Generates labels based on which barrier is touched first within
    the time horizon defined by vertical_barrier_bars.

    The barrier distances are calculated as:
        upper_barrier = close + (volatility * pt_multiplier)
        lower_barrier = close - (volatility * sl_multiplier)

    Args:
        lf: Input LazyFrame with required columns:
            - 'close': Closing prices
            - 'high': High prices
            - 'low': Low prices
            - `volatility_col`: Volatility measure (e.g., rolling std dev)
        vertical_barrier_bars: Number of bars for the time horizon.
        pt_multiplier: Multiplier for upper barrier (take profit).
        sl_multiplier: Multiplier for lower barrier (stop loss).
        volatility_col: Column name containing volatility values.

    Returns:
        LazyFrame with additional columns:
            - 'label': Barrier label (+1, -1, or 0)
            - 'barrier_ret': Realized return at barrier touch
            - 'barrier_type': String description of barrier hit

    Example:
        ```python
        lf = pl.scan_parquet("data.parquet")
        labeled = triple_barrier_method(
            lf,
            vertical_barrier_bars=10,
            pt_multiplier=2.0,  # 2x volatility for TP
            sl_multiplier=1.0,  # 1x volatility for SL
        )
        df = labeled.collect()
        ```

    Note:
        This implementation uses a conservative approach when both barriers
        could be hit within the same window: the stop loss takes precedence.
        This is because without tick-level data, we cannot determine which
        barrier was hit first within a bar.
    """
    if vertical_barrier_bars < 1:
        msg = f"vertical_barrier_bars must be >= 1, got {vertical_barrier_bars}"
        raise ValueError(msg)

    if pt_multiplier <= 0 or sl_multiplier <= 0:
        msg = "pt_multiplier and sl_multiplier must be positive"
        raise ValueError(msg)

    # Calculate barrier distances based on volatility
    upper_dist = pl.col(volatility_col) * pt_multiplier
    lower_dist = pl.col(volatility_col) * sl_multiplier

    # Look-ahead for future max/min prices
    # Using reverse + rolling + reverse pattern for forward-looking window
    future_high = (
        pl.col("high")
        .reverse()
        .rolling_max(window_size=vertical_barrier_bars)
        .reverse()
        .shift(-1)  # Exclude current bar
    )

    future_low = (
        pl.col("low")
        .reverse()
        .rolling_min(window_size=vertical_barrier_bars)
        .reverse()
        .shift(-1)
    )

    # Price at vertical barrier
    vertical_price = pl.col("close").shift(-vertical_barrier_bars)

    # Barrier hit conditions
    hit_upper = future_high >= (pl.col("close") + upper_dist)
    hit_lower = future_low <= (pl.col("close") - lower_dist)

    # Assign labels (stop loss takes precedence when both hit)
    label_expr = (
        pl.when(hit_lower)
        .then(pl.lit(BarrierLabel.STOP_LOSS))
        .when(hit_upper)
        .then(pl.lit(BarrierLabel.TAKE_PROFIT))
        .otherwise(pl.lit(BarrierLabel.TIMEOUT))
    )

    # Calculate realized return
    return_expr = (
        pl.when(hit_lower)
        .then(-lower_dist / pl.col("close"))  # SL return
        .when(hit_upper)
        .then(upper_dist / pl.col("close"))  # TP return
        .otherwise((vertical_price - pl.col("close")) / pl.col("close"))  # Timeout return
    )

    # Barrier type description
    type_expr = (
        pl.when(hit_lower)
        .then(pl.lit("stop_loss"))
        .when(hit_upper)
        .then(pl.lit("take_profit"))
        .otherwise(pl.lit("timeout"))
    )

    return lf.with_columns(
        [
            label_expr.alias("label"),
            return_expr.alias("barrier_ret"),
            type_expr.alias("barrier_type"),
        ]
    )


def add_volatility(
    lf: pl.LazyFrame,
    window: int = 20,
    return_col: str = "close",
    output_col: str = "volatility",
) -> pl.LazyFrame:
    """Add rolling volatility column for triple barrier method.

    Calculates rolling standard deviation of returns as a volatility
    measure for barrier distance calculation.

    Args:
        lf: Input LazyFrame with price data.
        window: Rolling window size for volatility calculation.
        return_col: Column to calculate returns from.
        output_col: Name for the output volatility column.

    Returns:
        LazyFrame with volatility column added.

    Example:
        ```python
        lf = pl.scan_parquet("data.parquet")
        lf = add_volatility(lf, window=20)
        labeled = triple_barrier_method(lf, vertical_barrier_bars=10)
        ```
    """
    returns = pl.col(return_col).pct_change()
    rolling_vol = returns.rolling_std(window_size=window) * pl.col(return_col)

    return lf.with_columns(rolling_vol.alias(output_col))


def filter_valid_labels(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Filter out rows with invalid labels (null from look-ahead).

    After triple barrier labeling, the last `vertical_barrier_bars` rows
    will have null labels because we can't look ahead past the data.
    This function removes those rows.

    Args:
        lf: LazyFrame after triple_barrier_method.

    Returns:
        LazyFrame with null labels removed.
    """
    return lf.filter(pl.col("label").is_not_null() & pl.col("barrier_ret").is_not_null())


def calculate_label_distribution(df: pl.DataFrame) -> dict[str, float]:
    """Calculate distribution of labels.

    Args:
        df: DataFrame with 'label' column.

    Returns:
        Dictionary with label counts and percentages.
    """
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "take_profit_count": 0,
            "stop_loss_count": 0,
            "timeout_count": 0,
            "take_profit_pct": 0.0,
            "stop_loss_pct": 0.0,
            "timeout_pct": 0.0,
        }

    counts = df.group_by("label").agg(pl.len().alias("count"))

    tp_count = counts.filter(pl.col("label") == 1)["count"].to_list()
    sl_count = counts.filter(pl.col("label") == -1)["count"].to_list()
    to_count = counts.filter(pl.col("label") == 0)["count"].to_list()

    tp = tp_count[0] if tp_count else 0
    sl = sl_count[0] if sl_count else 0
    to = to_count[0] if to_count else 0

    return {
        "total": total,
        "take_profit_count": tp,
        "stop_loss_count": sl,
        "timeout_count": to,
        "take_profit_pct": tp / total,
        "stop_loss_pct": sl / total,
        "timeout_pct": to / total,
    }
