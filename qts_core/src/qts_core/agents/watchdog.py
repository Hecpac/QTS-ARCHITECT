"""Drawdown Watchdog — Equity curve monitor with automatic risk scaling.

Inspired by Polystrat architecture (AutoResearch 2026-03-29, score 85):
hardcode limits, separate decision from execution, auto-halt on drawdown.

The watchdog sits between supervisor consensus and execution, acting as a
circuit breaker that scales position size or halts trading based on
equity curve deterioration.

Risk Tiers:
    NORMAL:  drawdown < warn_threshold     → full size (1.0x)
    WARNING: warn_threshold <= dd < halt   → reduced size (reduce_factor)
    HALTED:  dd >= halt_threshold          → block all new entries
    WEEKLY:  weekly dd >= weekly_halt      → block + alert

EXIT signals are never blocked (always risk-reducing).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import structlog
from pydantic import BaseModel, ConfigDict, Field

log = structlog.get_logger()


class WatchdogState(str, Enum):
    """Current risk state of the watchdog."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HALTED = "HALTED"


class WatchdogVerdict(BaseModel):
    """Output of watchdog evaluation."""

    model_config = ConfigDict(frozen=True)

    state: WatchdogState
    size_multiplier: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Position size scaling factor",
    )
    reason: str
    drawdown_pct: float = Field(
        default=0.0, description="Current drawdown as percentage",
    )
    peak_equity: float = Field(
        default=0.0, description="High water mark",
    )
    current_equity: float = Field(
        default=0.0, description="Current portfolio value",
    )


class DrawdownWatchdog:
    """Monitors equity curve and enforces drawdown-based risk limits.

    Attributes:
        warn_threshold: Drawdown % to trigger size reduction (e.g., 0.05 = 5%).
        halt_threshold: Drawdown % to halt all new entries (e.g., 0.08 = 8%).
        weekly_halt_threshold: Weekly drawdown % to halt + alert (e.g., 0.15).
        reduce_factor: Size multiplier when in WARNING state (e.g., 0.5 = 50%).
        cooldown_bars: Bars to wait after HALT before resuming to WARNING.
    """

    def __init__(
        self,
        warn_threshold: float = 0.05,
        halt_threshold: float = 0.08,
        weekly_halt_threshold: float = 0.15,
        reduce_factor: float = 0.50,
        cooldown_bars: int = 30,
    ) -> None:
        self.warn_threshold = warn_threshold
        self.halt_threshold = halt_threshold
        self.weekly_halt_threshold = weekly_halt_threshold
        self.reduce_factor = reduce_factor
        self.cooldown_bars = cooldown_bars

        # State tracking
        self._peak_equity: float = 0.0
        self._weekly_start_equity: float = 0.0
        self._weekly_start_date: datetime | None = None
        self._state: WatchdogState = WatchdogState.NORMAL
        self._halt_bar_count: int = 0

    @property
    def state(self) -> WatchdogState:
        """Current watchdog state."""
        return self._state

    def reset(self, initial_equity: float) -> None:
        """Reset watchdog with initial equity (call at start of session)."""
        self._peak_equity = initial_equity
        self._weekly_start_equity = initial_equity
        self._weekly_start_date = datetime.now(timezone.utc)
        self._state = WatchdogState.NORMAL
        self._halt_bar_count = 0
        log.info(
            "Watchdog reset",
            initial_equity=initial_equity,
            warn=f"{self.warn_threshold:.0%}",
            halt=f"{self.halt_threshold:.0%}",
        )

    def evaluate(
        self,
        current_equity: float,
        timestamp: datetime | None = None,
        is_exit: bool = False,
    ) -> WatchdogVerdict:
        """Evaluate current equity against drawdown limits.

        Args:
            current_equity: Current portfolio value.
            timestamp: Current bar timestamp.
            is_exit: Whether the proposed action is an EXIT (never blocked).

        Returns:
            WatchdogVerdict with state and size multiplier.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Update high water mark
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        # Reset weekly tracker on new week
        if self._weekly_start_date is not None:
            days_elapsed = (timestamp - self._weekly_start_date).days
            if days_elapsed >= 7:
                self._weekly_start_equity = current_equity
                self._weekly_start_date = timestamp
                log.info("Watchdog weekly reset", equity=current_equity)

        # Calculate drawdowns
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
        else:
            drawdown = 0.0

        if self._weekly_start_equity > 0:
            weekly_dd = (
                self._weekly_start_equity - current_equity
            ) / self._weekly_start_equity
        else:
            weekly_dd = 0.0

        # EXIT signals are always allowed
        if is_exit:
            return WatchdogVerdict(
                state=self._state,
                size_multiplier=1.0,
                reason="EXIT always approved (risk-reducing)",
                drawdown_pct=drawdown,
                peak_equity=self._peak_equity,
                current_equity=current_equity,
            )

        # Check weekly halt first (most severe)
        if weekly_dd >= self.weekly_halt_threshold:
            self._state = WatchdogState.HALTED
            log.critical(
                "WATCHDOG WEEKLY HALT",
                weekly_drawdown=f"{weekly_dd:.2%}",
                threshold=f"{self.weekly_halt_threshold:.2%}",
                equity=current_equity,
            )
            return WatchdogVerdict(
                state=WatchdogState.HALTED,
                size_multiplier=0.0,
                reason=(
                    f"Weekly drawdown {weekly_dd:.2%} >= "
                    f"{self.weekly_halt_threshold:.2%} — HALTED"
                ),
                drawdown_pct=drawdown,
                peak_equity=self._peak_equity,
                current_equity=current_equity,
            )

        # Check intraday halt
        if drawdown >= self.halt_threshold:
            self._state = WatchdogState.HALTED
            self._halt_bar_count = 0
            log.warning(
                "WATCHDOG HALT",
                drawdown=f"{drawdown:.2%}",
                threshold=f"{self.halt_threshold:.2%}",
                equity=current_equity,
                peak=self._peak_equity,
            )
            return WatchdogVerdict(
                state=WatchdogState.HALTED,
                size_multiplier=0.0,
                reason=(
                    f"Drawdown {drawdown:.2%} >= "
                    f"{self.halt_threshold:.2%} — HALTED"
                ),
                drawdown_pct=drawdown,
                peak_equity=self._peak_equity,
                current_equity=current_equity,
            )

        # Cooldown: transition from HALTED back to WARNING
        if self._state == WatchdogState.HALTED:
            self._halt_bar_count += 1
            if self._halt_bar_count >= self.cooldown_bars:
                self._state = WatchdogState.WARNING
                log.info(
                    "Watchdog cooldown complete, transitioning to WARNING",
                    bars_waited=self._halt_bar_count,
                )
            else:
                return WatchdogVerdict(
                    state=WatchdogState.HALTED,
                    size_multiplier=0.0,
                    reason=(
                        f"Cooling down ({self._halt_bar_count}/{self.cooldown_bars} bars)"
                    ),
                    drawdown_pct=drawdown,
                    peak_equity=self._peak_equity,
                    current_equity=current_equity,
                )

        # Check warning threshold
        if drawdown >= self.warn_threshold:
            self._state = WatchdogState.WARNING
            log.info(
                "Watchdog WARNING — reducing size",
                drawdown=f"{drawdown:.2%}",
                size_scale=self.reduce_factor,
            )
            return WatchdogVerdict(
                state=WatchdogState.WARNING,
                size_multiplier=self.reduce_factor,
                reason=(
                    f"Drawdown {drawdown:.2%} >= "
                    f"{self.warn_threshold:.2%} — size reduced to "
                    f"{self.reduce_factor:.0%}"
                ),
                drawdown_pct=drawdown,
                peak_equity=self._peak_equity,
                current_equity=current_equity,
            )

        # Normal operation
        self._state = WatchdogState.NORMAL
        return WatchdogVerdict(
            state=WatchdogState.NORMAL,
            size_multiplier=1.0,
            reason="Normal operation",
            drawdown_pct=drawdown,
            peak_equity=self._peak_equity,
            current_equity=current_equity,
        )
