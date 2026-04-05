"""Signal Quality Evaluator — Capa 4 of QTS risk management.

Tracks recent trading signals and evaluates their consistency to detect
noisy/whipsaw conditions. When signals flip direction frequently, the
evaluator reduces position size or blocks entries entirely.

Metrics:
    - Direction consistency: fraction of recent signals agreeing with latest
    - Flip rate: how often direction changed in the window
    - Average confidence of agreeing signals

Quality tiers:
    HIGH      → ≥70% consistency, flip_rate ≤ 0.2 → 1.0× size
    MEDIUM    → ≥50% consistency, flip_rate ≤ 0.4 → configurable scale
    LOW       → below MEDIUM thresholds → block entry
    NO_DATA   → insufficient history → permissive (allow)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


class SignalQuality(str, Enum):
    """Quality tier of recent signal history."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NO_DATA = "NO_DATA"


@dataclass(frozen=True)
class QualityVerdict:
    """Result of signal quality evaluation."""

    quality: SignalQuality
    consistency: float  # 0-1, fraction agreeing with latest direction
    flip_rate: float  # 0-1, fraction of direction changes in window
    avg_confidence: float  # mean confidence of agreeing signals
    size_multiplier: float  # 1.0 for HIGH, reduced for MEDIUM, 0.0 for LOW
    block_entry: bool  # True when quality is LOW
    sample_count: int
    reason: str


@dataclass
class _SignalRecord:
    """Internal record of a signal."""

    direction: str  # "LONG", "SHORT", "EXIT", "NEUTRAL"
    confidence: float
    timestamp: datetime


class SignalQualityEvaluator:
    """Evaluates consistency and quality of recent trading signals.

    Parameters
    ----------
    window_signals : int
        Number of recent signals to consider. Default 10.
    high_consistency : float
        Minimum consistency for HIGH quality. Default 0.70.
    medium_consistency : float
        Minimum consistency for MEDIUM quality. Default 0.50.
    high_max_flip_rate : float
        Maximum flip rate for HIGH quality. Default 0.20.
    medium_max_flip_rate : float
        Maximum flip rate for MEDIUM quality. Default 0.40.
    medium_size_scale : float
        Size multiplier when quality is MEDIUM. Default 0.60.
    min_samples : int
        Minimum signals before evaluation. Below this → NO_DATA. Default 5.
    window_hours : float
        Time window; signals older than this are pruned. Default 24.0.
    """

    def __init__(
        self,
        *,
        window_signals: int = 10,
        high_consistency: float = 0.70,
        medium_consistency: float = 0.50,
        high_max_flip_rate: float = 0.20,
        medium_max_flip_rate: float = 0.40,
        medium_size_scale: float = 0.60,
        min_samples: int = 5,
        window_hours: float = 24.0,
    ) -> None:
        self._window_signals = max(2, window_signals)
        self._high_consistency = high_consistency
        self._medium_consistency = medium_consistency
        self._high_max_flip_rate = high_max_flip_rate
        self._medium_max_flip_rate = medium_max_flip_rate
        self._medium_size_scale = medium_size_scale
        self._min_samples = max(1, min_samples)
        self._window_hours = window_hours
        self._history: deque[_SignalRecord] = deque(maxlen=self._window_signals * 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_signal(
        self,
        direction: str,
        confidence: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a new trading signal."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        confidence = max(0.0, min(1.0, confidence))
        self._history.append(
            _SignalRecord(
                direction=direction,
                confidence=confidence,
                timestamp=timestamp,
            )
        )

    def evaluate(self, now: datetime | None = None) -> QualityVerdict:
        """Evaluate signal quality from recent history.

        Returns a QualityVerdict with quality tier, metrics, and
        whether to block the entry.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        cutoff = now - timedelta(hours=self._window_hours)
        recent = [r for r in self._history if r.timestamp >= cutoff]

        # Keep only the last N
        if len(recent) > self._window_signals:
            recent = recent[-self._window_signals:]

        if len(recent) < self._min_samples:
            return QualityVerdict(
                quality=SignalQuality.NO_DATA,
                consistency=0.0,
                flip_rate=0.0,
                avg_confidence=0.0,
                size_multiplier=1.0,
                block_entry=False,
                sample_count=len(recent),
                reason=f"Insufficient signals {len(recent)}/{self._min_samples}",
            )

        # --- Consistency: fraction matching latest direction ---
        latest_dir = recent[-1].direction
        agreeing = [r for r in recent if r.direction == latest_dir]
        consistency = len(agreeing) / len(recent)
        avg_conf = sum(r.confidence for r in agreeing) / len(agreeing) if agreeing else 0.0

        # --- Flip rate: direction changes / (n-1) ---
        flips = 0
        for i in range(1, len(recent)):
            if recent[i].direction != recent[i - 1].direction:
                flips += 1
        flip_rate = flips / (len(recent) - 1)

        # --- Classify ---
        if consistency >= self._high_consistency and flip_rate <= self._high_max_flip_rate:
            quality = SignalQuality.HIGH
            size_mult = 1.0
            block = False
            reason = (
                f"High quality: {consistency:.0%} consistent, "
                f"{flip_rate:.0%} flip rate, "
                f"avg conf {avg_conf:.2f} ({len(recent)} signals)"
            )
        elif consistency >= self._medium_consistency and flip_rate <= self._medium_max_flip_rate:
            quality = SignalQuality.MEDIUM
            size_mult = self._medium_size_scale
            block = False
            reason = (
                f"Medium quality: {consistency:.0%} consistent, "
                f"{flip_rate:.0%} flip rate → "
                f"{self._medium_size_scale:.0%} size ({len(recent)} signals)"
            )
        else:
            quality = SignalQuality.LOW
            size_mult = 0.0
            block = True
            reason = (
                f"Low quality (whipsaw): {consistency:.0%} consistent, "
                f"{flip_rate:.0%} flip rate — blocking entry ({len(recent)} signals)"
            )

        return QualityVerdict(
            quality=quality,
            consistency=consistency,
            flip_rate=flip_rate,
            avg_confidence=avg_conf,
            size_multiplier=size_mult,
            block_entry=block,
            sample_count=len(recent),
            reason=reason,
        )

    def reset(self) -> None:
        """Clear all signal history."""
        self._history.clear()

    @property
    def signal_count(self) -> int:
        """Total signals in buffer (including stale)."""
        return len(self._history)
