"""FinBERT Sentiment Risk Filter — blocks entries on extreme negative sentiment.

Inspired by AutoResearch 2026-03-29:
- Insight #4 (score 82): FinBERT + XGBoost + regime filters → 135% in 24 months
  "FinBERT blocks BUY signals when sentiment < -0.70"

The filter maintains a rolling window of sentiment scores from headlines.
When average sentiment drops below a threshold, new LONG entries are blocked.

Operates as a pre-execution gate: sentiment data is fed externally
(headlines, news, social), and the filter is queried before each trade.

Graceful degradation: if no sentiment data is available, the filter
approves all trades (permissive default).
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum

import structlog
from pydantic import BaseModel, ConfigDict, Field

log = structlog.get_logger()


class SentimentState(str, Enum):
    """Current sentiment regime."""

    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    NO_DATA = "NO_DATA"


class SentimentVerdict(BaseModel):
    """Output of sentiment filter evaluation."""

    model_config = ConfigDict(frozen=True)

    state: SentimentState
    block_longs: bool = False
    boost_confidence: bool = False
    avg_sentiment: float = Field(
        default=0.0, description="Average sentiment in window",
    )
    sample_count: int = Field(
        default=0, description="Number of scores in window",
    )
    reason: str = ""


class SentimentScore(BaseModel):
    """A single sentiment measurement."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(..., ge=-1.0, le=1.0)
    timestamp: datetime
    source: str = ""


class SentimentFilter:
    """Rolling sentiment filter that gates trade entries.

    Attributes:
        block_threshold: Sentiment below this blocks longs (e.g., -0.70).
        boost_threshold: Sentiment above this boosts confidence (e.g., 0.70).
        window_hours: Hours of sentiment data to consider.
        min_samples: Minimum samples required to act (else permissive).
    """

    def __init__(
        self,
        block_threshold: float = -0.70,
        boost_threshold: float = 0.70,
        window_hours: float = 6.0,
        min_samples: int = 3,
    ) -> None:
        self.block_threshold = block_threshold
        self.boost_threshold = boost_threshold
        self.window_hours = window_hours
        self.min_samples = min_samples
        self._scores: deque[SentimentScore] = deque(maxlen=500)

    @property
    def score_count(self) -> int:
        """Number of scores currently in the buffer."""
        return len(self._scores)

    def add_score(
        self,
        score: float,
        timestamp: datetime | None = None,
        source: str = "",
    ) -> None:
        """Add a sentiment score.

        Args:
            score: Sentiment value in [-1.0, 1.0].
            timestamp: When the sentiment was measured.
            source: Origin of the score (e.g., "finbert", "twitter").
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        clamped = max(-1.0, min(1.0, score))
        self._scores.append(
            SentimentScore(score=clamped, timestamp=timestamp, source=source)
        )

    def add_scores(self, scores: list[tuple[float, str]]) -> None:
        """Batch add scores with current timestamp.

        Args:
            scores: List of (score, source) tuples.
        """
        now = datetime.now(timezone.utc)
        for score, source in scores:
            self.add_score(score, timestamp=now, source=source)

    def _window_scores(self, now: datetime | None = None) -> list[float]:
        """Get scores within the time window."""
        if now is None:
            now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self.window_hours)
        return [s.score for s in self._scores if s.timestamp >= cutoff]

    def evaluate(self, now: datetime | None = None) -> SentimentVerdict:
        """Evaluate current sentiment state.

        Args:
            now: Current timestamp (for testability).

        Returns:
            SentimentVerdict with block/boost decisions.
        """
        scores = self._window_scores(now)

        if len(scores) < self.min_samples:
            return SentimentVerdict(
                state=SentimentState.NO_DATA,
                block_longs=False,
                boost_confidence=False,
                avg_sentiment=0.0,
                sample_count=len(scores),
                reason=(
                    f"Insufficient data ({len(scores)}/{self.min_samples}) "
                    "— permissive default"
                ),
            )

        avg = sum(scores) / len(scores)

        if avg <= self.block_threshold:
            log.warning(
                "SENTIMENT FILTER: blocking longs",
                avg_sentiment=f"{avg:.3f}",
                threshold=self.block_threshold,
                samples=len(scores),
            )
            return SentimentVerdict(
                state=SentimentState.NEGATIVE,
                block_longs=True,
                boost_confidence=False,
                avg_sentiment=avg,
                sample_count=len(scores),
                reason=(
                    f"Sentiment {avg:.3f} <= {self.block_threshold} "
                    f"— LONG entries blocked ({len(scores)} samples)"
                ),
            )

        if avg >= self.boost_threshold:
            return SentimentVerdict(
                state=SentimentState.POSITIVE,
                block_longs=False,
                boost_confidence=True,
                avg_sentiment=avg,
                sample_count=len(scores),
                reason=(
                    f"Sentiment {avg:.3f} >= {self.boost_threshold} "
                    f"— high confidence ({len(scores)} samples)"
                ),
            )

        return SentimentVerdict(
            state=SentimentState.NEUTRAL,
            block_longs=False,
            boost_confidence=False,
            avg_sentiment=avg,
            sample_count=len(scores),
            reason=f"Neutral sentiment {avg:.3f} ({len(scores)} samples)",
        )

    def reset(self) -> None:
        """Clear all stored scores."""
        self._scores.clear()
