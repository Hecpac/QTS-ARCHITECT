"""Base agent implementations.

This module provides base classes and concrete implementations for
strategy and risk agents. All agents are Protocol-compatible.

Design Principles:
- Agents are stateless where possible (easier to test and parallelize)
- Configuration via __init__ parameters (Hydra instantiate compatible)
- Async-first for non-blocking execution
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from qts_core.agents.protocol import (
    AgentPriority,
    AgentSignal,
    ReviewRequest,
    RiskStatus,
    RiskVerdict,
    SignalType,
)
from qts_core.common.types import InstrumentId

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


# ==============================================================================
# Base Strategy Agent
# ==============================================================================
class BaseStrategyAgent:
    """Base class for strategy agents.

    Provides common functionality and enforces the Protocol interface.
    Subclasses must implement `_generate_signal()`.

    Attributes:
        name: Unique agent identifier.
        priority: Signal priority level.
        min_confidence: Minimum confidence threshold to emit a signal.
    """

    def __init__(
        self,
        name: str,
        priority: AgentPriority = AgentPriority.NORMAL,
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize the agent.

        Args:
            name: Unique identifier for this agent.
            priority: Signal priority for consensus weighting.
            min_confidence: Signals below this threshold are suppressed.
        """
        self.name = name
        self.priority = priority
        self.min_confidence = min_confidence

    async def analyze(
        self,
        instrument_id: InstrumentId,
        current_price: float,
        timestamp: datetime,
        ohlcv_history: list[tuple[datetime, float, float, float, float, float]] | None = None,
    ) -> AgentSignal | None:
        """Analyze market data and produce a signal.

        Template method pattern: calls _generate_signal() and applies
        common post-processing (confidence filtering, logging).

        Args:
            instrument_id: The instrument being analyzed.
            current_price: Latest price.
            timestamp: Current timestamp.
            ohlcv_history: Optional historical OHLCV data.

        Returns:
            AgentSignal if above confidence threshold, None otherwise.
        """
        try:
            signal = await self._generate_signal(
                instrument_id=instrument_id,
                current_price=current_price,
                timestamp=timestamp,
                ohlcv_history=ohlcv_history,
            )

            if signal is None:
                return None

            # Apply minimum confidence filter
            if signal.confidence < self.min_confidence:
                log.debug(
                    "Signal suppressed (low confidence)",
                    agent=self.name,
                    confidence=signal.confidence,
                    threshold=self.min_confidence,
                )
                return None

            return signal

        except Exception as e:
            log.error(
                "Agent analysis failed",
                agent=self.name,
                error=str(e),
                exc_info=True,
            )
            return None

    async def _generate_signal(
        self,
        instrument_id: InstrumentId,
        current_price: float,
        timestamp: datetime,
        ohlcv_history: list[tuple[datetime, float, float, float, float, float]] | None = None,
    ) -> AgentSignal | None:
        """Generate a trading signal. Override in subclasses.

        Args:
            instrument_id: The instrument being analyzed.
            current_price: Latest price.
            timestamp: Current timestamp.
            ohlcv_history: Optional OHLCV history as list of tuples.

        Returns:
            AgentSignal or None if no opportunity.
        """
        raise NotImplementedError("Subclasses must implement _generate_signal()")


# ==============================================================================
# Base Risk Agent
# ==============================================================================
class BaseRiskAgent:
    """Base class for risk agents.

    Risk agents evaluate trading signals against risk constraints
    and have veto power over trades.

    Attributes:
        name: Unique agent identifier.
        max_position_size: Maximum position as fraction of portfolio.
        max_daily_loss: Maximum allowed daily loss percentage.
    """

    def __init__(
        self,
        name: str,
        max_position_size: float = 0.10,
        max_daily_loss: float = 0.02,
    ) -> None:
        """Initialize risk agent.

        Args:
            name: Agent identifier.
            max_position_size: Max position size (0-1).
            max_daily_loss: Max daily loss as fraction (e.g., 0.02 = 2%).
        """
        self.name = name
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss

    async def evaluate(self, request: ReviewRequest) -> RiskVerdict:
        """Evaluate a trading request.

        Template method: applies common risk checks then delegates
        to _custom_evaluation() for agent-specific logic.

        Args:
            request: The review request.

        Returns:
            RiskVerdict with approval status.
        """
        # EXIT is always risk-reducing and must not be blocked by exposure checks.
        if request.proposed_signal.signal_type == SignalType.EXIT:
            return RiskVerdict(
                status=RiskStatus.APPROVED,
                reason="Exit signal approved (risk reduction)",
                risk_metrics={"current_exposure": request.portfolio_exposure},
            )

        # Check hard daily loss guardrail for non-exit actions.
        if request.daily_pnl_fraction <= -self.max_daily_loss:
            return RiskVerdict(
                status=RiskStatus.REJECTED,
                reason=(
                    "Daily loss limit breached "
                    f"({request.daily_pnl_fraction:.2%} <= -{self.max_daily_loss:.2%})"
                ),
                risk_metrics={
                    "daily_pnl_fraction": request.daily_pnl_fraction,
                    "max_daily_loss": self.max_daily_loss,
                },
            )

        # Check exposure limits
        if request.portfolio_exposure > self.max_position_size:
            return RiskVerdict(
                status=RiskStatus.REJECTED,
                reason=f"Position size exceeds limit ({request.portfolio_exposure:.1%} > {self.max_position_size:.1%})",
                risk_metrics={"current_exposure": request.portfolio_exposure},
            )

        # Delegate to custom evaluation
        return await self._custom_evaluation(request)

    async def _custom_evaluation(self, request: ReviewRequest) -> RiskVerdict:
        """Custom risk evaluation. Override in subclasses."""
        return RiskVerdict(
            status=RiskStatus.APPROVED,
            reason="Passed basic risk checks",
            risk_metrics={},
        )


# ==============================================================================
# Concrete Implementations
# ==============================================================================
class TechnicalAgent(BaseStrategyAgent):
    """Simple technical analysis agent.

    Uses basic price action (green/red candle) as a demonstration.
    Replace with real technical indicators in production.

    Attributes:
        bullish_threshold: Minimum price change % for bullish signal.
    """

    def __init__(
        self,
        name: str,
        bullish_threshold: float = 0.0,
        priority: AgentPriority = AgentPriority.NORMAL,
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize TechnicalAgent.

        Args:
            name: Agent identifier.
            bullish_threshold: Min price change for bullish signal.
            priority: Signal priority.
            min_confidence: Confidence threshold.
        """
        super().__init__(name=name, priority=priority, min_confidence=min_confidence)
        self.bullish_threshold = bullish_threshold

    async def _generate_signal(
        self,
        instrument_id: InstrumentId,
        current_price: float,
        timestamp: datetime,
        ohlcv_history: list[tuple[datetime, float, float, float, float, float]] | None = None,
    ) -> AgentSignal | None:
        """Generate signal based on simple price action."""
        # Simulate some processing time
        await asyncio.sleep(0.001)

        if ohlcv_history is None or len(ohlcv_history) < 1:
            return None

        # Get latest candle: (timestamp, open, high, low, close, volume)
        latest = ohlcv_history[-1]
        _, open_price, _, _, close_price, _ = latest

        price_change = (close_price - open_price) / open_price

        if price_change > self.bullish_threshold:
            signal_type = SignalType.LONG
            confidence = min(0.5 + price_change * 10, 0.95)  # Scale by magnitude
            reason = "Bullish candle"
        elif price_change < -self.bullish_threshold:
            signal_type = SignalType.SHORT
            confidence = min(0.5 + abs(price_change) * 10, 0.95)
            reason = "Bearish candle"
        else:
            return None

        return AgentSignal(
            source_agent=self.name,
            signal_type=signal_type,
            confidence=confidence,
            priority=self.priority,
            timestamp=timestamp,
            metadata={"reason": reason, "price_change": price_change},
        )


class StrictRiskAgent(BaseRiskAgent):
    """Risk agent with strict confidence requirements.

    Rejects signals below a confidence threshold and enforces
    position size limits.

    Attributes:
        min_signal_confidence: Minimum required signal confidence.
    """

    def __init__(
        self,
        name: str,
        min_signal_confidence: float = 0.7,
        max_position_size: float = 0.10,
        max_daily_loss: float = 0.02,
    ) -> None:
        """Initialize StrictRiskAgent.

        Args:
            name: Agent identifier.
            min_signal_confidence: Minimum signal confidence to approve.
            max_position_size: Maximum position size fraction.
            max_daily_loss: Maximum daily loss fraction.
        """
        super().__init__(
            name=name,
            max_position_size=max_position_size,
            max_daily_loss=max_daily_loss,
        )
        self.min_signal_confidence = min_signal_confidence

    async def _custom_evaluation(self, request: ReviewRequest) -> RiskVerdict:
        """Apply strict confidence check."""
        # Simulate processing
        await asyncio.sleep(0.001)

        if request.proposed_signal.confidence < self.min_signal_confidence:
            log.warning(
                "Risk veto: low confidence",
                agent=self.name,
                signal_confidence=request.proposed_signal.confidence,
                threshold=self.min_signal_confidence,
            )
            return RiskVerdict(
                status=RiskStatus.REJECTED,
                reason=f"Confidence {request.proposed_signal.confidence:.2f} below threshold {self.min_signal_confidence:.2f}",
                risk_metrics={
                    "signal_confidence": request.proposed_signal.confidence,
                    "threshold": self.min_signal_confidence,
                },
            )

        return RiskVerdict(
            status=RiskStatus.APPROVED,
            reason="Signal meets confidence and risk requirements",
            risk_metrics={
                "signal_confidence": request.proposed_signal.confidence,
                "exposure": request.portfolio_exposure,
            },
        )


class PermissiveRiskAgent(BaseRiskAgent):
    """Risk agent that approves most trades.

    Useful for backtesting without risk constraints.
    Only rejects if position size is exceeded.
    """

    async def _custom_evaluation(self, request: ReviewRequest) -> RiskVerdict:
        """Approve with minimal checks."""
        return RiskVerdict(
            status=RiskStatus.APPROVED,
            reason="Permissive mode: approved",
            risk_metrics={"exposure": request.portfolio_exposure},
        )
