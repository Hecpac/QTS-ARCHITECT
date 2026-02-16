"""Multi-Agent Supervisor with configurable consensus strategies.

The Supervisor orchestrates the agent ensemble:
1. Broadcasts market data to all strategy agents (parallel)
2. Collects and synthesizes signals using a consensus strategy
3. Submits selected signal to risk agent for review
4. Issues final trading decision to OMS

Consensus Strategies:
- HIGHEST_CONFIDENCE: Take the signal with highest confidence
- MAJORITY_VOTE: Take direction with most votes (weighted by confidence)
- UNANIMOUS: Only trade if all agents agree on direction
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import structlog

from qts_core.agents.protocol import (
    AgentSignal,
    ReviewRequest,
    RiskAgentProtocol,
    RiskStatus,
    SignalType,
    StrategyAgentProtocol,
    TradingDecision,
)


if TYPE_CHECKING:
    from qts_core.common.types import InstrumentId, MarketData

log = structlog.get_logger()


# ==============================================================================
# Consensus Strategies
# ==============================================================================
class ConsensusStrategy(str, Enum):
    """Available consensus strategies for signal aggregation."""

    HIGHEST_CONFIDENCE = "highest_confidence"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    UNANIMOUS = "unanimous"


# ==============================================================================
# OHLCV History Type
# ==============================================================================
OHLCVTuple = tuple[datetime, float, float, float, float, float]


# ==============================================================================
# Supervisor
# ==============================================================================
class Supervisor:
    """Multi-agent orchestrator with configurable consensus.

    The Supervisor is the central coordinator of the trading system.
    It manages the flow from market data to trading decisions.

    Attributes:
        strategy_agents: List of strategy agents to query.
        risk_agent: Risk agent with veto power.
        consensus_strategy: How to aggregate signals.
        min_confidence: Minimum confidence threshold for signals.
        min_agents_required: Minimum agents that must respond.
    """

    def __init__(
        self,
        strategy_agents: list[StrategyAgentProtocol],
        risk_agent: RiskAgentProtocol,
        consensus_strategy: ConsensusStrategy = ConsensusStrategy.HIGHEST_CONFIDENCE,
        min_confidence: float = 0.6,
        min_agents_required: int = 1,
    ) -> None:
        """Initialize Supervisor.

        Args:
            strategy_agents: List of strategy agents.
            risk_agent: Risk agent for final approval.
            consensus_strategy: Strategy for aggregating signals.
            min_confidence: Minimum signal confidence to consider.
            min_agents_required: Minimum agents that must provide signals.
        """
        if not strategy_agents:
            raise ValueError("At least one strategy agent is required")

        self.strategy_agents = strategy_agents
        self.risk_agent = risk_agent
        self.consensus_strategy = consensus_strategy
        self.min_confidence = min_confidence
        self.min_agents_required = min_agents_required

        log.info(
            "Supervisor initialized",
            agents=len(strategy_agents),
            consensus=consensus_strategy.value,
        )

    async def run(
        self,
        market_data: MarketData,
        ohlcv_history: list[OHLCVTuple] | None = None,
        portfolio_exposure: float = 0.0,
    ) -> TradingDecision | None:
        """Execute one decision cycle.

        This is the canonical interface: a single MarketData event plus optional
        OHLCV history (for strategies that require context).

        Args:
            market_data: Current market event (OHLCV bar).
            ohlcv_history: Optional OHLCV history for agents.
            portfolio_exposure: Current position as fraction of portfolio.

        Returns:
            TradingDecision if trade approved, None otherwise.
        """
        instrument_id = market_data.instrument_id
        current_price = market_data.close
        timestamp = market_data.timestamp

        ohlcv_history_for_agents = ohlcv_history or [
            (
                market_data.timestamp,
                market_data.open,
                market_data.high,
                market_data.low,
                market_data.close,
                market_data.volume,
            )
        ]

        correlation_id = timestamp.isoformat()
        log.info(
            "Supervisor cycle started",
            correlation_id=correlation_id,
            instrument=str(instrument_id),
            price=current_price,
        )

        # 1. Parallel Query (Scatter)
        signals = await self._gather_signals(
            instrument_id=instrument_id,
            current_price=current_price,
            timestamp=timestamp,
            ohlcv_history=ohlcv_history_for_agents,
        )

        if len(signals) < self.min_agents_required:
            log.warning(
                "Insufficient agent responses",
                received=len(signals),
                required=self.min_agents_required,
            )
            return None

        # 2. Prioritize explicit EXIT signals for risk-off behavior.
        exit_signals = [s for s in signals if s.signal_type == SignalType.EXIT]
        if exit_signals:
            consensus_signal = max(exit_signals, key=lambda s: s.confidence)
            log.info(
                "Exit signal prioritized",
                source=consensus_signal.source_agent,
                confidence=consensus_signal.confidence,
            )
            valid_signals = exit_signals
        else:
            # 3. Filter directional actionable signals
            valid_signals = [
                s for s in signals
                if s.signal_type != SignalType.NEUTRAL
                and s.confidence >= self.min_confidence
            ]

            if not valid_signals:
                log.debug("No actionable signals", total=len(signals))
                return None

            # 4. Apply consensus strategy
            consensus_signal = self._apply_consensus(valid_signals)
        if consensus_signal is None:
            log.info("No consensus reached")
            return None

        signal_value = getattr(
            consensus_signal.signal_type,
            "value",
            consensus_signal.signal_type,
        )
        log.info(
            "Consensus signal selected",
            signal=signal_value,
            confidence=f"{consensus_signal.confidence:.2%}",
            source=consensus_signal.source_agent,
        )

        # 4. Risk Review
        review_request = ReviewRequest(
            proposed_signal=consensus_signal,
            instrument_id=instrument_id,
            current_price=current_price,
            portfolio_exposure=portfolio_exposure,
        )

        verdict = await self.risk_agent.evaluate(review_request)

        if verdict.status == RiskStatus.REJECTED:
            log.warning(
                "Trade rejected by risk agent",
                reason=verdict.reason,
                metrics=verdict.risk_metrics,
            )
            return None

        # 5. Generate Trading Decision
        size_modifier = consensus_signal.confidence
        if verdict.status == RiskStatus.REDUCED:
            size_modifier *= verdict.adjusted_size
            log.info(
                "Position size reduced by risk agent",
                original=consensus_signal.confidence,
                adjusted=size_modifier,
            )

        # Collect contributing agents
        contributing = [
            s.source_agent for s in valid_signals
            if s.signal_type == consensus_signal.signal_type
        ]

        decision = TradingDecision(
            instrument_id=instrument_id,
            action=consensus_signal.signal_type,
            quantity_modifier=size_modifier,
            rationale=(
                f"Consensus: {self.consensus_strategy.value}. "
                f"Signal from {consensus_signal.source_agent} "
                f"(confidence: {consensus_signal.confidence:.2%}). "
                f"Risk: {verdict.reason}"
            ),
            contributing_agents=contributing,
        )

        action_value = getattr(decision.action, "value", decision.action)
        log.info(
            "Trading decision generated",
            decision_id=str(decision.decision_id),
            action=action_value,
            size_modifier=f"{decision.quantity_modifier:.2%}",
        )

        return decision

    async def _gather_signals(
        self,
        instrument_id: InstrumentId,
        current_price: float,
        timestamp: datetime,
        ohlcv_history: list[OHLCVTuple] | None,
    ) -> list[AgentSignal]:
        """Query all strategy agents in parallel."""
        tasks = [
            agent.analyze(
                instrument_id=instrument_id,
                current_price=current_price,
                timestamp=timestamp,
                ohlcv_history=ohlcv_history,
            )
            for agent in self.strategy_agents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: list[AgentSignal] = []
        for agent, result in zip(self.strategy_agents, results, strict=True):
            if isinstance(result, Exception):
                log.error(
                    "Agent analysis failed",
                    agent=agent.name,
                    error=str(result),
                )
                continue
            if result is not None:
                signals.append(result)

        log.debug(
            "Signals collected",
            total_agents=len(self.strategy_agents),
            signals_received=len(signals),
        )

        return signals

    def _apply_consensus(self, signals: list[AgentSignal]) -> AgentSignal | None:
        """Apply consensus strategy to aggregate signals."""
        if not signals:
            return None

        if self.consensus_strategy == ConsensusStrategy.HIGHEST_CONFIDENCE:
            return self._consensus_highest_confidence(signals)
        elif self.consensus_strategy == ConsensusStrategy.MAJORITY_VOTE:
            return self._consensus_majority_vote(signals)
        elif self.consensus_strategy == ConsensusStrategy.WEIGHTED_AVERAGE:
            return self._consensus_weighted_average(signals)
        elif self.consensus_strategy == ConsensusStrategy.UNANIMOUS:
            return self._consensus_unanimous(signals)
        else:
            # Default to highest confidence
            return self._consensus_highest_confidence(signals)

    def _consensus_highest_confidence(self, signals: list[AgentSignal]) -> AgentSignal:
        """Select signal with highest confidence."""
        # Weight by priority * confidence
        def score(s: AgentSignal) -> float:
            priority_val = int(s.priority)
            return priority_val * s.confidence

        return max(signals, key=score)

    def _consensus_majority_vote(
        self,
        signals: list[AgentSignal],
    ) -> AgentSignal | None:
        """Select direction with most votes, return highest confidence signal."""
        # Count votes weighted by confidence
        votes: Counter[SignalType] = Counter()
        for s in signals:
            votes[s.signal_type] += s.confidence

        if not votes:
            return None

        # Get winning direction
        winning_direction = votes.most_common(1)[0][0]

        # Return highest confidence signal with winning direction
        matching = [s for s in signals if s.signal_type == winning_direction]
        return max(matching, key=lambda s: s.confidence)

    def _consensus_weighted_average(
        self,
        signals: list[AgentSignal],
    ) -> AgentSignal | None:
        """Combine signals using weighted average confidence."""
        # Group by direction
        by_direction: dict[SignalType, list[AgentSignal]] = {}
        for s in signals:
            by_direction.setdefault(s.signal_type, []).append(s)

        # Calculate weighted confidence per direction
        best_direction: SignalType | None = None
        best_score = 0.0

        for direction, group in by_direction.items():
            total_weight = sum(int(s.priority) for s in group)
            weighted_conf = (
                sum(s.confidence * int(s.priority) for s in group) / total_weight
                if total_weight > 0
                else 0.0
            )

            if weighted_conf > best_score:
                best_score = weighted_conf
                best_direction = direction

        if best_direction is None:
            return None

        # Return representative signal
        matching = [s for s in signals if s.signal_type == best_direction]
        return max(matching, key=lambda s: s.confidence)

    def _consensus_unanimous(self, signals: list[AgentSignal]) -> AgentSignal | None:
        """Only return signal if all agents agree on direction."""
        directions = {s.signal_type for s in signals}

        if len(directions) != 1:
            log.debug(
                "No unanimous consensus",
                directions=[getattr(d, "value", d) for d in directions],
            )
            return None

        # All agree - return highest confidence
        return max(signals, key=lambda s: s.confidence)
