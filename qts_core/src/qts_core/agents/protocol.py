"""Agent communication protocols and message types.

This module defines the contract between agents in the multi-agent system.
All messages are immutable Pydantic models for type safety and serialization.

Message Flow:
    MarketData → StrategyAgent → AgentSignal → Supervisor → ReviewRequest
    → RiskAgent → RiskVerdict → TradingDecision → OMS
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from qts_core.common.types import MarketData

from qts_core.common.types import InstrumentId


# ==============================================================================
# Enumerations
# ==============================================================================
class SignalType(str, Enum):
    """Directional conviction of an agent.

    Attributes:
        LONG: Bullish signal - expect price to rise.
        SHORT: Bearish signal - expect price to fall.
        NEUTRAL: No directional bias - stay flat or exit.
        EXIT: Close existing position.
    """

    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"
    EXIT = "EXIT"


class RiskStatus(str, Enum):
    """Outcome of a risk review.

    Attributes:
        APPROVED: Trade passes all risk checks.
        REJECTED: Trade violates risk limits.
        REDUCED: Trade approved with reduced size.
    """

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REDUCED = "REDUCED"


class AgentPriority(int, Enum):
    """Priority levels for agent signals.

    Higher priority signals take precedence in consensus.
    """

    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 100


# ==============================================================================
# Agent Signal (Output from Strategy Agents)
# ==============================================================================
class AgentSignal(BaseModel):
    """Standardized output from a Strategy Agent.

    Signals represent the agent's view on market direction with confidence.
    They are collected by the Supervisor for consensus building.

    Attributes:
        source_agent: Name of the agent that generated this signal.
        signal_type: Directional bias.
        confidence: Confidence score in [0, 1] range.
        priority: Signal priority for consensus weighting.
        timestamp: When the signal was generated (UTC).
        metadata: Agent-specific contextual data.
    """

    model_config = ConfigDict(frozen=True, use_enum_values=True)

    source_agent: str = Field(..., min_length=1, description="Agent identifier")
    signal_type: SignalType
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score in [0, 1]",
    )
    priority: AgentPriority = Field(
        default=AgentPriority.NORMAL,
        description="Signal priority for consensus",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Signal generation time (UTC)",
    )
    metadata: dict[str, str | float | int | bool] = Field(
        default_factory=dict,
        description="Agent-specific context",
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


# ==============================================================================
# Risk Review Request (Supervisor → Risk Agent)
# ==============================================================================
class ReviewRequest(BaseModel):
    """Packet sent to the Risk Agent for validation.

    Contains the proposed signal and market context for risk evaluation.

    Attributes:
        request_id: Unique identifier for tracing.
        timestamp: Request creation time.
        proposed_signal: The signal to be reviewed.
        instrument_id: Target instrument.
        current_price: Latest market price.
        portfolio_exposure: Current exposure to this instrument.
        daily_pnl_fraction: Session/day PnL as fraction of start-of-day value.
            Negative values indicate losses (e.g., -0.02 = -2%).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    proposed_signal: AgentSignal
    instrument_id: InstrumentId
    current_price: float = Field(..., gt=0, description="Current market price")
    portfolio_exposure: float = Field(
        default=0.0,
        description="Current position value as fraction of portfolio",
    )
    daily_pnl_fraction: float = Field(
        default=0.0,
        description="Session/day PnL fraction from start-of-day portfolio value",
    )


# ==============================================================================
# Risk Verdict (Risk Agent → Supervisor)
# ==============================================================================
class RiskVerdict(BaseModel):
    """The definitive judgment from the Risk Agent.

    Attributes:
        status: Approval status.
        reason: Human-readable explanation.
        adjusted_size: If REDUCED, the allowed position size multiplier.
        risk_metrics: Calculated risk metrics for logging.
    """

    model_config = ConfigDict(frozen=True, use_enum_values=True)

    status: RiskStatus
    reason: str = Field(..., min_length=1, description="Explanation")
    adjusted_size: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Size multiplier if REDUCED",
    )
    risk_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Risk calculations (VaR, exposure, etc.)",
    )


# ==============================================================================
# Trading Decision (Supervisor → OMS)
# ==============================================================================
class TradingDecision(BaseModel):
    """The final output of the Supervisor, ready for execution.

    This is the immutable record of the decision-making process.

    Attributes:
        decision_id: Unique identifier for audit trail.
        created_at: Decision timestamp.
        instrument_id: Target instrument.
        action: Trade direction.
        quantity_modifier: Size scaling factor (0-1).
        rationale: Human-readable explanation.
        contributing_agents: Agents that contributed to this decision.
    """

    model_config = ConfigDict(frozen=True, use_enum_values=True)

    decision_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    instrument_id: InstrumentId
    action: SignalType
    quantity_modifier: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Size scaling factor",
    )
    rationale: str = Field(..., min_length=1)
    contributing_agents: list[str] = Field(
        default_factory=list,
        description="Agents that supported this decision",
    )


# ==============================================================================
# Agent Protocols (Structural Subtyping)
# ==============================================================================
@runtime_checkable
class StrategyAgentProtocol(Protocol):
    """Protocol for strategy agents.

    Any class with a matching `analyze` method signature is compatible.
    This enables duck typing while maintaining type safety.
    """

    name: str

    async def analyze(
        self,
        instrument_id: InstrumentId,
        current_price: float,
        timestamp: datetime,
        ohlcv_history: list[tuple[datetime, float, float, float, float, float]] | None = None,
    ) -> AgentSignal | None:
        """Analyze market data and produce a signal.

        Args:
            instrument_id: The instrument being analyzed.
            current_price: Latest price.
            timestamp: Current timestamp.
            ohlcv_history: Optional historical OHLCV data as list of tuples.

        Returns:
            AgentSignal if there's a trading opportunity, None otherwise.
        """
        ...


@runtime_checkable
class RiskAgentProtocol(Protocol):
    """Protocol for risk agents.

    Risk agents have veto power over trading decisions.
    """

    name: str

    async def evaluate(self, request: ReviewRequest) -> RiskVerdict:
        """Evaluate a trading request against risk parameters.

        Args:
            request: The review request containing proposed signal and context.

        Returns:
            RiskVerdict with approval status and metrics.
        """
        ...
