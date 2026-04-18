"""Claude Agent SDK integration for QTS-Architect.

Provides LLM-powered strategy agents that use Claude to analyze market data,
interpret macro context, and generate trading signals. Compatible with the
existing StrategyAgentProtocol and RiskAgentProtocol interfaces.

Architecture:
- ClaudeStrategyAgent: Uses Claude to analyze OHLCV + context → AgentSignal
- ClaudeRiskAgent: Uses Claude to evaluate risk with natural language reasoning
- ClaudeSentimentAgent: Analyzes news/sentiment via Claude tool_use

All agents use the Anthropic Python SDK with structured tool_use for
type-safe signal extraction (no free-text parsing).

Design Decisions:
- Agents are stateless per-call (context passed each invocation)
- Structured outputs via tool_use ensure valid SignalType/confidence
- Timeout + fallback to None on API errors (never blocks trading loop)
- Token budget caps prevent runaway costs
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Final

import structlog

from qts_core.agents.base import BaseRiskAgent, BaseStrategyAgent
from qts_core.agents.protocol import (
    AgentPriority,
    AgentSignal,
    ReviewRequest,
    RiskStatus,
    RiskVerdict,
    SignalType,
)
from qts_core.common.types import InstrumentId

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL: Final[str] = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS: Final[int] = 1024
DEFAULT_TIMEOUT_SECONDS: Final[float] = 15.0

# Tool definition for structured signal extraction
_SIGNAL_TOOL: dict[str, Any] = {
    "name": "emit_trading_signal",
    "description": (
        "Emit a trading signal after analyzing market data. "
        "Call this tool with your analysis conclusion."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "signal_type": {
                "type": "string",
                "enum": ["LONG", "SHORT", "NEUTRAL", "EXIT"],
                "description": "Directional conviction based on analysis.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score between 0 and 1.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the signal rationale.",
            },
            "pattern_detected": {
                "type": "string",
                "description": "Name of the technical pattern detected, if any.",
            },
        },
        "required": ["signal_type", "confidence", "reasoning"],
    },
}

_RISK_TOOL: dict[str, Any] = {
    "name": "emit_risk_verdict",
    "description": (
        "Emit a risk verdict after evaluating a proposed trade. "
        "Call this tool with your risk assessment."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["APPROVED", "REJECTED", "REDUCED"],
                "description": "Risk verdict status.",
            },
            "reason": {
                "type": "string",
                "description": "Human-readable explanation of the verdict.",
            },
            "adjusted_size": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Position size multiplier if REDUCED (default 1.0).",
            },
        },
        "required": ["status", "reason"],
    },
}


def _format_ohlcv_for_prompt(
    ohlcv_history: list[tuple[datetime, float, float, float, float, float]] | None,
    max_candles: int = 20,
) -> str:
    """Format OHLCV history into a compact text table for the LLM prompt."""
    if not ohlcv_history:
        return "No OHLCV history available."

    # Take the most recent candles
    candles = ohlcv_history[-max_candles:]
    lines = ["Timestamp | Open | High | Low | Close | Volume"]
    lines.append("-" * 60)
    for ts, o, h, l, c, v in candles:
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
        lines.append(f"{ts_str} | {o:.2f} | {h:.2f} | {l:.2f} | {c:.2f} | {v:.2f}")

    return "\n".join(lines)


# ==============================================================================
# Claude Strategy Agent
# ==============================================================================
class ClaudeStrategyAgent(BaseStrategyAgent):
    """LLM-powered strategy agent using Claude via the Anthropic SDK.

    Uses structured tool_use to extract typed signals from Claude's analysis.
    Falls back to None (no signal) on API errors or timeouts.

    Attributes:
        model: Claude model identifier.
        api_key: Anthropic API key (from env or config).
        system_prompt: Custom system prompt for the agent's persona.
        max_tokens: Maximum response tokens.
        timeout: API call timeout in seconds.
        extra_context: Optional additional context (e.g., news, macro data).
    """

    def __init__(
        self,
        name: str,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        extra_context: str = "",
        priority: AgentPriority = AgentPriority.HIGH,
        min_confidence: float = 0.6,
    ) -> None:
        """Initialize Claude Strategy Agent.

        Args:
            name: Agent identifier.
            model: Claude model to use.
            api_key: Anthropic API key. If None, reads ANTHROPIC_API_KEY env var.
            system_prompt: Custom system prompt. If None, uses a default.
            max_tokens: Max response tokens per call.
            timeout: API timeout in seconds.
            extra_context: Additional context injected into every prompt.
            priority: Signal priority for consensus.
            min_confidence: Minimum confidence to emit signals.
        """
        super().__init__(name=name, priority=priority, min_confidence=min_confidence)
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_context = extra_context

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(f"Agent '{name}' requires a valid Anthropic API key")

        # Lazy import to avoid hard dependency at module level
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

        self._system_prompt = system_prompt or (
            "You are a quantitative trading analyst agent. Your job is to analyze "
            "market data (OHLCV candles) and detect actionable patterns.\n\n"
            "Analysis framework:\n"
            "1. Identify trend direction (higher highs/lows vs lower highs/lows)\n"
            "2. Look for Fair Value Gaps, order blocks, and liquidity sweeps\n"
            "3. Assess momentum and volume confirmation\n"
            "4. Consider the kill zone timing context if provided\n\n"
            "Rules:\n"
            "- Only emit LONG or SHORT when you have clear evidence\n"
            "- Use NEUTRAL when the market is choppy or unclear\n"
            "- Confidence should reflect the quality of the setup (0.6-0.95 range)\n"
            "- Never force a signal — missing a trade is better than a bad entry\n"
            "- Always call the emit_trading_signal tool with your conclusion"
        )

    async def _generate_signal(
        self,
        instrument_id: InstrumentId,
        current_price: float,
        timestamp: datetime,
        ohlcv_history: list[tuple[datetime, float, float, float, float, float]]
        | None = None,
    ) -> AgentSignal | None:
        """Generate signal using Claude's analysis.

        Sends OHLCV data to Claude with tool_use, extracts structured signal.
        Returns None on any error (API timeout, invalid response, etc.).
        """
        ohlcv_text = _format_ohlcv_for_prompt(ohlcv_history)

        user_message = (
            f"Analyze the following market data for {instrument_id}.\n"
            f"Current price: {current_price:.4f}\n"
            f"Timestamp (UTC): {timestamp.isoformat()}\n\n"
            f"OHLCV History (most recent candles):\n{ohlcv_text}\n"
        )

        if self.extra_context:
            user_message += f"\nAdditional context:\n{self.extra_context}\n"

        user_message += (
            "\nAnalyze this data and call the emit_trading_signal tool "
            "with your conclusion."
        )

        try:
            response = await asyncio.wait_for(
                self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self._system_prompt,
                    tools=[_SIGNAL_TOOL],
                    tool_choice={"type": "tool", "name": "emit_trading_signal"},
                    messages=[{"role": "user", "content": user_message}],
                ),  # type: ignore[call-overload]
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            log.warning(
                "Claude API call timed out",
                agent=self.name,
                timeout=self.timeout,
            )
            return None
        except Exception as exc:
            log.error(
                "Claude API call failed",
                agent=self.name,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None

        # Extract tool_use result
        tool_result = self._extract_tool_result(response, "emit_trading_signal")
        if tool_result is None:
            log.warning(
                "Claude did not produce a valid tool_use response",
                agent=self.name,
            )
            return None

        # Parse structured output
        try:
            signal_type = SignalType(tool_result["signal_type"])
            confidence = float(tool_result["confidence"])
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(tool_result.get("reasoning", ""))
            pattern = str(tool_result.get("pattern_detected", ""))
        except (KeyError, ValueError) as exc:
            log.warning(
                "Invalid signal from Claude",
                agent=self.name,
                error=str(exc),
                raw=tool_result,
            )
            return None

        return AgentSignal(
            source_agent=self.name,
            signal_type=signal_type,
            confidence=confidence,
            priority=self.priority,
            timestamp=timestamp,
            metadata={
                "reasoning": reasoning,
                "pattern": pattern,
                "model": self.model,
            },
        )

    @staticmethod
    def _extract_tool_result(
        response: Any,
        tool_name: str,
    ) -> dict[str, Any] | None:
        """Extract tool_use input from Claude's response."""
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
                return block.input  # type: ignore[no-any-return]
        return None


# ==============================================================================
# Claude Risk Agent
# ==============================================================================
class ClaudeRiskAgent(BaseRiskAgent):
    """LLM-powered risk agent that uses Claude for nuanced risk evaluation.

    Combines hard-coded risk limits (from BaseRiskAgent) with Claude's
    reasoning for edge cases the rules can't handle (e.g., correlated
    positions, macro events, unusual volatility regimes).

    Attributes:
        model: Claude model identifier.
        api_key: Anthropic API key.
        timeout: API call timeout.
    """

    def __init__(
        self,
        name: str,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_position_size: float = 0.10,
        max_daily_loss: float = 0.02,
        max_short_exposure: float | None = None,
    ) -> None:
        """Initialize Claude Risk Agent."""
        super().__init__(
            name=name,
            max_position_size=max_position_size,
            max_daily_loss=max_daily_loss,
            max_short_exposure=max_short_exposure,
        )
        self.model = model
        self.timeout = timeout

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(f"Agent '{name}' requires a valid Anthropic API key")

        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

        self._system_prompt = (
            "You are a risk management officer for a quantitative trading system.\n\n"
            "Your job is to evaluate proposed trades and determine if they should be:\n"
            "- APPROVED: The trade passes risk checks\n"
            "- REJECTED: The trade violates risk limits or presents unacceptable risk\n"
            "- REDUCED: The trade is acceptable but should be smaller\n\n"
            "Consider:\n"
            "1. Current portfolio exposure and daily PnL\n"
            "2. Signal confidence relative to the risk being taken\n"
            "3. Correlation risk (are we doubling down on the same bet?)\n"
            "4. Market regime context\n\n"
            "Be conservative. When in doubt, REJECT or REDUCE.\n"
            "Always call the emit_risk_verdict tool with your conclusion."
        )

    async def _custom_evaluation(self, request: ReviewRequest) -> RiskVerdict:
        """Use Claude for nuanced risk evaluation beyond hard rules.

        The base class already handles hard limits (max_position_size,
        max_daily_loss). This method adds LLM reasoning for subtler risks.
        """
        user_message = (
            f"Evaluate this proposed trade:\n"
            f"- Instrument: {request.instrument_id}\n"
            f"- Signal: {request.proposed_signal.signal_type}\n"
            f"- Confidence: {request.proposed_signal.confidence:.2%}\n"
            f"- Current price: {request.current_price:.4f}\n"
            f"- Portfolio exposure: {request.portfolio_exposure:.2%}\n"
            f"- Daily PnL: {request.daily_pnl_fraction:.2%}\n"
            f"- Short exposure: {request.short_exposure_fraction:.2%}\n"
            f"- Source agent: {request.proposed_signal.source_agent}\n"
            f"- Signal metadata: {request.proposed_signal.metadata}\n\n"
            "Call emit_risk_verdict with your assessment."
        )

        try:
            response = await asyncio.wait_for(
                self._client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    system=self._system_prompt,
                    tools=[_RISK_TOOL],
                    tool_choice={"type": "tool", "name": "emit_risk_verdict"},
                    messages=[{"role": "user", "content": user_message}],
                ),  # type: ignore[call-overload]
                timeout=self.timeout,
            )
        except Exception as exc:
            log.error(
                "Claude risk evaluation failed, defaulting to REJECTED for safety",
                agent=self.name,
                error=str(exc),
            )
            # Fail-closed for risk: reject trade when we can't evaluate
            return RiskVerdict(
                status=RiskStatus.REJECTED,
                reason="Claude risk evaluation unavailable; rejecting for safety",
                risk_metrics={},
            )

        tool_result = ClaudeStrategyAgent._extract_tool_result(
            response, "emit_risk_verdict"
        )
        if tool_result is None:
            log.warning("Claude did not emit risk verdict", agent=self.name)
            return RiskVerdict(
                status=RiskStatus.REJECTED,
                reason="Claude did not emit verdict; rejecting for safety",
                risk_metrics={},
            )

        try:
            status = RiskStatus(tool_result["status"])
            reason = str(tool_result["reason"])
            adjusted_size = float(tool_result.get("adjusted_size", 1.0))
        except (KeyError, ValueError):
            log.warning("Invalid Claude risk output", agent=self.name, raw=tool_result)
            return RiskVerdict(
                status=RiskStatus.REJECTED,
                reason="Invalid Claude risk output; rejecting for safety",
                risk_metrics={},
            )

        return RiskVerdict(
            status=status,
            reason=f"[Claude Risk] {reason}",
            adjusted_size=max(0.0, min(1.0, adjusted_size)),
            risk_metrics={
                "signal_confidence": request.proposed_signal.confidence,
                "exposure": request.portfolio_exposure,
            },
        )


# ==============================================================================
# Claude Sentiment Agent
# ==============================================================================
class ClaudeSentimentAgent(BaseStrategyAgent):
    """Sentiment analysis agent that uses Claude to interpret market context.

    Unlike ClaudeStrategyAgent which analyzes price action, this agent
    focuses on qualitative inputs: news headlines, social sentiment,
    macro events, etc. Useful as a complementary signal in the ensemble.

    Attributes:
        model: Claude model identifier.
        news_fetcher: Optional callable that returns current news headlines.
    """

    def __init__(
        self,
        name: str,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        priority: AgentPriority = AgentPriority.NORMAL,
        min_confidence: float = 0.6,
    ) -> None:
        """Initialize Claude Sentiment Agent."""
        super().__init__(name=name, priority=priority, min_confidence=min_confidence)
        self.model = model
        self.timeout = timeout

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(f"Agent '{name}' requires a valid Anthropic API key")

        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

        self._system_prompt = (
            "You are a market sentiment analyst. Based on the price action "
            "and any provided news/context, assess the prevailing market "
            "sentiment for the given instrument.\n\n"
            "Focus on:\n"
            "1. Recent price momentum (last few candles)\n"
            "2. Volume patterns (increasing/decreasing)\n"
            "3. Any news or macro context provided\n\n"
            "Be conservative with confidence — sentiment is inherently noisy.\n"
            "Always call the emit_trading_signal tool."
        )

    async def _generate_signal(
        self,
        instrument_id: InstrumentId,
        current_price: float,
        timestamp: datetime,
        ohlcv_history: list[tuple[datetime, float, float, float, float, float]]
        | None = None,
    ) -> AgentSignal | None:
        """Generate sentiment-based signal using Claude."""
        ohlcv_text = _format_ohlcv_for_prompt(ohlcv_history, max_candles=10)

        user_message = (
            f"Assess market sentiment for {instrument_id}.\n"
            f"Current price: {current_price:.4f}\n"
            f"Timestamp: {timestamp.isoformat()}\n\n"
            f"Recent price action:\n{ohlcv_text}\n\n"
            "Based on the price action and volume, what is the sentiment? "
            "Call emit_trading_signal."
        )

        try:
            response = await asyncio.wait_for(
                self._client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    system=self._system_prompt,
                    tools=[_SIGNAL_TOOL],
                    tool_choice={"type": "tool", "name": "emit_trading_signal"},
                    messages=[{"role": "user", "content": user_message}],
                ),  # type: ignore[call-overload]
                timeout=self.timeout,
            )
        except Exception as exc:
            log.warning(
                "Claude sentiment analysis failed",
                agent=self.name,
                error=str(exc),
            )
            return None

        tool_result = ClaudeStrategyAgent._extract_tool_result(
            response, "emit_trading_signal"
        )
        if tool_result is None:
            return None

        try:
            signal_type = SignalType(tool_result["signal_type"])
            confidence = max(0.0, min(1.0, float(tool_result["confidence"])))
            reasoning = str(tool_result.get("reasoning", ""))
        except (KeyError, ValueError):
            return None

        return AgentSignal(
            source_agent=self.name,
            signal_type=signal_type,
            confidence=confidence,
            priority=self.priority,
            timestamp=timestamp,
            metadata={
                "reasoning": reasoning,
                "analysis_type": "sentiment",
                "model": self.model,
            },
        )
