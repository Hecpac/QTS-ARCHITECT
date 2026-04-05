"""Tests for Claude Agent SDK integration.

Tests the ClaudeStrategyAgent, ClaudeRiskAgent, and ClaudeSentimentAgent
using mocked Anthropic API responses to avoid real API calls.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qts_core.agents.claude_agent import (
    ClaudeRiskAgent,
    ClaudeSentimentAgent,
    ClaudeStrategyAgent,
    _format_ohlcv_for_prompt,
)
from qts_core.agents.protocol import (
    AgentSignal,
    RiskStatus,
    RiskVerdict,
    SignalType,
)
from qts_core.common.types import InstrumentId


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def instrument_id() -> InstrumentId:
    return InstrumentId("BTC/USDT")


@pytest.fixture
def timestamp() -> datetime:
    return datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def ohlcv_history(timestamp: datetime) -> list[tuple[datetime, float, float, float, float, float]]:
    """3 candles with a bullish pattern."""
    from datetime import timedelta

    return [
        (timestamp - timedelta(hours=2), 100.0, 102.0, 99.0, 101.0, 500.0),
        (timestamp - timedelta(hours=1), 101.0, 105.0, 100.5, 104.0, 800.0),
        (timestamp, 104.0, 108.0, 103.0, 107.0, 1200.0),
    ]


def _make_tool_use_response(
    tool_name: str,
    tool_input: dict[str, Any],
) -> SimpleNamespace:
    """Create a mock Claude API response with a tool_use block."""
    tool_block = SimpleNamespace(
        type="tool_use",
        name=tool_name,
        input=tool_input,
    )
    return SimpleNamespace(content=[tool_block])


# ---------------------------------------------------------------------------
# ClaudeStrategyAgent Tests
# ---------------------------------------------------------------------------
class TestClaudeStrategyAgent:
    """Tests for ClaudeStrategyAgent."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        client = AsyncMock()
        client.messages = AsyncMock()
        return client

    @pytest.fixture
    def agent(self, mock_client: AsyncMock) -> ClaudeStrategyAgent:
        with patch("qts_core.agents.claude_agent.anthropic", create=True) as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            agent = ClaudeStrategyAgent(
                name="test_claude",
                api_key="test-key",
                min_confidence=0.5,
            )
        agent._client = mock_client
        return agent

    async def test_generates_long_signal(
        self,
        agent: ClaudeStrategyAgent,
        mock_client: AsyncMock,
        instrument_id: InstrumentId,
        timestamp: datetime,
        ohlcv_history: list,
    ) -> None:
        """Claude returns a LONG signal via tool_use."""
        mock_client.messages.create = AsyncMock(
            return_value=_make_tool_use_response(
                "emit_trading_signal",
                {
                    "signal_type": "LONG",
                    "confidence": 0.85,
                    "reasoning": "Bullish FVG detected with momentum",
                    "pattern_detected": "Fair Value Gap",
                },
            )
        )

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=107.0,
            timestamp=timestamp,
            ohlcv_history=ohlcv_history,
        )

        assert signal is not None
        assert signal.signal_type == SignalType.LONG
        assert signal.confidence == 0.85
        assert signal.source_agent == "test_claude"
        assert signal.metadata["reasoning"] == "Bullish FVG detected with momentum"
        assert signal.metadata["pattern"] == "Fair Value Gap"

    async def test_returns_none_on_timeout(
        self,
        agent: ClaudeStrategyAgent,
        mock_client: AsyncMock,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """Agent returns None when Claude API times out."""
        mock_client.messages.create = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=100.0,
            timestamp=timestamp,
        )

        assert signal is None

    async def test_returns_none_on_api_error(
        self,
        agent: ClaudeStrategyAgent,
        mock_client: AsyncMock,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """Agent returns None when Claude API raises an error."""
        mock_client.messages.create = AsyncMock(
            side_effect=Exception("API rate limit exceeded")
        )

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=100.0,
            timestamp=timestamp,
        )

        assert signal is None

    async def test_returns_none_on_invalid_tool_response(
        self,
        agent: ClaudeStrategyAgent,
        mock_client: AsyncMock,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """Agent returns None when Claude returns invalid tool_use data."""
        mock_client.messages.create = AsyncMock(
            return_value=_make_tool_use_response(
                "emit_trading_signal",
                {"signal_type": "INVALID", "confidence": 999},
            )
        )

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=100.0,
            timestamp=timestamp,
        )

        assert signal is None

    async def test_neutral_signal_suppressed_by_confidence(
        self,
        agent: ClaudeStrategyAgent,
        mock_client: AsyncMock,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """NEUTRAL signals with low confidence are filtered out."""
        mock_client.messages.create = AsyncMock(
            return_value=_make_tool_use_response(
                "emit_trading_signal",
                {
                    "signal_type": "LONG",
                    "confidence": 0.3,
                    "reasoning": "Weak signal",
                },
            )
        )

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=100.0,
            timestamp=timestamp,
        )

        # min_confidence is 0.5, signal confidence is 0.3 → suppressed
        assert signal is None

    async def test_confidence_clamped_to_valid_range(
        self,
        agent: ClaudeStrategyAgent,
        mock_client: AsyncMock,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """Confidence values outside [0,1] are clamped."""
        mock_client.messages.create = AsyncMock(
            return_value=_make_tool_use_response(
                "emit_trading_signal",
                {
                    "signal_type": "SHORT",
                    "confidence": 1.5,  # Over 1.0
                    "reasoning": "Very confident",
                },
            )
        )

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=100.0,
            timestamp=timestamp,
        )

        assert signal is not None
        assert signal.confidence == 1.0


# ---------------------------------------------------------------------------
# ClaudeRiskAgent Tests
# ---------------------------------------------------------------------------
class TestClaudeRiskAgent:
    """Tests for ClaudeRiskAgent."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        client = AsyncMock()
        client.messages = AsyncMock()
        return client

    @pytest.fixture
    def agent(self, mock_client: AsyncMock) -> ClaudeRiskAgent:
        with patch("qts_core.agents.claude_agent.anthropic", create=True) as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = mock_client
            agent = ClaudeRiskAgent(
                name="test_risk",
                api_key="test-key",
                max_position_size=0.15,
            )
        agent._client = mock_client
        return agent

    async def test_approves_valid_trade(
        self,
        agent: ClaudeRiskAgent,
        mock_client: AsyncMock,
    ) -> None:
        """Claude approves a trade that passes all checks."""
        from qts_core.agents.protocol import AgentPriority, ReviewRequest

        mock_client.messages.create = AsyncMock(
            return_value=_make_tool_use_response(
                "emit_risk_verdict",
                {
                    "status": "APPROVED",
                    "reason": "Trade within acceptable risk parameters",
                },
            )
        )

        signal = AgentSignal(
            source_agent="test",
            signal_type=SignalType.LONG,
            confidence=0.8,
            priority=AgentPriority.HIGH,
            metadata={},
        )

        request = ReviewRequest(
            proposed_signal=signal,
            instrument_id=InstrumentId("BTC/USDT"),
            current_price=100.0,
            portfolio_exposure=0.05,
            daily_pnl_fraction=0.0,
        )

        verdict = await agent.evaluate(request)
        assert verdict.status == RiskStatus.APPROVED

    async def test_falls_back_to_rejected_on_api_failure(
        self,
        agent: ClaudeRiskAgent,
        mock_client: AsyncMock,
    ) -> None:
        """Risk agent defaults to REJECTED on API failure (fail-closed for safety)."""
        from qts_core.agents.protocol import AgentPriority, ReviewRequest

        mock_client.messages.create = AsyncMock(
            side_effect=Exception("API down")
        )

        signal = AgentSignal(
            source_agent="test",
            signal_type=SignalType.LONG,
            confidence=0.8,
            priority=AgentPriority.HIGH,
            metadata={},
        )

        request = ReviewRequest(
            proposed_signal=signal,
            instrument_id=InstrumentId("BTC/USDT"),
            current_price=100.0,
            portfolio_exposure=0.05,
            daily_pnl_fraction=0.0,
        )

        verdict = await agent.evaluate(request)
        assert verdict.status == RiskStatus.REJECTED
        assert "safety" in verdict.reason.lower()


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------
class TestFormatOHLCV:
    """Tests for _format_ohlcv_for_prompt."""

    def test_empty_history(self) -> None:
        assert "No OHLCV" in _format_ohlcv_for_prompt(None)
        assert "No OHLCV" in _format_ohlcv_for_prompt([])

    def test_formats_candles(self, timestamp: datetime) -> None:
        history = [
            (timestamp, 100.0, 105.0, 95.0, 102.0, 500.0),
        ]
        result = _format_ohlcv_for_prompt(history)
        assert "100.00" in result
        assert "105.00" in result
        assert "500.00" in result

    def test_respects_max_candles(self, timestamp: datetime) -> None:
        from datetime import timedelta

        history = [
            (timestamp + timedelta(hours=i), 100.0, 105.0, 95.0, 102.0, 500.0)
            for i in range(50)
        ]
        result = _format_ohlcv_for_prompt(history, max_candles=5)
        # Header + separator + 5 data lines = 7 lines
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 7
