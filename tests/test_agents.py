"""Tests for the agent system.

Tests cover:
- Agent signal generation
- Risk evaluation
- Supervisor consensus strategies
- ICT FVG detection
"""

from datetime import datetime, timezone

import pytest

from qts_core.agents import (
    AgentPriority,
    AgentSignal,
    ConsensusStrategy,
    ICTSmartMoneyAgent,
    PermissiveRiskAgent,
    ReviewRequest,
    RiskStatus,
    SignalType,
    StrictRiskAgent,
    Supervisor,
    TechnicalAgent,
)
from qts_core.common.types import InstrumentId, MarketData


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def instrument_id() -> InstrumentId:
    """Sample instrument ID."""
    return InstrumentId("BTC/USDT")


@pytest.fixture
def timestamp() -> datetime:
    """Sample timestamp in kill zone (14:00 UTC)."""
    return datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def bullish_ohlcv_history() -> list[tuple[datetime, float, float, float, float, float]]:
    """OHLCV history with bullish FVG pattern."""
    base_ts = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)
    return [
        # (timestamp, open, high, low, close, volume)
        # Candle 1: 100-102
        (base_ts, 100.0, 102.0, 99.5, 101.0, 1000.0),
        # Candle 2: Impulsive bullish (101-108)
        (base_ts, 101.0, 108.0, 100.5, 107.0, 2000.0),
        # Candle 3: Gap up, low (105) > candle1 high (102)
        (base_ts, 106.0, 110.0, 105.0, 109.0, 1500.0),
    ]


@pytest.fixture
def neutral_ohlcv_history() -> list[tuple[datetime, float, float, float, float, float]]:
    """OHLCV history with no FVG pattern."""
    base_ts = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)
    return [
        (base_ts, 100.0, 102.0, 99.0, 101.0, 1000.0),
        (base_ts, 101.0, 103.0, 100.0, 102.0, 1000.0),
        (base_ts, 102.0, 104.0, 101.0, 103.0, 1000.0),
    ]


# ==============================================================================
# Technical Agent Tests
# ==============================================================================
class TestTechnicalAgent:
    """Tests for TechnicalAgent."""

    @pytest.mark.asyncio
    async def test_bullish_signal_on_green_candle(
        self,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """Should emit LONG signal on green candle."""
        agent = TechnicalAgent(name="test_tech", min_confidence=0.0)

        # Green candle: close > open
        history = [(timestamp, 100.0, 105.0, 99.0, 104.0, 1000.0)]

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=104.0,
            timestamp=timestamp,
            ohlcv_history=history,
        )

        assert signal is not None
        assert signal.signal_type == SignalType.LONG
        assert signal.confidence > 0.5

    @pytest.mark.asyncio
    async def test_bearish_signal_on_red_candle(
        self,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """Should emit SHORT signal on red candle."""
        agent = TechnicalAgent(name="test_tech", min_confidence=0.0)

        # Red candle: close < open
        history = [(timestamp, 105.0, 106.0, 99.0, 100.0, 1000.0)]

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=100.0,
            timestamp=timestamp,
            ohlcv_history=history,
        )

        assert signal is not None
        assert signal.signal_type == SignalType.SHORT

    @pytest.mark.asyncio
    async def test_no_signal_without_history(
        self,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """Should return None without OHLCV history."""
        agent = TechnicalAgent(name="test_tech")

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=100.0,
            timestamp=timestamp,
            ohlcv_history=None,
        )

        assert signal is None


# ==============================================================================
# ICT Agent Tests
# ==============================================================================
class TestICTSmartMoneyAgent:
    """Tests for ICTSmartMoneyAgent."""

    @pytest.mark.asyncio
    async def test_detects_bullish_fvg_in_kill_zone(
        self,
        instrument_id: InstrumentId,
        timestamp: datetime,
        bullish_ohlcv_history: list,
    ) -> None:
        """Should detect bullish FVG during kill zone."""
        agent = ICTSmartMoneyAgent(
            name="ict_test",
            symbol="BTC/USDT",
            session_start=13,
            session_end=16,
            min_fvg_size=0.001,
        )

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=109.0,
            timestamp=timestamp,  # 14:00 UTC - in kill zone
            ohlcv_history=bullish_ohlcv_history,
        )

        assert signal is not None
        assert signal.signal_type == SignalType.LONG
        assert signal.metadata.get("pattern") == "Bullish FVG"

    @pytest.mark.asyncio
    async def test_no_signal_outside_kill_zone(
        self,
        instrument_id: InstrumentId,
        bullish_ohlcv_history: list,
    ) -> None:
        """Should not emit signal outside kill zone."""
        agent = ICTSmartMoneyAgent(
            name="ict_test",
            symbol="BTC/USDT",
            session_start=13,
            session_end=16,
        )

        # 10:00 UTC - outside kill zone
        outside_ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=109.0,
            timestamp=outside_ts,
            ohlcv_history=bullish_ohlcv_history,
        )

        assert signal is None

    @pytest.mark.asyncio
    async def test_no_signal_without_fvg(
        self,
        instrument_id: InstrumentId,
        timestamp: datetime,
        neutral_ohlcv_history: list,
    ) -> None:
        """Should not emit signal without FVG pattern."""
        agent = ICTSmartMoneyAgent(
            name="ict_test",
            symbol="BTC/USDT",
            session_start=13,
            session_end=16,
        )

        signal = await agent.analyze(
            instrument_id=instrument_id,
            current_price=103.0,
            timestamp=timestamp,
            ohlcv_history=neutral_ohlcv_history,
        )

        assert signal is None


# ==============================================================================
# Risk Agent Tests
# ==============================================================================
class TestRiskAgents:
    """Tests for risk agents."""

    @pytest.mark.asyncio
    async def test_strict_agent_rejects_low_confidence(
        self,
        instrument_id: InstrumentId,
    ) -> None:
        """StrictRiskAgent should reject low confidence signals."""
        agent = StrictRiskAgent(
            name="strict_risk",
            min_signal_confidence=0.7,
        )

        signal = AgentSignal(
            source_agent="test",
            signal_type=SignalType.LONG,
            confidence=0.5,  # Below threshold
        )

        request = ReviewRequest(
            proposed_signal=signal,
            instrument_id=instrument_id,
            current_price=100.0,
        )

        verdict = await agent.evaluate(request)

        assert verdict.status == RiskStatus.REJECTED

    @pytest.mark.asyncio
    async def test_strict_agent_approves_high_confidence(
        self,
        instrument_id: InstrumentId,
    ) -> None:
        """StrictRiskAgent should approve high confidence signals."""
        agent = StrictRiskAgent(
            name="strict_risk",
            min_signal_confidence=0.7,
        )

        signal = AgentSignal(
            source_agent="test",
            signal_type=SignalType.LONG,
            confidence=0.85,
        )

        request = ReviewRequest(
            proposed_signal=signal,
            instrument_id=instrument_id,
            current_price=100.0,
        )

        verdict = await agent.evaluate(request)

        assert verdict.status == RiskStatus.APPROVED

    @pytest.mark.asyncio
    async def test_permissive_agent_approves_all(
        self,
        instrument_id: InstrumentId,
    ) -> None:
        """PermissiveRiskAgent should approve all signals."""
        agent = PermissiveRiskAgent(name="permissive")

        signal = AgentSignal(
            source_agent="test",
            signal_type=SignalType.LONG,
            confidence=0.3,  # Very low
        )

        request = ReviewRequest(
            proposed_signal=signal,
            instrument_id=instrument_id,
            current_price=100.0,
        )

        verdict = await agent.evaluate(request)

        assert verdict.status == RiskStatus.APPROVED

    @pytest.mark.asyncio
    async def test_exit_signal_bypasses_exposure_rejection(
        self,
        instrument_id: InstrumentId,
    ) -> None:
        """EXIT should be approved even when exposure is above max limit."""
        agent = StrictRiskAgent(
            name="strict_risk",
            min_signal_confidence=0.99,
            max_position_size=0.10,
        )

        signal = AgentSignal(
            source_agent="test",
            signal_type=SignalType.EXIT,
            confidence=0.1,
        )

        request = ReviewRequest(
            proposed_signal=signal,
            instrument_id=instrument_id,
            current_price=100.0,
            portfolio_exposure=0.50,
        )

        verdict = await agent.evaluate(request)

        assert verdict.status == RiskStatus.APPROVED


# ==============================================================================
# Supervisor Tests
# ==============================================================================
class TestSupervisor:
    """Tests for Supervisor."""

    @pytest.mark.asyncio
    async def test_highest_confidence_consensus(
        self,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """Should select signal with highest confidence."""
        # Create agents with different confidence outputs
        agent1 = TechnicalAgent(name="agent1", min_confidence=0.0)
        agent2 = TechnicalAgent(name="agent2", min_confidence=0.0)
        risk = PermissiveRiskAgent(name="risk")

        supervisor = Supervisor(
            strategy_agents=[agent1, agent2],
            risk_agent=risk,
            consensus_strategy=ConsensusStrategy.HIGHEST_CONFIDENCE,
            min_confidence=0.0,
        )

        # Strong bullish candle
        history = [(timestamp, 100.0, 110.0, 99.0, 109.0, 1000.0)]

        market_data = MarketData(
            instrument_id=instrument_id,
            timestamp=timestamp,
            open=100.0,
            high=110.0,
            low=99.0,
            close=109.0,
            volume=1000.0,
        )

        decision = await supervisor.run(market_data, ohlcv_history=history)

        assert decision is not None
        assert decision.action == SignalType.LONG

    @pytest.mark.asyncio
    async def test_no_decision_without_signals(
        self,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """Should return None if no signals generated."""
        agent = TechnicalAgent(name="agent1", min_confidence=0.99)  # Very high threshold
        risk = PermissiveRiskAgent(name="risk")

        supervisor = Supervisor(
            strategy_agents=[agent],
            risk_agent=risk,
        )

        # Weak candle - won't meet confidence threshold
        history = [(timestamp, 100.0, 100.1, 99.9, 100.05, 1000.0)]

        market_data = MarketData(
            instrument_id=instrument_id,
            timestamp=timestamp,
            open=100.0,
            high=100.1,
            low=99.9,
            close=100.05,
            volume=1000.0,
        )

        decision = await supervisor.run(market_data, ohlcv_history=history)

        assert decision is None

    @pytest.mark.asyncio
    async def test_risk_veto_prevents_decision(
        self,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """Risk agent veto should prevent trading decision."""
        agent = TechnicalAgent(name="agent1", min_confidence=0.0)
        risk = StrictRiskAgent(name="risk", min_signal_confidence=0.99)  # Very strict

        supervisor = Supervisor(
            strategy_agents=[agent],
            risk_agent=risk,
            min_confidence=0.0,
        )

        history = [(timestamp, 100.0, 105.0, 99.0, 104.0, 1000.0)]

        market_data = MarketData(
            instrument_id=instrument_id,
            timestamp=timestamp,
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000.0,
        )

        decision = await supervisor.run(market_data, ohlcv_history=history)

        # Signal generated but rejected by risk
        assert decision is None

    @pytest.mark.asyncio
    async def test_exit_signal_is_prioritized(
        self,
        instrument_id: InstrumentId,
        timestamp: datetime,
    ) -> None:
        """Supervisor should prioritize EXIT signals over directional consensus."""

        class ExitAgent:
            name = "exit_agent"

            async def analyze(self, **_kwargs):
                return AgentSignal(
                    source_agent=self.name,
                    signal_type=SignalType.EXIT,
                    confidence=0.2,
                    priority=AgentPriority.HIGH,
                    timestamp=timestamp,
                )

        long_agent = TechnicalAgent(name="long_agent", min_confidence=0.0)
        risk = StrictRiskAgent(name="risk", min_signal_confidence=0.99)

        supervisor = Supervisor(
            strategy_agents=[long_agent, ExitAgent()],
            risk_agent=risk,
            consensus_strategy=ConsensusStrategy.HIGHEST_CONFIDENCE,
            min_confidence=0.95,
        )

        # Strong bullish candle would normally emit LONG if EXIT were not prioritized.
        history = [(timestamp, 100.0, 110.0, 99.0, 109.0, 1000.0)]
        market_data = MarketData(
            instrument_id=instrument_id,
            timestamp=timestamp,
            open=100.0,
            high=110.0,
            low=99.0,
            close=109.0,
            volume=1000.0,
        )

        decision = await supervisor.run(market_data, ohlcv_history=history)

        assert decision is not None
        assert decision.action == SignalType.EXIT

    def test_supervisor_requires_at_least_one_agent(self) -> None:
        """Supervisor should require at least one strategy agent."""
        risk = PermissiveRiskAgent(name="risk")

        with pytest.raises(ValueError, match="At least one strategy agent"):
            Supervisor(strategy_agents=[], risk_agent=risk)
