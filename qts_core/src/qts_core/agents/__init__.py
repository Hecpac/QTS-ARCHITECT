"""Multi-agent trading system for QTS-Architect.

This module provides the agent framework for signal generation,
risk management, and decision orchestration.

Architecture:
    StrategyAgents → Supervisor → RiskAgent → TradingDecision

Quick Start:
    ```python
    from qts_core.agents import (
        Supervisor,
        ICTSmartMoneyAgent,
        StrictRiskAgent,
        ConsensusStrategy,
    )

    # Create agents
    ict = ICTSmartMoneyAgent(name="ICT_NY", symbol="BTC/USDT")
    risk = StrictRiskAgent(name="RiskOfficer")

    # Create supervisor
    supervisor = Supervisor(
        strategy_agents=[ict],
        risk_agent=risk,
        consensus_strategy=ConsensusStrategy.HIGHEST_CONFIDENCE,
    )

    # Run decision cycle
    market_data = MarketData(
        instrument_id=InstrumentId("BTC/USDT"),
        timestamp=datetime.now(timezone.utc),
        open=50000.0,
        high=50000.0,
        low=50000.0,
        close=50000.0,
        volume=0.0,
    )
    decision = await supervisor.run(market_data)
    ```
"""

from qts_core.agents.base import (
    BaseRiskAgent,
    BaseStrategyAgent,
    PermissiveRiskAgent,
    StrictRiskAgent,
    TechnicalAgent,
)
from qts_core.agents.ict import ICTSmartMoneyAgent
from qts_core.agents.protocol import (
    AgentPriority,
    AgentSignal,
    ReviewRequest,
    RiskAgentProtocol,
    RiskStatus,
    RiskVerdict,
    SignalType,
    StrategyAgentProtocol,
    TradingDecision,
)
from qts_core.agents.supervisor import ConsensusStrategy, Supervisor


__all__ = [
    "AgentPriority",
    "AgentSignal",
    "BaseRiskAgent",
    "BaseStrategyAgent",
    "ConsensusStrategy",
    "ICTSmartMoneyAgent",
    "PermissiveRiskAgent",
    "ReviewRequest",
    "RiskAgentProtocol",
    "RiskStatus",
    "RiskVerdict",
    "SignalType",
    "StrategyAgentProtocol",
    "StrictRiskAgent",
    "Supervisor",
    "TechnicalAgent",
    "TradingDecision",
]
