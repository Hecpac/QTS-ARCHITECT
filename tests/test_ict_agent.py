import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Ensure src/ is on the path for direct pytest runs
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from qts_core.agents.ict import ICTSmartMoneyAgent
from qts_core.agents.protocol import SignalType
from qts_core.common.types import InstrumentId


def test_ict_agent_detects_bullish_fvg():
    """Detecta un Fair Value Gap (FVG) alcista con 3 velas OHLCV."""

    async def run_flow() -> None:
        agent = ICTSmartMoneyAgent(name="Sniper_Test", symbol="BTC/USDT")

        # Ensure we're inside the default kill zone (13-16 UTC)
        now = datetime.now(timezone.utc).replace(hour=14, minute=0, second=0, microsecond=0)

        # OHLCV tuples: (timestamp, open, high, low, close, volume)
        candles = [
            (now - timedelta(hours=2), 90.0, 100.0, 85.0, 95.0, 1000.0),
            (now - timedelta(hours=1), 101.0, 112.0, 101.0, 110.0, 5000.0),
            (now, 108.0, 115.0, 105.0, 112.0, 2000.0),
        ]

        signal = await agent.analyze(
            InstrumentId("BTC/USDT"),
            current_price=112.0,
            timestamp=now,
            ohlcv_history=candles,
        )

        assert signal is not None, "El agente debería haber detectado el patrón alcista"
        assert signal.signal_type == SignalType.LONG
        assert signal.confidence >= 0.8
        assert signal.metadata.get("pattern") == "Bullish FVG"

    asyncio.run(run_flow())
