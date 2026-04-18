"""Backtest: Gold EMA/RSI optimized strategy via QTS engine.

Uses XAUT/USDT from OKX (gold-backed token) with the grid-search
optimized parameters: EMA 12/21, RSI 14, entry <70, exit >85,
SL 3%, trailing 1.5%.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


async def download_ohlcv(
    symbol: str = "XAUT/USDT",
    timeframe: str = "1h",
    since_iso: str = "2025-10-01T00:00:00Z",
    limit_per_call: int = 100,
) -> pl.DataFrame:
    import ccxt.async_support as ccxt_async

    exchange = ccxt_async.okx({"enableRateLimit": True})
    try:
        since_ms = int(
            datetime.fromisoformat(since_iso.replace("Z", "+00:00")).timestamp() * 1000
        )
        all_rows: list[list] = []
        while True:
            candles = await exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=limit_per_call,
            )
            if not candles:
                break
            all_rows.extend(candles)
            last_ts = candles[-1][0]
            if last_ts <= since_ms:
                break
            since_ms = last_ts + 1
            if len(candles) < limit_per_call:
                break
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc)
                    for r in all_rows
                ],
                "instrument_id": [symbol] * len(all_rows),
                "open": [float(r[1]) for r in all_rows],
                "high": [float(r[2]) for r in all_rows],
                "low": [float(r[3]) for r in all_rows],
                "close": [float(r[4]) for r in all_rows],
                "volume": [float(r[5]) for r in all_rows],
            }
        )
    finally:
        await exchange.close()


async def run() -> None:
    from qts_core.agents.base import StrictRiskAgent
    from qts_core.agents.ema_rsi import EmaRsiAgent
    from qts_core.agents.supervisor import Supervisor
    from qts_core.backtest.engine import BacktestConfig, EventEngine

    print("Downloading XAUT/USDT 1h (6+ months from OKX)...")
    df = await download_ohlcv(since_iso="2025-10-01T00:00:00Z")
    print(f"  -> {len(df)} bars from {df['timestamp'][0]} to {df['timestamp'][-1]}")

    # --- Strategy agent ---
    ema_rsi = EmaRsiAgent(
        name="Gold_EMA_RSI",
        symbol="XAUT/USDT",
        ema_fast=12,
        ema_slow=21,
        rsi_period=14,
        rsi_entry_max=70.0,
        rsi_exit_min=85.0,
        stop_loss_pct=0.03,
        trailing_stop_pct=0.015,
        min_confidence=0.55,
    )

    risk = StrictRiskAgent(
        name="Gold_Risk_Guard",
        min_signal_confidence=0.55,
        max_position_size=0.50,
        max_short_exposure=0.0,
    )

    supervisor = Supervisor(
        strategy_agents=[ema_rsi],
        risk_agent=risk,
        min_confidence=0.55,
    )

    config = BacktestConfig(
        initial_capital=100_000.0,
        trade_size=50_000.0,       # 50% per trade
        slippage_bps=5.0,
        commission_bps=10.0,
        stop_loss_pct=0.03,
        risk_fraction=0.10,        # 10% risk fraction (was 4.5%)
    )

    print("\n> Running Gold EMA/RSI optimized backtest...")
    engine = EventEngine(supervisor, config)
    result = await engine.run(df)

    m = result.metrics
    ret = (result.final_capital / result.initial_capital - 1)
    bh_ret = (df["close"][-1] / df["close"][0] - 1)

    print(f"""
{'='*60}
  GOLD EMA/RSI OPTIMIZED — BACKTEST REPORT
{'='*60}
  Symbol:     XAUT/USDT (OKX)
  Timeframe:  1H
  Period:     {df['timestamp'][0].strftime('%Y-%m-%d')} -> {df['timestamp'][-1].strftime('%Y-%m-%d')}
  Bars:       {len(df):,}
  Params:     EMA 12/21, RSI 14, entry<70, exit>85
              SL 3%, Trailing 1.5%
{'='*60}
  Capital inicial:    $100,000.00
  Capital final:      ${result.final_capital:>12,.2f}
  Retorno estrategia: {ret:>+9.2%}
  Retorno B&H:        {bh_ret:>+9.2%}
  Alpha vs B&H:       {(ret - bh_ret)*100:>+9.2f} pp
{'-'*60}
  Sharpe:       {m.sharpe_ratio:>8.3f}   |  Sortino:  {m.sortino_ratio:>8.3f}
  Max DD:       {m.max_drawdown:>8.2%}  |  Win Rate: {m.win_rate:>8.1%}
  Profit Factor:{m.profit_factor:>8.3f}   |  Calmar:   {m.calmar_ratio:>8.3f}
  Trades:       {result.total_trades:>8}
{'='*60}
""")

    # Save report
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "symbol": "XAUT/USDT",
        "strategy": "Gold_EMA_RSI_optimized",
        "timeframe": "1h",
        "bars": len(df),
        "params": {
            "ema_fast": 12, "ema_slow": 21, "rsi_period": 14,
            "rsi_entry_max": 70.0, "rsi_exit_min": 85.0,
            "stop_loss_pct": 0.03, "trailing_stop_pct": 0.015,
        },
        "results": {
            "final_capital": round(result.final_capital, 2),
            "total_return": f"{ret:.2%}",
            "bh_return": f"{bh_ret:.2%}",
            "alpha_pp": round((ret - bh_ret) * 100, 2),
            "sharpe": round(m.sharpe_ratio, 3),
            "sortino": round(m.sortino_ratio, 3),
            "max_drawdown": f"{m.max_drawdown:.2%}",
            "win_rate": f"{m.win_rate:.1%}",
            "profit_factor": round(m.profit_factor, 2),
            "calmar": round(m.calmar_ratio, 3),
            "trades": result.total_trades,
        },
    }
    out = Path(__file__).parent.parent / "reports" / "gold-ema-rsi-optimized-backtest.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report saved: {out}")


if __name__ == "__main__":
    asyncio.run(run())
