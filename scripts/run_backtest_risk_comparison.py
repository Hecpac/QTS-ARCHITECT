"""Backtest: Risk Layers ON vs OFF comparison.

Downloads 6 months of BTC/USDT 1h from OKX, runs backtest with and without
the 5-layer risk management system, and compares results.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# ── Data download ──────────────────────────────────────────────────────────


async def download_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since_iso: str = "2025-09-01T00:00:00Z",
    limit_per_call: int = 100,
) -> pl.DataFrame:
    """Download OHLCV from OKX via ccxt."""
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


# ── Backtest runner ────────────────────────────────────────────────────────


async def run_comparison() -> None:
    from qts_core.agents.base import StrictRiskAgent
    from qts_core.agents.ict import ICTSmartMoneyAgent
    from qts_core.agents.supervisor import Supervisor
    from qts_core.backtest.engine import BacktestConfig, EventEngine

    print("Downloading BTC/USDT 1h (6 months)...")
    df = await download_ohlcv(since_iso="2025-09-01T00:00:00Z")
    print(f"  → {len(df)} bars from {df['timestamp'][0]} to {df['timestamp'][-1]}")

    # Shared agent setup
    def make_supervisor() -> Supervisor:
        ict = ICTSmartMoneyAgent(name="ICT_BT", symbol="BTC/USDT")
        risk = StrictRiskAgent(name="Risk")
        return Supervisor(
            strategy_agents=[ict],
            risk_agent=risk,
            min_confidence=0.6,
        )

    base_params = dict(
        initial_capital=100_000.0,
        trade_size=10_000.0,
        slippage_bps=5.0,
        commission_bps=10.0,
        stop_loss_pct=0.05,
        risk_fraction=0.045,
    )

    # ── Run 1: Baseline (no risk layers) ──────────────────────
    print("\n▸ Backtest 1: BASELINE (no risk layers)...")
    engine_base = EventEngine(make_supervisor(), BacktestConfig(**base_params))
    result_base = await engine_base.run(df)

    # ── Run 2: With risk layers ───────────────────────────────
    print("▸ Backtest 2: WITH RISK LAYERS...")
    engine_risk = EventEngine(
        make_supervisor(),
        BacktestConfig(
            **base_params,
            risk_layers_enabled=True,
            watchdog_warn_threshold=0.05,
            watchdog_halt_threshold=0.08,
            regime_crisis_percentile=0.95,
            regime_high_vol_size_scale=0.60,
            signal_quality_window=10,
            signal_quality_min_samples=5,
            signal_quality_medium_size_scale=0.60,
        ),
    )
    result_risk = await engine_risk.run(df)

    # ── Report ────────────────────────────────────────────────
    def fmt(r) -> dict:
        m = r.metrics
        return {
            "final_capital": round(r.final_capital, 2),
            "total_return": f"{((r.final_capital / r.initial_capital) - 1):.2%}",
            "trades": r.total_trades,
            "sharpe": round(m.sharpe_ratio, 3) if m else None,
            "sortino": round(m.sortino_ratio, 3) if m else None,
            "max_drawdown": f"{m.max_drawdown:.2%}" if m else None,
            "win_rate": f"{m.win_rate:.1%}" if m else None,
            "profit_factor": round(m.profit_factor, 2) if m else None,
            "calmar": round(m.calmar_ratio, 3) if m else None,
        }

    base_stats = fmt(result_base)
    risk_stats = fmt(result_risk)

    print("\n" + "=" * 60)
    print("BACKTEST COMPARISON: BTC/USDT 1h — 6 months")
    print("=" * 60)
    header = f"{'Metric':<22} {'Baseline':>16} {'Risk Layers':>16}"
    print(header)
    print("-" * 60)
    for key in base_stats:
        print(f"{key:<22} {str(base_stats[key]):>16} {str(risk_stats[key]):>16}")
    print("=" * 60)

    # Save to JSON
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "bars": len(df),
        "baseline": base_stats,
        "risk_layers": risk_stats,
    }
    out_path = Path(__file__).parent.parent / "reports" / "risk-layers-comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_comparison())
