# QTS Realism Pass — 30d vs 1y (2026-02-26)

## What changed for realism

1. **Forced close at end of backtest** (`force_close_positions_at_end=True`):
   - open positions are closed on the last bar using last known price.
2. **Carry accounting preserved on close**:
   - short borrow fees are charged when the forced close covers short exposure.
3. **Cross-instrument mark-to-market fix**:
   - equity uses last known prices across all symbols (not only current tick symbol).
4. **Annualization frequency inferred from bar spacing**:
   - hourly bars annualize with ~8760 periods/year instead of daily default.

## Real-data runs (with transaction-cost assumptions)

- **30d**: Kraken BTC/USDT 1h, taker fee assumption **40 bps**, slippage **5 bps**
- **1y**: OKX BTC/USDT 1h, taker fee assumption **15 bps**, slippage **5 bps**
- Strategy/risk profile: strict (same as current live profile)

| Metric | 30d (Kraken) | 1y (OKX) |
|---|---:|---:|
| Bars | 720 | 8760 |
| Trades | 4 | 4 |
| Final capital | 104,431.73 | 102,644.65 |
| PnL | +4,431.73 | +2,644.65 |
| Total return | 4.43% | 2.70% |
| Annualized return | 69.61% | 2.70% |
| Sharpe | 4.62 | 0.29 |
| Max drawdown | -2.33% | -13.56% |
| Short borrow fees paid | 113.42 | 1,318.72 |

## Interpretation

- The prior “very similar” 30d vs 1y behavior came from an unrealistically open-ended short lifecycle.
- With forced close + carry + cost assumptions, **1y underperforms 30d clearly**.
- Risk reject-rate remains high in strict mode, so trade count is still low; this is now a strategy-selectivity issue (not only an accounting artifact).
