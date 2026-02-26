# QTS A/B Backtest (real OHLCV) — 2026-02-26

Dataset: Kraken BTC/USDT 1h | rows=721 | 2026-01-27T17:00:00+00:00 → 2026-02-26T17:00:00+00:00

## Results

| Metric | Baseline (64336c2) | Current (87a557c) | Delta |
|---|---:|---:|---:|
| Final capital | 104933.334356 | 104933.334356 | +0.000000 |
| PnL abs | 4933.334356 | 4933.334356 | +0.000000 |
| Total trades | 3.000000 | 3.000000 | +0.000000 |
| Decision rate | 0.004161 | 0.004161 | +0.000000 |
| Risk reject rate | 0.938776 | 0.938776 | +0.000000 |
| Total return | 0.04933334 | 0.04933334 | +0.00000000 |
| Annualized return | 0.01699710 | 0.01699710 | +0.00000000 |
| Sharpe | 0.87189240 | 0.87189240 | +0.00000000 |
| Sortino | 0.93180040 | 0.93180040 | +0.00000000 |
| Max drawdown | -0.02328825 | -0.02328825 | +0.00000000 |
| Win rate | 0.49166667 | 0.49166667 | +0.00000000 |
| Profit factor | 1.17513308 | 1.17513308 | +0.00000000 |
| VaR95 | -0.00184295 | -0.00184295 | +0.00000000 |
| CVaR95 | -0.00262161 | -0.00262161 | +0.00000000 |

## Notes
- Same dataset, same strategy/risk config, same Python env; only code revision changes.
- Baseline is pre-wave2/3 commit; current includes wave2+wave3.
- Risk rejection counters are measured by instrumented risk wrapper during backtest run.
