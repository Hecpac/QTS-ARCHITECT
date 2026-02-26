# QTS A/B Backtest (1Y real OHLCV) — 2026-02-26

Dataset: OKX BTC/USDT 1h | rows=8760 | 2025-02-26T18:00:00+00:00 → 2026-02-26T17:00:00+00:00

## Results

| Metric | Baseline (64336c2) | Current (87a557c) | Delta |
|---|---:|---:|---:|
| Final capital | 104221.251257 | 104221.251257 | +0.000000 |
| PnL abs | 4221.251257 | 4221.251257 | +0.000000 |
| Total trades | 3.000000 | 3.000000 | +0.000000 |
| Decision rate | 0.000342 | 0.000342 | +0.000000 |
| Risk reject rate | 0.990854 | 0.990854 | +0.000000 |
| Total return | 0.04221251 | 0.04221251 | +0.00000000 |
| Annualized return | 0.00119025 | 0.00119025 | +0.00000000 |
| Sharpe | 0.07155787 | 0.07155787 | +0.00000000 |
| Sortino | 0.07343269 | 0.07343269 | +0.00000000 |
| Max drawdown | -0.13534964 | -0.13534964 | +0.00000000 |
| Win rate | 0.49765955 | 0.49765955 | +0.00000000 |
| Profit factor | 1.01350875 | 1.01350875 | +0.00000000 |
| VaR95 | -0.00178638 | -0.00178638 | +0.00000000 |
| CVaR95 | -0.00276789 | -0.00276789 | +0.00000000 |

## Notes
- Same dataset and config; only code revision differs.
- Baseline pre-wave2/3 vs current with wave2/3 integrated.
- Data source switched to OKX because Kraken OHLC endpoint is capped (~721 1h candles).
