# QTS Dynamic Extreme Volatility Filter Backtest — 2026-02-26

Strategy profile: sup=0.50, risk=0.60, threshold=0.012, ICT 13-17, min_fvg=0.0008, cap=0.15

Configs: baseline(no guard), dynamic extreme scale(z>=2.5, scale=0.8), dynamic extreme block(z>=2.5, block=true)

## 1y_okx

| Case | Trades | Guarded trades | Ann.Return | Max DD | Sharpe | Final capital |
|---|---:|---:|---:|---:|---:|---:|
| baseline_no_guard | 4 | 0 | 2.89% | -12.92% | 0.31 | 102834.69 |
| dynamic_extreme_scale_z2.5 | 5 | 1 | 1.75% | -12.19% | 0.21 | 101697.04 |
| dynamic_extreme_block_z2.5 | 11 | 0 | 4.34% | -10.76% | 0.50 | 104285.69 |

## 30d_kraken

| Case | Trades | Guarded trades | Ann.Return | Max DD | Sharpe | Final capital |
|---|---:|---:|---:|---:|---:|---:|
| baseline_no_guard | 4 | 0 | 66.04% | -2.25% | 4.48 | 104249.46 |
| dynamic_extreme_scale_z2.5 | 5 | 1 | 58.23% | -2.85% | 3.52 | 103838.32 |
| dynamic_extreme_block_z2.5 | 8 | 0 | 52.79% | -2.43% | 3.73 | 103540.60 |

