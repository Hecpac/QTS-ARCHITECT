# QTS Dynamic Regime Guardrail Eval — 2026-02-26

Symbol: BTC/USDT | Timeframe: 1h

Dynamic settings: window=48 bars, min_obs=12, zscore>=1.5, normal hours [14-17], high hours [13-18].

## 30d Kraken

| Case | Trades | Guarded trades | Ann.Return | Max DD | Sharpe | Final capital |
|---|---:|---:|---:|---:|---:|---:|
| no_guard_scale_1.0 | 4 | 0 | 68.66% | -2.25% | 4.63 | 104383.75 |
| static_guard_scale_0.8 | 5 | 3 | 61.45% | -2.90% | 3.64 | 104009.80 |
| dynamic_regime_guard_scale_0.8 | 5 | 3 | 61.45% | -2.90% | 3.64 | 104009.80 |

## 1y OKX

| Case | Trades | Guarded trades | Ann.Return | Max DD | Sharpe | Final capital |
|---|---:|---:|---:|---:|---:|---:|
| no_guard_scale_1.0 | 4 | 0 | 2.89% | -12.92% | 0.31 | 102834.69 |
| static_guard_scale_0.8 | 5 | 3 | 2.05% | -11.36% | 0.25 | 102001.42 |
| dynamic_regime_guard_scale_0.8 | 5 | 1 | 1.75% | -12.19% | 0.21 | 101697.04 |

## Summary

- In this profile, `no_guard_scale_1.0` remains best in both 30d and 1y.
- Static or dynamic hour scaling at 0.8 reduced annualized return.
- Dynamic regime did not improve 1y efficiency under current sparse-trade behavior.
