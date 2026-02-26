# QTS Gold Backtest — XAUUSD (OKX) Breakout-Only

Symbol: XAU/USD:USD | Exchange: okx | Taker: 0.0005 | Commission used: 5.00 bps
Rule: NY 08:00-11:00 breakout-only (high break->SHORT to session low / low break->LONG to session high).

## Coverage
| Window | Bars | Expected | Coverage | Mean volume |
|---|---:|---:|---:|---:|
| 365d_1h | 541 | 8760 | 6.18% | 0.18 |
| 90d_15m | 2164 | 8640 | 25.05% | 0.04 |

## Backtest
| Case | Trades | Ann.Return | Max DD | Sharpe | Final capital |
|---|---:|---:|---:|---:|---:|
| 365d_1h | 2 | -0.08% | -0.01% | -2.76 | 99995.36 |
| 90d_15m | 5 | 2.96% | -0.26% | 1.59 | 100180.41 |
