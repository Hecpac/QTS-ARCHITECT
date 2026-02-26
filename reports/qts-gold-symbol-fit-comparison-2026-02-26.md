# Gold Symbol Fit Comparison (XAUUSD vs XAUTUSDT) — 2026-02-26

Exchange: OKX | Strategy: NY 08:00-11:00 breakout-reversal + dynamic guardrails

## Market fit

| Label | CCXT Symbol | Type | Spot | Swap | Taker fee |
|---|---|---|---:|---:|---:|
| XAUUSD | XAU/USD:USD | swap | False | True | 0.0005 |
| XAUTUSDT | XAUT/USDT | spot | True | False | 0.0015 |

## Data coverage

| Label | Window | Bars | Expected | Coverage | Mean volume |
|---|---|---:|---:|---:|---:|
| XAUUSD | 365d 1h | 541 | 8760 | 6.18% | 0.18 |
| XAUUSD | 90d 15m | 2161 | 8640 | 25.01% | 0.04 |
| XAUTUSDT | 365d 1h | 8760 | 8760 | 100.00% | 79.96 |
| XAUTUSDT | 90d 15m | 8640 | 8640 | 100.00% | 50.29 |

## Backtest comparison

| Case | Trades | Ann.Return | Max DD | Sharpe | Final capital |
|---|---:|---:|---:|---:|---:|
| XAUUSD_365d_1h | 3 | -6.33% | -0.41% | -5.40 | 99597.44 |
| XAUUSD_90d_15m | 5 | 2.27% | -0.28% | 1.22 | 100138.41 |
| XAUTUSDT_365d_1h | 441 | -8.21% | -9.16% | -2.84 | 91786.31 |
| XAUTUSDT_90d_15m | 207 | -24.84% | -7.08% | -6.03 | 93200.65 |
