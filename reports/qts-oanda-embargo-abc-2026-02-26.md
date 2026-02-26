# QTS OANDA XAU_USD — Embargo A/B/C Validation

A: control | B: hard embargo (08:45 + reversal@09:45) | C: B + soft risk (08:15/08:30 x0.5)

| Case | Window | Trades | Ann.Return | Max DD | Sharpe | PF | Worst trade | Max losing streak | Final capital |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_control | 365d_1h | 182 | 5.02% | -2.54% | 1.43 | 1.84 | -959.29 | 4 | 103360.31 |
| A_control | 90d_15m | 104 | 0.46% | -3.02% | 0.11 | 1.15 | -550.52 | 3 | 100074.85 |
| B_hard_embargo | 365d_1h | 182 | 5.02% | -2.54% | 1.43 | 1.84 | -959.29 | 4 | 103360.31 |
| B_hard_embargo | 90d_15m | 99 | 9.23% | -1.78% | 1.93 | 2.30 | -335.71 | 3 | 101437.42 |
| C_hard_plus_soft | 365d_1h | 182 | 5.02% | -2.54% | 1.43 | 1.84 | -959.29 | 4 | 103360.31 |
| C_hard_plus_soft | 90d_15m | 104 | 12.01% | -0.87% | 3.06 | 3.43 | -280.37 | 3 | 101850.84 |
