# QTS 1Y Parameter Sweep (realistic) — 2026-02-26

Dataset: OKX BTC/USDT 1h | rows=8760 | 2025-02-26 18:00:00+00:00 → 2026-02-26 17:00:00+00:00

Costs: commission 15 bps, slippage 5 bps, borrow 2 bps/day, force_close_positions_at_end=true

## Baseline constraint

- Baseline: strict cap=0.15
- Annualized return floor: 2.7554%
- Max drawdown floor (must be >=): -13.5558%

## Top 3 feasible (more return, no worse drawdown)

No config satisfied the constraint.

## Full sweep snapshot

| Profile | Cap | Ann.Return | Total Return | Max DD | Sharpe | Trades | Reject rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| aggressive | 0.12 | 2.09% | 2.09% | -13.37% | 0.24 | 6 | 99.11% |
| aggressive | 0.15 | 2.69% | 2.69% | -15.33% | 0.27 | 8 | 98.76% |
| aggressive | 0.18 | 2.19% | 2.19% | -13.21% | 0.24 | 9 | 98.58% |
| aggressive | 0.22 | -8.87% | -8.86% | -18.65% | -0.70 | 36 | 93.79% |
| balanced | 0.12 | 1.23% | 1.23% | -9.25% | 0.19 | 4 | 99.29% |
| balanced | 0.15 | -3.88% | -3.88% | -12.65% | -0.42 | 11 | 97.65% |
| balanced | 0.18 | -5.42% | -5.42% | -15.32% | -0.47 | 12 | 97.41% |
| balanced | 0.22 | -6.62% | -6.62% | -18.23% | -0.50 | 14 | 96.94% |
| strict | 0.12 | 1.23% | 1.23% | -9.25% | 0.19 | 4 | 99.09% |
| strict | 0.15 | 2.76% | 2.76% | -13.56% | 0.30 | 4 | 99.09% |
| strict | 0.18 | 1.48% | 1.48% | -16.58% | 0.18 | 16 | 95.43% |
| strict | 0.22 | -6.98% | -6.97% | -17.94% | -0.50 | 8 | 97.87% |
