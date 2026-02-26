# QTS 1Y Trade-Frequency Sweep (dynamic extreme block) — 2026-02-26

Objective: increase trade count in 1y while monitoring return/drawdown quality.

Grid: sup_min_conf ∈ [0.45,0.50,0.55], risk_min_conf ∈ [0.55,0.60], bullish_threshold ∈ [0.008,0.010,0.012], sessions ∈ {(13,17),(12,20),(0,24)}

Total cases: 54

## Top by trade count (DD>=-18% filter)

| Rank | sup | risk | thr | session | Trades | Ann.Return | Max DD | Sharpe | Final |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.45 | 0.55 | 0.008 | 13-17 | 17 | -4.33% | -12.68% | -0.46 | 95624.28 |
| 2 | 0.50 | 0.55 | 0.008 | 13-17 | 17 | -4.33% | -12.68% | -0.46 | 95624.28 |
| 3 | 0.55 | 0.55 | 0.008 | 13-17 | 17 | -4.33% | -12.68% | -0.46 | 95624.28 |
| 4 | 0.45 | 0.55 | 0.008 | 12-20 | 16 | -3.84% | -11.96% | -0.42 | 96107.83 |
| 5 | 0.50 | 0.55 | 0.008 | 12-20 | 16 | -3.84% | -11.96% | -0.42 | 96107.83 |

## Top balanced (return/sharpe under DD>=-18%)

| Rank | sup | risk | thr | session | Trades | Ann.Return | Max DD | Sharpe | Final |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.45 | 0.55 | 0.012 | 13-17 | 11 | 4.34% | -10.76% | 0.50 | 104285.69 |
| 2 | 0.45 | 0.60 | 0.012 | 13-17 | 11 | 4.34% | -10.76% | 0.50 | 104285.69 |
| 3 | 0.50 | 0.55 | 0.012 | 13-17 | 11 | 4.34% | -10.76% | 0.50 | 104285.69 |
| 4 | 0.50 | 0.60 | 0.012 | 13-17 | 11 | 4.34% | -10.76% | 0.50 | 104285.69 |
| 5 | 0.55 | 0.55 | 0.012 | 13-17 | 11 | 4.34% | -10.76% | 0.50 | 104285.69 |

## Conclusion
- Yes: trade count can be increased (up to 17), but those configs turned unprofitable.
- Best risk-adjusted/profitable region remained around bullish_threshold=0.012 with ~11 trades.
- Session 13-17 consistently dominated wider sessions for profitability in this sweep.
