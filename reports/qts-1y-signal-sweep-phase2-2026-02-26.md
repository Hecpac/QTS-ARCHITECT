# QTS 1Y Signal Sweep Phase 2 (realistic) — 2026-02-26

Dataset: OKX BTC/USDT 1h | rows=8760 | 2025-02-26 18:00:00+00:00 → 2026-02-26 17:00:00+00:00

Costs: commission 15 bps, slippage 5 bps, borrow 2 bps/day, force_close_positions_at_end=true

Grid size: 108

## Baseline (strict current-like)

- sup=0.60, risk=0.70, thr=0.015, session=13-17, min_fvg=0.0012
- Ann.Return: 2.76%
- Max DD: -13.56%

## Top 3 feasible (higher ann.return, no worse DD)

| Rank | Case | Ann.Return | Max DD | Sharpe | Trades | Reject rate |
|---|---|---:|---:|---:|---:|---:|
| 1 | `sup0.50_risk0.60_thr0.012_s13-17_fvg0.0008` | 2.89% | -12.92% | 0.31 | 4 | 99.32% |
| 2 | `sup0.50_risk0.60_thr0.012_s13-17_fvg0.0012` | 2.89% | -12.92% | 0.31 | 4 | 99.28% |
| 3 | `sup0.55_risk0.60_thr0.012_s13-17_fvg0.0008` | 2.89% | -12.92% | 0.31 | 4 | 99.32% |

## Top 3 with DD tolerance (+1.5pp max DD)

| Rank | Case | Ann.Return | Max DD | Sharpe | Trades | Reject rate |
|---|---|---:|---:|---:|---:|---:|
| 1 | `sup0.50_risk0.60_thr0.012_s13-17_fvg0.0008` | 2.89% | -12.92% | 0.31 | 4 | 99.32% |
| 2 | `sup0.50_risk0.60_thr0.012_s13-17_fvg0.0012` | 2.89% | -12.92% | 0.31 | 4 | 99.28% |
| 3 | `sup0.55_risk0.60_thr0.012_s13-17_fvg0.0008` | 2.89% | -12.92% | 0.31 | 4 | 99.32% |

## Best by annualized return (overall)

- `sup0.50_risk0.60_thr0.012_s13-17_fvg0.0008` | Ann.Return 2.89% | Max DD -12.92% | Sharpe 0.31
