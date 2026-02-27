# QTS OANDA XAU_USD — Walk-Forward (Proxy) C Policy

> Method: trade-level proxy from closed roundtrips in `qts-oanda-minute-forensics-longonly-plusfvg-2026-02-26.json`.

## Static WF (3x30d) — A vs C

| Block | Trades(A) | A PnL | C PnL | A PF | C PF | A DD(trade) | C DD(trade) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 13 | 181.39 | 202.73 | 1.58 | 1.78 | -0.20% | -0.15% |
| B2 | 17 | 964.07 | 907.35 | 16.58 | 15.67 | -0.06% | -0.06% |
| B3 | 37 | -819.65 | 378.90 | 0.54 | 1.86 | -1.44% | -0.30% |

### Aggregate (90d proxy)

- A: PnL **325.81**, PF **1.15**, DD(trade) **-1.42%**, worst **-550.52**
- C: PnL **1488.98**, PF **2.95**, DD(trade) **-0.30%**, worst **-175.64**
- Counts C: blocked 08:45=5, blocked pattern@09:45=2, soft_scaled=20
- Invariants: no 08:45 entries=True, no reversal@09:45=True

### Gate

- PF>1 blocks: **3/3**
- DD not worse blocks: **2/3**
- Worst trade improves blocks: **3/3**
- Combined PF>1: **True**
- Combined DD not worse: **True**
- Verdict: **PASS**

## Robustness checks

### Hard-minute ±5

| Hard minute | PnL | PF | DD(trade) |
|---|---:|---:|---:|
| 08:40 | 1058.53 | 1.88 | -0.62% |
| 08:45 | 1488.98 | 2.95 | -0.30% |
| 08:50 | 1058.53 | 1.88 | -0.62% |

### Soft multiplier sweep

| Mult | PnL | PF | DD(trade) |
|---:|---:|---:|---:|
| 0.4 | 1527.06 | 3.23 | -0.24% |
| 0.5 | 1488.98 | 2.95 | -0.30% |
| 0.6 | 1450.90 | 2.72 | -0.36% |

## Rolling refit (proxy, anti-leak within train/test)

- Config: train 45d / test 14d / step 14d / folds 3
- PF>1 folds: 2/3
- DD not worse folds: 2/3
- Worst trade improves folds: 3/3
