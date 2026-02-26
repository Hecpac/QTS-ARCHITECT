# QTS Volatility-Hour Audit — 2026-02-26

Symbol: BTC/USDT | Timeframe: 1h

## Windows analyzed
- 1y: OKX (8760h dataset)
- 30d: Kraken (~720h)

## Top volatility hours (ranked composite)

### 1y OKX top-6
- 14:00 UTC | mean|ret|=0.5158% | p95|ret|=1.4389% | tail-hit=15.34%
- 15:00 UTC | mean|ret|=0.4676% | p95|ret|=1.5990% | tail-hit=13.42%
- 17:00 UTC | mean|ret|=0.4035% | p95|ret|=1.2823% | tail-hit=8.22%
- 16:00 UTC | mean|ret|=0.3890% | p95|ret|=1.2409% | tail-hit=9.04%
- 13:00 UTC | mean|ret|=0.3785% | p95|ret|=1.1453% | tail-hit=8.77%
- 18:00 UTC | mean|ret|=0.3475% | p95|ret|=1.0480% | tail-hit=6.32%

### 30d Kraken top-6
- 01:00 UTC | mean|ret|=0.7124% | p95|ret|=3.3728% | tail-hit=13.33%
- 15:00 UTC | mean|ret|=0.7069% | p95|ret|=2.4143% | tail-hit=10.00%
- 17:00 UTC | mean|ret|=0.6684% | p95|ret|=2.0910% | tail-hit=13.33%
- 16:00 UTC | mean|ret|=0.6752% | p95|ret|=1.9429% | tail-hit=10.00%
- 14:00 UTC | mean|ret|=0.6873% | p95|ret|=1.4364% | tail-hit=10.00%
- 18:00 UTC | mean|ret|=0.6879% | p95|ret|=1.4569% | tail-hit=6.67%

Consensus intersection (top-6 vs top-6): [14, 15, 16, 17, 18]

## Recommendation
- Core high-vol guardrail hours UTC: **[14, 15, 16, 17]**
- Extended conservative window: **[13, 14, 15, 16, 17, 18]**
- Keep **01:00 UTC** as a secondary watch hour (episodic spikes in 30d, not structural in 1y).

Current config `[14,15,16,17,18]` is directionally correct and close to the recommended extended window.
