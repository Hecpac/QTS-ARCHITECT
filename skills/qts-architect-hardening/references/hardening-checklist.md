# QTS-ARCHITECT Hardening Checklist

## Scope
Apply this checklist when modifying OMS, EMS, live execution loop, or risk metrics under low-liquidity/high-volatility conditions.

## P0 — Accounting Integrity
- Keep settlement delta-based for partial fills (never settle full reservation per partial event).
- Enforce fill idempotency using `exchange_trade_id` or equivalent unique fill key.
- Keep revert/cancel operations releasing only remaining reservations.
- Preserve consistency between `portfolio` and order persistence (atomic save where possible).

## P1 — Execution Resilience
- Keep `CircuitBreaker` HALF_OPEN probe concurrency bounded to 1 in-flight request.
- Use submit timeouts in live loop (`asyncio.wait_for`) with explicit timeout path.
- On timeout/unknown submission, avoid blind rollback; rely on reconciliation.
- On clear submission failure/rejection, revert allocation immediately.

## P2 — Market Microstructure Guardrails
- Reject or resize orders when expected slippage/spread exceeds threshold.
- Cap participation rate in thin markets.
- Prefer LIMIT/IOC in stressed volatility regimes when MARKET impact is high.
- Scale risk fraction down with volatility regime.

## P3 — Metrics Efficiency
- Compute VaR/CVaR together in one pass for shared tail data.
- Prefer selection methods (`np.partition`) over full sort when exact rank order is unnecessary.
- Cache expensive intermediate calculations when reused in same run.

## Validation
- Run focused unit tests first (`tests/test_execution.py`, `tests/test_backtest.py`).
- Run full test suite when environment supports it.
- Compile touched files at minimum (`python3 -m py_compile ...`) when full tests unavailable.
- Document unresolved risks explicitly in final report.
