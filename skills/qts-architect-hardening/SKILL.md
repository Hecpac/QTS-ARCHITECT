---
name: qts-architect-hardening
description: Harden and audit QTS-ARCHITECT trading reliability for OMS/EMS/live-loop/risk-metrics changes. Use when working on low-liquidity/high-volatility robustness, order lifecycle consistency, circuit-breaker behavior, timeout/reconciliation handling, or VaR/CVaR performance in qts_core/src/qts_core/{execution,main_live,backtest} and tests/{test_execution,test_backtest}. Also use when producing prioritized remediation plans (P0-P3) and validation checklists for this repository.
---

# QTS-ARCHITECT Hardening

Follow this workflow whenever touching trading-critical logic.

## Workflow

1. **Locate risk hotspots first**
   - Run:
     - `python3 skills/qts-architect-hardening/scripts/find_hotspots.py .`
   - Review matches before editing to avoid patching secondary symptoms.

2. **Map change to priority lane**
   - Use `references/hardening-checklist.md`.
   - Classify work as P0/P1/P2/P3 and state the target lane in your notes.

3. **Implement minimal, testable patch**
   - Prefer small diffs in these files:
     - `qts_core/src/qts_core/execution/oms.py`
     - `qts_core/src/qts_core/execution/ems.py`
     - `qts_core/src/qts_core/main_live.py`
     - `qts_core/src/qts_core/backtest/metrics.py`
   - Preserve backward compatibility unless the task explicitly requests breaking changes.

4. **Add/adjust focused regression tests**
   - For OMS/EMS/live-loop logic: update `tests/test_execution.py`.
   - For metrics logic: update `tests/test_backtest.py`.
   - Prefer scenario tests that reproduce the original failure mode.

5. **Validate and report risk status**
   - Run unit tests when available.
   - If test tooling is missing, run `python3 -m py_compile` on touched files.
   - Report:
     - what was fixed,
     - what remains intentionally deferred,
     - any residual risk and recovery path.

## Guardrails

- Never treat timeout as guaranteed failure in live execution.
- Never double-settle a fill event.
- Never release full reservation on partial fill.
- Keep reconciliation path available for unknown/ambiguous execution outcomes.

## Resources

- `references/hardening-checklist.md` — priority lanes and validation gates.
- `scripts/find_hotspots.py` — fast locator for high-risk code sections.
