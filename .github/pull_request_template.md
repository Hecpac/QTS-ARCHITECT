## 🎯 Summary

<!-- What changed and why? Include impacted modules. -->

## ✅ Scope / Intent

- [ ] Change is aligned with ticket/objective
- [ ] No unrelated refactors mixed in
- [ ] Backward compatibility considered (or explicitly documented)

## 🧪 Validation (required)

- [ ] `poetry run ruff check .` passed
- [ ] `poetry run mypy qts_core/src` passed
- [ ] `poetry run pytest -q` passed (or targeted suite + rationale)

## 📈 Trading reliability checks (when execution/risk logic touched)

- [ ] Order intent paths reviewed (`OPEN_LONG`, `OPEN_SHORT`, `CLOSE_LONG`, `CLOSE_SHORT`)
- [ ] Timeout/reconcile/error paths validated
- [ ] Risk controls/limits impact reviewed (no silent regression)
- [ ] Exchange-specific params noted when applicable (`reduceOnly`, margin/perp)

## 🧭 Observability / Ops

- [ ] Logs/metrics impact documented
- [ ] Runbook/rollback notes updated when needed

## ⚠️ Risk & Rollback

- **Main risk:**
- **Rollback plan:**

## 📎 Reviewer notes

- **Files to review first:**
- **Suggested test focus:**
