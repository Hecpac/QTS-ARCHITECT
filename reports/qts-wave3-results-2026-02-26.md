# QTS-ARCHITECT — Wave 3 (T6-T7) Results

Fecha: 2026-02-26  
Scope: SOLO Wave 3 (T6/P2 + T7/P3), con cambios mínimos y configurables.

---

## Lane mapping

- **T6 → P2 (Microstructure guardrails)**
- **T7 → P3 (VaR/CVaR efficiency)**

---

## T6 (P2) — Guardrails de microestructura (slippage/spread/participation/risk-scaling)

### Archivos cambiados
- `qts_core/src/qts_core/main_live.py`
- `tests/test_execution.py`

### Cambios implementados
1. **Nuevos guardrails configurables en `apply_execution_guardrails`**
   - `max_estimated_spread_bps`
   - `spread_volatility_factor`
   - `max_participation_rate`
   - `projected_quantity`
   - `enable_dynamic_volatility_risk_scaling`
   - `min_dynamic_risk_scale`

2. **Spread/slippage proxy por volatilidad intrabar (OHLC)**
   - `estimated_slippage_bps = intrabar_volatility * slippage_volatility_factor * 10_000`
   - `estimated_spread_bps = intrabar_volatility * spread_volatility_factor * 10_000`
   - Rechazo de entrada cuando exceden thresholds configurados.

3. **Participation cap (thin markets)**
   - Se calcula participación proyectada usando `projected_quantity / market_data.volume`.
   - Si excede `max_participation_rate`, se reduce `quantity_modifier` proporcionalmente.

4. **Risk scaling dinámico por régimen de volatilidad**
   - Cuando `intrabar_volatility > max_intrabar_volatility`, se aplica un escalado adicional:
     - `dynamic_scale = max(min_dynamic_risk_scale, max_intrabar_volatility / intrabar_volatility)`
   - Mantiene compatibilidad con el escalado ya existente por alta volatilidad/hora volátil.

5. **Integración en live loop**
   - En `_process_symbol_tick` se calcula `projected_quantity` con OMS y se pasa al guardrail para dimensionar participación antes de reservar capital en OMS.

### Tests nuevos (focalizados)
En `tests/test_execution.py`:
- `test_rejects_when_spread_proxy_exceeds_threshold`
- `test_scales_quantity_modifier_for_participation_cap`
- `test_dynamic_risk_scaling_reduces_size_in_high_volatility`

---

## T7 (P3) — Optimización VaR/CVaR (cálculo conjunto y más eficiente)

### Archivos cambiados
- `qts_core/src/qts_core/backtest/metrics.py`
- `tests/test_backtest.py`

### Cambios implementados
1. **Cálculo conjunto VaR/CVaR 95 en un solo `partition`**
   - `_tail_risk_95()` ahora usa una copia y `working.partition(kth)` una sola vez.
   - VaR se toma directamente en `working[kth]`.
   - CVaR se calcula como media del mismo bucket tail (`working[:tail_count]`).

2. **Reuso por caché**
   - Se mantiene cache compartido `_tail_risk_95_cache` para `var_95()` y `cvar_95()` en el mismo run.

### Tests nuevos (focalizados)
En `tests/test_backtest.py`:
- `test_var_cvar_95_with_multi_element_tail_bucket`
- `test_tail_risk_cache_shared_between_var_and_cvar`

---

## Evidencia de validación

### 1) Hotspots (ejecutado primero)
Comando:
- `python3 skills/qts-architect-hardening/scripts/find_hotspots.py .`

Resultado:
- Hotspots detectados en rutas esperadas (`ems.py`, `main_live.py`, `metrics.py`, tests focalizados).

### 2) Tests requeridos
Comandos:
- `PYTHONPATH=qts_core/src .venv/bin/python -m pytest tests/test_execution.py -q`
- `PYTHONPATH=qts_core/src .venv/bin/python -m pytest tests/test_backtest.py -q`

Resultado:
- `tests/test_execution.py` ✅
- `tests/test_backtest.py` ✅

### 3) Compilación mínima requerida
Comando:
- `python3 -m py_compile qts_core/src/qts_core/execution/ems.py qts_core/src/qts_core/main_live.py qts_core/src/qts_core/backtest/metrics.py`

Resultado:
- ✅ sin errores.

### 4) Benchmark/impacto (T7)
Comando ad-hoc de micro-benchmark sobre 200k retornos:
- primer cálculo conjunto VaR+CVaR: **~1.41 ms**
- llamadas subsecuentes cacheadas: **~0.001 ms**
- consistencia cache: `True`

---

## Riesgos residuales

1. **Spread/slippage son estimaciones por proxy OHLC** (no L2/order book); útil como guardrail defensivo, pero no reemplaza microestructura tick-level.
2. **Participation depende de volumen del bar**; en activos con reporting de volumen irregular puede sobrerreducir o infrareducir tamaño.
3. **No se forzó cambio automático MARKET→LIMIT/IOC** para evitar breaking changes en el lifecycle OMS actual; queda como mejora incremental futura si se requiere.

---

## Rollback

1. Identificar commit local de Wave 3:
   - `git log --oneline --decorate -n 10`
2. Revertir:
   - `git revert <wave3_commit_sha>`
3. Revalidar:
   - `PYTHONPATH=qts_core/src .venv/bin/python -m pytest tests/test_execution.py -q`
   - `PYTHONPATH=qts_core/src .venv/bin/python -m pytest tests/test_backtest.py -q`
   - `python3 -m py_compile qts_core/src/qts_core/execution/ems.py qts_core/src/qts_core/main_live.py qts_core/src/qts_core/backtest/metrics.py`

---

## Estado Wave 3

- [x] T6 implementado con guardrails configurables de microestructura.
- [x] T7 implementado con cálculo VaR/CVaR conjunto y más eficiente.
- [x] Tests focalizados en verde.
- [x] Compilación mínima en verde.
