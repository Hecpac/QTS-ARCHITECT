# QTS Loop + Skills Operativo (Codex-first)

Objetivo: ejecutar hardening de QTS de forma repetible, rápida y con validación real.

## 1) Loop estándar (15–30 min por issue)

1. **Detectar riesgo/fallo**
   - Fuente: logs live, alerta, test roto o inconsistencia OMS/EMS.
2. **Clasificar prioridad**
   - **P0**: integridad contable/settlement/idempotencia.
   - **P1**: resiliencia de ejecución (timeout, reconcile, circuit breaker).
   - **P2**: guardrails de microestructura (slippage/spread/participación).
   - **P3**: eficiencia de métricas (VaR/CVaR, rendimiento).
3. **Parche mínimo**
   - Tocar solo el punto crítico.
   - Sin cambios amplios de arquitectura en la misma pasada.
4. **Test focal**
   - Reproducir el fallo original + test de no-regresión.
5. **Cierre con gate**
   - Si pasa validación: merge.
   - Si no: rollback o dejar detrás de flag.

## 2) Skills operativas (playbooks cortos)

### A. Timeout / ambiguous submission (P1)
- Nunca asumir `timeout == fallo definitivo`.
- No hacer rollback ciego cuando el estado es ambiguo.
- Disparar reconciliación y preservar reservas hasta confirmar estado real.

### B. Partial fill + idempotencia (P0)
- Settlement siempre por **delta** (no por total reservado).
- Duplicados bloqueados por `exchange_trade_id`.
- `revert/cancel` solo suelta remanente, nunca lo ya ejecutado.

### C. Circuit breaker (P1)
- `HALF_OPEN`: máximo **1 probe in-flight**.
- En fallo de probe: reabrir circuito.
- En éxito: cerrar circuito y reset de fallos.

### D. Risk metrics (P3)
- Calcular VaR/CVaR en la misma pasada cuando compartan cola.
- Evitar operaciones innecesarias de ordenamiento total cuando no haga falta.

## 3) Gate obligatorio de calidad

### Hotspots first
```bash
python3 skills/qts-architect-hardening/scripts/find_hotspots.py .
```

### Tests críticos
```bash
uv run pytest -q tests/test_execution.py tests/test_backtest.py
```

### Fallback mínimo (si no hay tooling)
```bash
python3 -m py_compile qts_core/src/qts_core/execution/oms.py \
  qts_core/src/qts_core/execution/ems.py \
  qts_core/src/qts_core/main_live.py \
  qts_core/src/qts_core/backtest/metrics.py
```

## 4) Plantilla de reporte de cada pasada

```text
Issue:
Priority lane: P0|P1|P2|P3
Root cause:
Patch scope (files):
Validation run:
Residual risk:
Rollback path:
Next step (accionable):
```

## 5) Política de rollback rápido

- Si rompe P0/P1 en pruebas: revert inmediato del commit.
- Si falla en entorno live con riesgo de estado ambiguo: detener nuevas entradas, ejecutar reconcile, luego decidir rollback/forward-fix.
- Nunca cerrar incidente sin evidencia de validación real.
