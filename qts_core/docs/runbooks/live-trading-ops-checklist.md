# Runbook — Live Trading Ops Checklist

## Pre-market / pre-session
- [ ] `conf/live.yaml` revisado (`execution_timeout`, `max_session_drawdown`, guardrails).
- [ ] Gateway en modo esperado (sandbox/paper/live).
- [ ] Conectividad exchange/Redis verificada.
- [ ] `SYSTEM:HALT` en `false`.

## During session
- [ ] Heartbeat actualizado (`SYSTEM:HEARTBEAT`).
- [ ] Métricas base válidas (`METRICS:TOTAL_VALUE`, `METRICS:CASH`, `METRICS:PNL_DAILY`).
- [ ] Latencia bajo umbral:
  - `METRICS:LATENCY:TICK_TO_DECISION_MS`
  - `METRICS:LATENCY:DECISION_TO_FILL_MS`
  - `METRICS:LATENCY:TICK_TO_FILL_MS`
- [ ] Revisar alertas recientes (`ALERTS:LAST`).

## Ambiguous execution handling
- [ ] Si hubo timeout/confirm fail, confirmar que se emitió reconciliación.
- [ ] Validar consistencia OMS/exchange antes de continuar sizing normal.

## End-of-session
- [ ] Guardar snapshot de portfolio (`oms:portfolio`).
- [ ] Confirmar no quedan órdenes abiertas inesperadas.
- [ ] Registrar incidencias y tiempos de recuperación.

## Escalation policy
- **SEV-1:** exposición residual no explicada / stop no efectivo → activar halt y no reanudar.
- **SEV-2:** latencia sostenida alta / reconciliaciones frecuentes → reducir riesgo y evaluar continuidad.
- **SEV-3:** alertas aisladas recuperadas automáticamente → monitoreo reforzado.
