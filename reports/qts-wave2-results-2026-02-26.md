# QTS-ARCHITECT — Wave 2 (T2-T5) Hardening Results

Fecha: 2026-02-26  
Scope: Wave 2 only (T2/T3/T4/T5) sobre ejecución OMS/EMS/live + tests focalizados.

---

## Resumen de cambios

### T2 (P0) — Settlement correcto para parciales (sin sobre-settlement)
- **Estado:** cubierto y validado.
- **Archivo:** `qts_core/src/qts_core/execution/oms.py` (**sin cambios nuevos en esta wave**).
- **Nota:** la lógica delta-based ya estaba aplicada en `confirm_execution` + `_settle_buy/_settle_sell`:
  - clamp de `fill_qty` al `remaining_quantity`.
  - liberación incremental de `reserved_cash_remaining` / `reserved_quantity_remaining`.
  - `revert_allocation` libera solo reserva remanente.
- **Validación adicional:** se mantuvieron y ejecutaron tests de parciales/revert parcial en `tests/test_execution.py`.

### T3 (P0) — Idempotencia de fills con trade-id real (evitar colisiones con order-id)
- **Archivos cambiados:**
  - `qts_core/src/qts_core/execution/ems.py`
  - `qts_core/src/qts_core/main_live.py`
  - `tests/test_execution.py`
- **Cambios implementados:**
  - `FillReport` ahora incluye `exchange_trade_id`.
  - En CCXT parse se extrae `trade_id/tradeId/lastTradeId/trades[].id` cuando exista.
  - Fallback de trade-id **no usa solo order-id**; usa componentes (`order_id + timestamp + qty`) para reducir colisión.
  - `main_live` confirma en OMS con `exchange_trade_id` preferente (ya no recicla `exchange_order_id` como fill-id).
  - `order views` ahora guardan `exchange_trade_id` y estado real del fill report.

### T4 (P1) — Separar reject duro vs ambiguous/timeout en flujo live + reconcile
- **Archivos cambiados:**
  - `qts_core/src/qts_core/execution/ems.py`
  - `qts_core/src/qts_core/main_live.py`
  - `tests/test_execution.py`
- **Cambios implementados:**
  - EMS ahora diferencia errores con `ExecutionError(recoverable=...)`:
    - `recoverable=False` ⇒ hard reject (ej. invalid/rejected/limit missing/unsupported type/rate-limit local).
    - `recoverable=True` ⇒ outcome ambiguo (errores de red/timeouts/excepciones no deterministas).
  - CCXT:
    - parse de status `rejected/canceled/expired` lanza hard reject.
    - respuestas sin fill confirmado retornan `None` (ambiguous path) en lugar de sobre-confirmar.
  - Alpaca:
    - `quantity` reportada ahora usa `filled_qty` real (no `qty` enviada).
    - si `filled_qty<=0` retorna `None` (ambiguous + reconcile), sin confirmar fill ficticio.
    - hard reject por status rechazado explícito.
  - LiveTrader:
    - timeout / no fill / error recoverable => **preserva reserva + reconcile**.
    - hard reject => **revert inmediato**.
    - aplicada misma semántica en `_process_symbol_tick` y `_liquidate_open_positions`.

### T5 (P1) — Ampliar pruebas timeout/no_fill/reconcile/partials
- **Archivo cambiado:** `tests/test_execution.py`
- **Nuevos tests focalizados añadidos:**
  - no fill report => orden queda abierta + reconcile.
  - submit timeout => orden abierta + reconcile.
  - hard reject => revert inmediato.
  - ambiguous execution error => reconcile sin revert ciego.
  - live usa `exchange_trade_id` para idempotencia.
  - CCXT parse prioriza `exchange_trade_id` y rechaza status hard reject.
- **Tests de parciales existentes** mantenidos y ejecutados (incremental settlement + revert remanente + idempotencia).

---

## Evidencia de validación

### 1) Hotspots
Comando:
- `python3 skills/qts-architect-hardening/scripts/find_hotspots.py .`

Resultado:
- Hotspots detectados en `oms.py`, `ems.py`, `main_live.py`, `tests/test_execution.py` (esperado para esta wave).

### 2) Tests focalizados
Comandos:
- `PYTHONPATH=qts_core/src .venv/bin/python -m pytest tests/test_execution.py -q`
- `PYTHONPATH=qts_core/src .venv/bin/python -m pytest tests/test_backtest.py -q`

Resultado:
- `tests/test_execution.py` ✅ (green)
- `tests/test_backtest.py` ✅ (green)

### 3) Compilación mínima
Comando:
- `python3 -m py_compile qts_core/src/qts_core/execution/oms.py qts_core/src/qts_core/execution/ems.py qts_core/src/qts_core/main_live.py`

Resultado:
- ✅ sin errores.

---

## Riesgos residuales

1. **Alpaca trade-id real por fill:** el objeto de orden de submit no siempre expone un `trade_id` canónico por fill; se usa fallback sintético (`order_id + timestamp + qty`) cuando aplica.
2. **Outcome ambiguo por conectividad:** sigue existiendo incertidumbre en errores de red/intermitencia; se mitiga preservando reserva y reconciliando, pero depende de disponibilidad del exchange para `fetch_order`.
3. **Coste de reconcile local:** `get_open_orders` mantiene scan por claves; funcional para recovery, pero con alta cardinalidad puede impactar latencia de recuperación.

---

## Rollback

Si Wave 2 debe revertirse:

1. Ver commit de Wave 2:
   - `git log --oneline --decorate -n 10`
2. Revertir commit local:
   - `git revert <wave2_commit_sha>`
3. Re-ejecutar validación mínima:
   - `PYTHONPATH=qts_core/src .venv/bin/python -m pytest tests/test_execution.py -q`
   - `PYTHONPATH=qts_core/src .venv/bin/python -m pytest tests/test_backtest.py -q`
   - `python3 -m py_compile qts_core/src/qts_core/execution/oms.py qts_core/src/qts_core/execution/ems.py qts_core/src/qts_core/main_live.py`

GO/NO-GO sugerido para esta wave: **GO** (P0/P1 objetivos cubiertos con tests focalizados en verde).
