# Runbook — Emergency Halt & Recovery

## Objetivo
Estandarizar la respuesta operativa ante un evento crítico en el live loop (`SYSTEM:HALT=true`).

## Triggers típicos
- Drawdown de sesión excede `loop.max_session_drawdown`.
- Error de ejecución repetido / degradación de exchange.
- Operador activa `SYSTEM:HALT` manualmente.

## Acciones automáticas del sistema
1. Publica alerta crítica en `ALERTS:LAST` y `ALERTS:EVENT:<epoch_ms>`.
2. Intenta liquidar posiciones abiertas (`EXIT`) usando precio conocido.
3. Ejecuta reconciliación tras paths ambiguos (timeout/confirm fail).
4. Detiene EMS y finaliza el loop.

## Respuesta del operador (paso a paso)
1. Confirmar estado:
   - `SYSTEM:HALT == true`
   - revisar `ALERTS:LAST`
2. Verificar posiciones y órdenes:
   - `VIEW:POSITIONS`
   - `VIEW:ORDERS`
   - `oms:portfolio`
3. Validar reconciliación:
   - re-ejecutar reconcile manual si hay duda operativa.
4. Confirmar que exposición residual sea cero o dentro de tolerancia.
5. Documentar incidente (causa, timestamps, acciones, resultado).

## Criterios para volver a operar
- Causa raíz identificada y mitigada.
- Exposición residual validada.
- Reconciliación OMS/exchange consistente.
- Alertas estabilizadas.
- Aprobación explícita del operador responsable.

## Roll-forward recomendado
1. Reiniciar proceso live.
2. Verificar heartbeat y métricas de latencia.
3. Mantener tamaño reducido durante ventana de observación.
