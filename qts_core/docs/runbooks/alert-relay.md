# Runbook — Alert Relay (Redis → Slack/Telegram)

## Objetivo
Propagar alertas críticas del live trader (`ALERTS:LAST`) hacia canales externos de operación.

## Componente
- Módulo: `qts_core.ops.alert_relay`
- Entrada: Redis key `ALERTS:LAST`
- Salidas opcionales:
  - Slack Incoming Webhook
  - Telegram Bot API

## Variables de entorno
- `REDIS_HOST` (default: `localhost`)
- `REDIS_PORT` (default: `6379`)
- `REDIS_DB` (default: `0`)
- `ALERT_KEY` (default: `ALERTS:LAST`)
- `ALERT_POLL_SECONDS` (default: `2`)
- `ALERT_MIN_LEVEL` (default: `WARNING`)
- `ALERT_HTTP_TIMEOUT_SECONDS` (default: `5`)
- `SLACK_WEBHOOK_URL` (opcional)
- `TELEGRAM_BOT_TOKEN` (opcional)
- `TELEGRAM_CHAT_ID` (opcional)

## Ejecución
```bash
PYTHONPATH=qts_core/src python -m qts_core.ops.alert_relay
```

O vía entrypoint de Poetry:
```bash
poetry run qts-alert-relay
```

## Reglas operativas recomendadas
- Iniciar con `ALERT_MIN_LEVEL=ERROR` para evitar ruido inicial.
- Promover a `WARNING` solo tras calibrar volumen de alertas.
- Mantener el relay como proceso independiente del live loop.
- Si el relay cae, el trader sigue operando; reanudar relay y revisar backlog en `ALERTS:EVENT:*`.

## Troubleshooting rápido
1. No llegan alertas:
   - Verificar que `ALERTS:LAST` cambia en Redis.
   - Revisar credenciales Slack/Telegram.
2. Alertas duplicadas:
   - Confirmar que no hay múltiples instancias del relay en paralelo.
3. Mucho ruido:
   - Subir `ALERT_MIN_LEVEL` y ajustar reglas de emisión en live loop.
