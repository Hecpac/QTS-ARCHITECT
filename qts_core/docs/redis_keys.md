# Redis Key Conventions

Claves usadas por el ciclo en vivo, el OMS y el dashboard.

- `SYSTEM:HALT` — String (`"true"` / `"false"`). **Productor:** dashboard (botón de stop). **Consumidor:** trader (loop en vivo) para apagar de inmediato.
- `SYSTEM:HEARTBEAT` — ISO timestamp. **Productor:** trader (cada iteración). **Consumidor:** dashboard para estado de salud.
- `ALERTS:LAST` — JSON del último evento crítico/advertencia emitido por el trader. **Productor:** trader. **Consumidor:** dashboard/notifier.
- `ALERTS:EVENT:<epoch_ms>` — JSON por evento (snapshot append-only por timestamp). **Productor:** trader. **Consumidor:** auditoría operativa.
- `METRICS:TOTAL_VALUE` — Float en string, equity aproximada (cash + posiciones al último precio). **Productor:** trader. **Consumidor:** dashboard.
- `METRICS:CASH` — Float en string, efectivo disponible. **Productor:** trader. **Consumidor:** dashboard.
- `METRICS:PNL_DAILY` — Float en string, PnL diario fraccional respecto al ancla del día (ej. `-0.012` = -1.2%). **Productor:** trader. **Consumidor:** dashboard.
- `METRICS:LATENCY:TICK_TO_DECISION_MS` — Float en string, latencia desde recepción del tick hasta decisión del supervisor (ms). **Productor:** trader. **Consumidor:** dashboard/alerting.
- `METRICS:LATENCY:DECISION_TO_FILL_MS` — Float en string, latencia desde envío de orden hasta fill/no-fill (ms). **Productor:** trader. **Consumidor:** dashboard/alerting.
- `METRICS:LATENCY:TICK_TO_FILL_MS` — Float en string, latencia extremo a extremo tick→fill/no-fill (ms). **Productor:** trader. **Consumidor:** dashboard/alerting.
- `METRICS:LATENCY:TICK_TO_DECISION_MS:<SYMBOL>` — Latencia por símbolo (suffix normalizado, ej. `ETH_USDT`). **Productor:** trader. **Consumidor:** dashboard multi-activo.
- `METRICS:LATENCY:DECISION_TO_FILL_MS:<SYMBOL>` — Latencia submit→fill por símbolo. **Productor:** trader. **Consumidor:** dashboard multi-activo.
- `METRICS:LATENCY:TICK_TO_FILL_MS:<SYMBOL>` — Latencia end-to-end por símbolo. **Productor:** trader. **Consumidor:** dashboard multi-activo.
- `MARKET:ACTIVE_SYMBOLS` — JSON array con símbolos activos (ej. `["BTC/USDT","ETH/USDT"]`). **Productor:** trader. **Consumidor:** dashboard.
- `MARKET:LAST_PRICE` — Float en string, último precio del símbolo activo (canónico/backward-compatible). **Productor:** trader. **Consumidor:** dashboard.
- `MARKET:LAST_PRICE:<SYMBOL>` — Float en string por símbolo. **Productor:** trader. **Consumidor:** dashboard multi-activo.
- `MARKET:LAST_TICK` — JSON `{symbol, price, timestamp}` del último tick conocido (canónico/backward-compatible). **Productor:** trader. **Consumidor:** dashboard.
- `MARKET:LAST_TICK:<SYMBOL>` — JSON del último tick por símbolo. **Productor:** trader. **Consumidor:** dashboard multi-activo.
- `MARKET:OHLCV:<SYMBOL>` — OHLCV serializado por símbolo para chart multi-activo. **Productor:** trader. **Consumidor:** dashboard multi-activo.
- `VIEW:ORDERS` — JSON array de órdenes visibles, p.ej. `[{"order_id": "...", "side": "BUY", "qty": 0.1, "price": 110.0, "status": "FILLED", "ts": "..."}]`. **Productor:** trader al registrar fills. **Consumidor:** dashboard (tabla de órdenes).
- `VIEW:POSITIONS` — JSON array de posiciones, p.ej. `[{"instrument_id": "BTC/USDT", "quantity": 0.5}]`. **Productor:** trader. **Consumidor:** dashboard.
- `oms:portfolio` — Modelo Pydantic `Portfolio` serializado. **Productor/Consumidor:** OMS.
- `oms:order:<id>` — Modelo `Order` serializado por ID. **Productor/Consumidor:** OMS.
