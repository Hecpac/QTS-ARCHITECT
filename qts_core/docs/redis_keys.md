# Redis Key Conventions

Claves usadas por el ciclo en vivo, el OMS y el dashboard.

- `SYSTEM:HALT` — String (`"true"` / `"false"`). **Productor:** dashboard (botón de stop). **Consumidor:** trader (loop en vivo) para apagar de inmediato.
- `SYSTEM:HEARTBEAT` — ISO timestamp. **Productor:** trader (cada iteración). **Consumidor:** dashboard para estado de salud.
- `METRICS:TOTAL_VALUE` — Float en string, equity aproximada (cash + posiciones al último precio). **Productor:** trader. **Consumidor:** dashboard.
- `METRICS:CASH` — Float en string, efectivo disponible. **Productor:** trader. **Consumidor:** dashboard.
- `METRICS:PNL_DAILY` — Float en string, PnL diario (placeholder por defecto). **Productor:** trader. **Consumidor:** dashboard.
- `MARKET:LAST_PRICE` — Float en string, último precio del símbolo activo. **Productor:** trader. **Consumidor:** dashboard.
- `MARKET:LAST_TICK` — JSON `{symbol, price, timestamp}` del último tick conocido. **Productor:** trader. **Consumidor:** dashboard.
- `VIEW:ORDERS` — JSON array de órdenes visibles, p.ej. `[{"order_id": "...", "side": "BUY", "qty": 0.1, "price": 110.0, "status": "FILLED", "ts": "..."}]`. **Productor:** trader al registrar fills. **Consumidor:** dashboard (tabla de órdenes).
- `VIEW:POSITIONS` — JSON array de posiciones, p.ej. `[{"instrument_id": "BTC/USDT", "quantity": 0.5}]`. **Productor:** trader. **Consumidor:** dashboard.
- `oms:portfolio` — Modelo Pydantic `Portfolio` serializado. **Productor/Consumidor:** OMS.
- `oms:order:<id>` — Modelo `Order` serializado por ID. **Productor/Consumidor:** OMS.
