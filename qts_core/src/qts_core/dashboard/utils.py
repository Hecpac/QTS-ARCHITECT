from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


def safe_float(raw: str | None, default: float = 0.0) -> float:
    """Parse a float value defensively."""
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def safe_json(raw: str | None, default: Any) -> Any:
    """Parse JSON defensively, returning default on errors."""
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def symbol_key_suffix(symbol: str) -> str:
    """Normalize symbol into Redis key suffix (e.g. BTC/USDT -> BTC_USDT)."""
    return symbol.upper().replace("/", "_").replace(":", "_").replace("-", "_")


def symbol_scoped_key(base_key: str, symbol: str) -> str:
    """Build symbol-scoped key from a base key."""
    return f"{base_key}:{symbol_key_suffix(symbol)}"


def parse_active_symbols(raw: str | None) -> list[str]:
    """Parse active symbols payload from Redis into a normalized list."""
    parsed = safe_json(raw, default=[])
    if not isinstance(parsed, list):
        return []

    symbols: list[str] = []
    for item in parsed:
        value = str(item).strip()
        if not value:
            continue
        if value in symbols:
            continue
        symbols.append(value)

    return symbols


def parse_yahoo_chart_payload(payload: dict[str, Any]) -> list[dict[str, float | str]]:
    """Parse Yahoo Finance chart payload into normalized OHLCV rows."""
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict):
        return []

    results = chart.get("result")
    if not isinstance(results, list) or not results:
        return []

    result = results[0]
    if not isinstance(result, dict):
        return []

    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, dict):
        return []

    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or not quotes:
        return []

    quote_payload = quotes[0]
    if not isinstance(quote_payload, dict):
        return []

    opens = quote_payload.get("open")
    highs = quote_payload.get("high")
    lows = quote_payload.get("low")
    closes = quote_payload.get("close")
    volumes = quote_payload.get("volume")

    if not all(isinstance(series, list) for series in (opens, highs, lows, closes, volumes)):
        return []

    rows: list[dict[str, float | str]] = []
    for index, timestamp in enumerate(timestamps):
        if not isinstance(timestamp, (int, float)):
            continue

        if index >= len(opens) or index >= len(highs) or index >= len(lows) or index >= len(closes):
            continue

        open_value = opens[index]
        high_value = highs[index]
        low_value = lows[index]
        close_value = closes[index]
        volume_value = volumes[index] if index < len(volumes) else 0

        if None in (open_value, high_value, low_value, close_value):
            continue

        try:
            row = {
                "timestamp": datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat(),
                "open": float(open_value),
                "high": float(high_value),
                "low": float(low_value),
                "close": float(close_value),
                "volume": float(volume_value or 0.0),
            }
        except (TypeError, ValueError):
            continue

        rows.append(row)

    return rows


def fetch_yahoo_chart_rows(
    symbol: str,
    *,
    interval: str = "1h",
    range_: str = "5d",
    timeout_seconds: float = 5.0,
) -> list[dict[str, float | str]]:
    """Fetch OHLCV rows from Yahoo Finance chart endpoint."""
    encoded_symbol = quote(symbol, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        f"?interval={quote(interval)}&range={quote(range_)}"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []

    return parse_yahoo_chart_payload(payload)


def heartbeat_age_seconds(
    heartbeat_iso: str | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Compute heartbeat age in seconds from an ISO timestamp."""
    if not heartbeat_iso:
        return None

    normalized = heartbeat_iso.replace("Z", "+00:00")
    try:
        heartbeat_dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if heartbeat_dt.tzinfo is None:
        heartbeat_dt = heartbeat_dt.replace(tzinfo=timezone.utc)
    else:
        heartbeat_dt = heartbeat_dt.astimezone(timezone.utc)

    now_dt = now or datetime.now(timezone.utc)
    age = (now_dt - heartbeat_dt).total_seconds()
    return max(0.0, age)


def _event_epoch_ms(key: str, *, prefix: str) -> int | None:
    if not key.startswith(f"{prefix}:"):
        return None

    suffix = key.rsplit(":", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def load_alert_events(
    redis_client: Any,
    *,
    event_prefix: str = "ALERTS:EVENT",
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Load recent alert events from Redis ordered by newest first."""
    if limit <= 0:
        return []

    candidates: list[tuple[int, str]] = []

    try:
        for key in redis_client.scan_iter(match=f"{event_prefix}:*", count=200):
            if not isinstance(key, str):
                continue
            epoch_ms = _event_epoch_ms(key, prefix=event_prefix)
            if epoch_ms is None:
                continue
            candidates.append((epoch_ms, key))
    except Exception:
        return []

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:limit]
    keys = [key for _, key in selected]

    try:
        payloads = redis_client.mget(keys)
    except Exception:
        payloads = [redis_client.get(key) for key in keys]

    events: list[dict[str, Any]] = []
    for (epoch_ms, key), payload in zip(selected, payloads, strict=False):
        if payload is None:
            continue

        parsed = safe_json(payload, default={})
        if isinstance(parsed, dict):
            event = parsed.copy()
        else:
            event = {"raw": str(payload)}

        event.setdefault("event_key", key)
        event.setdefault("event_epoch_ms", epoch_ms)
        events.append(event)

    return events
