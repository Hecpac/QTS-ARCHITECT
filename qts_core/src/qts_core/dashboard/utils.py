from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


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
