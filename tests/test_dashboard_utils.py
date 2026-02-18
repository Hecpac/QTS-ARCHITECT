from __future__ import annotations

import json
from datetime import datetime, timezone

from qts_core.dashboard.utils import heartbeat_age_seconds, load_alert_events, safe_float


class FakeRedis:
    def __init__(self, data: dict[str, str]) -> None:
        self.data = data

    def scan_iter(self, *, match: str, count: int = 200):  # noqa: ARG002
        prefix = match[:-1] if match.endswith("*") else match
        for key in self.data:
            if key.startswith(prefix):
                yield key

    def mget(self, keys: list[str]) -> list[str | None]:
        return [self.data.get(key) for key in keys]

    def get(self, key: str) -> str | None:
        return self.data.get(key)


def test_safe_float_returns_default_on_invalid_input() -> None:
    assert safe_float("12.5") == 12.5
    assert safe_float("bad", default=3.0) == 3.0
    assert safe_float(None, default=7.0) == 7.0


def test_heartbeat_age_seconds_parses_iso_timestamp() -> None:
    now = datetime(2026, 2, 17, 18, 0, tzinfo=timezone.utc)
    heartbeat = "2026-02-17T17:59:58+00:00"

    age = heartbeat_age_seconds(heartbeat, now=now)
    assert age is not None
    assert age == 2.0


def test_load_alert_events_returns_newest_first() -> None:
    payload_old = json.dumps({"level": "WARNING", "event": "A", "message": "older"})
    payload_new = json.dumps({"level": "ERROR", "event": "B", "message": "newer"})

    redis_client = FakeRedis(
        {
            "ALERTS:EVENT:100": payload_old,
            "ALERTS:EVENT:200": payload_new,
            "ALERTS:EVENT:bad": "not-used",
            "OTHER:KEY": "ignored",
        }
    )

    events = load_alert_events(redis_client, limit=5)

    assert len(events) == 2
    assert events[0]["event"] == "B"
    assert events[0]["event_epoch_ms"] == 200
    assert events[1]["event"] == "A"
    assert events[1]["event_epoch_ms"] == 100
