from __future__ import annotations

import json
from datetime import datetime, timezone

from qts_core.dashboard.utils import (
    heartbeat_age_seconds,
    load_alert_events,
    parse_active_symbols,
    parse_rss_news_items,
    parse_yahoo_chart_payload,
    safe_float,
    symbol_key_suffix,
    symbol_scoped_key,
)


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


def test_symbol_key_helpers() -> None:
    assert symbol_key_suffix("BTC/USDT") == "BTC_USDT"
    assert symbol_scoped_key("MARKET:LAST_PRICE", "ETH/USDT") == "MARKET:LAST_PRICE:ETH_USDT"


def test_parse_active_symbols_filters_invalid_entries() -> None:
    raw = json.dumps(["BTC/USDT", "ETH/USDT", "", "BTC/USDT"])
    assert parse_active_symbols(raw) == ["BTC/USDT", "ETH/USDT"]


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


def test_parse_yahoo_chart_payload_returns_normalized_rows() -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1708192800, 1708196400],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0],
                                "high": [102.0, 103.0],
                                "low": [99.0, 100.5],
                                "close": [101.5, 102.2],
                                "volume": [1000, 1200],
                            }
                        ]
                    },
                }
            ]
        }
    }

    rows = parse_yahoo_chart_payload(payload)

    assert len(rows) == 2
    assert rows[0]["open"] == 100.0
    assert rows[1]["close"] == 102.2
    assert "timestamp" in rows[0]


def test_parse_rss_news_items_parses_and_sorts() -> None:
    payload = """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<rss version=\"2.0\">
  <channel>
    <item>
      <title>Older</title>
      <link>https://example.com/older</link>
      <pubDate>Tue, 17 Feb 2026 17:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Newer</title>
      <link>https://example.com/newer</link>
      <pubDate>Tue, 17 Feb 2026 18:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

    items = parse_rss_news_items(payload, source="TestFeed", limit=10)

    assert len(items) == 2
    assert items[0]["title"] == "Newer"
    assert items[0]["source"] == "TestFeed"
    assert items[1]["title"] == "Older"
    assert items[0]["published_ts"] >= items[1]["published_ts"]
