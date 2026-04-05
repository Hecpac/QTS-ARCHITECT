"""Alert relay: polls Redis and forwards alerts to Slack / Telegram.

Security improvements:
- Uses httpx instead of urllib.request for proper TLS verification
- URL validation to prevent SSRF
- Redis connection with password support
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

import redis

Sender = Callable[[str], None]

_SEVERITY_SCORE = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

# Allowed HTTPS hosts for outbound webhook requests (SSRF prevention)
_ALLOWED_WEBHOOK_HOSTS: frozenset[str] = frozenset({
    "hooks.slack.com",
    "api.telegram.org",
})


def _validate_webhook_url(url: str) -> str:
    """Validate that a webhook URL is HTTPS and targets an allowed host.

    Args:
        url: The webhook URL to validate.

    Returns:
        The validated URL.

    Raises:
        ValueError: If the URL is invalid or targets a disallowed host.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Webhook URL must use HTTPS, got: {parsed.scheme}")

    if parsed.hostname not in _ALLOWED_WEBHOOK_HOSTS:
        raise ValueError(
            f"Webhook host '{parsed.hostname}' not in allowed list: "
            f"{_ALLOWED_WEBHOOK_HOSTS}"
        )

    return url


@dataclass(frozen=True)
class AlertRelayConfig:
    """Runtime configuration for alert relay polling."""

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    alert_key: str = "ALERTS:LAST"
    poll_interval_seconds: float = 2.0
    http_timeout_seconds: float = 5.0
    min_level: str = "WARNING"

    slack_webhook_url: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    @classmethod
    def from_env(cls) -> AlertRelayConfig:
        """Build config from environment variables."""
        return cls(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            redis_password=os.getenv("REDIS_PASSWORD"),
            alert_key=os.getenv("ALERT_KEY", "ALERTS:LAST"),
            poll_interval_seconds=float(os.getenv("ALERT_POLL_SECONDS", "2")),
            http_timeout_seconds=float(os.getenv("ALERT_HTTP_TIMEOUT_SECONDS", "5")),
            min_level=os.getenv("ALERT_MIN_LEVEL", "WARNING").upper(),
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        )


def parse_alert_payload(payload: str) -> dict[str, Any]:
    """Parse serialized alert payload from Redis."""
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {
        "level": "INFO",
        "event": "RAW_PAYLOAD",
        "message": payload,
    }


def format_alert_message(alert: dict[str, Any]) -> str:
    """Format alert data into a compact outbound message."""
    level = str(alert.get("level", "INFO")).upper()
    event = str(alert.get("event", "UNKNOWN"))
    message = str(alert.get("message", ""))
    timestamp = str(alert.get("timestamp", "unknown-ts"))

    detail_parts: list[str] = []
    for key in ("reason", "order_id", "error"):
        if key in alert and alert[key] not in (None, ""):
            detail_parts.append(f"{key}={alert[key]}")

    details = ""
    if detail_parts:
        details = f" | {'; '.join(detail_parts)}"

    return f"[{level}] {event} @ {timestamp} — {message}{details}"


def level_passes_filter(level: str, min_level: str) -> bool:
    """Return True when event level meets minimum relay threshold."""
    level_score = _SEVERITY_SCORE.get(level.upper(), _SEVERITY_SCORE["INFO"])
    min_score = _SEVERITY_SCORE.get(min_level.upper(), _SEVERITY_SCORE["WARNING"])
    return level_score >= min_score


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> None:
    """POST JSON payload via httpx with TLS verification."""
    import httpx

    _validate_webhook_url(url)

    response = httpx.post(
        url,
        json=payload,
        timeout=timeout_seconds,
        # httpx verifies TLS certificates by default
    )
    response.raise_for_status()


def send_to_slack(text: str, *, webhook_url: str, timeout_seconds: float) -> None:
    """Send alert text to Slack incoming webhook."""
    _post_json(
        webhook_url,
        payload={"text": text},
        timeout_seconds=timeout_seconds,
    )


def send_to_telegram(
    text: str,
    *,
    bot_token: str,
    chat_id: str,
    timeout_seconds: float,
) -> None:
    """Send alert text via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    _post_json(
        url,
        payload={"chat_id": chat_id, "text": text},
        timeout_seconds=timeout_seconds,
    )


class AlertRelay:
    """Poll Redis alert key and relay new events to configured sinks."""

    def __init__(
        self,
        redis_client: redis.Redis,  # type: ignore[type-arg]
        *,
        alert_key: str,
        senders: list[Sender],
        min_level: str = "WARNING",
    ) -> None:
        self.redis_client = redis_client
        self.alert_key = alert_key
        self.senders = senders
        self.min_level = min_level
        self._last_payload: str | None = None

    def poll_once(self) -> bool:
        """Poll alert key once and relay if a fresh payload is found."""
        try:
            payload = self.redis_client.get(self.alert_key)
        except Exception as exc:
            print(
                f"[alert-relay] redis read failed (key={self.alert_key}): {exc}",
                file=sys.stderr,
            )
            return False

        if payload is None or payload == self._last_payload:
            return False

        self._last_payload = payload

        alert = parse_alert_payload(payload)
        level = str(alert.get("level", "INFO")).upper()
        if not level_passes_filter(level, self.min_level):
            return False

        message = format_alert_message(alert)

        print(f"[alert-relay] {message}")
        for sender in self.senders:
            try:
                sender(message)
            except Exception as exc:
                print(f"[alert-relay] sender failed: {exc}", file=sys.stderr)

        return True


def build_senders(config: AlertRelayConfig) -> list[Sender]:
    """Create outbound sender callables from config."""
    senders: list[Sender] = []

    if config.slack_webhook_url:
        # Validate at build time, not at send time
        _validate_webhook_url(config.slack_webhook_url)
        senders.append(
            lambda text: send_to_slack(
                text,
                webhook_url=config.slack_webhook_url or "",
                timeout_seconds=config.http_timeout_seconds,
            )
        )

    if config.telegram_bot_token and config.telegram_chat_id:
        senders.append(
            lambda text: send_to_telegram(
                text,
                bot_token=config.telegram_bot_token or "",
                chat_id=config.telegram_chat_id or "",
                timeout_seconds=config.http_timeout_seconds,
            )
        )

    return senders


def main() -> None:
    """Run alert relay loop forever."""
    config = AlertRelayConfig.from_env()

    client = redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        db=config.redis_db,
        password=config.redis_password,
        decode_responses=True,
    )

    relay = AlertRelay(
        client,
        alert_key=config.alert_key,
        senders=build_senders(config),
        min_level=config.min_level,
    )

    print(
        "[alert-relay] started"
        f" redis={config.redis_host}:{config.redis_port}/{config.redis_db}"
        f" key={config.alert_key}"
        f" min_level={config.min_level}"
        f" poll={config.poll_interval_seconds}s"
    )

    while True:
        relay.poll_once()
        time.sleep(max(0.1, config.poll_interval_seconds))


if __name__ == "__main__":
    main()
