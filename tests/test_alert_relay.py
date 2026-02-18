from __future__ import annotations

from qts_core.ops.alert_relay import AlertRelay, format_alert_message, parse_alert_payload


class FakeRedis:
    def __init__(self, payloads: list[str | None]) -> None:
        self.payloads = payloads
        self._index = 0

    def get(self, key: str) -> str | None:  # noqa: ARG002
        if self._index >= len(self.payloads):
            return self.payloads[-1] if self.payloads else None

        value = self.payloads[self._index]
        self._index += 1
        return value


def test_parse_alert_payload_falls_back_to_raw_message() -> None:
    parsed = parse_alert_payload("not-json")

    assert parsed["event"] == "RAW_PAYLOAD"
    assert parsed["message"] == "not-json"


def test_format_alert_message_includes_core_fields() -> None:
    message = format_alert_message(
        {
            "level": "WARNING",
            "event": "ORDER_SUBMISSION_TIMEOUT",
            "message": "submit timed out",
            "timestamp": "2026-02-17T18:00:00Z",
            "order_id": "abc-123",
        }
    )

    assert "[WARNING]" in message
    assert "ORDER_SUBMISSION_TIMEOUT" in message
    assert "submit timed out" in message
    assert "order_id=abc-123" in message


def test_alert_relay_emits_only_for_new_payload() -> None:
    payload = (
        '{"level":"ERROR","event":"UNIT_TEST","message":"boom",'
        '"timestamp":"2026-02-17T18:01:00Z"}'
    )

    fake_redis = FakeRedis([payload, payload, None])
    sent: list[str] = []

    relay = AlertRelay(
        fake_redis,  # type: ignore[arg-type]
        alert_key="ALERTS:LAST",
        senders=[sent.append],
        min_level="WARNING",
    )

    assert relay.poll_once() is True
    assert relay.poll_once() is False
    assert relay.poll_once() is False
    assert len(sent) == 1
    assert "UNIT_TEST" in sent[0]
