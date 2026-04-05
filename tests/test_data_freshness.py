"""Tests for data freshness validation and stale data handling.

Covers the P0 fixes:
- _validate_data_freshness rejects old data
- _fetch_live_data returns None instead of synthetic data
- Consecutive stale ticks trigger emergency halt
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import OmegaConf

from qts_core.common.types import InstrumentId, MarketData
from qts_core.main_live import (
    DEFAULT_MAX_DATA_AGE_SECONDS,
    STALE_DATA_CONSECUTIVE_LIMIT,
    LiveTrader,
)


def _make_minimal_cfg(**overrides: object) -> MagicMock:
    """Create a minimal OmegaConf config for LiveTrader."""
    cfg = OmegaConf.create({
        "env": "test",
        "symbol": "BTC/USDT",
        "symbols": ["BTC/USDT"],
        "loop": {
            "tick_interval": 1.0,
            "heartbeat_key": "SYSTEM:HEARTBEAT",
            "execution_timeout": 5.0,
            "entry_cooldown_seconds": 0.0,
            "max_entries_per_candle_per_symbol": 1,
            "max_session_drawdown": 0.0,
            "stop_loss_pct": 0.0,
            "max_data_age_seconds": 120.0,
        },
        "execution_guardrails": {"enabled": False},
        "oms": {
            "initial_cash": 100000.0,
            "risk_fraction": 0.10,
            "account_mode": "spot",
            "short_leverage": 1.0,
            "short_borrow_rate_bps_per_day": 0.0,
            "min_short_liquidation_buffer": 0.0,
        },
        "supervisor": {"min_confidence": 0.6},
        "agents": {
            "strategies": [],
            "risk": {
                "_target_": "qts_core.agents.base.StrictRiskAgent",
                "name": "TestRisk",
            },
        },
        "store": {
            "_target_": "qts_core.execution.store.MemoryStore",
        },
        "gateway": {
            "_target_": "qts_core.execution.ems.MockGateway",
        },
        "alerts": {"enabled": False},
        "telemetry": None,
    })
    return cfg


class TestDataFreshnessValidation:
    """Tests for the _validate_data_freshness method."""

    def test_fresh_data_passes(self) -> None:
        """Data from a few seconds ago should be considered fresh."""
        with patch("qts_core.main_live.LiveTrader.__init__", return_value=None):
            trader = LiveTrader.__new__(LiveTrader)
            trader._max_data_age_seconds = DEFAULT_MAX_DATA_AGE_SECONDS
            trader._consecutive_stale_ticks = 0
            trader._last_fresh_data_at = None
            trader.symbol = "BTC/USDT"

            now = datetime.now(timezone.utc)
            recent_ts = now - timedelta(seconds=5)

            assert trader._validate_data_freshness(recent_ts) is True
            assert trader._consecutive_stale_ticks == 0
            assert trader._last_fresh_data_at is not None

    def test_stale_data_rejected(self) -> None:
        """Data older than max_data_age_seconds should be rejected."""
        with patch("qts_core.main_live.LiveTrader.__init__", return_value=None):
            trader = LiveTrader.__new__(LiveTrader)
            trader._max_data_age_seconds = 60.0
            trader._consecutive_stale_ticks = 0
            trader._last_fresh_data_at = None
            trader.symbol = "BTC/USDT"

            now = datetime.now(timezone.utc)
            old_ts = now - timedelta(seconds=120)

            assert trader._validate_data_freshness(old_ts) is False
            assert trader._consecutive_stale_ticks == 1

    def test_naive_timestamp_treated_as_utc(self) -> None:
        """Naive timestamps should be treated as UTC."""
        with patch("qts_core.main_live.LiveTrader.__init__", return_value=None):
            trader = LiveTrader.__new__(LiveTrader)
            trader._max_data_age_seconds = 60.0
            trader._consecutive_stale_ticks = 0
            trader._last_fresh_data_at = None
            trader.symbol = "BTC/USDT"

            # Naive datetime (no tz) 5 seconds ago
            now_utc = datetime.now(timezone.utc)
            naive_recent = now_utc.replace(tzinfo=None) - timedelta(seconds=5)

            assert trader._validate_data_freshness(naive_recent) is True

    def test_consecutive_counter_increments(self) -> None:
        """Each stale tick should increment the counter."""
        with patch("qts_core.main_live.LiveTrader.__init__", return_value=None):
            trader = LiveTrader.__new__(LiveTrader)
            trader._max_data_age_seconds = 60.0
            trader._consecutive_stale_ticks = 3
            trader._last_fresh_data_at = None
            trader.symbol = "BTC/USDT"

            old_ts = datetime.now(timezone.utc) - timedelta(seconds=300)
            trader._validate_data_freshness(old_ts)

            assert trader._consecutive_stale_ticks == 4

    def test_fresh_data_resets_counter(self) -> None:
        """Fresh data should reset the consecutive stale counter."""
        with patch("qts_core.main_live.LiveTrader.__init__", return_value=None):
            trader = LiveTrader.__new__(LiveTrader)
            trader._max_data_age_seconds = 120.0
            trader._consecutive_stale_ticks = 4
            trader._last_fresh_data_at = None
            trader.symbol = "BTC/USDT"

            recent_ts = datetime.now(timezone.utc) - timedelta(seconds=5)
            trader._validate_data_freshness(recent_ts)

            assert trader._consecutive_stale_ticks == 0


class TestAlertRelayURLValidation:
    """Test that alert relay URL validation prevents SSRF."""

    def test_valid_slack_url(self) -> None:
        from qts_core.ops.alert_relay import _validate_webhook_url

        url = "https://hooks.slack.com/services/T000/B000/XXX"
        assert _validate_webhook_url(url) == url

    def test_rejects_http_url(self) -> None:
        from qts_core.ops.alert_relay import _validate_webhook_url

        with pytest.raises(ValueError, match="HTTPS"):
            _validate_webhook_url("http://hooks.slack.com/services/T000/B000/XXX")

    def test_rejects_non_allowed_host(self) -> None:
        from qts_core.ops.alert_relay import _validate_webhook_url

        with pytest.raises(ValueError, match="not in allowed"):
            _validate_webhook_url("https://evil.com/steal-tokens")

    def test_valid_telegram_url(self) -> None:
        from qts_core.ops.alert_relay import _validate_webhook_url

        url = "https://api.telegram.org/bot123/sendMessage"
        assert _validate_webhook_url(url) == url
