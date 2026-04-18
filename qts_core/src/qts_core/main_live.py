from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import signal
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

import hydra
import structlog
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from qts_core.agents.prediction_gate import GateState, PredictionMarketGate
from qts_core.agents.protocol import SignalType, TradingDecision
from qts_core.agents.sentiment_filter import SentimentFilter
from qts_core.agents.signal_quality import SignalQuality, SignalQualityEvaluator
from qts_core.agents.supervisor import ConsensusStrategy, Supervisor
from qts_core.agents.watchdog import DrawdownWatchdog, WatchdogState
from qts_core.common.types import InstrumentId, MarketData
from qts_core.execution.ems import ExecutionError, ExecutionStatus
from qts_core.execution.oms import OrderManagementSystem, OrderRequest
from qts_core.models.regime import MarketRegime, RegimeDetector
from qts_core.polymarket import (
    ImpliedDirection,
    PolymarketLoader,
    SentimentLoader,
)


if TYPE_CHECKING:
    from qts_core.execution.ems import ExecutionGateway, FillReport
    from qts_core.execution.store import StateStore

OHLCVTuple = tuple[datetime, float, float, float, float, float]

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Data freshness constants
# ---------------------------------------------------------------------------
DEFAULT_MAX_DATA_AGE_SECONDS: float = 120.0  # 2 minutes
STALE_DATA_CONSECUTIVE_LIMIT: int = 5  # halt after N consecutive stale ticks


def _iter_target_strings(node: object) -> list[str]:
    """Collect Hydra `_target_` strings from a nested config subtree."""
    if isinstance(node, DictConfig):
        targets: list[str] = []
        target = node.get("_target_")
        if isinstance(target, str):
            targets.append(target)
        for value in node.values():
            targets.extend(_iter_target_strings(value))
        return targets

    if isinstance(node, dict):
        targets = []
        target = node.get("_target_")
        if isinstance(target, str):
            targets.append(target)
        for value in node.values():
            targets.extend(_iter_target_strings(value))
        return targets

    if isinstance(node, (list, tuple)):
        targets = []
        for value in node:
            targets.extend(_iter_target_strings(value))
        return targets

    return []


def _symbol_to_sentiment_query(symbol: str) -> str:
    """Map a tradable symbol to the query used by social/prediction loaders."""
    base_symbol = symbol.split(":")[-1].split("-")[0].split("/")[0].upper()
    query_map = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "SOL": "Solana",
        "GOLD": "Gold",
        "XAU": "Gold",
        "NDX": "Nasdaq",
        "SPX": "S&P 500",
    }
    return query_map.get(base_symbol, base_symbol)


def _infer_prediction_direction(question: str, query: str) -> ImpliedDirection | None:
    """Infer whether a Polymarket YES contract is bullish or bearish for the asset."""
    normalized = question.lower()
    query_lower = query.lower()

    if query_lower not in normalized:
        return None

    bullish_markers = (
        "above",
        "over",
        "rise",
        "rally",
        "gain",
        "bull",
        "surge",
        "higher",
        "hit",
    )
    bearish_markers = (
        "below",
        "under",
        "drop",
        "fall",
        "bear",
        "crash",
        "lower",
        "recession",
        "lose",
    )

    if any(marker in normalized for marker in bullish_markers):
        return ImpliedDirection.BULLISH
    if any(marker in normalized for marker in bearish_markers):
        return ImpliedDirection.BEARISH
    return None


def validate_api_keys(cfg: DictConfig) -> None:
    """Verify required API keys are available before starting the trading loop.

    Raises RuntimeError for critical missing keys, warns for optional ones.
    """
    targets = _iter_target_strings(cfg.get("strategy")) + _iter_target_strings(
        cfg.get("agents")
    )
    if any(target.startswith("qts_core.agents.claude_agent.") for target in targets):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set but Claude agents are enabled. "
                "Set the env var or remove the Claude strategy bundle."
            )
        log.info("api_key_check_passed", service="anthropic")

    # Reddit credentials — optional (sentiment degrades gracefully)
    if not (
        os.environ.get("REDDIT_CLIENT_ID")
        and os.environ.get("REDDIT_CLIENT_SECRET")
    ):
        log.warning(
            "api_key_missing_optional",
            service="reddit",
            impact="Sentiment loader will skip Reddit data",
        )

    # OANDA — required for forex/metals
    gateway_target = str(cfg.get("gateway", {}).get("_target_", ""))
    if "OANDAGateway" in gateway_target:
        if not os.environ.get("OANDA_API_TOKEN"):
            raise RuntimeError(
                "OANDA_API_TOKEN not set but OANDA exchange is configured."
            )


def apply_execution_guardrails(
    decision: TradingDecision,
    market_data: MarketData,
    *,
    enabled: bool = True,
    min_volume: float = 0.0,
    max_intrabar_volatility: float = 0.05,
    high_volatility_size_scale: float = 0.5,
    max_estimated_slippage_bps: float = 50.0,
    slippage_volatility_factor: float = 0.25,
    max_estimated_spread_bps: float = 0.0,
    spread_volatility_factor: float = 0.15,
    max_participation_rate: float = 0.0,
    projected_quantity: float | None = None,
    enable_dynamic_volatility_risk_scaling: bool = True,
    min_dynamic_risk_scale: float = 0.25,
    high_volatility_hours_utc: tuple[int, ...] = (),
    high_volatility_hours_size_scale: float = 1.0,
    high_volatility_hours_entry_block: bool = False,
) -> TradingDecision | None:
    """Apply market microstructure guardrails to a trading decision.

    Returns:
        - None if the trade should be rejected.
        - Original or adjusted TradingDecision otherwise.
    """
    if not enabled:
        return decision

    if market_data.close <= 0:
        return None

    if min_volume > 0 and market_data.volume < min_volume:
        return None

    timestamp_utc = market_data.timestamp
    if timestamp_utc.tzinfo is None:
        timestamp_utc = timestamp_utc.replace(tzinfo=timezone.utc)
    else:
        timestamp_utc = timestamp_utc.astimezone(timezone.utc)

    hour_utc = timestamp_utc.hour
    in_high_vol_hour = bool(high_volatility_hours_utc) and hour_utc in high_volatility_hours_utc
    if in_high_vol_hour and high_volatility_hours_entry_block:
        return None

    intrabar_volatility = max(0.0, (market_data.high - market_data.low) / market_data.close)
    estimated_slippage_bps = intrabar_volatility * slippage_volatility_factor * 10_000
    estimated_spread_bps = intrabar_volatility * spread_volatility_factor * 10_000

    if max_estimated_slippage_bps > 0 and estimated_slippage_bps > max_estimated_slippage_bps:
        return None

    if max_estimated_spread_bps > 0 and estimated_spread_bps > max_estimated_spread_bps:
        return None

    adjusted_modifier = decision.quantity_modifier
    rationale_suffix: list[str] = []

    if (
        enable_dynamic_volatility_risk_scaling
        and max_intrabar_volatility > 0
        and intrabar_volatility > max_intrabar_volatility
    ):
        dynamic_scale = max_intrabar_volatility / intrabar_volatility
        dynamic_scale = max(min_dynamic_risk_scale, min(1.0, dynamic_scale))
        if dynamic_scale < 1.0:
            adjusted_modifier *= dynamic_scale
            rationale_suffix.append(
                f"Guardrail: dynamic risk scale {dynamic_scale:.2f}"
            )

    if (
        max_intrabar_volatility > 0
        and intrabar_volatility > max_intrabar_volatility
        and high_volatility_size_scale < 1.0
    ):
        adjusted_modifier *= max(0.0, high_volatility_size_scale)
        rationale_suffix.append(
            f"Guardrail: high volatility {intrabar_volatility:.2%}"
        )

    if in_high_vol_hour and high_volatility_hours_size_scale < 1.0:
        adjusted_modifier *= max(0.0, high_volatility_hours_size_scale)
        rationale_suffix.append(
            f"Guardrail: high-vol hour {hour_utc:02d}:00 UTC"
        )

    if max_participation_rate > 0:
        if market_data.volume <= 0:
            return None

        if projected_quantity is not None and projected_quantity > 0:
            if decision.quantity_modifier > 0:
                quantity_scale = adjusted_modifier / decision.quantity_modifier
            else:
                quantity_scale = 0.0

            projected_effective_qty = projected_quantity * max(0.0, quantity_scale)
            projected_participation = projected_effective_qty / market_data.volume

            if projected_participation > max_participation_rate:
                participation_scale = max_participation_rate / projected_participation
                adjusted_modifier *= max(0.0, participation_scale)
                rationale_suffix.append(
                    "Guardrail: participation cap "
                    f"{projected_participation:.2%}>{max_participation_rate:.2%}"
                )

    adjusted_modifier = max(0.0, min(1.0, adjusted_modifier))
    if adjusted_modifier < 1e-8:
        return None

    if adjusted_modifier == decision.quantity_modifier and not rationale_suffix:
        return decision

    rationale = decision.rationale
    if rationale_suffix:
        rationale_detail = "; ".join(rationale_suffix)
        rationale = f"{rationale} | {rationale_detail}" if rationale else rationale_detail

    return decision.model_copy(
        update={
            "quantity_modifier": adjusted_modifier,
            "rationale": rationale,
        }
    )


class _TeeStream:
    """Mirror writes to both stdout and a local file handle."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def configure_logging(level: str = "INFO") -> None:
    """Configure logging for the live trader."""
    log_path = Path("main_live.log")
    file_stream = log_path.open("a", encoding="utf-8", buffering=1)
    atexit.register(file_stream.close)
    tee_stream = _TeeStream(sys.stdout, file_stream)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        stream=tee_stream,
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=cast(TextIO, tee_stream)),
        cache_logger_on_first_use=True,
    )


class LiveTrader:
    """Live trading loop: data feed → supervisor → OMS → EMS, with telemetry."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the live trader from a Hydra config."""
        self.cfg = cfg

        configured_symbols = cfg.get("symbols")
        if configured_symbols:
            if isinstance(configured_symbols, str):
                parsed_symbols = [configured_symbols]
            else:
                parsed_symbols = [str(item) for item in configured_symbols]
        else:
            parsed_symbols = [str(cfg.get("symbol", "BTC/USDT"))]

        normalized_symbols = [symbol.strip() for symbol in parsed_symbols if symbol.strip()]
        self.symbols: list[str] = list(dict.fromkeys(normalized_symbols)) or ["BTC/USDT"]
        self.symbol: str = self.symbols[0]

        self.tick_interval: float = float(cfg.loop.get("tick_interval", 1.0))
        self.heartbeat_key: str = cfg.loop.get("heartbeat_key", "SYSTEM:HEARTBEAT")
        self.execution_timeout: float = max(
            0.1,
            float(cfg.loop.get("execution_timeout", max(5.0, self.tick_interval * 2))),
        )
        self.entry_cooldown_seconds: float = max(
            0.0,
            float(cfg.loop.get("entry_cooldown_seconds", 0.0)),
        )
        self.max_entries_per_candle_per_symbol: int = max(
            1,
            int(cfg.loop.get("max_entries_per_candle_per_symbol", 1)),
        )

        guardrails_cfg = cfg.get("execution_guardrails")
        self.guardrails_enabled: bool = bool(
            guardrails_cfg.get("enabled", True) if guardrails_cfg else True
        )
        self.guardrails_min_volume: float = float(
            guardrails_cfg.get("min_volume", 0.0) if guardrails_cfg else 0.0
        )
        self.guardrails_max_intrabar_volatility: float = float(
            guardrails_cfg.get("max_intrabar_volatility", 0.05)
            if guardrails_cfg else 0.05
        )
        self.guardrails_high_volatility_size_scale: float = float(
            guardrails_cfg.get("high_volatility_size_scale", 0.5)
            if guardrails_cfg else 0.5
        )
        self.guardrails_max_estimated_slippage_bps: float = float(
            guardrails_cfg.get("max_estimated_slippage_bps", 50.0)
            if guardrails_cfg else 50.0
        )
        self.guardrails_slippage_volatility_factor: float = float(
            guardrails_cfg.get("slippage_volatility_factor", 0.25)
            if guardrails_cfg else 0.25
        )
        self.guardrails_max_estimated_spread_bps: float = float(
            guardrails_cfg.get("max_estimated_spread_bps", 0.0)
            if guardrails_cfg else 0.0
        )
        self.guardrails_spread_volatility_factor: float = float(
            guardrails_cfg.get("spread_volatility_factor", 0.15)
            if guardrails_cfg else 0.15
        )
        self.guardrails_max_participation_rate: float = max(
            0.0,
            float(guardrails_cfg.get("max_participation_rate", 0.0) if guardrails_cfg else 0.0),
        )
        self.guardrails_enable_dynamic_volatility_risk_scaling: bool = bool(
            guardrails_cfg.get("enable_dynamic_volatility_risk_scaling", True)
            if guardrails_cfg else True
        )
        self.guardrails_min_dynamic_risk_scale: float = max(
            0.0,
            min(
                1.0,
                float(guardrails_cfg.get("min_dynamic_risk_scale", 0.25) if guardrails_cfg else 0.25),
            ),
        )
        raw_high_vol_hours = (
            guardrails_cfg.get("high_volatility_hours_utc", []) if guardrails_cfg else []
        )
        parsed_high_vol_hours: list[int] = []
        for raw_hour in raw_high_vol_hours:
            try:
                hour = int(raw_hour)
            except (TypeError, ValueError):
                continue
            if 0 <= hour <= 23:
                parsed_high_vol_hours.append(hour)

        self.guardrails_high_volatility_hours_utc: tuple[int, ...] = tuple(
            sorted(set(parsed_high_vol_hours))
        )
        self.guardrails_high_volatility_hours_size_scale: float = max(
            0.0,
            min(
                1.0,
                float(
                    guardrails_cfg.get("high_volatility_hours_size_scale", 1.0)
                    if guardrails_cfg else 1.0
                ),
            ),
        )
        self.guardrails_high_volatility_hours_entry_block: bool = bool(
            guardrails_cfg.get("high_volatility_hours_entry_block", False)
            if guardrails_cfg else False
        )

        self._last_market_data: MarketData | None = None
        self._last_ohlcv_history: list[OHLCVTuple] | None = None
        self._last_ohlcv_payload: list[dict[str, Any]] | None = None

        # Async lock protecting mutable shared state
        self._state_lock = asyncio.Lock()

        # Data freshness tracking
        self._max_data_age_seconds: float = float(
            cfg.loop.get("max_data_age_seconds", DEFAULT_MAX_DATA_AGE_SECONDS)
        )
        self._consecutive_stale_ticks: int = 0
        self._last_fresh_data_at: datetime | None = None

        # Persistence + components from Hydra config
        self.store: StateStore = instantiate(cfg.store)
        self.oms = OrderManagementSystem(
            self.store,
            initial_cash=float(cfg.oms.get("initial_cash", 100_000.0)),
            risk_fraction=float(cfg.oms.get("risk_fraction", 0.10)),
            account_mode=str(cfg.oms.get("account_mode", "spot")),
            short_leverage=float(cfg.oms.get("short_leverage", 1.0)),
            short_borrow_rate_bps_per_day=float(
                cfg.oms.get("short_borrow_rate_bps_per_day", 0.0)
            ),
            min_short_liquidation_buffer=float(
                cfg.oms.get("min_short_liquidation_buffer", 0.0)
            ),
        )
        self.ems: ExecutionGateway = instantiate(cfg.gateway)

        strategies_cfg = list(cfg.agents.strategies)
        risk_cfg = cfg.agents.risk
        supervisor_cfg = cfg.get("supervisor")
        strategy_cfg = cfg.get("strategy")
        if isinstance(strategy_cfg, DictConfig):
            if strategy_cfg.get("agents") is not None:
                bundle_agents = strategy_cfg.agents
                strategies_cfg = list(bundle_agents.get("strategies", strategies_cfg))
                risk_override = bundle_agents.get("risk")
                if risk_override is not None:
                    risk_cfg = risk_override
            elif strategy_cfg.get("_target_"):
                strategies_cfg = [strategy_cfg]

        strategies = [instantiate(agent_cfg) for agent_cfg in strategies_cfg]
        risk_agent = instantiate(risk_cfg)
        bundle_supervisor_cfg = (
            strategy_cfg.get("supervisor")
            if isinstance(strategy_cfg, DictConfig)
            else None
        )
        supervisor_min_confidence: float = float(
            bundle_supervisor_cfg.get(
                "min_confidence",
                supervisor_cfg.get("min_confidence", 0.6) if supervisor_cfg else 0.6,
            )
            if bundle_supervisor_cfg is not None
            else supervisor_cfg.get("min_confidence", 0.6) if supervisor_cfg else 0.6
        )
        supervisor_min_agents_required = int(
            bundle_supervisor_cfg.get(
                "min_agents_required",
                supervisor_cfg.get("min_agents_required", 1) if supervisor_cfg else 1,
            )
            if bundle_supervisor_cfg is not None
            else supervisor_cfg.get("min_agents_required", 1) if supervisor_cfg else 1
        )
        consensus_name = str(
            bundle_supervisor_cfg.get(
                "consensus_strategy",
                supervisor_cfg.get(
                    "consensus_strategy",
                    ConsensusStrategy.HIGHEST_CONFIDENCE.value,
                )
                if supervisor_cfg else ConsensusStrategy.HIGHEST_CONFIDENCE.value,
            )
            if bundle_supervisor_cfg is not None
            else supervisor_cfg.get(
                "consensus_strategy",
                ConsensusStrategy.HIGHEST_CONFIDENCE.value,
            )
            if supervisor_cfg else ConsensusStrategy.HIGHEST_CONFIDENCE.value
        )
        try:
            consensus_strategy = ConsensusStrategy(consensus_name)
        except ValueError:
            log.warning(
                "Invalid supervisor consensus strategy; using default",
                configured=consensus_name,
                default=ConsensusStrategy.HIGHEST_CONFIDENCE.value,
            )
            consensus_strategy = ConsensusStrategy.HIGHEST_CONFIDENCE

        self.supervisor = Supervisor(
            strategy_agents=strategies,
            risk_agent=risk_agent,
            consensus_strategy=consensus_strategy,
            min_confidence=supervisor_min_confidence,
            min_agents_required=supervisor_min_agents_required,
        )
        self.max_portfolio_exposure_forced_exit: float = max(
            0.0,
            float(
                cfg.loop.get(
                    "max_portfolio_exposure_forced_exit",
                    getattr(risk_agent, "max_position_size", 0.0),
                )
            ),
        )

        self._order_views: list[dict[str, Any]] = []
        self._entry_cooldown_until: dict[str, datetime] = {}
        self._entry_candle_key_by_symbol: dict[str, str] = {}
        self._entry_count_by_symbol: dict[str, int] = {}
        self.running = True

        # Daily PnL anchor for max_daily_loss enforcement in risk review.
        self._daily_anchor_date: date | None = None
        self._daily_anchor_value: float | None = None

        # Last known marks by instrument for multi-instrument exposure accounting.
        self._instrument_marks: dict[InstrumentId, float] = {}

        # Optional session drawdown breaker (0 disables).
        self.max_session_drawdown: float = max(
            0.0,
            float(cfg.loop.get("max_session_drawdown", 0.0)),
        )
        self._session_peak_value: float | None = None

        # Per-position stop-loss (0 disables).
        self.stop_loss_pct: float = max(
            0.0,
            min(1.0, float(cfg.loop.get("stop_loss_pct", 0.0))),
        )
        self._position_entry_prices: dict[InstrumentId, float] = {}

        # Drawdown Watchdog (AutoResearch 2026-03-29: risk > intelligence)
        watchdog_cfg = cfg.get("watchdog")
        self.watchdog = DrawdownWatchdog(
            warn_threshold=float(
                watchdog_cfg.get("warn_threshold", 0.05) if watchdog_cfg else 0.05
            ),
            halt_threshold=float(
                watchdog_cfg.get("halt_threshold", 0.08) if watchdog_cfg else 0.08
            ),
            weekly_halt_threshold=float(
                watchdog_cfg.get("weekly_halt_threshold", 0.15)
                if watchdog_cfg else 0.15
            ),
            reduce_factor=float(
                watchdog_cfg.get("reduce_factor", 0.50) if watchdog_cfg else 0.50
            ),
            cooldown_bars=int(
                watchdog_cfg.get("cooldown_bars", 30) if watchdog_cfg else 30
            ),
        )

        # Strategy-specific gates must keep independent state per symbol.
        sentiment_cfg = cfg.get("sentiment")
        quality_cfg = cfg.get("signal_quality")
        prediction_cfg = cfg.get("prediction_gate")
        regime_cfg = cfg.get("regime")

        self._sentiment_filter_kwargs: dict[str, float | int] = {
            "block_threshold": float(
                sentiment_cfg.get("block_threshold", -0.70)
                if sentiment_cfg else -0.70
            ),
            "boost_threshold": float(
                sentiment_cfg.get("boost_threshold", 0.70) if sentiment_cfg else 0.70
            ),
            "window_hours": float(
                sentiment_cfg.get("window_hours", 6.0) if sentiment_cfg else 6.0
            ),
            "min_samples": int(
                sentiment_cfg.get("min_samples", 3) if sentiment_cfg else 3
            ),
        }
        self._signal_quality_kwargs: dict[str, float | int] = {
            "window_signals": int(
                quality_cfg.get("window_signals", 10) if quality_cfg else 10
            ),
            "high_consistency": float(
                quality_cfg.get("high_consistency", 0.70) if quality_cfg else 0.70
            ),
            "medium_consistency": float(
                quality_cfg.get("medium_consistency", 0.50) if quality_cfg else 0.50
            ),
            "high_max_flip_rate": float(
                quality_cfg.get("high_max_flip_rate", 0.20) if quality_cfg else 0.20
            ),
            "medium_max_flip_rate": float(
                quality_cfg.get("medium_max_flip_rate", 0.40) if quality_cfg else 0.40
            ),
            "medium_size_scale": float(
                quality_cfg.get("medium_size_scale", 0.60) if quality_cfg else 0.60
            ),
            "min_samples": int(
                quality_cfg.get("min_samples", 5) if quality_cfg else 5
            ),
            "window_hours": float(
                quality_cfg.get("window_hours", 24.0) if quality_cfg else 24.0
            ),
        }
        self._prediction_gate_kwargs: dict[str, float | int] = {
            "block_threshold": float(
                prediction_cfg.get("block_threshold", 0.80)
                if prediction_cfg else 0.80
            ),
            "warn_threshold": float(
                prediction_cfg.get("warn_threshold", 0.65)
                if prediction_cfg else 0.65
            ),
            "warn_size_scale": float(
                prediction_cfg.get("warn_size_scale", 0.50)
                if prediction_cfg else 0.50
            ),
            "window_hours": float(
                prediction_cfg.get("window_hours", 12.0)
                if prediction_cfg else 12.0
            ),
            "min_signals": int(
                prediction_cfg.get("min_signals", 1) if prediction_cfg else 1
            ),
        }
        self._regime_detector_kwargs: dict[str, float | int] = {
            "lookback": int(regime_cfg.get("lookback", 100) if regime_cfg else 100),
            "atr_period": int(regime_cfg.get("atr_period", 14) if regime_cfg else 14),
            "low_percentile": float(
                regime_cfg.get("low_percentile", 0.25) if regime_cfg else 0.25
            ),
            "high_percentile": float(
                regime_cfg.get("high_percentile", 0.75) if regime_cfg else 0.75
            ),
            "crisis_percentile": float(
                regime_cfg.get("crisis_percentile", 0.95) if regime_cfg else 0.95
            ),
            "high_vol_size_scale": float(
                regime_cfg.get("high_vol_size_scale", 0.60) if regime_cfg else 0.60
            ),
        }

        self._sentiment_filters = {
            symbol_name: self._new_sentiment_filter()
            for symbol_name in self.symbols
        }
        self._signal_qualities = {
            symbol_name: self._new_signal_quality()
            for symbol_name in self.symbols
        }
        self._prediction_gates = {
            symbol_name: self._new_prediction_gate()
            for symbol_name in self.symbols
        }
        self._regime_detectors = {
            symbol_name: self._new_regime_detector()
            for symbol_name in self.symbols
        }

        # Preserve the original single-symbol attributes for existing tests/callers.
        self.sentiment_filter = self._sentiment_filters[self.symbol]
        self.signal_quality = self._signal_qualities[self.symbol]
        self.prediction_gate = self._prediction_gates[self.symbol]
        self.regime_detector = self._regime_detectors[self.symbol]

        self._sentiment_loader: SentimentLoader | None = None
        self._prediction_loader: PolymarketLoader | None = None
        self._sentiment_refresh_interval_seconds = float(
            sentiment_cfg.get("refresh_interval_seconds", 900.0)
            if sentiment_cfg else 900.0
        )
        self._prediction_refresh_interval_seconds = float(
            prediction_cfg.get("refresh_interval_seconds", 900.0)
            if prediction_cfg else 900.0
        )
        self._sentiment_fetch_timeout_seconds = float(
            sentiment_cfg.get("fetch_timeout_seconds", 15.0)
            if sentiment_cfg else 15.0
        )
        self._prediction_fetch_timeout_seconds = float(
            prediction_cfg.get("fetch_timeout_seconds", 10.0)
            if prediction_cfg else 10.0
        )
        self._prediction_market_limit = int(
            prediction_cfg.get("market_limit", 5) if prediction_cfg else 5
        )
        self._last_sentiment_refresh_at: dict[str, datetime] = {}
        self._last_prediction_refresh_at: dict[str, datetime] = {}

        if self._sentiment_refresh_interval_seconds > 0:
            self._sentiment_loader = SentimentLoader()
        if self._prediction_refresh_interval_seconds > 0:
            self._prediction_loader = PolymarketLoader()

    def _new_sentiment_filter(self) -> SentimentFilter:
        return SentimentFilter(
            block_threshold=float(self._sentiment_filter_kwargs["block_threshold"]),
            boost_threshold=float(self._sentiment_filter_kwargs["boost_threshold"]),
            window_hours=float(self._sentiment_filter_kwargs["window_hours"]),
            min_samples=int(self._sentiment_filter_kwargs["min_samples"]),
        )

    def _new_signal_quality(self) -> SignalQualityEvaluator:
        return SignalQualityEvaluator(
            window_signals=int(self._signal_quality_kwargs["window_signals"]),
            high_consistency=float(
                self._signal_quality_kwargs["high_consistency"]
            ),
            medium_consistency=float(
                self._signal_quality_kwargs["medium_consistency"]
            ),
            high_max_flip_rate=float(
                self._signal_quality_kwargs["high_max_flip_rate"]
            ),
            medium_max_flip_rate=float(
                self._signal_quality_kwargs["medium_max_flip_rate"]
            ),
            medium_size_scale=float(
                self._signal_quality_kwargs["medium_size_scale"]
            ),
            min_samples=int(self._signal_quality_kwargs["min_samples"]),
            window_hours=float(self._signal_quality_kwargs["window_hours"]),
        )

    def _new_prediction_gate(self) -> PredictionMarketGate:
        return PredictionMarketGate(
            block_threshold=float(self._prediction_gate_kwargs["block_threshold"]),
            warn_threshold=float(self._prediction_gate_kwargs["warn_threshold"]),
            warn_size_scale=float(self._prediction_gate_kwargs["warn_size_scale"]),
            window_hours=float(self._prediction_gate_kwargs["window_hours"]),
            min_signals=int(self._prediction_gate_kwargs["min_signals"]),
        )

    def _new_regime_detector(self) -> RegimeDetector:
        return RegimeDetector(
            lookback=int(self._regime_detector_kwargs["lookback"]),
            atr_period=int(self._regime_detector_kwargs["atr_period"]),
            low_percentile=float(self._regime_detector_kwargs["low_percentile"]),
            high_percentile=float(self._regime_detector_kwargs["high_percentile"]),
            crisis_percentile=float(
                self._regime_detector_kwargs["crisis_percentile"]
            ),
            high_vol_size_scale=float(
                self._regime_detector_kwargs["high_vol_size_scale"]
            ),
        )

    def _get_sentiment_filter(self, symbol: str) -> SentimentFilter:
        return self._sentiment_filters[symbol]

    def _get_signal_quality(self, symbol: str) -> SignalQualityEvaluator:
        return self._signal_qualities[symbol]

    def _get_prediction_gate(self, symbol: str) -> PredictionMarketGate:
        return self._prediction_gates[symbol]

    def _get_regime_detector(self, symbol: str) -> RegimeDetector:
        return self._regime_detectors[symbol]

    @staticmethod
    def _refresh_due(
        last_refresh: datetime | None,
        now: datetime,
        interval_seconds: float,
    ) -> bool:
        if interval_seconds <= 0:
            return False
        if last_refresh is None:
            return True
        return (now - last_refresh).total_seconds() >= interval_seconds

    async def _refresh_sentiment_context(
        self,
        symbol: str,
        now: datetime,
    ) -> None:
        if self._sentiment_loader is None:
            return

        last_refresh = self._last_sentiment_refresh_at.get(symbol)
        if not self._refresh_due(
            last_refresh,
            now,
            self._sentiment_refresh_interval_seconds,
        ):
            return

        query = _symbol_to_sentiment_query(symbol)
        try:
            snapshot = await asyncio.wait_for(
                self._sentiment_loader.get_sentiment(query),
                timeout=self._sentiment_fetch_timeout_seconds,
            )
        except Exception as exc:
            log.warning(
                "Sentiment refresh failed",
                symbol=symbol,
                query=query,
                error=str(exc),
            )
            self._last_sentiment_refresh_at[symbol] = now
            return

        if snapshot.volume > 0:
            self._get_sentiment_filter(symbol).add_score(
                snapshot.score,
                timestamp=snapshot.timestamp,
                source=snapshot.source.value,
            )

        self._last_sentiment_refresh_at[symbol] = now

    async def _refresh_prediction_context(
        self,
        symbol: str,
        now: datetime,
    ) -> None:
        if self._prediction_loader is None:
            return

        last_refresh = self._last_prediction_refresh_at.get(symbol)
        if not self._refresh_due(
            last_refresh,
            now,
            self._prediction_refresh_interval_seconds,
        ):
            return

        query = _symbol_to_sentiment_query(symbol)
        try:
            markets = await asyncio.wait_for(
                self._prediction_loader.search_markets(
                    query,
                    limit=self._prediction_market_limit,
                ),
                timeout=self._prediction_fetch_timeout_seconds,
            )
        except Exception as exc:
            log.warning(
                "Prediction market refresh failed",
                symbol=symbol,
                query=query,
                error=str(exc),
            )
            self._last_prediction_refresh_at[symbol] = now
            return

        gate = self._get_prediction_gate(symbol)
        gate.reset()
        for market in markets:
            implied_direction = _infer_prediction_direction(market.question, query)
            if implied_direction is None:
                continue
            gate.add_signal(
                event_id=market.condition_id,
                direction=implied_direction,
                probability=market.implied_probability,
                source="polymarket",
                timestamp=now,
            )

        self._last_prediction_refresh_at[symbol] = now

    async def _refresh_external_risk_context(
        self,
        symbol: str,
        now: datetime,
    ) -> None:
        await asyncio.gather(
            self._refresh_sentiment_context(symbol, now),
            self._refresh_prediction_context(symbol, now),
        )

    def _validate_data_freshness(self, data_ts: datetime) -> bool:
        """Validate that market data is not stale.

        Args:
            data_ts: Timestamp of the received market data.

        Returns:
            True if data is fresh, False if stale.
        """
        now = datetime.now(timezone.utc)
        if data_ts.tzinfo is None:
            data_ts = data_ts.replace(tzinfo=timezone.utc)

        age_seconds = (now - data_ts).total_seconds()

        if age_seconds > self._max_data_age_seconds:
            self._consecutive_stale_ticks += 1
            log.warning(
                "Stale market data detected",
                symbol=self.symbol,
                age_seconds=age_seconds,
                max_age_seconds=self._max_data_age_seconds,
                consecutive_stale=self._consecutive_stale_ticks,
            )
            return False

        # Data is fresh — reset counter
        self._consecutive_stale_ticks = 0
        self._last_fresh_data_at = now
        return True

    async def _fetch_live_data(self) -> MarketData | None:
        """Fetch live market data with freshness validation.

        Returns MarketData if fresh data is available, None if all sources
        failed or data is stale. Never generates synthetic/random data.

        Notes:
        - Domain models remain DataFrame-free (no pandas/polars inside MarketData).
        - OHLCV is published separately via telemetry.
        """
        exchange = getattr(self.ems, "exchange", None)
        if exchange is None:
            log.error(
                "No exchange gateway configured — cannot fetch live data",
                symbol=self.symbol,
            )
            return None

        timeframe = "1h"
        if "trading" in self.cfg and self.cfg.trading.get("timeframe"):
            timeframe = str(self.cfg.trading.timeframe)

        # --- Source 1: OHLCV candles (preferred) ---
        try:
            raw_ohlcv = await exchange.fetch_ohlcv(
                self.symbol,
                timeframe=timeframe,
                limit=50,
            )
        except Exception as exc:
            log.warning(
                "fetch_ohlcv failed",
                error=str(exc),
                error_type=type(exc).__name__,
                symbol=self.symbol,
            )
            raw_ohlcv = []

        ohlcv_history: list[OHLCVTuple] = []
        ohlcv_payload: list[dict[str, Any]] = []

        for row in raw_ohlcv:
            ts_ms, open_, high, low, close, volume = row
            ts = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
            ohlcv_history.append(
                (
                    ts,
                    float(open_),
                    float(high),
                    float(low),
                    float(close),
                    float(volume),
                )
            )
            ohlcv_payload.append(
                {
                    "timestamp": ts.isoformat(),
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(volume),
                }
            )

        if ohlcv_history:
            async with self._state_lock:
                self._last_ohlcv_history = ohlcv_history
                self._last_ohlcv_payload = ohlcv_payload

            latest_ts, open_, high, low, close, volume = ohlcv_history[-1]

            if not self._validate_data_freshness(latest_ts):
                self._emit_alert(
                    level="WARNING",
                    event="STALE_OHLCV_DATA",
                    message=f"OHLCV data is stale for {self.symbol}",
                    details={
                        "symbol": self.symbol,
                        "data_age_seconds": (
                            datetime.now(timezone.utc) - latest_ts
                        ).total_seconds(),
                        "consecutive_stale": self._consecutive_stale_ticks,
                    },
                )
                # Stale data: skip this tick (don't trade on old prices)
                return None

            return MarketData(
                instrument_id=InstrumentId(self.symbol),
                timestamp=latest_ts,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )

        # --- Source 2: Ticker snapshot fallback ---
        price: float | None = None
        try:
            ticker = await exchange.fetch_ticker(self.symbol)
            last_price = ticker.get("last") or ticker.get("close")
            if last_price:
                price = float(last_price)
        except Exception as exc:
            log.warning(
                "fetch_ticker failed",
                error=str(exc),
                error_type=type(exc).__name__,
                symbol=self.symbol,
            )

        # --- Source 3: Cached mark (with age check) ---
        if price is None or price <= 0:
            if self._last_fresh_data_at is not None:
                cache_age = (
                    datetime.now(timezone.utc) - self._last_fresh_data_at
                ).total_seconds()
                if cache_age <= self._max_data_age_seconds:
                    cached_mark = self._instrument_marks.get(
                        InstrumentId(self.symbol)
                    )
                    if cached_mark and cached_mark > 0:
                        price = float(cached_mark)
                        log.info(
                            "Using cached mark (within freshness window)",
                            symbol=self.symbol,
                            cache_age_seconds=cache_age,
                        )
                else:
                    log.warning(
                        "Cached mark too old, skipping",
                        symbol=self.symbol,
                        cache_age_seconds=cache_age,
                    )

        if price is not None and price > 0:
            ts = datetime.now(timezone.utc)
            async with self._state_lock:
                self._last_ohlcv_history = [(ts, price, price, price, price, 0.0)]
                self._last_ohlcv_payload = [
                    {
                        "timestamp": ts.isoformat(),
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 0.0,
                    }
                ]
            return MarketData(
                instrument_id=InstrumentId(self.symbol),
                timestamp=ts,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0.0,
            )

        # --- All sources exhausted: return None instead of synthetic data ---
        self._consecutive_stale_ticks += 1
        log.error(
            "All data sources exhausted — no fresh price available",
            symbol=self.symbol,
            consecutive_stale=self._consecutive_stale_ticks,
        )
        self._emit_alert(
            level="ERROR",
            event="DATA_FEED_EXHAUSTED",
            message=f"No live price available for {self.symbol}",
            details={
                "symbol": self.symbol,
                "consecutive_stale": self._consecutive_stale_ticks,
            },
        )
        return None

    def _halt_requested(self) -> bool:
        halt_flag = self.store.get("SYSTEM:HALT")
        return bool(halt_flag and str(halt_flag).lower() == "true")

    @staticmethod
    def _signal_type_str(action: SignalType | str) -> str:
        if isinstance(action, SignalType):
            return action.value
        return str(action).upper()

    @classmethod
    def _is_entry_action(cls, action: SignalType | str) -> bool:
        return cls._signal_type_str(action) in {SignalType.LONG.value, SignalType.SHORT.value}

    def _sync_entry_candle_window(self, symbol: str, candle_timestamp: datetime) -> None:
        candle_key = candle_timestamp.astimezone(timezone.utc).isoformat()
        if self._entry_candle_key_by_symbol.get(symbol) == candle_key:
            return

        self._entry_candle_key_by_symbol[symbol] = candle_key
        self._entry_count_by_symbol[symbol] = 0

    def _entry_is_throttled(
        self,
        *,
        symbol: str,
        action: SignalType | str,
        candle_timestamp: datetime,
    ) -> bool:
        if not self._is_entry_action(action):
            return False

        action_label = self._signal_type_str(action)

        now = datetime.now(timezone.utc)
        cooldown_until = self._entry_cooldown_until.get(symbol)
        if cooldown_until and now < cooldown_until:
            remaining_seconds = (cooldown_until - now).total_seconds()
            log.info(
                "Entry throttled by cooldown",
                instrument=symbol,
                action=action_label,
                cooldown_seconds_remaining=max(0.0, remaining_seconds),
            )
            return True

        self._sync_entry_candle_window(symbol, candle_timestamp)
        entries_in_candle = self._entry_count_by_symbol.get(symbol, 0)
        if entries_in_candle >= self.max_entries_per_candle_per_symbol:
            log.info(
                "Entry throttled by per-candle limit",
                instrument=symbol,
                action=action_label,
                entries_in_candle=entries_in_candle,
                per_candle_limit=self.max_entries_per_candle_per_symbol,
            )
            return True

        return False

    def _mark_entry_executed(
        self,
        *,
        symbol: str,
        action: SignalType | str,
        candle_timestamp: datetime,
    ) -> None:
        if not self._is_entry_action(action):
            return

        self._sync_entry_candle_window(symbol, candle_timestamp)
        self._entry_count_by_symbol[symbol] = self._entry_count_by_symbol.get(symbol, 0) + 1

        if self.entry_cooldown_seconds <= 0:
            return

        self._entry_cooldown_until[symbol] = datetime.now(timezone.utc) + timedelta(
            seconds=self.entry_cooldown_seconds
        )

    @staticmethod
    def _resolve_fill_trade_id(fill_report: FillReport) -> str | None:
        if fill_report.exchange_trade_id:
            return fill_report.exchange_trade_id
        if fill_report.exchange_order_id:
            return (
                f"{fill_report.exchange_order_id}:"
                f"{fill_report.timestamp.timestamp():.6f}:"
                f"{fill_report.quantity:.12f}"
            )
        return None

    def _record_order_view(
        self,
        order_request: OrderRequest,
        fill_report: FillReport,
    ) -> None:
        self._order_views.append(
            {
                "order_id": order_request.oms_order_id,
                "instrument_id": str(order_request.instrument_id),
                "side": order_request.side.value,
                "intent": order_request.intent.value,
                "qty": order_request.quantity,
                "fill_qty": fill_report.quantity,
                "price": fill_report.price,
                "fee": fill_report.fee,
                "exchange_order_id": fill_report.exchange_order_id,
                "exchange_trade_id": self._resolve_fill_trade_id(fill_report),
                "status": fill_report.status.value,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

    @staticmethod
    def _symbol_key_suffix(symbol: str) -> str:
        return symbol.upper().replace("/", "_").replace(":", "_").replace("-", "_")

    @classmethod
    def _symbol_scoped_key(cls, base_key: str, symbol: str) -> str:
        return f"{base_key}:{cls._symbol_key_suffix(symbol)}"

    def _emit_alert(
        self,
        *,
        level: str,
        event: str,
        message: str,
        details: dict[str, str | float | int | bool] | None = None,
    ) -> None:
        alerts_cfg = self.cfg.get("alerts")
        if alerts_cfg and not bool(alerts_cfg.get("enabled", True)):
            return

        payload: dict[str, str | float | int | bool] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            "message": message,
        }
        if details:
            payload.update(details)

        try:
            if alerts_cfg:
                last_key = alerts_cfg.get("last_key", "ALERTS:LAST")
                event_prefix = alerts_cfg.get("event_prefix", "ALERTS:EVENT")
            else:
                last_key = "ALERTS:LAST"
                event_prefix = "ALERTS:EVENT"

            event_key = f"{event_prefix}:{int(time.time() * 1000)}"
            serialized = json.dumps(payload)
            self.store.set(last_key, serialized)
            self.store.set(event_key, serialized)
        except Exception as exc:
            log.warning("Alert emission failed", event=event, error=str(exc))

    def _publish_latency_metrics(
        self,
        *,
        tick_to_decision_ms: float | None = None,
        decision_to_fill_ms: float | None = None,
        tick_to_fill_ms: float | None = None,
        symbol: str | None = None,
    ) -> None:
        telemetry_cfg = self.cfg.get("telemetry")
        if not telemetry_cfg:
            return

        latency_cfg = telemetry_cfg.get("latency_keys")
        if not latency_cfg:
            return

        try:
            tick_to_decision_key = latency_cfg.get(
                "tick_to_decision_ms",
                "METRICS:LATENCY:TICK_TO_DECISION_MS",
            )
            decision_to_fill_key = latency_cfg.get(
                "decision_to_fill_ms",
                "METRICS:LATENCY:DECISION_TO_FILL_MS",
            )
            tick_to_fill_key = latency_cfg.get(
                "tick_to_fill_ms",
                "METRICS:LATENCY:TICK_TO_FILL_MS",
            )

            if tick_to_decision_ms is not None:
                serialized = str(tick_to_decision_ms)
                self.store.set(tick_to_decision_key, serialized)
                if symbol:
                    self.store.set(
                        self._symbol_scoped_key(tick_to_decision_key, symbol),
                        serialized,
                    )

            if decision_to_fill_ms is not None:
                serialized = str(decision_to_fill_ms)
                self.store.set(decision_to_fill_key, serialized)
                if symbol:
                    self.store.set(
                        self._symbol_scoped_key(decision_to_fill_key, symbol),
                        serialized,
                    )

            if tick_to_fill_ms is not None:
                serialized = str(tick_to_fill_ms)
                self.store.set(tick_to_fill_key, serialized)
                if symbol:
                    self.store.set(
                        self._symbol_scoped_key(tick_to_fill_key, symbol),
                        serialized,
                    )
        except Exception as exc:
            log.warning("Latency metrics publish failed", error=str(exc))

    def _mark_for_instrument(
        self,
        instrument_id: InstrumentId,
        default_mark: float,
    ) -> float:
        mark = self._instrument_marks.get(instrument_id)
        if mark is None or mark <= 0:
            return default_mark
        return mark

    def _effective_position_qty(self, instrument_id: InstrumentId) -> float:
        # Include blocked long inventory so pending CLOSE_LONG doesn't hide exposure.
        held = self.oms.portfolio.positions.get(instrument_id, 0.0)
        blocked = self.oms.portfolio.blocked_positions.get(instrument_id, 0.0)
        return held + blocked

    def _compute_total_value(
        self,
        default_mark: float,
    ) -> float:
        positions_value = 0.0
        instruments = set(self.oms.portfolio.positions) | set(self.oms.portfolio.blocked_positions)
        for instrument_id in instruments:
            qty = self._effective_position_qty(instrument_id)
            mark = self._mark_for_instrument(instrument_id, default_mark)
            positions_value += qty * mark

        return (
            self.oms.portfolio.cash
            + self.oms.portfolio.blocked_cash
            + positions_value
        )

    def _compute_portfolio_exposure_fraction(
        self,
        default_mark: float,
        total_value: float,
    ) -> float:
        if total_value <= 0:
            return 0.0

        gross_notional = 0.0
        instruments = set(self.oms.portfolio.positions) | set(self.oms.portfolio.blocked_positions)
        for instrument_id in instruments:
            qty = self._effective_position_qty(instrument_id)
            if abs(qty) <= 1e-8:
                continue
            mark = self._mark_for_instrument(instrument_id, default_mark)
            gross_notional += abs(qty * mark)

        return gross_notional / total_value

    def _compute_short_exposure_fraction(
        self,
        default_mark: float,
        total_value: float,
    ) -> float:
        if total_value <= 0:
            return 0.0

        short_notional = 0.0
        instruments = set(self.oms.portfolio.positions) | set(self.oms.portfolio.blocked_positions)
        for instrument_id in instruments:
            qty = self._effective_position_qty(instrument_id)
            if qty >= -1e-8:
                continue
            mark = self._mark_for_instrument(instrument_id, default_mark)
            short_notional += abs(qty * mark)

        return short_notional / total_value

    def _compute_daily_pnl_fraction(
        self,
        total_value: float,
        asof: datetime,
    ) -> float:
        asof_date = asof.astimezone(timezone.utc).date()

        if (
            self._daily_anchor_date != asof_date
            or self._daily_anchor_value is None
            or self._daily_anchor_value <= 0
        ):
            self._daily_anchor_date = asof_date
            self._daily_anchor_value = max(total_value, 0.0)
            return 0.0

        return (total_value - self._daily_anchor_value) / self._daily_anchor_value

    def _compute_session_drawdown_fraction(self, total_value: float) -> float:
        if total_value <= 0:
            return 0.0

        if self._session_peak_value is None or total_value > self._session_peak_value:
            self._session_peak_value = total_value
            return 0.0

        if self._session_peak_value <= 0:
            return 0.0

        drawdown = (self._session_peak_value - total_value) / self._session_peak_value
        return max(0.0, drawdown)

    def _drawdown_limit_breached(self, total_value: float) -> bool:
        if self.max_session_drawdown <= 0:
            return False
        return self._compute_session_drawdown_fraction(total_value) >= self.max_session_drawdown

    async def _trigger_emergency_halt(
        self,
        *,
        reason: str,
        total_value: float | None = None,
    ) -> None:
        try:
            self.store.set("SYSTEM:HALT", "true")
        except Exception as exc:
            log.warning("Failed to persist SYSTEM:HALT flag", error=str(exc))

        log.critical(
            "Emergency halt triggered",
            reason=reason,
            total_value=total_value,
            max_session_drawdown=self.max_session_drawdown,
        )
        self._emit_alert(
            level="CRITICAL",
            event="SYSTEM_HALT",
            message="Emergency halt triggered",
            details={
                "reason": reason,
                "total_value": total_value if total_value is not None else -1.0,
                "max_session_drawdown": self.max_session_drawdown,
            },
        )

        await self._liquidate_open_positions()
        await self.ems.stop()
        self.running = False

    async def _reconcile_after_ambiguous_submission(
        self,
        *,
        reason: str,
        order_id: str,
    ) -> None:
        try:
            exchange = getattr(self.ems, "exchange", None)
            if exchange is not None:
                await self.oms.reconcile_with_exchange(exchange)
            else:
                self.oms.reconcile()

            log.warning(
                "OMS reconciliation executed after ambiguous submission",
                reason=reason,
                order_id=order_id,
            )
            self._emit_alert(
                level="WARNING",
                event="OMS_RECONCILE_AMBIGUOUS",
                message="Reconciliation run after ambiguous submission",
                details={"reason": reason, "order_id": order_id},
            )
        except Exception as exc:
            log.error(
                "OMS reconciliation failed after ambiguous submission",
                reason=reason,
                order_id=order_id,
                error=str(exc),
                exc_info=True,
            )
            self._emit_alert(
                level="ERROR",
                event="OMS_RECONCILE_FAILED",
                message="Reconciliation failed after ambiguous submission",
                details={"reason": reason, "order_id": order_id, "error": str(exc)},
            )

    async def _resolve_liquidation_price(self, instrument_id: InstrumentId) -> float | None:
        cached_mark = self._instrument_marks.get(instrument_id)
        if cached_mark is not None and cached_mark > 0:
            return cached_mark

        if (
            self._last_market_data is not None
            and self._last_market_data.instrument_id == instrument_id
        ):
            return self._last_market_data.close

        exchange = getattr(self.ems, "exchange", None)
        if exchange is None:
            return None

        try:
            ticker = await exchange.fetch_ticker(str(instrument_id))
        except Exception as exc:
            log.error(
                "Failed to fetch ticker during halt liquidation",
                instrument=str(instrument_id),
                error=str(exc),
            )
            return None

        last_price = ticker.get("last") or ticker.get("close")
        if not last_price:
            return None

        return float(last_price)

    async def _liquidate_open_positions(self) -> None:
        open_positions = {
            instrument: qty
            for instrument, qty in self.oms.portfolio.positions.items()
            if abs(qty) > 1e-8
        }

        if not open_positions:
            return

        log.critical(
            "Emergency halt liquidation started",
            open_positions={str(k): v for k, v in open_positions.items()},
        )

        for instrument_id in list(open_positions):
            liquidation_price = await self._resolve_liquidation_price(instrument_id)
            if liquidation_price is None or liquidation_price <= 0:
                log.error(
                    "Skipping halt liquidation due to missing price",
                    instrument=str(instrument_id),
                )
                continue

            decision = TradingDecision(
                instrument_id=instrument_id,
                action=SignalType.EXIT,
                quantity_modifier=1.0,
                rationale="Emergency halt liquidation",
                contributing_agents=["SYSTEM_HALT"],
            )
            order_request = self.oms.process_decision(decision, current_price=liquidation_price)
            if order_request is None:
                continue

            try:
                fill_report = await asyncio.wait_for(
                    self.ems.submit_order(order_request),
                    timeout=self.execution_timeout,
                )
            except asyncio.TimeoutError:
                log.error(
                    "Halt liquidation timed out; preserving reservation for reconciliation",
                    order_id=order_request.oms_order_id,
                    timeout_seconds=self.execution_timeout,
                )
                await self._reconcile_after_ambiguous_submission(
                    reason="halt_liquidation_timeout",
                    order_id=order_request.oms_order_id,
                )
                continue
            except ExecutionError as exc:
                if exc.recoverable:
                    log.warning(
                        "Halt liquidation had ambiguous submission error; preserving reservation",
                        order_id=order_request.oms_order_id,
                        error=str(exc),
                    )
                    await self._reconcile_after_ambiguous_submission(
                        reason="halt_liquidation_ambiguous_error",
                        order_id=order_request.oms_order_id,
                    )
                else:
                    log.error(
                        "Halt liquidation hard rejected; reverting allocation",
                        order_id=order_request.oms_order_id,
                        error=str(exc),
                    )
                    self.oms.revert_allocation(order_request.oms_order_id)
                continue
            except Exception as exc:
                log.error(
                    "Halt liquidation submission failed; preserving reservation for reconciliation",
                    order_id=order_request.oms_order_id,
                    error=str(exc),
                )
                await self._reconcile_after_ambiguous_submission(
                    reason="halt_liquidation_submit_exception",
                    order_id=order_request.oms_order_id,
                )
                continue

            if fill_report is None:
                log.warning(
                    "Halt liquidation returned no fill report; preserving reservation for reconciliation",
                    order_id=order_request.oms_order_id,
                )
                await self._reconcile_after_ambiguous_submission(
                    reason="halt_liquidation_no_fill_report",
                    order_id=order_request.oms_order_id,
                )
                continue

            if fill_report.status in {
                ExecutionStatus.REJECTED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CIRCUIT_OPEN,
            }:
                log.error(
                    "Halt liquidation hard rejected by gateway; reverting allocation",
                    order_id=order_request.oms_order_id,
                    status=fill_report.status.value,
                )
                self.oms.revert_allocation(order_request.oms_order_id)
                continue

            if fill_report.status == ExecutionStatus.TIMEOUT:
                await self._reconcile_after_ambiguous_submission(
                    reason="halt_liquidation_timeout_status",
                    order_id=order_request.oms_order_id,
                )
                continue

            try:
                self.oms.confirm_execution(
                    oms_order_id=fill_report.oms_order_id,
                    fill_price=fill_report.price,
                    fill_qty=fill_report.quantity,
                    fee=fill_report.fee,
                    exchange_trade_id=self._resolve_fill_trade_id(fill_report),
                    exchange_order_id=fill_report.exchange_order_id,
                )
            except Exception as exc:
                log.error(
                    "Halt liquidation confirmation failed; preserving reservation for reconciliation",
                    order_id=order_request.oms_order_id,
                    error=str(exc),
                )
                await self._reconcile_after_ambiguous_submission(
                    reason="halt_liquidation_confirm_failed",
                    order_id=order_request.oms_order_id,
                )

    def _publish_heartbeat(self) -> None:
        try:
            self.store.set(self.heartbeat_key, datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            log.warning("Heartbeat publish failed", error=str(exc))

    def _publish_telemetry(
        self,
        market_data: MarketData | None = None,
        total_value_override: float | None = None,
    ) -> None:
        telemetry_cfg = self.cfg.get("telemetry")
        if not telemetry_cfg:
            return

        if market_data is None:
            market_data = self._last_market_data
            if market_data is None:
                return

        try:
            # Remember last good tick so we can publish even if the next fetch fails.
            self._last_market_data = market_data
            total_value = (
                total_value_override
                if total_value_override is not None
                else self._compute_total_value(market_data.close)
            )
            daily_pnl_fraction = self._compute_daily_pnl_fraction(
                total_value,
                market_data.timestamp,
            )

            metrics_keys = telemetry_cfg.metrics_keys
            self.store.set(metrics_keys.total_value, str(total_value))
            self.store.set(metrics_keys.cash, str(self.oms.portfolio.cash))
            self.store.set(metrics_keys.pnl_daily, str(daily_pnl_fraction))

            last_price_key = telemetry_cfg.get("last_price_key", "MARKET:LAST_PRICE")
            last_tick_key = telemetry_cfg.get("last_tick_key", "MARKET:LAST_TICK")
            ohlcv_key = telemetry_cfg.get("ohlcv_key", "MARKET:OHLCV")
            positions_key = telemetry_cfg.get("positions_key", "VIEW:POSITIONS")
            orders_key = telemetry_cfg.get("orders_key", "VIEW:ORDERS")

            instrument_symbol = str(market_data.instrument_id)
            self.store.set("MARKET:ACTIVE_SYMBOLS", json.dumps(self.symbols))

            # Always publish canonical keys for backwards compatibility.
            self.store.set(last_price_key, str(market_data.close))
            last_tick_payload = json.dumps(
                {
                    "symbol": instrument_symbol,
                    "price": market_data.close,
                    "timestamp": market_data.timestamp.isoformat(),
                }
            )
            self.store.set(last_tick_key, last_tick_payload)

            # Also publish per-symbol keys for multi-asset dashboards.
            scoped_last_price_key = self._symbol_scoped_key(last_price_key, instrument_symbol)
            scoped_last_tick_key = self._symbol_scoped_key(last_tick_key, instrument_symbol)
            self.store.set(scoped_last_price_key, str(market_data.close))
            self.store.set(scoped_last_tick_key, last_tick_payload)

            if self._last_ohlcv_payload:
                serialized_ohlcv = json.dumps(self._last_ohlcv_payload)
                self.store.set(ohlcv_key, serialized_ohlcv)
                scoped_ohlcv_key = self._symbol_scoped_key(ohlcv_key, instrument_symbol)
                self.store.set(scoped_ohlcv_key, serialized_ohlcv)

            positions_payload = [
                {"instrument_id": str(instr), "quantity": qty}
                for instr, qty in self.oms.portfolio.positions.items()
            ]
            self.store.set(positions_key, json.dumps(positions_payload))

            if telemetry_cfg.publish_views and self._order_views:
                self.store.set(orders_key, json.dumps(self._order_views[-50:]))
        except Exception as exc:
            log.error("Telemetry publish failed", error=str(exc), exc_info=True)

    async def _process_symbol_tick(self, symbol: str) -> None:  # noqa: PLR0915
        self.symbol = symbol

        try:
            market_data = await self._fetch_live_data()
        except Exception as exc:
            log.error("Data Feed Error", symbol=symbol, error=str(exc))
            return

        if market_data is None:
            # Check if consecutive stale ticks warrant an emergency halt
            if self._consecutive_stale_ticks >= STALE_DATA_CONSECUTIVE_LIMIT:
                await self._trigger_emergency_halt(
                    reason=(
                        f"Data feed dead: {self._consecutive_stale_ticks} consecutive "
                        f"stale ticks for {symbol}"
                    ),
                )
            return

        instrument_symbol = str(market_data.instrument_id)
        async with self._state_lock:
            self._instrument_marks[market_data.instrument_id] = market_data.close
        self._publish_telemetry(market_data)
        log.info(
            "Tick Received",
            symbol=market_data.instrument_id,
            price=market_data.close,
        )

        tick_started_at = time.monotonic()

        # Portfolio-level exposure for risk layer (multi-instrument aware,
        # includes blocked inventory from pending CLOSE_LONG orders).
        # Protected by _state_lock to prevent concurrent mutation.
        async with self._state_lock:
            total_value = self._compute_total_value(market_data.close)
            daily_pnl_fraction = self._compute_daily_pnl_fraction(
                total_value,
                market_data.timestamp,
            )
            portfolio_exposure = self._compute_portfolio_exposure_fraction(
                market_data.close,
                total_value,
            )
            short_exposure_fraction = self._compute_short_exposure_fraction(
                market_data.close,
                total_value,
            )

        await self._refresh_external_risk_context(
            instrument_symbol,
            market_data.timestamp,
        )
        sentiment_filter = self._get_sentiment_filter(instrument_symbol)
        signal_quality = self._get_signal_quality(instrument_symbol)
        prediction_gate = self._get_prediction_gate(instrument_symbol)
        regime_detector = self._get_regime_detector(instrument_symbol)

        session_drawdown = self._compute_session_drawdown_fraction(total_value)
        if (
            self.max_session_drawdown > 0
            and session_drawdown >= self.max_session_drawdown
        ):
            await self._trigger_emergency_halt(
                reason="session_drawdown_limit_breached",
                total_value=total_value,
            )
            return

        # ── Drawdown Watchdog gate ──────────────────────────────────
        watchdog_verdict = self.watchdog.evaluate(
            current_equity=total_value,
            timestamp=market_data.timestamp,
        )
        if watchdog_verdict.state == WatchdogState.HALTED:
            log.warning(
                "Watchdog HALTED — skipping tick",
                reason=watchdog_verdict.reason,
                drawdown=f"{watchdog_verdict.drawdown_pct:.2%}",
                equity=total_value,
                peak=watchdog_verdict.peak_equity,
            )
            self._publish_telemetry(
                market_data,
                total_value_override=total_value,
            )
            return

        # ── Regime Detection gate ──────────────────────────────────
        regime_verdict = regime_detector.update(
            high=market_data.high,
            low=market_data.low,
            close=market_data.close,
        )
        if regime_verdict.regime == MarketRegime.CRISIS:
            log.warning(
                "REGIME CRISIS — skipping tick",
                reason=regime_verdict.reason,
                atr=f"{regime_verdict.current_atr:.6f}",
                percentile=f"{regime_verdict.atr_percentile:.0%}",
            )
            self._publish_telemetry(
                market_data,
                total_value_override=total_value,
            )
            return

        # ── Sentiment Filter evaluation ───────────────────────────────
        sentiment_verdict = sentiment_filter.evaluate(
            now=market_data.timestamp,
        )

        # Per-position stop-loss check
        if self.stop_loss_pct > 0:
            for _sl_instr, _sl_qty in list(
                self.oms.portfolio.positions.items()
            ):
                if abs(_sl_qty) < 1e-8:
                    continue
                _sl_entry = self._position_entry_prices.get(_sl_instr)
                if not _sl_entry or _sl_entry <= 0:
                    continue
                _sl_mark = self._mark_for_instrument(
                    _sl_instr, market_data.close
                )
                if _sl_mark <= 0:
                    continue
                if _sl_qty > 0:
                    _sl_loss = (_sl_entry - _sl_mark) / _sl_entry
                else:
                    _sl_loss = (_sl_mark - _sl_entry) / _sl_entry
                if _sl_loss >= self.stop_loss_pct:
                    log.warning(
                        "Per-position stop-loss triggered",
                        instrument=_sl_instr,
                        entry=_sl_entry,
                        mark=_sl_mark,
                        loss_pct=f"{_sl_loss:.2%}",
                    )
                    _sl_decision = TradingDecision(
                        instrument_id=_sl_instr,
                        action=SignalType.EXIT,
                        quantity_modifier=1.0,
                        rationale=f"Stop-loss {self.stop_loss_pct:.1%}",
                        contributing_agents=["SYSTEM_STOP_LOSS"],
                    )
                    _sl_order = self.oms.process_decision(
                        _sl_decision, current_price=_sl_mark,
                    )
                    if _sl_order:
                        try:
                            await asyncio.wait_for(
                                self.ems.submit_order(_sl_order),
                                timeout=self.execution_timeout,
                            )
                        except Exception as _sl_exc:
                            log.error(
                                "Stop-loss order failed",
                                instrument=_sl_instr,
                                error=str(_sl_exc),
                            )

        decision: TradingDecision | None = None
        max_exposure_forced_exit = self.max_portfolio_exposure_forced_exit
        instrument_position = self._effective_position_qty(market_data.instrument_id)

        if (
            max_exposure_forced_exit > 0
            and portfolio_exposure > max_exposure_forced_exit
            and abs(instrument_position) > 1e-8
        ):
            decision = TradingDecision(
                instrument_id=market_data.instrument_id,
                action=SignalType.EXIT,
                quantity_modifier=1.0,
                rationale=(
                    "Forced de-risk exit: portfolio exposure "
                    f"{portfolio_exposure:.2%} exceeds {max_exposure_forced_exit:.2%}"
                ),
                contributing_agents=["SYSTEM_DE_RISK"],
            )
            log.warning(
                "Exposure-based de-risk EXIT triggered",
                instrument=instrument_symbol,
                exposure=portfolio_exposure,
                max_exposure=max_exposure_forced_exit,
                instrument_position=instrument_position,
            )
        else:
            try:
                decision = await self.supervisor.run(
                    market_data,
                    ohlcv_history=self._last_ohlcv_history,
                    portfolio_exposure=portfolio_exposure,
                    daily_pnl_fraction=daily_pnl_fraction,
                    short_exposure_fraction=short_exposure_fraction,
                )
            except Exception as exc:
                self._publish_latency_metrics(
                    tick_to_decision_ms=(time.monotonic() - tick_started_at) * 1000.0,
                    symbol=instrument_symbol,
                )
                log.error("Strategy Execution Failed", symbol=instrument_symbol, error=str(exc))
                self._emit_alert(
                    level="ERROR",
                    event="STRATEGY_EXECUTION_FAILED",
                    message="Supervisor execution failed",
                    details={"error": str(exc), "symbol": instrument_symbol},
                )
                return

        self._publish_latency_metrics(
            tick_to_decision_ms=(time.monotonic() - tick_started_at) * 1000.0,
            symbol=instrument_symbol,
        )

        if not decision:
            return

        # ── Sentiment Filter gate (block LONG entries) ────────────
        if (
            sentiment_verdict.block_longs
            and decision.action == SignalType.LONG
        ):
            log.warning(
                "SENTIMENT FILTER — blocking LONG entry",
                reason=sentiment_verdict.reason,
                avg_sentiment=f"{sentiment_verdict.avg_sentiment:.3f}",
                samples=sentiment_verdict.sample_count,
            )
            return

        # ── Signal Quality gate (whipsaw detection) ─────────────────
        signal_quality.record_signal(
            direction=decision.action,
            confidence=decision.quantity_modifier,
            timestamp=market_data.timestamp,
        )
        if decision.action != SignalType.EXIT:
            quality_verdict = signal_quality.evaluate(now=market_data.timestamp)
            if quality_verdict.block_entry:
                log.warning(
                    "SIGNAL QUALITY — blocking entry (whipsaw)",
                    reason=quality_verdict.reason,
                    consistency=f"{quality_verdict.consistency:.0%}",
                    flip_rate=f"{quality_verdict.flip_rate:.0%}",
                )
                return

        # ── Prediction Market gate ────────────────────────────────
        if decision.action != SignalType.EXIT:
            pred_verdict = prediction_gate.evaluate(
                trade_direction=decision.action,
                now=market_data.timestamp,
            )
            if pred_verdict.block_trade:
                log.warning(
                    "PREDICTION GATE — blocking trade",
                    reason=pred_verdict.reason,
                    contradiction=f"{pred_verdict.strongest_contradiction:.0%}",
                    signals=pred_verdict.signal_count,
                )
                return
        else:
            pred_verdict = None

        # ── Compound risk scaling ─────────────────────────────────
        if decision.action != SignalType.EXIT:
            quality_mult = (
                quality_verdict.size_multiplier
                if quality_verdict.quality == SignalQuality.MEDIUM
                else 1.0
            )
            pred_mult = (
                pred_verdict.size_multiplier
                if pred_verdict and pred_verdict.state == GateState.REDUCED
                else 1.0
            )
            combined_multiplier = (
                watchdog_verdict.size_multiplier
                * regime_verdict.size_multiplier
                * quality_mult
                * pred_mult
            )
            if combined_multiplier < 1.0:
                scaled_modifier = decision.quantity_modifier * combined_multiplier
                parts: list[str] = []
                if watchdog_verdict.size_multiplier < 1.0:
                    parts.append(
                        f"Watchdog {watchdog_verdict.state.value} "
                        f"{watchdog_verdict.size_multiplier:.0%}"
                    )
                if regime_verdict.size_multiplier < 1.0:
                    parts.append(
                        f"Regime {regime_verdict.regime.value} "
                        f"{regime_verdict.size_multiplier:.0%}"
                    )
                if quality_mult < 1.0:
                    parts.append(
                        f"SignalQuality {quality_verdict.quality.value} "
                        f"{quality_mult:.0%}"
                    )
                if pred_verdict is not None and pred_mult < 1.0:
                    parts.append(
                        f"PredictionGate {pred_verdict.state.value} "
                        f"{pred_mult:.0%}"
                    )
                decision = decision.model_copy(
                    update={
                        "quantity_modifier": scaled_modifier,
                        "rationale": (
                            f"{decision.rationale} | "
                            + "; ".join(parts)
                        ),
                    }
                )
                log.info(
                    "Risk scaling applied",
                    watchdog=watchdog_verdict.size_multiplier,
                    regime=regime_verdict.size_multiplier,
                    combined=combined_multiplier,
                )

        if decision.action == SignalType.EXIT:
            guarded_decision: TradingDecision | None = decision
        else:
            projected_quantity = self.oms.calculate_order_quantity(
                market_data.close,
                decision.quantity_modifier,
            )
            guarded_decision = apply_execution_guardrails(
                decision,
                market_data,
                enabled=self.guardrails_enabled,
                min_volume=self.guardrails_min_volume,
                max_intrabar_volatility=self.guardrails_max_intrabar_volatility,
                high_volatility_size_scale=self.guardrails_high_volatility_size_scale,
                max_estimated_slippage_bps=self.guardrails_max_estimated_slippage_bps,
                slippage_volatility_factor=self.guardrails_slippage_volatility_factor,
                max_estimated_spread_bps=self.guardrails_max_estimated_spread_bps,
                spread_volatility_factor=self.guardrails_spread_volatility_factor,
                max_participation_rate=self.guardrails_max_participation_rate,
                projected_quantity=projected_quantity,
                enable_dynamic_volatility_risk_scaling=(
                    self.guardrails_enable_dynamic_volatility_risk_scaling
                ),
                min_dynamic_risk_scale=self.guardrails_min_dynamic_risk_scale,
                high_volatility_hours_utc=self.guardrails_high_volatility_hours_utc,
                high_volatility_hours_size_scale=self.guardrails_high_volatility_hours_size_scale,
                high_volatility_hours_entry_block=self.guardrails_high_volatility_hours_entry_block,
            )
            if not guarded_decision:
                log.warning(
                    "Order rejected by execution guardrails",
                    instrument=market_data.instrument_id,
                    volume=market_data.volume,
                    intrabar_volatility=max(
                        0.0,
                        (market_data.high - market_data.low) / market_data.close,
                    ),
                )
                return

        assert guarded_decision is not None
        if self._entry_is_throttled(
            symbol=instrument_symbol,
            action=guarded_decision.action,
            candle_timestamp=market_data.timestamp,
        ):
            return

        order_request = self.oms.process_decision(
            guarded_decision,
            current_price=market_data.close,
        )
        if not order_request:
            return

        submit_started_at = time.monotonic()

        try:
            fill_report = await asyncio.wait_for(
                self.ems.submit_order(order_request),
                timeout=self.execution_timeout,
            )
        except asyncio.TimeoutError:
            self._publish_latency_metrics(
                decision_to_fill_ms=(time.monotonic() - submit_started_at) * 1000.0,
                tick_to_fill_ms=(time.monotonic() - tick_started_at) * 1000.0,
                symbol=instrument_symbol,
            )
            log.error(
                "Order submission timed out; preserving reservation for reconciliation",
                order_id=order_request.oms_order_id,
                timeout_seconds=self.execution_timeout,
            )
            self._emit_alert(
                level="WARNING",
                event="ORDER_SUBMISSION_TIMEOUT",
                message="Order submission timed out",
                details={
                    "order_id": order_request.oms_order_id,
                    "timeout_seconds": self.execution_timeout,
                    "symbol": instrument_symbol,
                },
            )
            await self._reconcile_after_ambiguous_submission(
                reason="submit_timeout",
                order_id=order_request.oms_order_id,
            )
            return
        except ExecutionError as exc:
            self._publish_latency_metrics(
                decision_to_fill_ms=(time.monotonic() - submit_started_at) * 1000.0,
                tick_to_fill_ms=(time.monotonic() - tick_started_at) * 1000.0,
                symbol=instrument_symbol,
            )
            if exc.recoverable:
                log.warning(
                    "Order submission returned ambiguous error; preserving reservation",
                    error=str(exc),
                    order_id=order_request.oms_order_id,
                )
                self._emit_alert(
                    level="WARNING",
                    event="ORDER_SUBMISSION_AMBIGUOUS",
                    message="Order submission failed with ambiguous outcome",
                    details={
                        "order_id": order_request.oms_order_id,
                        "error": str(exc),
                        "symbol": instrument_symbol,
                    },
                )
                await self._reconcile_after_ambiguous_submission(
                    reason="submit_ambiguous_error",
                    order_id=order_request.oms_order_id,
                )
            else:
                log.error(
                    "Order hard rejected; reverting allocation",
                    error=str(exc),
                    order_id=order_request.oms_order_id,
                )
                self._emit_alert(
                    level="ERROR",
                    event="ORDER_SUBMISSION_REJECTED",
                    message="Order submission was hard rejected",
                    details={
                        "order_id": order_request.oms_order_id,
                        "error": str(exc),
                        "symbol": instrument_symbol,
                    },
                )
                self.oms.revert_allocation(order_request.oms_order_id)
            return
        except Exception as exc:
            self._publish_latency_metrics(
                decision_to_fill_ms=(time.monotonic() - submit_started_at) * 1000.0,
                tick_to_fill_ms=(time.monotonic() - tick_started_at) * 1000.0,
                symbol=instrument_symbol,
            )
            log.error(
                "Order submission raised unexpected exception; preserving reservation",
                error=str(exc),
                order_id=order_request.oms_order_id,
            )
            self._emit_alert(
                level="WARNING",
                event="ORDER_SUBMISSION_EXCEPTION",
                message="Unexpected submission exception; reconciliation triggered",
                details={
                    "order_id": order_request.oms_order_id,
                    "error": str(exc),
                    "symbol": instrument_symbol,
                },
            )
            await self._reconcile_after_ambiguous_submission(
                reason="submit_exception",
                order_id=order_request.oms_order_id,
            )
            return

        if fill_report:
            self._publish_latency_metrics(
                decision_to_fill_ms=(time.monotonic() - submit_started_at) * 1000.0,
                tick_to_fill_ms=(time.monotonic() - tick_started_at) * 1000.0,
                symbol=instrument_symbol,
            )

            if fill_report.status in {
                ExecutionStatus.REJECTED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CIRCUIT_OPEN,
            }:
                log.error(
                    "Order hard rejected in fill report; reverting allocation",
                    order_id=order_request.oms_order_id,
                    status=fill_report.status.value,
                )
                self._emit_alert(
                    level="ERROR",
                    event="ORDER_REJECTED",
                    message="Gateway returned hard reject status",
                    details={
                        "order_id": order_request.oms_order_id,
                        "status": fill_report.status.value,
                        "symbol": instrument_symbol,
                    },
                )
                self.oms.revert_allocation(order_request.oms_order_id)
                return

            if fill_report.status == ExecutionStatus.TIMEOUT:
                self._emit_alert(
                    level="WARNING",
                    event="ORDER_TIMEOUT_STATUS",
                    message="Gateway returned timeout status; reconciliation triggered",
                    details={
                        "order_id": order_request.oms_order_id,
                        "symbol": instrument_symbol,
                    },
                )
                await self._reconcile_after_ambiguous_submission(
                    reason="fill_timeout_status",
                    order_id=order_request.oms_order_id,
                )
                return

            try:
                self.oms.confirm_execution(
                    oms_order_id=fill_report.oms_order_id,
                    fill_price=fill_report.price,
                    fill_qty=fill_report.quantity,
                    fee=fill_report.fee,
                    exchange_trade_id=self._resolve_fill_trade_id(fill_report),
                    exchange_order_id=fill_report.exchange_order_id,
                )
            except Exception as exc:
                log.error(
                    "Order confirmation failed; preserving reservation for reconciliation",
                    error=str(exc),
                    order_id=order_request.oms_order_id,
                )
                self._emit_alert(
                    level="ERROR",
                    event="ORDER_CONFIRMATION_FAILED",
                    message="Order confirmation failed",
                    details={
                        "order_id": order_request.oms_order_id,
                        "error": str(exc),
                        "symbol": instrument_symbol,
                    },
                )
                await self._reconcile_after_ambiguous_submission(
                    reason="confirm_failed",
                    order_id=order_request.oms_order_id,
                )
                return
            self._record_order_view(order_request, fill_report)

            # Track entry price for stop-loss
            if fill_report.price and fill_report.price > 0:
                _fill_instr = market_data.instrument_id
                _fill_qty = self._effective_position_qty(_fill_instr)
                if abs(_fill_qty) < 1e-8:
                    self._position_entry_prices.pop(_fill_instr, None)
                elif guarded_decision.action in {
                    SignalType.LONG,
                    SignalType.SHORT,
                }:
                    self._position_entry_prices[_fill_instr] = fill_report.price

            self._mark_entry_executed(
                symbol=instrument_symbol,
                action=guarded_decision.action,
                candle_timestamp=market_data.timestamp,
            )
        else:
            self._publish_latency_metrics(
                decision_to_fill_ms=(time.monotonic() - submit_started_at) * 1000.0,
                tick_to_fill_ms=(time.monotonic() - tick_started_at) * 1000.0,
                symbol=instrument_symbol,
            )
            log.warning(
                "Order submission returned no fill report; preserving reservation for reconciliation",
                order_id=order_request.oms_order_id,
            )
            self._emit_alert(
                level="WARNING",
                event="ORDER_NO_FILL_REPORT",
                message="Order submission returned no fill report; reconciliation triggered",
                details={"order_id": order_request.oms_order_id, "symbol": instrument_symbol},
            )
            await self._reconcile_after_ambiguous_submission(
                reason="submit_no_fill_report",
                order_id=order_request.oms_order_id,
            )
            return

        self._publish_telemetry(market_data)

    async def run(self) -> None:  # noqa: PLR0915
        """Run the live trading loop until stopped or halted."""
        log.info("Starting Live Trading Engine", env=self.cfg.get("env", "unknown"))

        try:
            await self.ems.start()

            try:
                exchange = getattr(self.ems, "exchange", None)
                if exchange is not None:
                    await self.oms.reconcile_with_exchange(exchange)
                else:
                    self.oms.reconcile()

                log.info("Startup OMS reconciliation completed")
            except Exception as exc:
                log.error(
                    "Startup OMS reconciliation failed",
                    error=str(exc),
                    exc_info=True,
                )
                self._emit_alert(
                    level="ERROR",
                    event="STARTUP_RECONCILE_FAILED",
                    message="Startup reconciliation failed",
                    details={"error": str(exc)},
                )

            # Initialize watchdog with current portfolio value
            initial_equity = self.oms.portfolio.cash + self.oms.portfolio.blocked_cash
            self.watchdog.reset(initial_equity)

            while self.running:
                cycle_started_at = time.monotonic()

                # Heartbeat even if no data/decisions
                self._publish_heartbeat()

                if self._halt_requested():
                    await self._trigger_emergency_halt(reason="external_halt_flag")
                    return

                for symbol in self.symbols:
                    if not self.running:
                        break

                    if self._halt_requested():
                        await self._trigger_emergency_halt(reason="external_halt_flag")
                        return

                    try:
                        await self._process_symbol_tick(symbol)
                    except Exception as exc:
                        log.error(
                            "Unhandled symbol tick failure",
                            symbol=symbol,
                            error=str(exc),
                            exc_info=True,
                        )
                        self._emit_alert(
                            level="ERROR",
                            event="SYMBOL_TICK_FAILED",
                            message="Unhandled exception while processing symbol tick",
                            details={"symbol": symbol, "error": str(exc)},
                        )

                if getattr(self.ems, "exchange", None) is not None:
                    elapsed = time.monotonic() - cycle_started_at
                    await asyncio.sleep(max(0.0, self.tick_interval - elapsed))

        except KeyboardInterrupt:
            log.info("Manual Shutdown Requested")
        except Exception:
            log.exception("Fatal Crash in Live Loop")
        finally:
            log.info("Shutting down services...")
            if self._sentiment_loader is not None:
                await self._sentiment_loader.close()
            await self.ems.stop()


@hydra.main(version_base=None, config_path="../../../conf", config_name="live")
def app(cfg: DictConfig) -> None:
    """Hydra entrypoint for the live trader with graceful shutdown."""
    configure_logging()
    log.info(
        "Hydra configuration loaded",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    validate_api_keys(cfg)
    trader = LiveTrader(cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_signal(sig: int, _frame: Any) -> None:
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        sig_name = signal.Signals(sig).name
        log.info("Received signal, initiating graceful shutdown", signal=sig_name)
        trader.running = False

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        loop.run_until_complete(trader.run())
    finally:
        loop.close()


if __name__ == "__main__":
    app()
