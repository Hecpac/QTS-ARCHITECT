from __future__ import annotations

import asyncio
import atexit
import json
import logging
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import structlog
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from qts_core.agents.protocol import SignalType, TradingDecision
from qts_core.agents.supervisor import Supervisor
from qts_core.common.types import InstrumentId, MarketData
from qts_core.execution.oms import OrderManagementSystem, OrderRequest


if TYPE_CHECKING:
    from qts_core.execution.ems import ExecutionGateway, FillReport
    from qts_core.execution.store import StateStore

OHLCVTuple = tuple[datetime, float, float, float, float, float]

log = structlog.get_logger()


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

    intrabar_volatility = max(0.0, (market_data.high - market_data.low) / market_data.close)
    estimated_slippage_bps = intrabar_volatility * slippage_volatility_factor * 10_000

    if max_estimated_slippage_bps > 0 and estimated_slippage_bps > max_estimated_slippage_bps:
        return None

    adjusted_modifier = decision.quantity_modifier
    rationale_suffix: list[str] = []

    if (
        max_intrabar_volatility > 0
        and intrabar_volatility > max_intrabar_volatility
        and high_volatility_size_scale < 1.0
    ):
        adjusted_modifier *= max(0.0, high_volatility_size_scale)
        rationale_suffix.append(
            f"Guardrail: high volatility {intrabar_volatility:.2%}"
        )

    adjusted_modifier = max(0.0, min(1.0, adjusted_modifier))
    if adjusted_modifier < 1e-8:
        return None

    if adjusted_modifier == decision.quantity_modifier and not rationale_suffix:
        return decision

    rationale = decision.rationale
    if rationale_suffix:
        rationale = f"{rationale} | {'; '.join(rationale_suffix)}"

    return decision.model_copy(
        update={
            "quantity_modifier": adjusted_modifier,
            "rationale": rationale,
        }
    )


class _TeeStream:
    """Mirror writes to both stdout and a local file handle."""

    def __init__(self, *streams: Any) -> None:
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
        logger_factory=structlog.PrintLoggerFactory(file=tee_stream),
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

        self._last_market_data: MarketData | None = None
        self._last_ohlcv_history: list[OHLCVTuple] | None = None
        self._last_ohlcv_payload: list[dict[str, Any]] | None = None

        # Persistence + components from Hydra config
        self.store: StateStore = instantiate(cfg.store)
        self.oms = OrderManagementSystem(
            self.store,
            initial_cash=float(cfg.oms.get("initial_cash", 100_000.0)),
            risk_fraction=float(cfg.oms.get("risk_fraction", 0.10)),
            account_mode=str(cfg.oms.get("account_mode", "spot")),
            short_leverage=float(cfg.oms.get("short_leverage", 1.0)),
        )
        self.ems: ExecutionGateway = instantiate(cfg.gateway)

        strategies = [instantiate(agent_cfg) for agent_cfg in cfg.agents.strategies]
        risk_agent = instantiate(cfg.agents.risk)
        self.supervisor = Supervisor(strategy_agents=strategies, risk_agent=risk_agent)
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

    async def _fetch_live_data(self) -> MarketData:
        """Fetch live market data.

        If a CCXT gateway is available, fetch a small OHLCV window and use the
        latest candle as the current MarketData event.

        Notes:
        - Domain models remain DataFrame-free (no pandas/polars inside MarketData).
        - OHLCV is published separately via telemetry.
        """
        exchange = getattr(self.ems, "exchange", None)
        if exchange:
            timeframe = "1h"
            if "trading" in self.cfg and self.cfg.trading.get("timeframe"):
                timeframe = str(self.cfg.trading.timeframe)

            try:
                raw_ohlcv = await exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe=timeframe,
                    limit=50,
                )
            except Exception as exc:
                log.warning("fetch_ohlcv failed", error=str(exc), symbol=self.symbol)
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
                self._last_ohlcv_history = ohlcv_history
                self._last_ohlcv_payload = ohlcv_payload

                latest_ts, open_, high, low, close, volume = ohlcv_history[-1]
                return MarketData(
                    instrument_id=InstrumentId(self.symbol),
                    timestamp=latest_ts,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )

            # Data-fetch fallback for live gateways: do NOT sleep or inject synthetic prices.
            # Prefer ticker snapshot, then cached mark, then last market data for this symbol.
            price: float | None = None

            try:
                ticker = await exchange.fetch_ticker(self.symbol)
                last_price = ticker.get("last") or ticker.get("close")
                if last_price:
                    price = float(last_price)
            except Exception as exc:
                log.warning("fetch_ticker failed", error=str(exc), symbol=self.symbol)

            if price is None or price <= 0:
                cached_mark = self._instrument_marks.get(InstrumentId(self.symbol))
                if cached_mark and cached_mark > 0:
                    price = float(cached_mark)

            if (
                (price is None or price <= 0)
                and self._last_market_data is not None
                and str(self._last_market_data.instrument_id) == self.symbol
            ):
                price = float(self._last_market_data.close)

            if price is not None and price > 0:
                ts = datetime.now(timezone.utc)
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

        # Fallback simulation (non-live/mock flows only).
        await asyncio.sleep(self.tick_interval)
        price = 100_000.0 + random.uniform(-100, 100)  # noqa: S311
        ts = datetime.now(timezone.utc)

        self._last_ohlcv_history = [(ts, price, price + 50, price - 50, price, 0.0)]
        self._last_ohlcv_payload = [
            {
                "timestamp": ts.isoformat(),
                "open": price,
                "high": price + 50,
                "low": price - 50,
                "close": price,
                "volume": 0.0,
            }
        ]

        return MarketData(
            instrument_id=InstrumentId(self.symbol),
            timestamp=ts,
            open=price,
            high=price + 50,
            low=price - 50,
            close=price,
            volume=random.uniform(0.1, 5.0),  # noqa: S311
        )

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
                "status": "FILLED",
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
            except Exception as exc:
                log.error(
                    "Halt liquidation submission failed; reverting allocation",
                    order_id=order_request.oms_order_id,
                    error=str(exc),
                )
                self.oms.revert_allocation(order_request.oms_order_id)
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

            try:
                self.oms.confirm_execution(
                    oms_order_id=fill_report.oms_order_id,
                    fill_price=fill_report.price,
                    fill_qty=fill_report.quantity,
                    fee=fill_report.fee,
                    exchange_trade_id=fill_report.exchange_order_id,
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

    def _publish_telemetry(self, market_data: MarketData | None = None) -> None:
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
            total_value = self._compute_total_value(market_data.close)
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

        instrument_symbol = str(market_data.instrument_id)
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
        total_value = self._compute_total_value(market_data.close)
        daily_pnl_fraction = self._compute_daily_pnl_fraction(
            total_value,
            market_data.timestamp,
        )
        portfolio_exposure = self._compute_portfolio_exposure_fraction(
            market_data.close,
            total_value,
        )

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

        if decision.action == SignalType.EXIT:
            guarded_decision = decision
        else:
            guarded_decision = apply_execution_guardrails(
                decision,
                market_data,
                enabled=self.guardrails_enabled,
                min_volume=self.guardrails_min_volume,
                max_intrabar_volatility=self.guardrails_max_intrabar_volatility,
                high_volatility_size_scale=self.guardrails_high_volatility_size_scale,
                max_estimated_slippage_bps=self.guardrails_max_estimated_slippage_bps,
                slippage_volatility_factor=self.guardrails_slippage_volatility_factor,
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
        except Exception as exc:
            self._publish_latency_metrics(
                decision_to_fill_ms=(time.monotonic() - submit_started_at) * 1000.0,
                tick_to_fill_ms=(time.monotonic() - tick_started_at) * 1000.0,
                symbol=instrument_symbol,
            )
            log.error(
                "Order Submission Failed",
                error=str(exc),
                order_id=order_request.oms_order_id,
            )
            self._emit_alert(
                level="ERROR",
                event="ORDER_SUBMISSION_FAILED",
                message="Order submission raised an exception",
                details={
                    "order_id": order_request.oms_order_id,
                    "error": str(exc),
                    "symbol": instrument_symbol,
                },
            )
            self.oms.revert_allocation(order_request.oms_order_id)
            return

        if fill_report:
            self._publish_latency_metrics(
                decision_to_fill_ms=(time.monotonic() - submit_started_at) * 1000.0,
                tick_to_fill_ms=(time.monotonic() - tick_started_at) * 1000.0,
                symbol=instrument_symbol,
            )
            try:
                self.oms.confirm_execution(
                    oms_order_id=fill_report.oms_order_id,
                    fill_price=fill_report.price,
                    fill_qty=fill_report.quantity,
                    fee=fill_report.fee,
                    exchange_trade_id=fill_report.exchange_order_id,
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
            await self.ems.stop()


@hydra.main(version_base=None, config_path="../../../conf", config_name="live")
def app(cfg: DictConfig) -> None:
    """Hydra entrypoint for the live trader."""
    configure_logging()
    log.info(
        "Hydra configuration loaded",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    trader = LiveTrader(cfg)
    asyncio.run(trader.run())


if __name__ == "__main__":
    app()
