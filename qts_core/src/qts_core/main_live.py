from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import date, datetime, timezone
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


def configure_logging(level: str = "INFO") -> None:
    """Configure logging for the live trader."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


class LiveTrader:
    """Live trading loop: data feed → supervisor → OMS → EMS, with telemetry."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the live trader from a Hydra config."""
        self.cfg = cfg
        self.symbol: str = cfg.get("symbol", "BTC/USDT")
        self.tick_interval: float = float(cfg.loop.get("tick_interval", 1.0))
        self.heartbeat_key: str = cfg.loop.get("heartbeat_key", "SYSTEM:HEARTBEAT")
        self.execution_timeout: float = max(
            0.1,
            float(cfg.loop.get("execution_timeout", max(5.0, self.tick_interval * 2))),
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

        self._order_views: list[dict[str, Any]] = []
        self.running = True

        # Daily PnL anchor for max_daily_loss enforcement in risk review.
        self._daily_anchor_date: date | None = None
        self._daily_anchor_value: float | None = None

        # Last known marks by instrument for multi-instrument exposure accounting.
        self._instrument_marks: dict[InstrumentId, float] = {}

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
                log.warning("fetch_ohlcv failed", error=str(exc))
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

        # Fallback simulation
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

    def _record_order_view(
        self,
        order_request: OrderRequest,
        fill_report: FillReport,
    ) -> None:
        self._order_views.append(
            {
                "order_id": order_request.oms_order_id,
                "side": order_request.side.value,
                "qty": order_request.quantity,
                "price": fill_report.price,
                "status": "FILLED",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

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

    def _reconcile_after_ambiguous_submission(
        self,
        *,
        reason: str,
        order_id: str,
    ) -> None:
        try:
            self.oms.reconcile()
            log.warning(
                "OMS reconciliation executed after ambiguous submission",
                reason=reason,
                order_id=order_id,
            )
        except Exception as exc:
            log.error(
                "OMS reconciliation failed after ambiguous submission",
                reason=reason,
                order_id=order_id,
                error=str(exc),
                exc_info=True,
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
                self._reconcile_after_ambiguous_submission(
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
                log.error(
                    "Halt liquidation returned no fill; reverting allocation",
                    order_id=order_request.oms_order_id,
                )
                self.oms.revert_allocation(order_request.oms_order_id)
                continue

            try:
                self.oms.confirm_execution(
                    oms_order_id=fill_report.oms_order_id,
                    fill_price=fill_report.price,
                    fill_qty=fill_report.quantity,
                    fee=fill_report.fee,
                    exchange_trade_id=fill_report.exchange_order_id,
                )
            except Exception as exc:
                log.error(
                    "Halt liquidation confirmation failed; preserving reservation for reconciliation",
                    order_id=order_request.oms_order_id,
                    error=str(exc),
                )
                self._reconcile_after_ambiguous_submission(
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

            # Always publish price keys
            self.store.set(last_price_key, str(market_data.close))
            self.store.set(
                last_tick_key,
                json.dumps(
                    {
                        "symbol": str(market_data.instrument_id),
                        "price": market_data.close,
                        "timestamp": market_data.timestamp.isoformat(),
                    }
                ),
            )

            if self._last_ohlcv_payload:
                self.store.set(ohlcv_key, json.dumps(self._last_ohlcv_payload))

            positions_payload = [
                {"instrument_id": str(instr), "quantity": qty}
                for instr, qty in self.oms.portfolio.positions.items()
            ]
            self.store.set(positions_key, json.dumps(positions_payload))

            if telemetry_cfg.publish_views and self._order_views:
                self.store.set(orders_key, json.dumps(self._order_views[-50:]))
        except Exception as exc:
            log.error("Telemetry publish failed", error=str(exc), exc_info=True)

    async def run(self) -> None:  # noqa: PLR0915
        """Run the live trading loop until stopped or halted."""
        log.info("Starting Live Trading Engine", env=self.cfg.get("env", "unknown"))

        try:
            await self.ems.start()

            try:
                self.oms.reconcile()
                log.info("Startup OMS reconciliation completed")
            except Exception as exc:
                log.error(
                    "Startup OMS reconciliation failed",
                    error=str(exc),
                    exc_info=True,
                )

            while self.running:
                # Heartbeat even if no data/decisions
                self._publish_heartbeat()

                if self._halt_requested():
                    log.critical("SYSTEM HALT DETECTED. Starting emergency liquidation.")
                    await self._liquidate_open_positions()
                    await self.ems.stop()
                    return

                try:
                    market_data = await self._fetch_live_data()
                except Exception as exc:
                    log.error("Data Feed Error", error=str(exc))
                    await asyncio.sleep(self.tick_interval)
                    continue

                self._instrument_marks[market_data.instrument_id] = market_data.close
                self._publish_telemetry(market_data)
                log.info(
                    "Tick Received",
                    symbol=market_data.instrument_id,
                    price=market_data.close,
                )

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

                try:
                    decision = await self.supervisor.run(
                        market_data,
                        ohlcv_history=self._last_ohlcv_history,
                        portfolio_exposure=portfolio_exposure,
                        daily_pnl_fraction=daily_pnl_fraction,
                    )
                except Exception as exc:
                    log.error("Strategy Execution Failed", error=str(exc))
                    continue

                if not decision:
                    continue

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
                    continue

                order_request = self.oms.process_decision(
                    guarded_decision,
                    current_price=market_data.close,
                )
                if not order_request:
                    continue

                try:
                    fill_report = await asyncio.wait_for(
                        self.ems.submit_order(order_request),
                        timeout=self.execution_timeout,
                    )
                except asyncio.TimeoutError:
                    log.error(
                        "Order submission timed out; preserving reservation for reconciliation",
                        order_id=order_request.oms_order_id,
                        timeout_seconds=self.execution_timeout,
                    )
                    self._reconcile_after_ambiguous_submission(
                        reason="submit_timeout",
                        order_id=order_request.oms_order_id,
                    )
                    continue
                except Exception as exc:
                    log.error(
                        "Order Submission Failed",
                        error=str(exc),
                        order_id=order_request.oms_order_id,
                    )
                    self.oms.revert_allocation(order_request.oms_order_id)
                    continue

                if fill_report:
                    try:
                        self.oms.confirm_execution(
                            oms_order_id=fill_report.oms_order_id,
                            fill_price=fill_report.price,
                            fill_qty=fill_report.quantity,
                            fee=fill_report.fee,
                            exchange_trade_id=fill_report.exchange_order_id,
                        )
                    except Exception as exc:
                        log.error(
                            "Order confirmation failed; preserving reservation for reconciliation",
                            error=str(exc),
                            order_id=order_request.oms_order_id,
                        )
                        self._reconcile_after_ambiguous_submission(
                            reason="confirm_failed",
                            order_id=order_request.oms_order_id,
                        )
                        continue
                    self._record_order_view(order_request, fill_report)
                else:
                    log.error(
                        "Order Submission Failed or No Report",
                        order_id=order_request.oms_order_id,
                    )
                    self.oms.revert_allocation(order_request.oms_order_id)

                self._publish_telemetry(market_data)

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
