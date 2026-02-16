from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import hydra
import structlog
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from qts_core.agents.supervisor import Supervisor
from qts_core.common.types import InstrumentId, MarketData
from qts_core.execution.oms import OrderManagementSystem, OrderRequest


if TYPE_CHECKING:
    from qts_core.execution.ems import ExecutionGateway, FillReport
    from qts_core.execution.store import StateStore

OHLCVTuple = tuple[datetime, float, float, float, float, float]

log = structlog.get_logger()


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
        self._last_market_data: MarketData | None = None
        self._last_ohlcv_history: list[OHLCVTuple] | None = None
        self._last_ohlcv_payload: list[dict[str, Any]] | None = None

        # Persistence + components from Hydra config
        self.store: StateStore = instantiate(cfg.store)
        self.oms = OrderManagementSystem(
            self.store,
            initial_cash=float(cfg.oms.get("initial_cash", 100_000.0)),
            risk_fraction=float(cfg.oms.get("risk_fraction", 0.10)),
        )
        self.ems: ExecutionGateway = instantiate(cfg.gateway)

        strategies = [instantiate(agent_cfg) for agent_cfg in cfg.agents.strategies]
        risk_agent = instantiate(cfg.agents.risk)
        self.supervisor = Supervisor(strategy_agents=strategies, risk_agent=risk_agent)

        self._order_views: list[dict[str, Any]] = []
        self.running = True

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
            positions_value = sum(
                qty * market_data.close for qty in self.oms.portfolio.positions.values()
            )
            total_value = (
                self.oms.portfolio.cash
                + self.oms.portfolio.blocked_cash
                + positions_value
            )

            metrics_keys = telemetry_cfg.metrics_keys
            self.store.set(metrics_keys.total_value, str(total_value))
            self.store.set(metrics_keys.cash, str(self.oms.portfolio.cash))
            self.store.set(metrics_keys.pnl_daily, str(0.0))

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

            while self.running:
                # Heartbeat even if no data/decisions
                self._publish_heartbeat()

                if self._halt_requested():
                    log.critical("SYSTEM HALT DETECTED. Shutting down immediately.")
                    await self.ems.stop()
                    return

                try:
                    market_data = await self._fetch_live_data()
                except Exception as exc:
                    log.error("Data Feed Error", error=str(exc))
                    await asyncio.sleep(self.tick_interval)
                    continue

                self._publish_telemetry(market_data)
                log.info(
                    "Tick Received",
                    symbol=market_data.instrument_id,
                    price=market_data.close,
                )

                # Approximate exposure for risk layer
                qty = self.oms.portfolio.positions.get(market_data.instrument_id, 0.0)
                positions_value = sum(
                    pos_qty * market_data.close
                    for pos_qty in self.oms.portfolio.positions.values()
                )
                total_value = (
                    self.oms.portfolio.cash
                    + self.oms.portfolio.blocked_cash
                    + positions_value
                )
                portfolio_exposure = (
                    abs(qty * market_data.close) / total_value
                    if total_value > 0
                    else 0.0
                )

                try:
                    decision = await self.supervisor.run(
                        market_data,
                        ohlcv_history=self._last_ohlcv_history,
                        portfolio_exposure=portfolio_exposure,
                    )
                except Exception as exc:
                    log.error("Strategy Execution Failed", error=str(exc))
                    continue

                if not decision:
                    continue

                order_request = self.oms.process_decision(
                    decision,
                    current_price=market_data.close,
                )
                if not order_request:
                    continue

                try:
                    fill_report = await self.ems.submit_order(order_request)
                except Exception as exc:
                    log.error(
                        "Order Submission Failed",
                        error=str(exc),
                        order_id=order_request.oms_order_id,
                    )
                    self.oms.revert_allocation(order_request.oms_order_id)
                    continue

                if fill_report:
                    self.oms.confirm_execution(
                        oms_order_id=fill_report.oms_order_id,
                        fill_price=fill_report.price,
                        fill_qty=fill_report.quantity,
                        fee=fill_report.fee,
                        exchange_trade_id=fill_report.exchange_order_id,
                    )
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
