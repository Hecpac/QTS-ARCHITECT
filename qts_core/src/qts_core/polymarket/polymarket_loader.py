"""Polymarket data loader for QTS-Architect.

Fetches prediction market data (odds, volume, orderbook) from Polymarket
via the CLI binary or the py-clob-client SDK.

This loader does NOT conform to the OHLCV DataLoader protocol because
prediction markets have fundamentally different data shapes. Instead, it
provides its own schema optimized for binary outcome markets.

Usage:
    loader = PolymarketLoader(cli_path="/usr/local/bin/polymarket")
    markets = await loader.search_markets("Bitcoin")
    book = await loader.get_orderbook(token_id="0xabc...")
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import structlog

log = structlog.get_logger()

# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_CLI_PATH: Final[str] = "/usr/local/bin/polymarket"
FALLBACK_CLI_PATH: Final[str] = str(
    Path.home() / "claw_workspace" / "polymarket"
)


# ==============================================================================
# Data Models
# ==============================================================================
@dataclass(frozen=True)
class PolymarketMarket:
    """A single prediction market."""

    condition_id: str
    question: str
    slug: str
    outcome_yes_price: float
    outcome_no_price: float
    volume_total: float
    volume_24h: float
    liquidity: float
    end_date: str
    active: bool
    token_id_yes: str = ""
    token_id_no: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def implied_probability(self) -> float:
        """YES price as implied probability."""
        return self.outcome_yes_price

    @property
    def spread(self) -> float:
        """Bid-ask spread (1 - yes - no)."""
        return max(0.0, 1.0 - self.outcome_yes_price - self.outcome_no_price)


@dataclass(frozen=True)
class OrderbookLevel:
    """A single price level in the orderbook."""

    price: float
    size: float


@dataclass(frozen=True)
class PolymarketOrderbook:
    """Full orderbook for a token."""

    token_id: str
    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def bid_depth(self) -> float:
        return sum(level.size for level in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(level.size for level in self.asks)


# ==============================================================================
# Loader
# ==============================================================================
class PolymarketLoader:
    """Data loader for Polymarket prediction markets.

    Uses the Polymarket CLI for data fetching. Falls back to direct
    HTTP API if CLI is not available.

    Args:
        cli_path: Path to the polymarket CLI binary.
        timeout: Command timeout in seconds.
    """

    def __init__(
        self,
        cli_path: str = DEFAULT_CLI_PATH,
        timeout: float = 30.0,
    ) -> None:
        self.cli_path = cli_path
        self.timeout = timeout

        # Resolve CLI path
        if not Path(self.cli_path).exists():
            if Path(FALLBACK_CLI_PATH).exists():
                self.cli_path = FALLBACK_CLI_PATH
                log.info("Using fallback CLI path", path=self.cli_path)
            else:
                log.warning(
                    "Polymarket CLI not found",
                    primary=cli_path,
                    fallback=FALLBACK_CLI_PATH,
                )

    async def _run_cli(self, *args: str) -> dict[str, Any] | list[Any]:
        """Execute a CLI command and parse JSON output.

        Args:
            *args: CLI subcommand and arguments.

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: If CLI execution fails.
        """
        cmd = [self.cli_path, *args, "--output", "json"]
        log.debug("polymarket_cli", cmd=" ".join(cmd))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            if proc.returncode != 0:
                err_msg = stderr.decode().strip()
                raise RuntimeError(
                    f"polymarket CLI failed (rc={proc.returncode}): {err_msg}"
                )

            return json.loads(stdout.decode())

        except asyncio.TimeoutError:
            raise RuntimeError(
                f"polymarket CLI timed out after {self.timeout}s"
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse CLI output: {e}")

    # ------------------------------------------------------------------
    # Market Discovery
    # ------------------------------------------------------------------
    async def search_markets(
        self,
        query: str,
        limit: int = 20,
    ) -> list[PolymarketMarket]:
        """Search for markets by keyword.

        Args:
            query: Search term (e.g., "Bitcoin", "Iran", "Nasdaq").
            limit: Maximum results to return.

        Returns:
            List of matching markets.
        """
        raw = await self._run_cli(
            "markets", "search", query, "--limit", str(limit)
        )

        markets = []
        items = raw if isinstance(raw, list) else raw.get("markets", [])

        for m in items[:limit]:
            markets.append(self._parse_market(m))

        log.info(
            "polymarket_search",
            query=query,
            results=len(markets),
        )
        return markets

    async def list_markets(
        self,
        order_by: str = "volume",
        limit: int = 20,
        active_only: bool = True,
    ) -> list[PolymarketMarket]:
        """List markets sorted by a field.

        Args:
            order_by: Sort field (volume, liquidity, end_date).
            limit: Maximum results.
            active_only: Only return active markets.

        Returns:
            List of markets.
        """
        args = ["markets", "list", "--order", order_by, "--limit", str(limit)]
        if active_only:
            args.append("--active")

        raw = await self._run_cli(*args)
        items = raw if isinstance(raw, list) else raw.get("markets", [])
        return [self._parse_market(m) for m in items[:limit]]

    async def get_market(self, condition_id: str) -> PolymarketMarket:
        """Get a specific market by condition ID.

        Args:
            condition_id: The market's condition ID.

        Returns:
            Market details.
        """
        raw = await self._run_cli("markets", "get", condition_id)
        return self._parse_market(raw)

    # ------------------------------------------------------------------
    # Orderbook
    # ------------------------------------------------------------------
    async def get_orderbook(self, token_id: str) -> PolymarketOrderbook:
        """Fetch the full orderbook for a token.

        Args:
            token_id: The YES or NO token ID (hex string).

        Returns:
            Orderbook with bids and asks.
        """
        raw = await self._run_cli("clob", "book", token_id)

        bids = [
            OrderbookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in (raw.get("bids") or [])
        ]
        asks = [
            OrderbookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in (raw.get("asks") or [])
        ]

        book = PolymarketOrderbook(
            token_id=token_id,
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
        )

        log.info(
            "polymarket_orderbook",
            token_id=token_id[:16] + "...",
            bids=len(book.bids),
            asks=len(book.asks),
            spread=book.spread,
        )
        return book

    async def get_price(self, token_id: str) -> float:
        """Get the current mid price for a token.

        Args:
            token_id: Token ID.

        Returns:
            Mid price (average of best bid and best ask).
        """
        raw = await self._run_cli("clob", "price", token_id)
        return float(raw.get("price", 0.0))

    async def get_spread(self, token_id: str) -> dict[str, float]:
        """Get bid-ask spread for a token.

        Args:
            token_id: Token ID.

        Returns:
            Dict with bid, ask, spread keys.
        """
        raw = await self._run_cli("clob", "spread", token_id)
        return {
            "bid": float(raw.get("bid", 0.0)),
            "ask": float(raw.get("ask", 0.0)),
            "spread": float(raw.get("spread", 0.0)),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_market(raw: dict[str, Any]) -> PolymarketMarket:
        """Parse raw API/CLI response into a PolymarketMarket."""
        # Handle different response formats (CLI vs API)
        tokens = raw.get("tokens", [])
        yes_token = ""
        no_token = ""
        yes_price = 0.0
        no_price = 0.0

        if tokens:
            for t in tokens:
                outcome = t.get("outcome", "").upper()
                if outcome == "YES":
                    yes_token = t.get("token_id", "")
                    yes_price = float(t.get("price", 0.0))
                elif outcome == "NO":
                    no_token = t.get("token_id", "")
                    no_price = float(t.get("price", 0.0))
        else:
            yes_price = float(raw.get("outcomePrices", [0, 0])[0])
            no_price = float(raw.get("outcomePrices", [0, 0])[1])

        return PolymarketMarket(
            condition_id=raw.get("condition_id", raw.get("conditionId", "")),
            question=raw.get("question", ""),
            slug=raw.get("slug", raw.get("market_slug", "")),
            outcome_yes_price=yes_price,
            outcome_no_price=no_price,
            volume_total=float(raw.get("volume", raw.get("volumeNum", 0))),
            volume_24h=float(raw.get("volume_num_24hr", raw.get("volume24hr", 0))),
            liquidity=float(raw.get("liquidity", raw.get("liquidityNum", 0))),
            end_date=raw.get("end_date_iso", raw.get("endDate", "")),
            active=raw.get("active", True),
            token_id_yes=yes_token,
            token_id_no=no_token,
            tags=raw.get("tags", []),
        )
