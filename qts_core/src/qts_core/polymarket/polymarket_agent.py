"""Polymarket Strategy Agent for QTS-Architect.

A Claude-powered agent that analyzes prediction markets and generates
trading signals compatible with the QTS agent protocol.

The agent:
1. Scans Polymarket for high-value opportunities
2. Analyzes orderbook depth and liquidity
3. Cross-references with QTS market data (forex, crypto, stocks)
4. Generates AgentSignals for the Supervisor

Integration:
    This agent plugs into the existing QTS multi-agent supervisor.
    It extends BaseStrategyAgent and follows the same Protocol interface.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from qts_core.agents.base import BaseStrategyAgent
from qts_core.agents.protocol import AgentPriority, AgentSignal, SignalType
from qts_core.common.types import InstrumentId

from .polymarket_loader import PolymarketLoader, PolymarketMarket, PolymarketOrderbook
from .polymarket_oms import PolymarketOMS, PolymarketOrder, PolymarketSide
from .sentiment_loader import SentimentLoader, SentimentSnapshot

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


# ==============================================================================
# Polymarket Instrument ID Convention
# ==============================================================================
# Format: POLYMARKET:<condition_id_short>
# Example: POLYMARKET:0x5e83ac5e
def make_instrument_id(condition_id: str) -> InstrumentId:
    """Create a QTS InstrumentId from a Polymarket condition ID."""
    short_id = condition_id[:10] if len(condition_id) > 10 else condition_id
    return InstrumentId(f"POLYMARKET:{short_id}")


# ==============================================================================
# Market Opportunity
# ==============================================================================
class MarketOpportunity:
    """A scored prediction market opportunity."""

    def __init__(
        self,
        market: PolymarketMarket,
        book: PolymarketOrderbook | None = None,
    ) -> None:
        self.market = market
        self.book = book
        self._score: float | None = None

    @property
    def score(self) -> float:
        """Opportunity score (0-1). Higher = better opportunity."""
        if self._score is not None:
            return self._score

        score = 0.0

        # Factor 1: Volume (higher = more legitimate market)
        if self.market.volume_24h > 100_000:
            score += 0.2
        elif self.market.volume_24h > 10_000:
            score += 0.1

        # Factor 2: Mispricing potential (odds near 50% = most uncertainty)
        yes_price = self.market.outcome_yes_price
        uncertainty = 1.0 - abs(yes_price - 0.5) * 2  # 1.0 at 50%, 0.0 at 0/100%
        score += uncertainty * 0.3

        # Factor 3: Orderbook quality
        if self.book:
            if self.book.spread is not None and self.book.spread < 0.05:
                score += 0.2  # Tight spread
            if self.book.bid_depth > 10_000 and self.book.ask_depth > 10_000:
                score += 0.15  # Deep book
            # Asymmetric book = potential edge
            if self.book.bid_depth > 0 and self.book.ask_depth > 0:
                ratio = self.book.bid_depth / self.book.ask_depth
                if ratio > 3.0 or ratio < 0.33:
                    score += 0.15  # Heavy asymmetry

        self._score = min(score, 1.0)
        return self._score

    @property
    def signal_type(self) -> SignalType:
        """Determine signal direction based on analysis."""
        yes_price = self.market.outcome_yes_price

        # Strong conviction zones
        if yes_price < 0.15:
            return SignalType.LONG  # Cheap YES, potential value
        if yes_price > 0.85:
            return SignalType.SHORT  # Expensive YES, sell/fade

        # Orderbook asymmetry
        if self.book:
            if self.book.bid_depth > self.book.ask_depth * 3:
                return SignalType.LONG  # Heavy buying pressure
            if self.book.ask_depth > self.book.bid_depth * 3:
                return SignalType.SHORT  # Heavy selling pressure

        return SignalType.NEUTRAL


# ==============================================================================
# Polymarket Agent
# ==============================================================================
class PolymarketAgent(BaseStrategyAgent):
    """Strategy agent for Polymarket prediction markets.

    Scans markets, analyzes orderbooks, and generates signals
    compatible with the QTS supervisor.

    Args:
        name: Agent identifier.
        loader: Polymarket data loader.
        oms: Polymarket order management system (optional).
        watch_queries: Keywords to monitor (e.g., ["Bitcoin", "Iran", "Nasdaq"]).
        min_volume_24h: Minimum 24h volume to consider a market.
        priority: Signal priority.
        min_confidence: Minimum confidence to emit signal.
    """

    def __init__(
        self,
        name: str = "polymarket_agent",
        loader: PolymarketLoader | None = None,
        oms: PolymarketOMS | None = None,
        sentiment: SentimentLoader | None = None,
        watch_queries: list[str] | None = None,
        min_volume_24h: float = 5_000.0,
        priority: AgentPriority = AgentPriority.NORMAL,
        min_confidence: float = 0.5,
        kelly_bankroll: float = 10_000.0,
        kelly_max_fraction: float = 0.05,
    ) -> None:
        super().__init__(name=name, priority=priority, min_confidence=min_confidence)
        self.loader = loader or PolymarketLoader()
        self.oms = oms
        self.sentiment = sentiment or SentimentLoader()
        self.watch_queries = watch_queries or ["Bitcoin", "Nasdaq", "Gold", "Iran"]
        self.min_volume_24h = min_volume_24h
        self.kelly_bankroll = kelly_bankroll
        self.kelly_max_fraction = kelly_max_fraction
        self._cache: dict[str, MarketOpportunity] = {}

    async def _generate_signal(
        self,
        instrument_id: InstrumentId,
        current_price: float,
        timestamp: datetime,
        ohlcv_history: list[tuple[datetime, float, float, float, float, float]]
        | None = None,
    ) -> AgentSignal | None:
        """Generate signal from Polymarket analysis.

        This method is called by the supervisor with a QTS instrument.
        The agent checks if any Polymarket markets are relevant to the
        instrument and generates a signal based on prediction market sentiment.

        For example, if analyzing BTC/USDT, the agent checks Polymarket
        for "Bitcoin" markets and uses their odds as a sentiment indicator.
        """
        # Extract base asset from instrument_id (e.g., "BINANCE:BTC-USDT" → "BTC")
        symbol = str(instrument_id).split(":")[-1].split("-")[0].split("/")[0]

        # Map common symbols to Polymarket queries
        query_map = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            "SOL": "Solana",
            "GOLD": "Gold",
            "XAU": "Gold",
            "NDX": "Nasdaq",
            "SPX": "S&P 500",
        }
        query = query_map.get(symbol.upper(), symbol)

        try:
            markets = await self.loader.search_markets(query, limit=5)
        except Exception as e:
            log.warning("polymarket_search_failed", query=query, error=str(e))
            return None

        if not markets:
            return None

        # Filter by volume
        viable = [m for m in markets if m.volume_24h >= self.min_volume_24h]
        if not viable:
            return None

        # Analyze top market
        top = viable[0]

        # Try to get orderbook for deeper analysis
        book = None
        if top.token_id_yes:
            try:
                book = await self.loader.get_orderbook(top.token_id_yes)
            except Exception:
                pass

        opp = MarketOpportunity(market=top, book=book)

        if opp.score < self.min_confidence:
            return None

        # Map Polymarket signal to QTS signal
        signal_type = opp.signal_type
        if signal_type == SignalType.NEUTRAL:
            return None

        # Fetch social sentiment for this asset
        sentiment_snap: SentimentSnapshot | None = None
        try:
            sentiment_snap = await self.sentiment.get_sentiment(query)
        except Exception as e:
            log.warning("sentiment_fetch_failed", query=query, error=str(e))

        # Boost/penalize confidence with sentiment
        confidence = opp.score
        if sentiment_snap and sentiment_snap.volume > 0:
            # Sentiment alignment bonus: if social agrees with signal, boost
            if signal_type == SignalType.LONG and sentiment_snap.score > 0.2:
                confidence = min(1.0, confidence + sentiment_snap.signal_strength * 0.2)
            elif signal_type == SignalType.SHORT and sentiment_snap.score < -0.2:
                confidence = min(1.0, confidence + sentiment_snap.signal_strength * 0.2)
            # Sentiment contradiction penalty
            elif signal_type == SignalType.LONG and sentiment_snap.score < -0.3:
                confidence *= 0.7
            elif signal_type == SignalType.SHORT and sentiment_snap.score > 0.3:
                confidence *= 0.7

        if confidence < self.min_confidence:
            return None

        return AgentSignal(
            source_agent=self.name,
            signal_type=signal_type,
            confidence=confidence,
            priority=self.priority,
            timestamp=timestamp,
            metadata={
                "polymarket_question": top.question[:100],
                "yes_price": top.outcome_yes_price,
                "volume_24h": top.volume_24h,
                "opportunity_score": opp.score,
                "spread": book.spread if book and book.spread else 0.0,
                "sentiment_score": sentiment_snap.score if sentiment_snap else 0.0,
                "sentiment_volume": sentiment_snap.volume if sentiment_snap else 0,
                "sentiment_consensus": sentiment_snap.consensus if sentiment_snap else "N/A",
            },
        )

    # ------------------------------------------------------------------
    # Standalone Scanning (not part of supervisor flow)
    # ------------------------------------------------------------------
    async def scan_all(self) -> list[MarketOpportunity]:
        """Scan all watched markets and return scored opportunities.

        This can be used independently of the QTS supervisor for
        a standalone Polymarket scanner.

        Returns:
            List of opportunities sorted by score (highest first).
        """
        all_opportunities: list[MarketOpportunity] = []

        for query in self.watch_queries:
            try:
                markets = await self.loader.search_markets(query, limit=10)
                for market in markets:
                    if market.volume_24h < self.min_volume_24h:
                        continue

                    book = None
                    if market.token_id_yes:
                        try:
                            book = await self.loader.get_orderbook(
                                market.token_id_yes
                            )
                        except Exception:
                            pass

                    opp = MarketOpportunity(market=market, book=book)
                    all_opportunities.append(opp)

            except Exception as e:
                log.warning("polymarket_scan_error", query=query, error=str(e))

        # Sort by score descending
        all_opportunities.sort(key=lambda x: x.score, reverse=True)

        log.info(
            "polymarket_scan_complete",
            queries=len(self.watch_queries),
            opportunities=len(all_opportunities),
            top_score=all_opportunities[0].score if all_opportunities else 0,
        )
        return all_opportunities

    async def execute_opportunity(
        self,
        opp: MarketOpportunity,
        size_usd: float = 10.0,
    ) -> PolymarketOrder | None:
        """Execute a trade on a market opportunity.

        Requires OMS to be configured. Respects paper/live mode.

        Args:
            opp: The opportunity to trade.
            size_usd: Position size in USD.

        Returns:
            The created order, or None if OMS not configured.
        """
        if self.oms is None:
            log.warning("polymarket_no_oms", msg="OMS not configured")
            return None

        token_id = opp.market.token_id_yes
        if not token_id:
            log.warning("polymarket_no_token", market=opp.market.question[:50])
            return None

        # Determine side and price
        if opp.signal_type == SignalType.LONG:
            side = PolymarketSide.BUY
            price = opp.market.outcome_yes_price
        elif opp.signal_type == SignalType.SHORT:
            side = PolymarketSide.SELL
            price = opp.market.outcome_yes_price
        else:
            return None

        # Create and submit order
        order = self.oms.create_order(
            token_id=token_id,
            side=side,
            price=price,
            size=size_usd,
            market_question=opp.market.question,
        )
        await self.oms.submit_order(order.order_id)
        return order
