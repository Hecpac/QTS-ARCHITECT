"""Polymarket integration for QTS-Architect.

Modules:
    polymarket_loader: Data fetching (markets, orderbooks, prices)
    polymarket_oms: Order management (paper + live trading)
    polymarket_agent: Strategy agent (plugs into QTS supervisor)
"""

from .polymarket_agent import MarketOpportunity, PolymarketAgent, make_instrument_id
from .polymarket_loader import (
    OrderbookLevel,
    PolymarketLoader,
    PolymarketMarket,
    PolymarketOrderbook,
)
from qts_core.agents.prediction_gate import ImpliedDirection
from .polymarket_oms import (
    PolymarketOMS,
    PolymarketOrder,
    PolymarketOrderStatus,
    PolymarketSide,
)
from .sentiment_loader import (
    SentimentLoader,
    SentimentPost,
    SentimentSnapshot,
    SentimentSource,
    XClient,
    RedditClient,
)

__all__ = [
    "MarketOpportunity",
    "OrderbookLevel",
    "ImpliedDirection",
    "PolymarketAgent",
    "PolymarketLoader",
    "PolymarketMarket",
    "PolymarketOMS",
    "PolymarketOrder",
    "PolymarketOrderbook",
    "PolymarketOrderStatus",
    "PolymarketSide",
    "RedditClient",
    "SentimentLoader",
    "SentimentPost",
    "SentimentSnapshot",
    "SentimentSource",
    "XClient",
    "make_instrument_id",
]
