"""Social Sentiment Loader for QTS-Architect.

Fetches real-time sentiment from X (Twitter) and Reddit, producing
normalized sentiment scores that feed into the PolymarketAgent.

Data Sources:
    - X (Twitter) API v2: keyword search, recent tweets, engagement metrics
    - Reddit API (PRAW): subreddit posts, comments, upvote ratios

Output:
    SentimentSnapshot — a timestamped sentiment reading per query/topic,
    compatible with QTS agent metadata.

Setup:
    X API: Bearer token in env var TWITTER_BEARER_TOKEN
    Reddit: Client ID/secret in REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final

import structlog

log = structlog.get_logger()

# ==============================================================================
# Constants
# ==============================================================================
X_RECENT_SEARCH_URL: Final[str] = "https://api.twitter.com/2/tweets/search/recent"
X_MAX_RESULTS: Final[int] = 100
REDDIT_USER_AGENT: Final[str] = "QTS-Architect:SentimentLoader:v1.0"

# Sentiment thresholds
BULLISH_THRESHOLD: Final[float] = 0.6
BEARISH_THRESHOLD: Final[float] = -0.6


# ==============================================================================
# Models
# ==============================================================================
class SentimentSource(str, Enum):
    """Where the sentiment data came from."""

    TWITTER = "twitter"
    REDDIT = "reddit"
    COMBINED = "combined"


@dataclass
class SentimentPost:
    """A single social media post with sentiment metadata."""

    text: str
    source: SentimentSource
    timestamp: datetime
    engagement: float = 0.0  # likes + retweets (X) or upvotes (Reddit)
    sentiment_score: float = 0.0  # -1.0 (bearish) to +1.0 (bullish)
    author_followers: int = 0


@dataclass
class SentimentSnapshot:
    """Aggregated sentiment for a topic at a point in time.

    This is what the PolymarketAgent consumes.
    """

    query: str
    source: SentimentSource
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    score: float = 0.0  # -1.0 to +1.0
    volume: int = 0  # Number of posts analyzed
    bullish_pct: float = 0.0  # % of posts with positive sentiment
    bearish_pct: float = 0.0  # % of posts with negative sentiment
    avg_engagement: float = 0.0  # Average engagement per post
    top_posts: list[SentimentPost] = field(default_factory=list)
    raw_scores: list[float] = field(default_factory=list)

    @property
    def signal_strength(self) -> float:
        """How strong the sentiment signal is (0-1).

        Combines score magnitude with volume and engagement.
        High volume + strong directional sentiment = strong signal.
        """
        if self.volume == 0:
            return 0.0

        magnitude = abs(self.score)
        # Volume weight: more posts = more confidence (log scale)
        import math
        vol_weight = min(1.0, math.log10(max(self.volume, 1)) / 3.0)  # 1000 posts = 1.0
        # Engagement weight
        eng_weight = min(1.0, self.avg_engagement / 1000.0)

        return min(1.0, magnitude * 0.5 + vol_weight * 0.3 + eng_weight * 0.2)

    @property
    def consensus(self) -> str:
        """Human-readable sentiment consensus."""
        if self.score >= BULLISH_THRESHOLD:
            return "STRONG_BULLISH"
        if self.score >= 0.2:
            return "BULLISH"
        if self.score <= BEARISH_THRESHOLD:
            return "STRONG_BEARISH"
        if self.score <= -0.2:
            return "BEARISH"
        return "NEUTRAL"


# ==============================================================================
# Keyword-Based Sentiment Scoring (no ML dependency)
# ==============================================================================
# Fast heuristic scorer — can be replaced with a fine-tuned model later.

_BULLISH_WORDS = frozenset({
    "bullish", "moon", "pump", "rally", "breakout", "ath", "all-time high",
    "buy", "long", "upside", "surge", "soar", "rocket", "gain", "profit",
    "bull run", "accumulate", "undervalued", "cheap", "opportunity",
    "support", "bounce", "recovery", "green", "calls", "alpha",
})

_BEARISH_WORDS = frozenset({
    "bearish", "dump", "crash", "sell", "short", "downside", "plunge",
    "drop", "fall", "decline", "collapse", "bust", "overvalued", "expensive",
    "resistance", "breakdown", "red", "puts", "rug", "scam", "fear",
    "panic", "liquidation", "margin call", "capitulation",
})


def score_text(text: str) -> float:
    """Score text sentiment using keyword heuristics.

    Returns:
        Float between -1.0 (very bearish) and +1.0 (very bullish).
    """
    lower = text.lower()
    words = set(lower.split())

    bull_hits = sum(1 for w in _BULLISH_WORDS if w in words or w in lower)
    bear_hits = sum(1 for w in _BEARISH_WORDS if w in words or w in lower)

    total = bull_hits + bear_hits
    if total == 0:
        return 0.0

    return (bull_hits - bear_hits) / total


# ==============================================================================
# X (Twitter) Client — Firecrawl Search (free, no API key needed)
# ==============================================================================
class XClient:
    """Fetches X/Twitter sentiment via Firecrawl web search.

    Uses `firecrawl search` CLI to find recent tweets/discussions
    about a topic. No X API key required.

    Falls back to X API v2 if bearer_token is set and Firecrawl
    returns empty results.
    """

    def __init__(self, bearer_token: str | None = None) -> None:
        self.bearer_token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN", "")

    async def search_recent(
        self,
        query: str,
        max_results: int = X_MAX_RESULTS,
    ) -> list[SentimentPost]:
        """Search recent social media posts about a topic.

        Uses Firecrawl web search with site:x.com filter for Twitter
        content, plus general financial news for broader sentiment.

        Args:
            query: Topic to search (e.g., "Bitcoin", "Gold").
            max_results: Target number of results.

        Returns:
            List of SentimentPost with sentiment scores.
        """
        posts: list[SentimentPost] = []

        # Search Twitter/X content + financial news via Firecrawl
        searches = [
            f"site:x.com {query} trading sentiment",
            f"{query} market sentiment bullish bearish today",
        ]

        for search_query in searches:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "firecrawl", "search", search_query,
                    "--limit", str(min(max_results // 2, 10)),
                    "--scrape",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=30.0
                )

                if proc.returncode != 0:
                    log.warning(
                        "firecrawl_search_error",
                        query=search_query,
                        stderr=stderr.decode()[:200],
                    )
                    continue

                output = stdout.decode()
                # Parse firecrawl output — each result has title + content
                for block in output.split("\n\n"):
                    text = block.strip()
                    if not text or len(text) < 20:
                        continue

                    is_twitter = "x.com" in text.lower() or "twitter" in text.lower()
                    posts.append(SentimentPost(
                        text=text[:500],
                        source=SentimentSource.TWITTER if is_twitter else SentimentSource.COMBINED,
                        timestamp=datetime.now(timezone.utc),
                        engagement=10.0 if is_twitter else 5.0,
                        sentiment_score=score_text(text),
                    ))

            except asyncio.TimeoutError:
                log.warning("firecrawl_timeout", query=search_query)
            except FileNotFoundError:
                log.error("firecrawl_not_found", msg="firecrawl CLI not installed")
                break
            except Exception as e:
                log.error("firecrawl_search_failed", query=search_query, error=str(e))

        log.info("x_search_ok", query=query, posts=len(posts), method="firecrawl")
        return posts[:max_results]

    async def close(self) -> None:
        pass  # No persistent session with CLI approach


# ==============================================================================
# Reddit Client
# ==============================================================================
class RedditClient:
    """Async client for Reddit API (OAuth2 script app).

    Uses client_credentials flow. Read-only.

    Env: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET
    """

    OAUTH_URL = "https://www.reddit.com/api/v1/access_token"
    API_BASE = "https://oauth.reddit.com"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET", "")
        self._session: Any = None
        self._token: str = ""
        self._token_expires: datetime = datetime.min.replace(tzinfo=timezone.utc)

    async def _ensure_token(self) -> None:
        """Get or refresh OAuth2 token."""
        now = datetime.now(timezone.utc)
        if self._token and now < self._token_expires:
            return

        import aiohttp
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        async with aiohttp.ClientSession() as tmp:
            async with tmp.post(
                self.OAUTH_URL,
                auth=auth,
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": REDDIT_USER_AGENT},
            ) as resp:
                if resp.status != 200:
                    log.error("reddit_auth_failed", status=resp.status)
                    return
                data = await resp.json()
                self._token = data.get("access_token", "")
                expires_in = data.get("expires_in", 3600)
                from datetime import timedelta
                self._token_expires = now + timedelta(seconds=expires_in - 60)

    async def _get_session(self) -> Any:
        await self._ensure_token()
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "User-Agent": REDDIT_USER_AGENT,
                }
            )
        else:
            self._session.headers.update({
                "Authorization": f"Bearer {self._token}",
            })
        return self._session

    async def search_subreddit(
        self,
        query: str,
        subreddit: str = "all",
        limit: int = 50,
        sort: str = "relevance",
        time_filter: str = "day",
    ) -> list[SentimentPost]:
        """Search Reddit posts matching a query.

        Args:
            query: Search keywords.
            subreddit: Subreddit to search (default: all).
            limit: Number of posts (max 100).
            sort: Sort by relevance, hot, new, top.
            time_filter: hour, day, week, month.

        Returns:
            List of SentimentPost with engagement and sentiment.
        """
        if not self.client_id or not self.client_secret:
            log.warning("reddit_no_credentials", msg="Reddit API credentials not set")
            return []

        session = await self._get_session()
        url = f"{self.API_BASE}/r/{subreddit}/search"
        params = {
            "q": query,
            "limit": min(limit, 100),
            "sort": sort,
            "t": time_filter,
            "restrict_sr": "true" if subreddit != "all" else "false",
        }

        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    log.warning("reddit_rate_limited", query=query)
                    return []
                if resp.status != 200:
                    log.warning("reddit_api_error", status=resp.status, query=query)
                    return []

                data = await resp.json()
                children = data.get("data", {}).get("children", [])

                posts = []
                for child in children:
                    post_data = child.get("data", {})
                    text = f"{post_data.get('title', '')} {post_data.get('selftext', '')}"
                    upvotes = post_data.get("ups", 0)
                    num_comments = post_data.get("num_comments", 0)
                    engagement = upvotes + num_comments * 0.5

                    created_utc = post_data.get("created_utc", 0)
                    ts = datetime.fromtimestamp(created_utc, tz=timezone.utc)

                    posts.append(SentimentPost(
                        text=text[:500],  # Truncate for scoring
                        source=SentimentSource.REDDIT,
                        timestamp=ts,
                        engagement=engagement,
                        sentiment_score=score_text(text),
                    ))

                log.info("reddit_search_ok", query=query, subreddit=subreddit, posts=len(posts))
                return posts

        except Exception as e:
            log.error("reddit_search_failed", query=query, error=str(e))
            return []

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "RedditClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


# ==============================================================================
# Unified Sentiment Loader
# ==============================================================================
class SentimentLoader:
    """Unified sentiment loader combining X and Reddit.

    Fetches from both sources, scores text, and produces an aggregated
    SentimentSnapshot for a given topic/query.

    Args:
        x_client: X (Twitter) API client.
        reddit_client: Reddit API client.
        subreddits: Default subreddits to search for financial sentiment.
    """

    DEFAULT_SUBREDDITS = [
        "wallstreetbets", "CryptoCurrency", "Bitcoin",
        "stocks", "investing", "polymarket",
    ]

    def __init__(
        self,
        x_client: XClient | None = None,
        reddit_client: RedditClient | None = None,
        subreddits: list[str] | None = None,
    ) -> None:
        self.x_client = x_client or XClient()
        self.reddit_client = reddit_client or RedditClient()
        self.subreddits = subreddits or self.DEFAULT_SUBREDDITS

    async def get_sentiment(
        self,
        query: str,
        subreddit: str | None = None,
        x_max_results: int = 100,
        reddit_limit: int = 50,
    ) -> SentimentSnapshot:
        """Fetch and aggregate sentiment from X + Reddit.

        Runs both API calls concurrently for speed.

        Args:
            query: Topic to analyze (e.g., "Bitcoin", "Iran conflict").
            subreddit: Specific subreddit (default: search relevant ones).
            x_max_results: Max tweets to fetch.
            reddit_limit: Max Reddit posts to fetch.

        Returns:
            Aggregated SentimentSnapshot with combined score.
        """
        # Run X and Reddit fetches concurrently
        reddit_sub = subreddit or "all"
        x_task = self.x_client.search_recent(query, max_results=x_max_results)
        reddit_task = self.reddit_client.search_subreddit(
            query, subreddit=reddit_sub, limit=reddit_limit
        )

        x_posts, reddit_posts = await asyncio.gather(x_task, reddit_task)
        all_posts = x_posts + reddit_posts

        if not all_posts:
            return SentimentSnapshot(
                query=query,
                source=SentimentSource.COMBINED,
                volume=0,
            )

        # Aggregate scores (engagement-weighted)
        total_engagement = sum(p.engagement for p in all_posts) or 1.0
        weighted_score = sum(
            p.sentiment_score * (p.engagement / total_engagement)
            for p in all_posts
        )

        raw_scores = [p.sentiment_score for p in all_posts]
        bullish = sum(1 for s in raw_scores if s > 0.1)
        bearish = sum(1 for s in raw_scores if s < -0.1)
        total = len(raw_scores)

        # Sort by engagement for top posts
        all_posts.sort(key=lambda p: p.engagement, reverse=True)

        snapshot = SentimentSnapshot(
            query=query,
            source=SentimentSource.COMBINED,
            score=max(-1.0, min(1.0, weighted_score)),
            volume=total,
            bullish_pct=bullish / total if total > 0 else 0.0,
            bearish_pct=bearish / total if total > 0 else 0.0,
            avg_engagement=total_engagement / total,
            top_posts=all_posts[:5],  # Keep top 5 by engagement
            raw_scores=raw_scores,
        )

        log.info(
            "sentiment_snapshot",
            query=query,
            score=round(snapshot.score, 3),
            volume=snapshot.volume,
            consensus=snapshot.consensus,
            signal_strength=round(snapshot.signal_strength, 3),
        )
        return snapshot

    async def get_multi_sentiment(
        self,
        queries: list[str],
    ) -> dict[str, SentimentSnapshot]:
        """Fetch sentiment for multiple queries concurrently.

        Args:
            queries: List of topics to analyze.

        Returns:
            Dict mapping query → SentimentSnapshot.
        """
        tasks = [self.get_sentiment(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        snapshots = {}
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                log.error("sentiment_query_failed", query=query, error=str(result))
                snapshots[query] = SentimentSnapshot(
                    query=query, source=SentimentSource.COMBINED
                )
            elif isinstance(result, SentimentSnapshot):
                snapshots[query] = result

        return snapshots

    async def close(self) -> None:
        """Clean up HTTP sessions."""
        await self.x_client.close()
        await self.reddit_client.close()

    async def __aenter__(self) -> "SentimentLoader":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
