from __future__ import annotations

import os
import time

import json

import plotly.graph_objects as go
import redis
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from qts_core.dashboard.utils import (
    compute_orderbook_imbalance,
    fetch_orderbook_with_fallback,
    fetch_rss_news,
    fetch_yahoo_chart_rows,
    heartbeat_age_seconds,
    load_alert_events,
    parse_active_symbols,
    safe_float,
    safe_json,
    symbol_scoped_key,
    to_binance_symbol,
)


# --------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------
st.set_page_config(page_title="QTS Pro Terminal", layout="wide", page_icon="🚀")

# --------------------------------------------------------------------
# REDIS CONNECTION
# --------------------------------------------------------------------
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
redis_db = int(os.getenv("REDIS_DB", "0"))

redis_client = redis.Redis(
    host=redis_host,
    port=redis_port,
    db=redis_db,
    decode_responses=True,
)


def _compute_ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    if period <= 1:
        return values.copy()

    multiplier = 2 / (period + 1)
    ema = [values[0]]
    for value in values[1:]:
        ema.append((value - ema[-1]) * multiplier + ema[-1])
    return ema


def _compute_rsi(values: list[float], period: int = 14) -> list[float | None]:
    if len(values) < 2:
        return [None for _ in values]

    deltas = [values[index] - values[index - 1] for index in range(1, len(values))]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [abs(min(delta, 0.0)) for delta in deltas]

    avg_gain = sum(gains[:period]) / period if len(gains) >= period else (sum(gains) / len(gains))
    avg_loss = sum(losses[:period]) / period if len(losses) >= period else (sum(losses) / len(losses))

    rsi: list[float | None] = [None]
    for index in range(len(deltas)):
        if index >= period:
            avg_gain = ((avg_gain * (period - 1)) + gains[index]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[index]) / period

        if avg_loss == 0:
            rsi.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))

    return rsi


def _build_candlestick_figure(
    rows: list[dict[str, float | str]],
    *,
    title: str,
    height: int = 360,
) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Candlestick(
                x=[row.get("timestamp") for row in rows],
                open=[row.get("open") for row in rows],
                high=[row.get("high") for row in rows],
                low=[row.get("low") for row in rows],
                close=[row.get("close") for row in rows],
                name="Price",
            )
        ]
    )
    figure.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )
    return figure


def _build_tradingview_style_figure(
    rows: list[dict[str, float | str]],
    *,
    title: str,
    height: int = 720,
) -> go.Figure:
    timestamps = [row.get("timestamp") for row in rows]
    opens = [safe_float(row.get("open")) for row in rows]
    highs = [safe_float(row.get("high")) for row in rows]
    lows = [safe_float(row.get("low")) for row in rows]
    closes = [safe_float(row.get("close")) for row in rows]
    volumes = [safe_float(row.get("volume")) for row in rows]

    ema_20 = _compute_ema(closes, 20)
    ema_50 = _compute_ema(closes, 50)
    rsi_14 = _compute_rsi(closes, 14)
    candle_colors = ["#00c087" if close >= open_ else "#ff4976" for open_, close in zip(opens, closes, strict=False)]

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.62, 0.22, 0.16],
        subplot_titles=("Price", "Volume", "RSI (14)"),
    )

    figure.add_trace(
        go.Candlestick(
            x=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Candles",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=ema_20,
            mode="lines",
            line={"color": "#f5b700", "width": 1.5},
            name="EMA 20",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=ema_50,
            mode="lines",
            line={"color": "#7c4dff", "width": 1.5},
            name="EMA 50",
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Bar(
            x=timestamps,
            y=volumes,
            marker_color=candle_colors,
            name="Volume",
            opacity=0.65,
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=rsi_14,
            mode="lines",
            line={"color": "#29b6f6", "width": 1.4},
            name="RSI 14",
        ),
        row=3,
        col=1,
    )

    figure.add_hline(y=70, row=3, col=1, line_width=1, line_dash="dash", line_color="#ff7043")
    figure.add_hline(y=30, row=3, col=1, line_width=1, line_dash="dash", line_color="#66bb6a")

    figure.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
    )
    figure.update_yaxes(fixedrange=False)
    figure.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

    return figure


@st.cache_data(ttl=60, show_spinner=False)
def _load_nasdaq_rows(interval: str, range_: str) -> list[dict[str, float | str]]:
    return fetch_yahoo_chart_rows("^IXIC", interval=interval, range_=range_)


@st.cache_data(ttl=45, show_spinner=False)
def _load_market_news() -> list[dict[str, str]]:
    feeds = [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Cointelegraph", "https://cointelegraph.com/rss"),
        ("Nasdaq", "https://www.nasdaq.com/feed/rssoutbound?category=Markets"),
        ("Investing", "https://www.investing.com/rss/news_25.rss"),
    ]
    return fetch_rss_news(feeds, per_feed_limit=10, total_limit=30)


@st.cache_data(ttl=5, show_spinner=False)
def _load_orderbook(
    symbol: str,
    depth: int,
) -> tuple[dict[str, list[dict[str, float]]], str]:
    return fetch_orderbook_with_fallback(symbol, limit=depth)


def _build_heatmap_figure(rows: list[dict[str, str | float]]) -> go.Figure:
    if not rows:
        return go.Figure()

    labels = [str(row.get("symbol", "")) for row in rows]
    values = [abs(safe_float(row.get("change_pct"))) + 1 for row in rows]
    changes = [safe_float(row.get("change_pct")) for row in rows]
    texts = [f"{label}<br>{change:+.2f}%" for label, change in zip(labels, changes, strict=False)]

    figure = go.Figure(
        go.Treemap(
            labels=labels,
            parents=["" for _ in labels],
            values=values,
            text=texts,
            textinfo="text",
            marker={
                "colors": changes,
                "colorscale": "RdYlGn",
                "cmid": 0,
                "line": {"width": 1, "color": "#1f1f1f"},
                "showscale": True,
                "colorbar": {"title": "Change %"},
            },
        )
    )
    figure.update_layout(
        title="Heatmap · Change %",
        template="plotly_dark",
        height=280,
        margin={"l": 0, "r": 0, "t": 35, "b": 0},
    )
    return figure


def _plotly_interaction_config() -> dict[str, object]:
    return {
        "displaylogo": False,
        "modeBarButtonsToAdd": [
            "drawline",
            "drawopenpath",
            "drawrect",
            "drawcircle",
            "eraseshape",
        ],
        "scrollZoom": True,
        "responsive": True,
    }


def _render_websocket_ticker(symbols: list[str]) -> None:
    streams = [f"{to_binance_symbol(symbol).lower()}@trade" for symbol in symbols if symbol]
    streams = streams[:8]
    if not streams:
        return

    stream_param = "/".join(streams)
    label_map = {to_binance_symbol(symbol).lower(): symbol for symbol in symbols}

    payload = json.dumps({"stream": stream_param, "labels": label_map})
    components.html(
        f"""
        <div style=\"padding:8px 12px;border:1px solid #2d2d2d;border-radius:8px;background:#111;margin:6px 0 10px 0;\">
          <div style=\"font-size:12px;color:#9aa0a6;margin-bottom:6px;\">WebSocket ticks (Binance stream)</div>
          <div id=\"ticker-wrap\" style=\"display:flex;gap:12px;flex-wrap:wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;\"></div>
        </div>
        <script>
          const cfg = {payload};
          const labels = cfg.labels || {{}};
          const wrap = document.getElementById('ticker-wrap');
          const format = (n) => Number(n).toLocaleString(undefined, {{maximumFractionDigits: 6}});

          Object.values(labels).forEach((label) => {{
            const node = document.createElement('span');
            node.id = `tick-${{label.replace(/[^a-zA-Z0-9]/g,'_')}}`;
            node.textContent = `${{label}} --`;
            node.style.color = '#cfd8dc';
            wrap.appendChild(node);
          }});

          const socket = new WebSocket(`wss://stream.binance.com:9443/stream?streams=${{cfg.stream}}`);
          socket.onmessage = (event) => {{
            try {{
              const parsed = JSON.parse(event.data);
              const stream = String(parsed.stream || '');
              const data = parsed.data || {{}};
              const symbol = stream.split('@')[0] || '';
              const label = labels[symbol] || symbol.toUpperCase();
              const nodeId = `tick-${{label.replace(/[^a-zA-Z0-9]/g,'_')}}`;
              const node = document.getElementById(nodeId);
              if (!node) return;

              const price = Number(data.p || data.c || 0);
              const isBuyerMaker = Boolean(data.m);
              node.textContent = `${{label}} ${{format(price)}}`;
              node.style.color = isBuyerMaker ? '#ef5350' : '#26a69a';
            }} catch (_) {{}}
          }};
          socket.onerror = () => {{
            wrap.style.opacity = 0.6;
          }};
        </script>
        """,
        height=84,
    )


def _render_audio_alert(trigger: bool, token: str) -> None:
    if not trigger:
        return

    components.html(
        f"""
        <script>
          const token = {json.dumps(token)};
          const storageKey = 'qts-audio-alert-token';
          const previous = sessionStorage.getItem(storageKey);
          if (previous !== token) {{
            sessionStorage.setItem(storageKey, token);
            try {{
              const ctx = new (window.AudioContext || window.webkitAudioContext)();
              const oscillator = ctx.createOscillator();
              const gainNode = ctx.createGain();
              oscillator.type = 'sine';
              oscillator.frequency.value = 880;
              gainNode.gain.value = 0.035;
              oscillator.connect(gainNode);
              gainNode.connect(ctx.destination);
              oscillator.start();
              oscillator.stop(ctx.currentTime + 0.25);
            }} catch (_) {{}}
          }}
        </script>
        """,
        height=0,
    )


# --------------------------------------------------------------------
# STYLES
# --------------------------------------------------------------------
st.markdown(
    """
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stAlert {margin-top: 10px;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------
# HEADER & KILL SWITCH
# --------------------------------------------------------------------
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("⚡ QTS Architect | Live Terminal")
    st.caption("Quantitative Trading System • Multi-Agent • ICT Logic")

with col_head2:
    system_status = redis_client.get("SYSTEM:HALT")
    if system_status == "true":
        st.error("🔴 SYSTEM HALTED")
        if st.button("♻️ RESUME OPERATIONS", use_container_width=True):
            redis_client.set("SYSTEM:HALT", "false")
            st.rerun()
    else:
        st.success("🟢 SYSTEM RUNNING")
        if st.button("🚨 EMERGENCY STOP", type="primary", use_container_width=True):
            redis_client.set("SYSTEM:HALT", "true")
            st.rerun()

st.divider()


# --------------------------------------------------------------------
# DATA FETCH
# --------------------------------------------------------------------
default_symbols_env = os.getenv("DEFAULT_ACTIVE_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT")
default_symbols = [item.strip() for item in default_symbols_env.split(",") if item.strip()]
if not default_symbols:
    default_symbols = ["BTC/USDT"]

active_symbols = default_symbols
try:
    active_symbols = parse_active_symbols(redis_client.get("MARKET:ACTIVE_SYMBOLS")) or default_symbols
except Exception:
    active_symbols = default_symbols

selected_symbol = st.selectbox("Activo", options=active_symbols, index=0)

try:
    total_val = safe_float(redis_client.get("METRICS:TOTAL_VALUE"))
    cash = safe_float(redis_client.get("METRICS:CASH"))
    daily_pnl_fraction = safe_float(redis_client.get("METRICS:PNL_DAILY"))

    symbol_last_price_key = symbol_scoped_key("MARKET:LAST_PRICE", selected_symbol)

    last_price = safe_float(
        redis_client.get(symbol_last_price_key) or redis_client.get("MARKET:LAST_PRICE")
    )

    latency_tick_to_decision = safe_float(
        redis_client.get(
            symbol_scoped_key("METRICS:LATENCY:TICK_TO_DECISION_MS", selected_symbol)
        )
        or redis_client.get("METRICS:LATENCY:TICK_TO_DECISION_MS")
    )
    latency_decision_to_fill = safe_float(
        redis_client.get(
            symbol_scoped_key("METRICS:LATENCY:DECISION_TO_FILL_MS", selected_symbol)
        )
        or redis_client.get("METRICS:LATENCY:DECISION_TO_FILL_MS")
    )
    latency_tick_to_fill = safe_float(
        redis_client.get(symbol_scoped_key("METRICS:LATENCY:TICK_TO_FILL_MS", selected_symbol))
        or redis_client.get("METRICS:LATENCY:TICK_TO_FILL_MS")
    )

    ohlcv_rows_by_symbol: dict[str, list[dict[str, float | str]]] = {}

    raw_pos = redis_client.get("VIEW:POSITIONS")
    raw_ord = redis_client.get("VIEW:ORDERS")

    asset_snapshot_rows: list[dict[str, str | float]] = []
    for symbol in active_symbols:
        scoped_price = redis_client.get(symbol_scoped_key("MARKET:LAST_PRICE", symbol))
        if scoped_price is None:
            continue

        parsed_price = safe_float(scoped_price)

        raw_symbol_ohlcv = redis_client.get(symbol_scoped_key("MARKET:OHLCV", symbol))
        if raw_symbol_ohlcv is None and symbol == selected_symbol:
            raw_symbol_ohlcv = redis_client.get("MARKET:OHLCV")

        symbol_rows = safe_json(raw_symbol_ohlcv, default=[])
        if isinstance(symbol_rows, list) and symbol_rows:
            ohlcv_rows_by_symbol[symbol] = symbol_rows

        pct_change = 0.0
        if isinstance(symbol_rows, list) and len(symbol_rows) >= 2:
            prev_close = safe_float(symbol_rows[-2].get("close"))
            if abs(prev_close) > 1e-12:
                pct_change = ((parsed_price - prev_close) / prev_close) * 100

        asset_snapshot_rows.append(
            {
                "symbol": symbol,
                "last_price": parsed_price,
                "change_pct": pct_change,
            }
        )

    last_heartbeat_raw = redis_client.get("SYSTEM:HEARTBEAT")
    last_alert_raw = redis_client.get("ALERTS:LAST")
except Exception:
    total_val = 0.0
    cash = 0.0
    last_price = 0.0
    daily_pnl_fraction = 0.0

    latency_tick_to_decision = 0.0
    latency_decision_to_fill = 0.0
    latency_tick_to_fill = 0.0

    ohlcv_rows_by_symbol = {}
    raw_pos = None
    raw_ord = None
    asset_snapshot_rows = []

    last_heartbeat_raw = None
    last_alert_raw = None

selected_change_pct = 0.0
for row in asset_snapshot_rows:
    if str(row.get("symbol")) == selected_symbol:
        selected_change_pct = safe_float(row.get("change_pct"))
        break

last_alert_payload = safe_json(last_alert_raw, default={})
if not isinstance(last_alert_payload, dict):
    last_alert_payload = {}

last_alert_level = str(last_alert_payload.get("level", "INFO")).upper()
last_alert_event = str(last_alert_payload.get("event", ""))

heartbeat_age = heartbeat_age_seconds(last_heartbeat_raw)
heartbeat_stale_seconds = int(os.getenv("HEARTBEAT_STALE_SECONDS", "15"))

# --------------------------------------------------------------------
# METRIC ROW
# --------------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("💰 Total Equity", f"${total_val:,.2f}")
m2.metric("💵 Cash Available", f"${cash:,.2f}")
m3.metric(f"{selected_symbol} Price", f"${last_price:,.2f}")
m4.metric("Daily PnL", f"{daily_pnl_fraction * 100:+.2f}%")
if heartbeat_age is None:
    m5.metric("Heartbeat Age", "N/A")
else:
    m5.metric("Heartbeat Age", f"{heartbeat_age:,.1f}s")

_render_websocket_ticker(active_symbols)

# --------------------------------------------------------------------
# LATENCY ROW
# --------------------------------------------------------------------
l1, l2, l3 = st.columns(3)
l1.metric("Latency Tick→Decision", f"{latency_tick_to_decision:,.1f} ms")
l2.metric("Latency Decision→Fill", f"{latency_decision_to_fill:,.1f} ms")
l3.metric("Latency Tick→Fill", f"{latency_tick_to_fill:,.1f} ms")

if heartbeat_age is None:
    st.warning("Heartbeat no disponible todavía.")
elif heartbeat_age > heartbeat_stale_seconds:
    st.error(
        f"Heartbeat stale: {heartbeat_age:,.1f}s (> {heartbeat_stale_seconds}s)."
    )

with st.expander("🔔 Alertas sonoras", expanded=False):
    audio_enabled = st.checkbox("Activar audio", value=True)
    audio_threshold_pct = st.slider(
        "Umbral de variación % para beep",
        min_value=0.2,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

audio_trigger = False
if audio_enabled:
    stale_trigger = heartbeat_age is not None and heartbeat_age > heartbeat_stale_seconds
    critical_trigger = last_alert_level in {"ERROR", "CRITICAL"}
    move_trigger = abs(selected_change_pct) >= audio_threshold_pct
    audio_trigger = stale_trigger or critical_trigger or move_trigger

    if audio_trigger:
        trigger_token = "|".join(
            [
                selected_symbol,
                f"{selected_change_pct:.4f}",
                last_alert_level,
                last_alert_event,
                str(last_heartbeat_raw),
            ]
        )
        _render_audio_alert(True, trigger_token)

if asset_snapshot_rows:
    st.subheader("Market Watch")
    st.dataframe(
        asset_snapshot_rows,
        use_container_width=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol"),
            "last_price": st.column_config.NumberColumn("Last", format="%.6f"),
            "change_pct": st.column_config.NumberColumn("Change %", format="%.2f%%"),
        },
        hide_index=True,
    )

# --------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------
tab_chart, tab_multichart, tab_news, tab_pos, tab_alerts, tab_logs = st.tabs(
    [
        "📈 TradingView",
        "🧩 Multi-Chart",
        "📰 News Live",
        "💼 Portfolio",
        "🚨 Alerts",
        "🧠 System Brain",
    ]
)

with tab_chart:
    col_chart_main, col_chart_side = st.columns([3.4, 1.6])

    selected_rows = ohlcv_rows_by_symbol.get(selected_symbol, [])
    with col_chart_main:
        if selected_rows:
            tradingview_figure = _build_tradingview_style_figure(
                selected_rows,
                title=f"{selected_symbol} — TradingView Style",
                height=760,
            )
            st.plotly_chart(
                tradingview_figure,
                use_container_width=True,
                config=_plotly_interaction_config(),
            )

            with st.expander("Ver datos crudos (OHLCV) del activo seleccionado"):
                st.dataframe(selected_rows[-10:], use_container_width=True)
        else:
            st.info(f"Esperando datos de velas (OHLCV) para {selected_symbol}...")

    with col_chart_side:
        st.subheader("DOM / Orderbook")
        dom_depth = st.select_slider("Depth", options=[5, 10, 20, 30], value=20)
        orderbook, orderbook_source = _load_orderbook(selected_symbol, dom_depth)
        imbalance = compute_orderbook_imbalance(orderbook)
        st.metric("Book imbalance", f"{imbalance:+.3f}")

        bid_rows = orderbook.get("bids", [])
        ask_rows = orderbook.get("asks", [])

        st.caption("Asks")
        if ask_rows:
            st.dataframe(ask_rows[:10], use_container_width=True, hide_index=True)
        else:
            st.caption("Sin asks")

        st.caption("Bids")
        if bid_rows:
            st.dataframe(bid_rows[:10], use_container_width=True, hide_index=True)
        else:
            st.caption("Sin bids")

        st.caption(f"Fuente DOM: {orderbook_source}")

with tab_multichart:
    st.subheader("Panel multi-activo")

    if asset_snapshot_rows:
        st.plotly_chart(
            _build_heatmap_figure(asset_snapshot_rows),
            use_container_width=True,
        )

    for symbol in active_symbols:
        symbol_rows = ohlcv_rows_by_symbol.get(symbol, [])
        if not symbol_rows:
            st.caption(f"Sin OHLCV todavía para {symbol}.")
            continue

        st.plotly_chart(
            _build_candlestick_figure(
                symbol_rows,
                title=f"{symbol} - Live Action",
                height=320,
            ),
            use_container_width=True,
            config=_plotly_interaction_config(),
        )

    st.subheader("NASDAQ Composite (^IXIC)")
    nasdaq_interval = os.getenv("NASDAQ_INTERVAL", "1h")
    nasdaq_range = os.getenv("NASDAQ_RANGE", "5d")
    nasdaq_rows = _load_nasdaq_rows(nasdaq_interval, nasdaq_range)

    if nasdaq_rows:
        nasdaq_last_close = nasdaq_rows[-1].get("close", 0.0)
        st.caption(
            f"Fuente: Yahoo Finance · interval={nasdaq_interval} · range={nasdaq_range} "
            f"· last={nasdaq_last_close}"
        )
        st.plotly_chart(
            _build_tradingview_style_figure(
                nasdaq_rows,
                title="NASDAQ Composite (^IXIC) — TradingView Style",
                height=680,
            ),
            use_container_width=True,
            config=_plotly_interaction_config(),
        )
    else:
        st.warning("No se pudo cargar el gráfico de NASDAQ por ahora.")

with tab_news:
    st.subheader("Noticias de mercado en tiempo real")
    st.caption("Actualiza automáticamente (cache ~45s por feed RSS).")

    news_items = _load_market_news()
    if not news_items:
        st.warning("No se pudieron cargar noticias por ahora.")
    else:
        for item in news_items:
            title = item.get("title", "(sin título)")
            source = item.get("source", "unknown")
            published = item.get("published", "")
            link = item.get("link", "")

            st.markdown(f"**{source}** · {published}")
            if link:
                st.markdown(f"[{title}]({link})")
            else:
                st.markdown(title)
            st.divider()

with tab_pos:
    col_p, col_o = st.columns(2)
    with col_p:
        st.subheader("Open Positions")
        pos_rows = safe_json(raw_pos, default=[])
        if isinstance(pos_rows, list) and pos_rows:
            st.dataframe(pos_rows)
        elif raw_pos:
            st.code(raw_pos)
        else:
            st.caption("No active positions.")

    with col_o:
        st.subheader("Active Orders")
        ord_rows = safe_json(raw_ord, default=[])
        if isinstance(ord_rows, list) and ord_rows:
            st.dataframe(ord_rows)
        elif raw_ord:
            st.code(raw_ord)
        else:
            st.caption("No open orders.")

with tab_alerts:
    st.subheader("Last Alert")
    if last_alert_payload:
        level = str(last_alert_payload.get("level", "INFO")).upper()
        event = str(last_alert_payload.get("event", "UNKNOWN"))
        message = str(last_alert_payload.get("message", ""))
        caption = f"[{level}] {event} — {message}"

        if level in {"CRITICAL", "ERROR"}:
            st.error(caption)
        elif level == "WARNING":
            st.warning(caption)
        else:
            st.info(caption)

        st.json(last_alert_payload)
    else:
        st.caption("Sin alertas recientes.")

    st.subheader("Recent Alert Events")
    event_limit = int(os.getenv("ALERT_EVENTS_LIMIT", "25"))
    alert_events = load_alert_events(redis_client, limit=event_limit)
    if alert_events:
        st.dataframe(alert_events, use_container_width=True)
    else:
        st.caption("No se encontraron eventos en ALERTS:EVENT:*.")

with tab_logs:
    st.text("System Logs & Heartbeat")
    st.code(f"Last Heartbeat: {last_heartbeat_raw}")

    if heartbeat_age is None:
        st.warning("Heartbeat no disponible.")
    elif heartbeat_age > heartbeat_stale_seconds:
        st.error(
            f"Heartbeat stale ({heartbeat_age:,.1f}s). "
            "Revisar conectividad EMS/loop o estado del proceso live."
        )
    else:
        st.success(f"Heartbeat healthy ({heartbeat_age:,.1f}s)")

    if last_alert_raw:
        st.text("Raw last alert payload")
        st.code(last_alert_raw)

# --------------------------------------------------------------------
# AUTO REFRESH
# --------------------------------------------------------------------
auto_refresh_seconds = max(1, int(os.getenv("DASHBOARD_REFRESH_SECONDS", "2")))
time.sleep(auto_refresh_seconds)
st.rerun()
