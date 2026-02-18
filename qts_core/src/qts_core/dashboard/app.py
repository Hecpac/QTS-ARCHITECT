from __future__ import annotations

import os
import time

import plotly.graph_objects as go
import redis
import streamlit as st

from qts_core.dashboard.utils import (
    heartbeat_age_seconds,
    load_alert_events,
    parse_active_symbols,
    safe_float,
    safe_json,
    symbol_scoped_key,
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
    symbol_ohlcv_key = symbol_scoped_key("MARKET:OHLCV", selected_symbol)

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

    raw_ohlcv = redis_client.get(symbol_ohlcv_key) or redis_client.get("MARKET:OHLCV")
    raw_pos = redis_client.get("VIEW:POSITIONS")
    raw_ord = redis_client.get("VIEW:ORDERS")

    asset_snapshot_rows: list[dict[str, str | float]] = []
    for symbol in active_symbols:
        scoped_price = redis_client.get(symbol_scoped_key("MARKET:LAST_PRICE", symbol))
        if scoped_price is None:
            continue
        asset_snapshot_rows.append(
            {
                "symbol": symbol,
                "last_price": safe_float(scoped_price),
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

    raw_ohlcv = None
    raw_pos = None
    raw_ord = None
    asset_snapshot_rows = []

    last_heartbeat_raw = None
    last_alert_raw = None

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

if asset_snapshot_rows:
    st.subheader("Market Watch")
    st.dataframe(asset_snapshot_rows, use_container_width=True)

# --------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------
tab_chart, tab_pos, tab_alerts, tab_logs = st.tabs(
    ["📈 Market Chart", "💼 Portfolio", "🚨 Alerts", "🧠 System Brain"]
)

with tab_chart:
    ohlcv_rows = safe_json(raw_ohlcv, default=[])
    if isinstance(ohlcv_rows, list) and ohlcv_rows:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=[row.get("timestamp") for row in ohlcv_rows],
                    open=[row.get("open") for row in ohlcv_rows],
                    high=[row.get("high") for row in ohlcv_rows],
                    low=[row.get("low") for row in ohlcv_rows],
                    close=[row.get("close") for row in ohlcv_rows],
                    name="Price",
                )
            ]
        )
        fig.update_layout(
            title=f"{selected_symbol} - Live Action",
            height=500,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Ver datos crudos (OHLCV)"):
            st.dataframe(ohlcv_rows[-5:])
    else:
        st.info("Esperando datos de velas (OHLCV)...")

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
    last_alert = safe_json(last_alert_raw, default={})
    if isinstance(last_alert, dict) and last_alert:
        level = str(last_alert.get("level", "INFO")).upper()
        event = str(last_alert.get("event", "UNKNOWN"))
        message = str(last_alert.get("message", ""))
        caption = f"[{level}] {event} — {message}"

        if level in {"CRITICAL", "ERROR"}:
            st.error(caption)
        elif level == "WARNING":
            st.warning(caption)
        else:
            st.info(caption)

        st.json(last_alert)
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
