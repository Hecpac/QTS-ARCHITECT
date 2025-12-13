import json
import os
import time

import plotly.graph_objects as go
import redis
import streamlit as st


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

r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

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
    system_status = r.get("SYSTEM:HALT")
    if system_status == "true":
        st.error("🔴 SYSTEM HALTED")
        if st.button("♻️ RESUME OPERATIONS", use_container_width=True):
            r.set("SYSTEM:HALT", "false")
            st.rerun()
    else:
        st.success("🟢 SYSTEM RUNNING")
        if st.button("🚨 EMERGENCY STOP", type="primary", use_container_width=True):
            r.set("SYSTEM:HALT", "true")
            st.rerun()

st.divider()

# --------------------------------------------------------------------
# DATA FETCH
# --------------------------------------------------------------------
try:
    total_val = float(r.get("METRICS:TOTAL_VALUE") or 0.0)
    cash = float(r.get("METRICS:CASH") or 0.0)
    last_price = float(r.get("MARKET:LAST_PRICE") or 0.0)
    raw_ohlcv = r.get("MARKET:OHLCV")
except Exception:
    total_val, cash, last_price, raw_ohlcv = 0.0, 0.0, 0.0, None

# --------------------------------------------------------------------
# METRIC ROW
# --------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("💰 Total Equity", f"${total_val:,.2f}")
m2.metric("💵 Cash Available", f"${cash:,.2f}")
m3.metric("BTC Price", f"${last_price:,.2f}")
m4.metric("Daily PnL", "$0.00", delta_color="normal")  # TODO: conectar PnL real

# --------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------
tab_chart, tab_pos, tab_logs = st.tabs(
    ["📈 Market Chart", "💼 Portfolio", "🧠 System Brain"]
)

with tab_chart:
    if raw_ohlcv:
        try:
            ohlcv_rows = json.loads(raw_ohlcv)
        except Exception:
            ohlcv_rows = []

        if ohlcv_rows:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=[r.get("timestamp") for r in ohlcv_rows],
                        open=[r.get("open") for r in ohlcv_rows],
                        high=[r.get("high") for r in ohlcv_rows],
                        low=[r.get("low") for r in ohlcv_rows],
                        close=[r.get("close") for r in ohlcv_rows],
                        name="Price",
                    )
                ]
            )
            fig.update_layout(
                title="BTC/USD - Live Action",
                height=500,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                margin={"l": 0, "r": 0, "t": 30, "b": 0},
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Ver datos crudos (OHLCV)"):
                st.dataframe(ohlcv_rows[-5:])
        else:
            st.warning("OHLCV vacío recibido.")
    else:
        st.info("Esperando datos de velas (OHLCV)...")

with tab_pos:
    col_p, col_o = st.columns(2)
    with col_p:
        st.subheader("Open Positions")
        raw_pos = r.get("VIEW:POSITIONS")
        if raw_pos and raw_pos != "[]":
            try:
                st.dataframe(json.loads(raw_pos))
            except Exception:
                st.code(raw_pos)
        else:
            st.caption("No active positions.")

    with col_o:
        st.subheader("Active Orders")
        raw_ord = r.get("VIEW:ORDERS")
        if raw_ord and raw_ord != "[]":
            try:
                st.dataframe(json.loads(raw_ord))
            except Exception:
                st.code(raw_ord)
        else:
            st.caption("No open orders.")

with tab_logs:
    st.text("System Logs & Heartbeat")
    st.code(f"Last Heartbeat: {r.get('SYSTEM:HEARTBEAT')}")

# --------------------------------------------------------------------
# AUTO REFRESH
# --------------------------------------------------------------------
time.sleep(2)
st.rerun()
