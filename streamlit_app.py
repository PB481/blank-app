# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time, date
from dataclasses import dataclass
from typing import List, Dict, Optional

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="UCITS NAV Lifecycle – Timeline Modeler",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0a1628 0%, #1a2744 50%, #0d2137 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border-left: 4px solid #00d4aa;
    }
    .main-header h1 {
        color: #ffffff; font-size: 1.6rem; margin: 0 0 0.3rem 0;
        font-weight: 700; letter-spacing: -0.02em;
    }
    .main-header p { color: #8899aa; font-size: 0.85rem; margin: 0; }
    .sla-card { padding: 1.1rem 1.3rem; border-radius: 10px; text-align: center; font-weight: 600; }
    .sla-met { background: linear-gradient(135deg, #0a2e1a, #0d3d22); border: 1px solid #00d4aa; color: #00d4aa; }
    .sla-breach { background: linear-gradient(135deg, #3d0a0a, #4d1111); border: 1px solid #ff4444; color: #ff4444; }
    .sla-card .sla-label { font-size: 0.70rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.7; margin-bottom: 0.2rem; }
    .sla-card .sla-value { font-size: 1.35rem; }
    .info-card { background: #0e1a2e; border: 1px solid #1e2d44; border-radius: 8px; padding: 0.85rem 1rem; text-align: center; }
    .info-card .label { font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em; color: #667788; margin-bottom: 0.15rem; }
    .info-card .value { font-size: 1rem; font-weight: 600; color: #c8d8e8; }
    .info-card .sub { font-size: 0.72rem; color: #ff8844; margin-top: 0.1rem; }
    .tz-strip { display: flex; gap: 0; border-radius: 10px; overflow: hidden; border: 1px solid #1e2d44; margin-bottom: 1rem; }
    .tz-cell { flex: 1; padding: 0.7rem 0.6rem; text-align: center; background: #0e1a2e; border-right: 1px solid #1e2d44; }
    .tz-cell:last-child { border-right: none; }
    .tz-city { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.07em; color: #667788; margin-bottom: 0.15rem; }
    .tz-time { font-size: 1.05rem; font-weight: 700; color: #c8d8e8; }
    .tz-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
    .tz-dot.on { background: #00d4aa; }
    .tz-dot.off { background: #ff4444; opacity: 0.5; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Hub & Timezone Definitions
# ---------------------------------------------------------------------
@dataclass
class HubInfo:
    short: str
    tz_name: str
    gmt_offset: float
    window_start_gmt: int
    window_end_gmt: int
    city: str

HUB_DATA = {
    "EMEA – Dublin":       HubInfo("EMEA-DUB", "GMT", 0, 7*60, 19*60, "Dublin"),
    "EMEA – Luxembourg":   HubInfo("EMEA-LUX", "CET", +1, 6*60, 18*60, "Luxembourg"),
    "APAC – India":        HubInfo("APAC-IND", "IST", +5.5, 3*60+30, 14*60+30, "Mumbai"),
    "APAC – Philippines":  HubInfo("APAC-PHL", "PHT", +8, 1*60, 12*60, "Manila"),
    "NAM – New York":      HubInfo("NAM-NYC", "EST", -5, 13*60, 24*60, "New York"),
    "NAM – US East":       HubInfo("NAM-USE", "EST", -5, 13*60, 29*60, "US East"),
}

HUBS = list(HUB_DATA.keys())
CATEGORY_COLORS = {
    "Data Ingestion": "#3b82f6", "Batch Run": "#8b5cf6", "Trade Date Processing": "#f59e0b",
    "Reconciliation": "#06b6d4", "Valuation": "#ec4899", "T+1 Review": "#10b981", "Publication": "#00d4aa",
}

T_DATE = date(2025, 1, 15)
T1_DATE = T_DATE + timedelta(days=1)
VALUATION_POINT = datetime.combine(T_DATE, time(16, 0))
NAV_DEADLINE = datetime.combine(T1_DATE, time(9, 0))
TIMELINE_START = datetime.combine(T_DATE, time(8, 0))
TIMELINE_END = datetime.combine(T1_DATE, time(12, 0))
US_OPEN_GMT = datetime.combine(T_DATE, time(13, 0))

# ---------------------------------------------------------------------
# Timezone Helpers
# ---------------------------------------------------------------------
def gmt_to_local(gmt_dt: datetime, offset_hours: float) -> datetime:
    return gmt_dt + timedelta(hours=offset_hours)

def gmt_to_est(gmt_dt: datetime) -> datetime:
    return gmt_to_local(gmt_dt, -5)

def fmt_gmt(dt_val: datetime) -> str:
    return dt_val.strftime("%H:%M")

def fmt_est(gmt_dt: datetime) -> str:
    return gmt_to_est(gmt_dt).strftime("%H:%M")

def fmt_local(gmt_dt: datetime, hub_name: str) -> str:
    info = HUB_DATA.get(hub_name)
    if not info: return fmt_gmt(gmt_dt)
    return gmt_to_local(gmt_dt, info.gmt_offset).strftime("%H:%M")

def is_in_operating_window(gmt_dt: datetime, hub_name: str) -> bool:
    info = HUB_DATA.get(hub_name)
    if not info: return True
    mins = gmt_dt.hour * 60 + gmt_dt.minute
    if gmt_dt.date() > T_DATE: mins += 24 * 60
    return info.window_start_gmt <= mins <= info.window_end_gmt

def add_mins(d: datetime, minutes: int) -> datetime:
    return d + timedelta(minutes=minutes)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    hub_trade_processing = st.selectbox("Trade Processing", HUBS, index=2)
    hub_recon = st.selectbox("Reconciliations", HUBS, index=0)
    hub_accruals = st.selectbox("Income & Expense Accruals", HUBS, index=2)
    hub_corp_actions = st.selectbox("Corporate Actions", HUBS, index=0)
    hub_derivatives = st.selectbox("Derivatives Pricing", HUBS, index=0)
    hub_nav_review = st.selectbox("NAV Review & Publication", HUBS, index=0)

    st.markdown("### 📥 Data Ingestion Cutoffs (GMT)")
    ta_file_time = st.time_input("TA Cap Stock Files", value=time(17, 0))
    broker_file_time = st.time_input("Broker Files", value=time(16, 30))
    pricing_file_time = st.time_input("Pricing Feed", value=time(16, 15))

    st.markdown("### 🔄 Accounting System Batches")
    batch_1_time = st.time_input("Batch Run 1 (T)", value=time(18, 0))
    batch_2_time = st.time_input("Batch Run 2 (Overnight)", value=time(2, 0))
    batch_3_time = st.time_input("Batch Run 3 (Final)", value=time(5, 30))
    batch_duration = st.slider("Batch Duration (mins)", 10, 90, 30)

    st.markdown("### ⏱️ Processing Durations")
    dur_trade = st.slider("Trade Processing", 15, 120, 45)
    dur_recon = st.slider("Cash Recon", 15, 120, 60)
    dur_nav_review = st.slider("Final NAV Review", 15, 120, 60)
    dur_nav_publish = st.slider("NAV Publication", 5, 30, 15)

# ---------------------------------------------------------------------
# Logic Calculations
# ---------------------------------------------------------------------
warnings, errors = [], []
ta_file_dt = datetime.combine(T_DATE, ta_file_time)
broker_file_dt = datetime.combine(T_DATE, broker_file_time)
pricing_file_dt = datetime.combine(T_DATE, pricing_file_time)

batch1_start = datetime.combine(T_DATE, batch_1_time)
batch1_end = add_mins(batch1_start, batch_duration)
batch2_start = datetime.combine(T1_DATE if batch_2_time.hour < 12 else T_DATE, batch_2_time)
batch2_end = add_mins(batch2_start, batch_duration)
batch3_start = datetime.combine(T1_DATE if batch_3_time.hour < 12 else T_DATE, batch_3_time)
batch3_end = add_mins(batch3_start, batch_duration)

# T-Day sequence
trade_proc_start = max(broker_file_dt, pricing_file_dt, VALUATION_POINT)
trade_proc_end = add_mins(trade_proc_start, dur_trade)
recon_start = max(trade_proc_end, ta_file_dt)
recon_end = add_mins(recon_start, dur_recon)

# T+1 sequence
nav_review_start = batch3_end
nav_review_end = add_mins(nav_review_start, dur_nav_review)
nav_pub_start = nav_review_end
nav_pub_end = add_mins(nav_pub_start, dur_nav_publish)

sla_met = nav_pub_end <= NAV_DEADLINE
slack_minutes = (NAV_DEADLINE - nav_pub_end).total_seconds() / 60

# ---------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------
st.markdown('<div class="main-header"><h1>🏦 Irish UCITS – Daily NAV Lifecycle Modeler</h1><p>Critical-path timing from Valuation Point (16:00 GMT) to NAV Delivery (09:00 GMT)</p></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    status_class = "sla-met" if sla_met else "sla-breach"
    st.markdown(f'<div class="sla-card {status_class}"><div class="sla-label">SLA Status</div><div class="sla-value">{"✅ MET" if sla_met else "❌ BREACH"}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="info-card"><div class="label">NAV Published</div><div class="value">{fmt_gmt(nav_pub_end)} GMT</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="info-card"><div class="label">SLA Buffer</div><div class="value">{int(slack_minutes)} mins</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="info-card"><div class="label">Valuation Point</div><div class="value">16:00 GMT</div></div>', unsafe_allow_html=True)

# Gantt Chart
st.markdown("### 📊 Lifecycle Timeline")
tasks_list = [
    dict(Task="Batch 1", Start=batch1_start, Finish=batch1_end, Category="Batch Run"),
    dict(Task="Batch 3", Start=batch3_start, Finish=batch3_end, Category="Batch Run"),
    dict(Task="Trade Processing", Start=trade_proc_start, Finish=trade_proc_end, Category="Trade Date Processing"),
    dict(Task="Reconciliation", Start=recon_start, Finish=recon_end, Category="Reconciliation"),
    dict(Task="NAV Review", Start=nav_review_start, Finish=nav_review_end, Category="T+1 Review"),
    dict(Task="Publication", Start=nav_pub_start, Finish=nav_pub_end, Category="Publication"),
]
df_gantt = pd.DataFrame(tasks_list)
fig = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Task", color="Category", color_discrete_map=CATEGORY_COLORS)
fig.update_yaxes(autorange="reversed")
fig.add_vline(x=NAV_DEADLINE, line_dash="dash", line_color="red")
fig.update_layout(plot_bgcolor="#0a1628", paper_bgcolor="#0a1628", font=dict(color="#c8d8e8"), height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Irish UCITS NAV Lifecycle Modeler | For modelling purposes only.")
