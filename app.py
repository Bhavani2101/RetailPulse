# src/dashboard/app.py
# RetailPulse – Streamlit Dashboard (4 Business Dashboards)

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import *

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailPulse Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 14px; }

    /* Main background */
    .main { background: #f6f8fa; }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #0969da;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #57606a;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-delta {
        font-size: 0.82rem;
        color: #1a7f37;
        margin-top: 3px;
        font-weight: 600;
    }

    /* Section headers */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #24292f;
        border-left: 3px solid #0969da;
        padding-left: 10px;
        margin: 18px 0 10px 0;
    }

    /* Page title */
    .page-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        color: #24292f;
    }
    .page-sub {
        font-size: 0.85rem;
        color: #57606a;
        margin-top: 2px;
        margin-bottom: 16px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 14px;
        font-weight: 600;
        color: #57606a;
    }
    .stTabs [aria-selected="true"] { color: #0969da !important; }

    /* Alert boxes */
    .alert-box {
        background: #fff8c5;
        border: 1px solid #d4a72c;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.84rem;
        color: #633c01;
        margin-bottom: 10px;
    }
    .info-box {
        background: #ddf4ff;
        border: 1px solid #54aeff;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.84rem;
        color: #0550ae;
        margin-bottom: 10px;
    }

    /* Divider */
    hr { border-color: #d0d7de; margin: 14px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_cleaned():
    if os.path.exists(CLEANED_FILE):
        return pd.read_csv(CLEANED_FILE, parse_dates=["InvoiceDate"])
    return None

@st.cache_data
def load_rfm():
    if os.path.exists(RFM_FILE):
        return pd.read_csv(RFM_FILE)
    return None

@st.cache_data
def load_segmented():
    p = os.path.join(DATA_PROCESSED, "segmented_customers.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

@st.cache_data
def load_forecast():
    for name in ["ensemble_forecast.csv", "prophet_forecast.csv"]:
        p = os.path.join(DATA_PROCESSED, name)
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=["ds"]), name
    return None, None

@st.cache_data
def load_churn():
    p = os.path.join(DATA_PROCESSED, "churn_scores.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

@st.cache_data
def load_inventory():
    p = os.path.join(DATA_PROCESSED, "inventory_recommendations.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

@st.cache_data
def load_daily():
    if os.path.exists(FORECAST_FILE):
        return pd.read_csv(FORECAST_FILE, parse_dates=["ds"])
    return None


cleaned   = load_cleaned()
rfm       = load_rfm()
segmented = load_segmented()
forecast, fc_name = load_forecast()
churn     = load_churn()
inventory = load_inventory()
daily     = load_daily()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 RetailPulse")
    st.markdown("<span style='font-size:12px;color:#8b949e'>AI-Powered Retail Analytics</span>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigate", [
        "💰 Sales Dashboard",
        "👥 Customer Dashboard",
        "📈 Forecast Dashboard",
        "📦 Inventory Dashboard",
    ], label_visibility="collapsed")

    st.markdown("---")

    # Global date filter (if cleaned data available)
    if cleaned is not None:
        min_date = cleaned["InvoiceDate"].min().date()
        max_date = cleaned["InvoiceDate"].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            cleaned = cleaned[
                (cleaned["InvoiceDate"].dt.date >= date_range[0]) &
                (cleaned["InvoiceDate"].dt.date <= date_range[1])
            ]

    st.markdown("---")
    st.markdown("<span style='font-size:11px;color:#8b949e'>Zidio Development · 2026<br>RetailPulse v2.0</span>", unsafe_allow_html=True)


def kpi(value, label, delta=None):
    delta_html = f'<div class="kpi-delta">▲ {delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD 1 — SALES
# ═════════════════════════════════════════════════════════════════════════════

if page == "💰 Sales Dashboard":
    st.markdown('<div class="page-title">💰 Sales Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Revenue performance, trends, and product analysis</div>', unsafe_allow_html=True)

    if cleaned is None:
        st.warning("Run `data_loader.py` first to generate cleaned data.")
        st.stop()

    total_rev    = cleaned["TotalAmount"].sum()
    total_orders = cleaned["InvoiceNo"].nunique()
    total_cust   = cleaned["CustomerID"].nunique()
    avg_order    = cleaned["TotalAmount"].mean()
    avg_daily    = cleaned.groupby(cleaned["InvoiceDate"].dt.date)["TotalAmount"].sum().mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi(f"£{total_rev:,.0f}", "Total Revenue"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{total_orders:,}", "Total Orders"), unsafe_allow_html=True)
    c3.markdown(kpi(f"{total_cust:,}", "Customers"), unsafe_allow_html=True)
    c4.markdown(kpi(f"£{avg_order:.2f}", "Avg Order Value"), unsafe_allow_html=True)
    c5.markdown(kpi(f"£{avg_daily:,.0f}", "Avg Daily Revenue"), unsafe_allow_html=True)

    st.markdown("---")

    # Revenue trend
    section("Revenue Over Time")
    freq = st.radio("Granularity", ["Daily", "Weekly", "Monthly"],
                    horizontal=True, key="sales_freq")
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    rev_trend = (cleaned.set_index("InvoiceDate")
                 .resample(freq_map[freq])["TotalAmount"].sum().reset_index())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rev_trend["InvoiceDate"], y=rev_trend["TotalAmount"],
                              fill="tozeroy", line=dict(color="#0969da", width=2),
                              fillcolor="rgba(9,105,218,0.10)", name="Revenue"))
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                      xaxis_title="", yaxis_title="Revenue (£)",
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                      xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"))
    st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)

    with col_a:
        section("Top 10 Products by Revenue")
        top_prod = (cleaned.groupby("Description")["TotalAmount"]
                    .sum().nlargest(10).reset_index())
        top_prod.columns = ["Product", "Revenue"]
        fig = px.bar(top_prod.sort_values("Revenue"), x="Revenue", y="Product",
                     orientation="h", color="Revenue",
                     color_continuous_scale=["#cfe2ff", "#0969da"],
                     labels={"Revenue": "Revenue (£)"})
        fig.update_layout(height=340, showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    with col_b:
        section("Revenue by Country")
        country_rev = (cleaned.groupby("Country")["TotalAmount"]
                       .sum().nlargest(10).reset_index())
        fig = px.bar(country_rev.sort_values("TotalAmount"),
                     x="TotalAmount", y="Country", orientation="h",
                     color="TotalAmount", color_continuous_scale=["#d1ecf1", "#0c7b93"],
                     labels={"TotalAmount": "Revenue (£)"})
        fig.update_layout(height=340, showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    col_c, col_d = st.columns(2)

    with col_c:
        section("Sales by Day of Week")
        cleaned["DayOfWeek"] = cleaned["InvoiceDate"].dt.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = cleaned.groupby("DayOfWeek")["TotalAmount"].sum().reindex(dow_order).reset_index()
        fig = px.bar(dow, x="DayOfWeek", y="TotalAmount",
                     color="TotalAmount", color_continuous_scale=["#d3f9d8","#1a7f37"],
                     labels={"TotalAmount": "Revenue (£)", "DayOfWeek": ""})
        fig.update_layout(height=280, showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    with col_d:
        section("Monthly Revenue Heatmap")
        pivot = (cleaned.assign(
                    Month=cleaned["InvoiceDate"].dt.strftime("%b"),
                    Year=cleaned["InvoiceDate"].dt.year)
                 .groupby(["Year","Month"])["TotalAmount"].sum().unstack(fill_value=0))
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot[[m for m in months if m in pivot.columns]]
        fig = px.imshow(pivot, color_continuous_scale="Blues", aspect="auto",
                        labels={"color": "Revenue (£)"})
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, width='stretch')

    section("Order Value Distribution")
    fig = px.histogram(cleaned[cleaned["TotalAmount"] < cleaned["TotalAmount"].quantile(0.99)],
                       x="TotalAmount", nbins=60,
                       color_discrete_sequence=["#0969da"],
                       labels={"TotalAmount": "Order Value (£)"})
    fig.update_layout(height=240, margin=dict(l=0,r=0,t=10,b=0),
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                      bargap=0.05)
    st.plotly_chart(fig, width='stretch')


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD 2 — CUSTOMER
# ═════════════════════════════════════════════════════════════════════════════

elif page == "👥 Customer Dashboard":
    st.markdown('<div class="page-title">👥 Customer Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Segmentation, RFM analysis, and churn intelligence</div>', unsafe_allow_html=True)

    data = segmented if segmented is not None else rfm

    if data is None:
        st.warning("Run `segmentation.py` first to generate customer data.")
        st.stop()

    seg_col = "Business_Segment" if "Business_Segment" in data.columns else \
              "Segment" if "Segment" in data.columns else None

    total_customers = len(data)
    churn_rate = 0
    high_risk  = 0
    champions  = 0
    if churn is not None and "churned" in churn.columns:
        churn_rate = churn["churned"].mean() * 100
        high_risk  = (churn["churn_risk_label"] == "High").sum() if "churn_risk_label" in churn.columns else 0
    if seg_col and seg_col in data.columns:
        champions = (data[seg_col] == "Champions").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi(f"{total_customers:,}", "Total Customers"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{churn_rate:.1f}%", "Churn Rate"), unsafe_allow_html=True)
    c3.markdown(kpi(f"{high_risk:,}", "High Churn Risk"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{champions:,}", "Champions"), unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🗂 Segmentation", "📊 RFM Analysis", "⚠️ Churn"])

    # — Tab 1: Segmentation —
    with tab1:
        if seg_col:
            col_a, col_b = st.columns(2)
            with col_a:
                section("Segment Distribution")
                seg_counts = data[seg_col].value_counts().reset_index()
                seg_counts.columns = ["Segment", "Count"]
                fig = px.pie(seg_counts, names="Segment", values="Count",
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             hole=0.45)
                fig.update_traces(textposition="outside", textinfo="label+percent")
                fig.update_layout(height=360, showlegend=False,
                                  margin=dict(l=20,r=20,t=20,b=20))
                st.plotly_chart(fig, width='stretch')

            with col_b:
                section("Revenue by Segment")
                seg_rev = data.groupby(seg_col)["Monetary"].sum().reset_index()
                seg_rev.columns = ["Segment", "Revenue"]
                fig = px.bar(seg_rev.sort_values("Revenue", ascending=True),
                             x="Revenue", y="Segment", orientation="h",
                             color="Revenue",
                             color_continuous_scale=["#d3f9d8", "#1a7f37"],
                             labels={"Revenue": "Revenue (£)"})
                fig.update_layout(height=360, showlegend=False, coloraxis_showscale=False,
                                  margin=dict(l=0,r=0,t=10,b=0),
                                  plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
                st.plotly_chart(fig, width='stretch')

            section("Segment Summary Table")
            grp_cols = {"CustomerID": "count", "Recency": "mean",
                        "Frequency": "mean", "Monetary": "mean"}
            grp_cols = {k: v for k, v in grp_cols.items() if k in data.columns}
            summary = data.groupby(seg_col).agg(grp_cols).round(2).reset_index()
            summary.columns = [seg_col, "Customers", "Avg Recency",
                                "Avg Frequency", "Avg Monetary"][:len(summary.columns)]
            st.dataframe(summary, width='stretch', hide_index=True)

            section("3D RFM Scatter")
            sample = data.sample(min(1500, len(data)), random_state=42)
            fig = px.scatter_3d(sample, x="Recency", y="Frequency", z="Monetary",
                                 color=seg_col, opacity=0.65,
                                 color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(height=480, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Segment column not found. Run `segmentation.py` for labelled segments.")

    # — Tab 2: RFM Analysis —
    with tab2:
        if rfm is not None:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                section("Recency Distribution")
                fig = px.histogram(rfm, x="Recency", nbins=40,
                                   color_discrete_sequence=["#0969da"])
                fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                                  plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
                st.plotly_chart(fig, width='stretch')
            with col_b:
                section("Frequency Distribution")
                fig = px.histogram(rfm, x="Frequency", nbins=40,
                                   color_discrete_sequence=["#1a7f37"])
                fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                                  plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
                st.plotly_chart(fig, width='stretch')
            with col_c:
                section("Monetary Distribution")
                fig = px.histogram(rfm, x="Monetary", nbins=40,
                                   color_discrete_sequence=["#9a3412"])
                fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                                  plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
                st.plotly_chart(fig, width='stretch')

            if "RFM_Score" in rfm.columns:
                section("RFM Score Distribution")
                score_counts = rfm["RFM_Score"].value_counts().sort_index().reset_index()
                score_counts.columns = ["Score", "Customers"]
                fig = px.bar(score_counts, x="Score", y="Customers",
                             color="Score", color_continuous_scale="Blues")
                fig.update_layout(height=280, showlegend=False,
                                  coloraxis_showscale=False,
                                  plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                                  margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig, width='stretch')

            section("Top 20 Customers by Monetary Value")
            top_cust = rfm.nlargest(20, "Monetary")[["CustomerID","Recency","Frequency","Monetary"]]
            st.dataframe(top_cust.reset_index(drop=True), width='stretch', hide_index=True)
        else:
            st.info("Run `rfm.py` to generate RFM scores.")

    # — Tab 3: Churn —
    with tab3:
        if churn is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                section("Churn Risk Distribution")
                rc = churn["churn_risk_label"].value_counts().reset_index()
                rc.columns = ["Risk", "Count"]
                color_map = {"High": "#cf222e", "Medium": "#d4a72c", "Low": "#1a7f37"}
                fig = px.bar(rc, x="Risk", y="Count", color="Risk",
                             color_discrete_map=color_map)
                fig.update_layout(height=300, showlegend=False,
                                  plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                                  margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig, width='stretch')

            with col_b:
                section("Churn Probability Distribution")
                fig = px.histogram(churn, x="churn_probability", nbins=40,
                                   color_discrete_sequence=["#cf222e"])
                fig.add_vline(x=0.5, line_dash="dash", line_color="#57606a",
                              annotation_text="Threshold 0.5")
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                                  plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
                st.plotly_chart(fig, width='stretch')

            threshold = st.slider("Risk Threshold", 0.1, 0.9, 0.5, 0.05)
            at_risk = churn[churn["churn_probability"] >= threshold]
            st.markdown(f'<div class="alert-box">⚠️ At threshold <b>{threshold:.2f}</b>: <b>{len(at_risk):,} customers</b> flagged at-risk ({len(at_risk)/len(churn)*100:.1f}%)</div>', unsafe_allow_html=True)

            section("High Risk Customers — Top 20")
            top20 = (churn.sort_values("churn_probability", ascending=False)
                     .head(20)[["CustomerID","churn_probability","churn_risk_label"]]
                     .reset_index(drop=True))
            top20["churn_probability"] = top20["churn_probability"].round(4)
            st.dataframe(top20, width='stretch', hide_index=True)

            shap_path = os.path.join(REPORTS_DIR, "shap_summary.png")
            if os.path.exists(shap_path):
                section("SHAP Feature Importance")
                st.image(shap_path, caption="Top drivers of churn prediction")
        else:
            st.info("Run `churn_model.py` to generate churn predictions.")


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD 3 — FORECAST
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📈 Forecast Dashboard":
    st.markdown('<div class="page-title">📈 Forecast Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Prophet + LSTM ensemble · 30-day demand predictions</div>', unsafe_allow_html=True)

    if daily is None or forecast is None:
        st.warning("Run `forecasting.py` to generate forecast data.")
        st.stop()

    yhat_col    = "yhat_ensemble" if "yhat_ensemble" in forecast.columns else "yhat"
    model_label = "Ensemble (Prophet + LSTM)" if yhat_col == "yhat_ensemble" else "Prophet"
    last_actual = daily["ds"].max()
    future_fc   = forecast[forecast["ds"] > last_actual].copy()

    # KPIs
    total_fc_rev  = future_fc[yhat_col].sum()
    avg_fc_daily  = future_fc[yhat_col].mean()
    peak_day      = future_fc.loc[future_fc[yhat_col].idxmax(), "ds"]
    actual_avg    = daily["y"].mean()
    lift          = ((avg_fc_daily - actual_avg) / actual_avg * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi(f"£{total_fc_rev:,.0f}", "30-Day Forecast Revenue"), unsafe_allow_html=True)
    c2.markdown(kpi(f"£{avg_fc_daily:,.0f}", "Avg Daily Forecast"), unsafe_allow_html=True)
    c3.markdown(kpi(peak_day.strftime("%b %d"), "Peak Forecast Day"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{lift:+.1f}%", "vs Historical Avg"), unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("⚙️ What-If Controls", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            demand_adj  = st.slider("Demand Adjustment (%)", -30, 50, 0, 5)
        with col2:
            days_ahead  = st.slider("Forecast Horizon (Days)", 7, 90, 30, 7)

    fc_adj = forecast.copy()
    fc_adj[yhat_col] = fc_adj[yhat_col] * (1 + demand_adj / 100)
    future_adj = fc_adj[fc_adj["ds"] > last_actual].head(days_ahead)

    section("Demand Forecast vs Actuals")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"],
                              name="Actual", line=dict(color="#0969da", width=1.5)))
    hist_fc = fc_adj[fc_adj["ds"] <= last_actual]
    fig.add_trace(go.Scatter(x=hist_fc["ds"], y=hist_fc[yhat_col],
                              name="Model Fit", line=dict(color="#d4a72c", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=future_adj["ds"], y=future_adj[yhat_col],
                              name=f"Forecast ({days_ahead}d)",
                              line=dict(color="#cf222e", width=2.5)))
    if "yhat_upper" in future_adj.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([future_adj["ds"], future_adj["ds"][::-1]]),
            y=pd.concat([future_adj["yhat_upper"], future_adj["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(207,34,46,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI"))
    fig.add_vline(x=last_actual.timestamp() * 1000, line_dash="dash", line_color="#8c8c8c",
                  annotation_text="Forecast Start")
    fig.update_layout(height=400, hovermode="x unified",
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                      margin=dict(l=0,r=0,t=10,b=0),
                      xaxis=dict(showgrid=False),
                      yaxis=dict(gridcolor="#f0f0f0", title="Revenue (£)"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)

    with col_a:
        section("Forecast by Week")
        future_adj["Week"] = future_adj["ds"].dt.to_period("W").astype(str)
        weekly = future_adj.groupby("Week")[yhat_col].sum().reset_index()
        weekly.columns = ["Week", "Revenue"]
        fig = px.bar(weekly, x="Week", y="Revenue",
                     color="Revenue", color_continuous_scale=["#cfe2ff","#0969da"])
        fig.update_layout(height=280, showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, width='stretch')

    with col_b:
        section("Daily Forecast Table")
        display = future_adj[["ds", yhat_col]].copy()
        display.columns = ["Date", "Forecast Revenue (£)"]
        display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
        display["Forecast Revenue (£)"] = display["Forecast Revenue (£)"].round(2)
        st.dataframe(display, width='stretch', height=280, hide_index=True)

    # Historical seasonality insight
    section("Historical Revenue by Month")
    daily["Month"] = daily["ds"].dt.strftime("%b")
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = daily.groupby("Month")["y"].mean().reindex(months).reset_index()
    monthly.columns = ["Month", "Avg Revenue"]
    fig = px.line(monthly, x="Month", y="Avg Revenue", markers=True,
                  color_discrete_sequence=["#0969da"])
    fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                      plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                      xaxis=dict(showgrid=False),
                      yaxis=dict(gridcolor="#f0f0f0", title="Avg Daily Revenue (£)"))
    st.plotly_chart(fig, use_container_width=True)

    # Drift / model images
    col_c, col_d = st.columns(2)
    with col_c:
        p = os.path.join(REPORTS_DIR, "prophet_forecast.png")
        if os.path.exists(p):
            section("Prophet Forecast Chart")
            st.image(p)
    with col_d:
        p = os.path.join(REPORTS_DIR, "elbow_silhouette.png")
        if os.path.exists(p):
            section("Elbow / Silhouette Chart")
            st.image(p)


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD 4 — INVENTORY
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📦 Inventory Dashboard":
    st.markdown('<div class="page-title">📦 Inventory Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">EOQ · Safety stock · Reorder recommendations</div>', unsafe_allow_html=True)

    if inventory is None:
        st.warning("Run `inventory.py` to generate inventory recommendations.")
        st.stop()

    reorder_now = (inventory["stock_status"] == "🔴 Reorder Now").sum()
    monitor     = (inventory["stock_status"] == "🟡 Monitor").sum()
    ok          = (inventory["stock_status"] == "🟢 OK").sum()
    total_prod  = len(inventory)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi(f"{total_prod:,}", "Total Products"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{reorder_now:,}", "🔴 Reorder Now"), unsafe_allow_html=True)
    c3.markdown(kpi(f"{monitor:,}", "🟡 Monitor"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{ok:,}", "🟢 OK"), unsafe_allow_html=True)

    if reorder_now > 0:
        st.markdown(f'<div class="alert-box">⚠️ <b>{reorder_now} products</b> need immediate reordering.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # What-if controls
    with st.expander("⚙️ What-If: Adjust Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            demand_change = st.slider("Demand Change (%)", -30, 50, 0, 5)
        with col2:
            lead_time = st.slider("Lead Time (Days)", 1, 30, LEAD_TIME_DAYS)
        with col3:
            svc = st.selectbox("Service Level",
                               ["90% (Z=1.28)", "95% (Z=1.65)", "99% (Z=2.33)"], index=1)
        z_val = {"90% (Z=1.28)": 1.28, "95% (Z=1.65)": 1.65, "99% (Z=2.33)": 2.33}[svc]

    inv = inventory.copy()
    inv["adj_avg_daily"]    = inv["avg_daily_demand"] * (1 + demand_change / 100)
    inv["adj_safety_stock"] = (z_val * inv["std_daily_demand"] * np.sqrt(lead_time)).round(0)
    inv["adj_reorder_point"]= (inv["adj_avg_daily"] * lead_time + inv["adj_safety_stock"]).round(0)

    col_a, col_b = st.columns(2)

    with col_a:
        section("Stock Status Breakdown")
        sc = inventory["stock_status"].value_counts().reset_index()
        sc.columns = ["Status", "Count"]
        cmap = {"🔴 Reorder Now": "#cf222e", "🟡 Monitor": "#d4a72c", "🟢 OK": "#1a7f37"}
        fig = px.pie(sc, names="Status", values="Count",
                     color="Status", color_discrete_map=cmap, hole=0.4)
        fig.update_traces(textinfo="label+percent+value")
        fig.update_layout(height=340, showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section("Demand vs Safety Stock (Top 80 Products)")
        top80 = inv.head(80)
        fig = px.scatter(top80, x="avg_daily_demand", y="adj_safety_stock",
                         size="total_revenue", color="stock_status",
                         hover_name="Description",
                         color_discrete_map=cmap,
                         labels={"avg_daily_demand": "Avg Daily Demand",
                                 "adj_safety_stock": "Safety Stock (units)"})
        fig.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        section("Top 15 — Reorder Point")
        top15 = inv.head(15)
        fig = px.bar(top15.sort_values("adj_reorder_point"),
                     x="adj_reorder_point", y="Description",
                     orientation="h", color="adj_reorder_point",
                     color_continuous_scale=["#cfe2ff","#0969da"],
                     labels={"adj_reorder_point": "Reorder Point (units)"})
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        section("Top 15 — Safety Stock")
        fig = px.bar(top15.sort_values("adj_safety_stock"),
                     x="adj_safety_stock", y="Description",
                     orientation="h", color="adj_safety_stock",
                     color_continuous_scale=["#ffd8b2","#d4a72c"],
                     labels={"adj_safety_stock": "Safety Stock (units)"})
        fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)

    section("All Products — Reorder Recommendations")
    status_filter = st.multiselect("Filter by Status",
                                   inventory["stock_status"].unique().tolist(),
                                   default=inventory["stock_status"].unique().tolist())
    filtered = inv[inv["stock_status"].isin(status_filter)]
    show_cols = ["StockCode","Description","avg_daily_demand","adj_safety_stock",
                 "adj_reorder_point","eoq","stock_status"]
    st.dataframe(filtered[show_cols].round(2).reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    section("Download Recommendations")
    csv = filtered[show_cols].round(2).to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV", csv,
                       file_name="inventory_recommendations.csv", mime="text/csv")
