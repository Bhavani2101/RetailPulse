# app.py
# RetailPulse – AI-Powered Customer Analytics & Demand Forecasting
# Streamlit Dashboard
#
# Pages:
#   🏠 Home              – Project overview and quick stats
#   📊 Dashboard         – Sales KPIs, trends, top products
#   👥 Segmentation      – RFM + K-Means customer segments
#   📈 Demand Forecast   – 30-day Prophet demand forecast
#   ⚠️  Churn Prediction  – Customer churn risk table
#   📦 Inventory         – EOQ reorder recommendations
#   ℹ️  About             – Tech stack and author info
#
# Run with:
#   streamlit run app.py

import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from config import (
    DATA_PROCESSED, REPORTS_DIR,
    CLEANED_FILE, RFM_FILE, FORECAST_FILE, INVENTORY_FILE,
    LEAD_TIME_DAYS,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailPulse Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* Main background */
    .main { background: #f6f8fa; }
    .block-container { padding: 1.5rem 2rem; }

    /* KPI card */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 10px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 8px;
    }
    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #0969da;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.74rem;
        color: #57606a;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .kpi-delta { font-size: 0.80rem; color: #1a7f37; margin-top: 2px; font-weight: 600; }

    /* Page title */
    .page-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #24292f;
    }
    .page-sub { font-size: 0.84rem; color: #57606a; margin-top: 2px; margin-bottom: 14px; }

    /* Section header */
    .section-hdr {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.98rem;
        font-weight: 700;
        color: #24292f;
        border-left: 3px solid #0969da;
        padding-left: 9px;
        margin: 16px 0 8px 0;
    }

    /* Alert and info boxes */
    .alert-box {
        background: #fff8c5;
        border: 1px solid #d4a72c;
        border-radius: 8px;
        padding: 9px 13px;
        font-size: 0.83rem;
        color: #633c01;
        margin-bottom: 10px;
    }
    .info-box {
        background: #ddf4ff;
        border: 1px solid #54aeff;
        border-radius: 8px;
        padding: 9px 13px;
        font-size: 0.83rem;
        color: #0550ae;
        margin-bottom: 10px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 14px;
        font-weight: 600;
        color: #57606a;
    }
    .stTabs [aria-selected="true"] { color: #0969da !important; }

    hr { border-color: #d0d7de; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_card(value: str, label: str, delta: str = None) -> str:
    """Return HTML for a KPI metric card."""
    delta_html = f'<div class="kpi-delta">▲ {delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""


def section_header(title: str) -> None:
    """Render a styled section header."""
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)


def chart_layout(height: int = 300) -> dict:
    """Standard Plotly layout settings for a clean white chart."""
    return dict(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#f0f0f0"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached so the app doesn't reload on every interaction)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_cleaned() -> pd.DataFrame | None:
    """Load cleaned retail transactions. Returns None if file is missing."""
    if os.path.exists(CLEANED_FILE):
        return pd.read_csv(CLEANED_FILE, parse_dates=["InvoiceDate"])
    return None


@st.cache_data(show_spinner=False)
def load_rfm() -> pd.DataFrame | None:
    """Load RFM scores. Returns None if file is missing."""
    if os.path.exists(RFM_FILE):
        return pd.read_csv(RFM_FILE)
    return None


@st.cache_data(show_spinner=False)
def load_segmented() -> pd.DataFrame | None:
    """Load segmented customers (with KMeans cluster labels)."""
    path = os.path.join(DATA_PROCESSED, "segmented_customers.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_forecast() -> tuple:
    """Load forecast data. Prefers ensemble, falls back to prophet."""
    for name in ["ensemble_forecast.csv", "prophet_forecast.csv"]:
        path = os.path.join(DATA_PROCESSED, name)
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["ds"]), name
    return None, None


@st.cache_data(show_spinner=False)
def load_churn() -> pd.DataFrame | None:
    """Load churn predictions (probability + risk label per customer)."""
    path = os.path.join(DATA_PROCESSED, "churn_scores.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_inventory() -> pd.DataFrame | None:
    """Load inventory recommendations."""
    if os.path.exists(INVENTORY_FILE):
        return pd.read_csv(INVENTORY_FILE)
    return None


@st.cache_data(show_spinner=False)
def load_daily() -> pd.DataFrame | None:
    """Load the daily sales time series used for forecasting."""
    if os.path.exists(FORECAST_FILE):
        return pd.read_csv(FORECAST_FILE, parse_dates=["ds"])
    return None


@st.cache_data(show_spinner=False)
def generate_and_cache_data() -> dict:
    """
    Generate synthetic data + run ML pipeline in-memory.
    Used when the user hasn't run run_pipeline.py yet.
    Returns a dict of DataFrames for all pages.
    """
    from src.data_preprocessing import (
        generate_synthetic_data, clean_data, build_rfm, build_daily_sales
    )
    from src.segmentation import preprocess_rfm, find_optimal_k, run_kmeans, interpret_clusters
    from src.feature_engineering import build_churn_features
    from src.inventory import compute_product_stats, compute_reorder_recommendations

    with st.spinner("⚙️ Generating demo data – this takes about 20 seconds …"):
        raw   = generate_synthetic_data(n_customers=800, n_transactions=30_000)
        clean = clean_data(raw)
        rfm   = build_rfm(clean)
        daily = build_daily_sales(clean)

        # Segmentation
        X_scaled, _, _ = preprocess_rfm(rfm)
        best_k         = find_optimal_k(X_scaled, k_range=range(2, 8))
        rfm, _         = run_kmeans(X_scaled, rfm, k=best_k)
        rfm, _         = interpret_clusters(rfm)

        # Churn (simplified: use RFM-based proxy without full model training)
        rfm["churned"]           = (rfm["R_Score"] <= 2).astype(int)
        rfm["churn_probability"] = 1 - (rfm["R_Score"] / 5)
        rfm["churn_risk_label"]  = pd.cut(
            rfm["churn_probability"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"],
        )
        churn = rfm[["CustomerID", "churn_probability", "churn_risk_label", "churned"]].copy()

        # Inventory
        stats     = compute_product_stats(clean)
        inventory = compute_reorder_recommendations(stats)

        # Simple linear forecast for demo
        n      = len(daily)
        t      = np.arange(n)
        slope  = np.polyfit(t, daily["y"].values, 1)
        future_t   = np.arange(n, n + 30)
        future_y   = np.polyval(slope, future_t)
        future_dates = pd.date_range(daily["ds"].max() + pd.Timedelta(days=1), periods=30)
        forecast = pd.DataFrame({
            "ds":         pd.concat([daily["ds"], pd.Series(future_dates)]).reset_index(drop=True),
            "yhat":       np.concatenate([np.polyval(slope, t), future_y]),
            "yhat_lower": np.concatenate([np.polyval(slope, t) * 0.85, future_y * 0.85]),
            "yhat_upper": np.concatenate([np.polyval(slope, t) * 1.15, future_y * 1.15]),
        })

    return {
        "cleaned":  clean,
        "rfm":      rfm,
        "segmented": rfm,
        "daily":    daily,
        "forecast": forecast,
        "churn":    churn,
        "inventory": inventory,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

cleaned   = load_cleaned()
rfm       = load_rfm()
segmented = load_segmented()
forecast, fc_name = load_forecast()
churn     = load_churn()
inventory = load_inventory()
daily     = load_daily()

# If no processed data exists, offer to generate demo data
data_ready = cleaned is not None

if not data_ready:
    demo_data = None  # will be set per-session if user clicks


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 RetailPulse")
    st.markdown(
        "<span style='font-size:12px;color:#8b949e'>AI-Powered Retail Analytics</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📊 Sales Dashboard",
            "👥 Customer Segmentation",
            "📈 Demand Forecast",
            "⚠️ Churn Prediction",
            "📦 Inventory",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Date filter (applies to Sales Dashboard)
    if cleaned is not None and page == "📊 Sales Dashboard":
        min_date = cleaned["InvoiceDate"].min().date()
        max_date = cleaned["InvoiceDate"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if len(date_range) == 2:
            cleaned = cleaned[
                (cleaned["InvoiceDate"].dt.date >= date_range[0]) &
                (cleaned["InvoiceDate"].dt.date <= date_range[1])
            ]

    st.markdown(
        "<span style='font-size:11px;color:#8b949e'>"
        "Zidio Development · 2026<br>RetailPulse v2.0</span>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Show "Run Pipeline" prompt when data is missing
# ─────────────────────────────────────────────────────────────────────────────

def _no_data_banner(step: str) -> bool:
    """
    Show a warning when required data files are missing.
    Returns True if data is available, False if missing.
    """
    if st.session_state.get("demo_data") is not None:
        return True   # demo data loaded
    st.warning(
        f"⚠️ **{step}** data not found.\n\n"
        "Run the pipeline to generate data:\n"
        "```bash\npython run_pipeline.py\n```\n"
        "Or click the button below to load a **demo** dataset."
    )
    if st.button("▶ Load Demo Data (Synthetic)"):
        st.session_state["demo_data"] = generate_and_cache_data()
        st.rerun()
    return False


def _get(key: str):
    """Return live data or fall back to demo data."""
    demo = st.session_state.get("demo_data")
    sources = {
        "cleaned":   cleaned,
        "rfm":       rfm,
        "segmented": segmented,
        "daily":     daily,
        "forecast":  forecast,
        "churn":     churn,
        "inventory": inventory,
    }
    live = sources.get(key)
    if live is not None:
        return live
    if demo is not None:
        return demo.get(key)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown(
        '<div class="page-title">📊 RetailPulse – AI-Powered Retail Analytics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-sub">End-to-end customer analytics & demand forecasting platform</div>',
        unsafe_allow_html=True,
    )

    # Hero summary
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### What is RetailPulse?

        RetailPulse is a data science portfolio project that transforms raw retail
        transaction data into actionable business intelligence.

        **Built for:**
        - Zidio Development Internship (Data Science – June 2026)
        - GitHub Portfolio
        - Interview Demonstrations

        ### What does it do?

        | Module | Description |
        |--------|-------------|
        | 📊 Sales Dashboard | Revenue trends, top products, country breakdown |
        | 👥 Customer Segmentation | RFM analysis + K-Means clustering |
        | 📈 Demand Forecast | Prophet-based 30-day revenue forecast |
        | ⚠️ Churn Prediction | XGBoost model with risk scoring |
        | 📦 Inventory | EOQ + Safety stock reorder recommendations |
        """)

    with col2:
        # Quick stats if data is available
        _cleaned  = _get("cleaned")
        _rfm      = _get("rfm")
        _churn    = _get("churn")
        _inv      = _get("inventory")

        if _cleaned is not None:
            st.markdown("#### 📈 Quick Stats")
            total_rev   = _cleaned["TotalAmount"].sum()
            total_cust  = _cleaned["CustomerID"].nunique()
            churn_rate  = (_churn["churned"].mean() * 100) if _churn is not None else 0
            reorder_cnt = ((_inv["stock_status"] == "🔴 Reorder Now").sum()
                           if _inv is not None else 0)

            c1, c2 = st.columns(2)
            c1.markdown(kpi_card(f"£{total_rev/1e6:.1f}M", "Total Revenue"),
                        unsafe_allow_html=True)
            c2.markdown(kpi_card(f"{total_cust:,}", "Customers"),
                        unsafe_allow_html=True)
            c1.markdown(kpi_card(f"{churn_rate:.1f}%", "Churn Rate"),
                        unsafe_allow_html=True)
            c2.markdown(kpi_card(f"{reorder_cnt}", "Reorder Alerts"),
                        unsafe_allow_html=True)
        else:
            st.info(
                "No data loaded yet.\n\n"
                "Run `python run_pipeline.py` or use the sidebar pages "
                "to load demo data."
            )

    st.markdown("---")
    st.markdown("""
    ### 🚀 Getting Started

    **Step 1:** Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

    **Step 2:** Generate processed data
    ```bash
    python run_pipeline.py
    ```

    **Step 3:** Launch the dashboard
    ```bash
    streamlit run app.py
    ```

    Or use the **▶ Load Demo Data** button on any page to try the app immediately.
    """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SALES DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📊 Sales Dashboard":
    st.markdown('<div class="page-title">📊 Sales Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Revenue performance, product trends and geographic breakdown</div>',
        unsafe_allow_html=True,
    )

    _cleaned = _get("cleaned")
    if _cleaned is None and not _no_data_banner("Sales"):
        st.stop()
    if _cleaned is None:
        _cleaned = st.session_state["demo_data"]["cleaned"]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_rev    = _cleaned["TotalAmount"].sum()
    total_orders = _cleaned["InvoiceNo"].nunique()
    total_cust   = _cleaned["CustomerID"].nunique()
    avg_order    = _cleaned["TotalAmount"].mean()
    avg_daily    = _cleaned.groupby(_cleaned["InvoiceDate"].dt.date)["TotalAmount"].sum().mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi_card(f"£{total_rev:,.0f}", "Total Revenue"), unsafe_allow_html=True)
    c2.markdown(kpi_card(f"{total_orders:,}", "Total Orders"),   unsafe_allow_html=True)
    c3.markdown(kpi_card(f"{total_cust:,}", "Customers"),        unsafe_allow_html=True)
    c4.markdown(kpi_card(f"£{avg_order:.2f}", "Avg Order"),      unsafe_allow_html=True)
    c5.markdown(kpi_card(f"£{avg_daily:,.0f}", "Avg Daily Rev"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Revenue Trend ─────────────────────────────────────────────────────────
    section_header("Revenue Over Time")
    freq_choice = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True)
    freq_map    = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    rev_trend   = (
        _cleaned.set_index("InvoiceDate")
        .resample(freq_map[freq_choice])["TotalAmount"]
        .sum()
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rev_trend["InvoiceDate"], y=rev_trend["TotalAmount"],
        fill="tozeroy", line=dict(color="#0969da", width=2),
        fillcolor="rgba(9,105,218,0.10)", name="Revenue",
    ))
    fig.update_layout(**chart_layout(300), yaxis_title="Revenue (£)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Top Products + Country ────────────────────────────────────────────────
    cola, colb = st.columns(2)

    with cola:
        section_header("Top 10 Products by Revenue")
        top_prod = (
            _cleaned.groupby("Description")["TotalAmount"]
            .sum().nlargest(10).reset_index()
        )
        top_prod.columns = ["Product", "Revenue"]
        fig = px.bar(
            top_prod.sort_values("Revenue"), x="Revenue", y="Product",
            orientation="h", color="Revenue",
            color_continuous_scale=["#cfe2ff", "#0969da"],
            labels={"Revenue": "Revenue (£)"},
        )
        fig.update_layout(**chart_layout(340), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with colb:
        section_header("Revenue by Country")
        country_rev = (
            _cleaned.groupby("Country")["TotalAmount"]
            .sum().nlargest(10).reset_index()
        )
        fig = px.bar(
            country_rev.sort_values("TotalAmount"),
            x="TotalAmount", y="Country", orientation="h",
            color="TotalAmount", color_continuous_scale=["#d1ecf1", "#0c7b93"],
            labels={"TotalAmount": "Revenue (£)"},
        )
        fig.update_layout(**chart_layout(340), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Day of Week + Heatmap ─────────────────────────────────────────────────
    colc, cold = st.columns(2)

    with colc:
        section_header("Sales by Day of Week")
        _cleaned["DayName"] = _cleaned["InvoiceDate"].dt.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = (
            _cleaned.groupby("DayName")["TotalAmount"]
            .sum().reindex(dow_order).reset_index()
        )
        fig = px.bar(
            dow, x="DayName", y="TotalAmount",
            color="TotalAmount", color_continuous_scale=["#d3f9d8", "#1a7f37"],
            labels={"TotalAmount": "Revenue (£)", "DayName": ""},
        )
        fig.update_layout(**chart_layout(280), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with cold:
        section_header("Monthly Revenue Heatmap")
        hm = _cleaned.assign(
            Month=_cleaned["InvoiceDate"].dt.strftime("%b"),
            Year=_cleaned["InvoiceDate"].dt.year,
        ).groupby(["Year","Month"])["TotalAmount"].sum().unstack(fill_value=0)
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        hm = hm[[m for m in month_order if m in hm.columns]]
        fig = px.imshow(hm, color_continuous_scale="Blues", aspect="auto",
                        labels={"color": "Revenue (£)"})
        fig.update_layout(**chart_layout(280))
        st.plotly_chart(fig, use_container_width=True)

    # ── Order Value Distribution ──────────────────────────────────────────────
    section_header("Order Value Distribution")
    q99 = _cleaned["TotalAmount"].quantile(0.99)
    fig = px.histogram(
        _cleaned[_cleaned["TotalAmount"] < q99],
        x="TotalAmount", nbins=60,
        color_discrete_sequence=["#0969da"],
        labels={"TotalAmount": "Order Value (£)"},
    )
    fig.update_layout(**chart_layout(240), bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

elif page == "👥 Customer Segmentation":
    st.markdown(
        '<div class="page-title">👥 Customer Segmentation</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-sub">RFM analysis · K-Means clustering · Business segment profiling</div>',
        unsafe_allow_html=True,
    )

    _seg   = _get("segmented") or _get("rfm")
    _rfm   = _get("rfm")
    _churn = _get("churn")

    if _seg is None and not _no_data_banner("Segmentation"):
        st.stop()
    if _seg is None:
        _seg = st.session_state["demo_data"].get("segmented")
        _rfm = _seg
        _churn = st.session_state["demo_data"].get("churn")

    # Detect which segment column exists
    seg_col = (
        "Business_Segment" if "Business_Segment" in _seg.columns
        else "Segment" if "Segment" in _seg.columns
        else None
    )

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_cust = len(_seg)
    churn_rate = (_churn["churned"].mean() * 100) if _churn is not None else 0
    high_risk  = ((_churn["churn_risk_label"] == "High").sum()
                  if _churn is not None and "churn_risk_label" in _churn.columns else 0)
    champions  = ((_seg[seg_col] == "Champions").sum()
                  if seg_col else 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(f"{total_cust:,}", "Total Customers"),   unsafe_allow_html=True)
    c2.markdown(kpi_card(f"{churn_rate:.1f}%", "Churn Rate"),     unsafe_allow_html=True)
    c3.markdown(kpi_card(f"{high_risk:,}", "High Churn Risk"),    unsafe_allow_html=True)
    c4.markdown(kpi_card(f"{champions:,}", "Champions"),          unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🗂 Segments", "📊 RFM Analysis", "🗺 3D Scatter"])

    # ── Tab 1: Segment Distribution ───────────────────────────────────────────
    with tab1:
        if seg_col:
            cola, colb = st.columns(2)
            with cola:
                section_header("Segment Distribution")
                seg_counts = _seg[seg_col].value_counts().reset_index()
                seg_counts.columns = ["Segment", "Count"]
                fig = px.pie(
                    seg_counts, names="Segment", values="Count",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    hole=0.45,
                )
                fig.update_traces(textposition="outside", textinfo="label+percent")
                fig.update_layout(height=360, showlegend=False,
                                  margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with colb:
                section_header("Monetary Value by Segment")
                seg_rev = _seg.groupby(seg_col)["Monetary"].sum().reset_index()
                seg_rev.columns = ["Segment", "Revenue"]
                fig = px.bar(
                    seg_rev.sort_values("Revenue", ascending=True),
                    x="Revenue", y="Segment", orientation="h",
                    color="Revenue", color_continuous_scale=["#d3f9d8", "#1a7f37"],
                    labels={"Revenue": "Revenue (£)"},
                )
                fig.update_layout(**chart_layout(360), showlegend=False,
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            section_header("Segment Summary Table")
            grp_cols = {
                c: fn for c, fn in
                [("CustomerID", "count"), ("Recency", "mean"),
                 ("Frequency", "mean"), ("Monetary", "mean")]
                if c in _seg.columns
            }
            summary = _seg.groupby(seg_col).agg(grp_cols).round(2).reset_index()
            st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.info("Segment column not found. Run segmentation.py first.")

    # ── Tab 2: RFM Distributions ──────────────────────────────────────────────
    with tab2:
        if _rfm is not None:
            cola, colb, colc = st.columns(3)
            for col, feat, color, label in [
                (cola, "Recency",   "#0969da", "Days since last purchase"),
                (colb, "Frequency", "#1a7f37", "Number of unique orders"),
                (colc, "Monetary",  "#9a3412", "Total spend (£)"),
            ]:
                with col:
                    section_header(feat)
                    fig = px.histogram(
                        _rfm, x=feat, nbins=40,
                        color_discrete_sequence=[color],
                        labels={feat: label},
                    )
                    fig.update_layout(**chart_layout(260))
                    st.plotly_chart(fig, use_container_width=True)

            if "RFM_Score" in _rfm.columns:
                section_header("RFM Score Distribution")
                score_counts = (
                    _rfm["RFM_Score"].value_counts()
                    .sort_index().reset_index()
                )
                score_counts.columns = ["Score", "Customers"]
                fig = px.bar(
                    score_counts, x="Score", y="Customers",
                    color="Score", color_continuous_scale="Blues",
                )
                fig.update_layout(**chart_layout(280), showlegend=False,
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run data_preprocessing.py to generate RFM data.")

    # ── Tab 3: 3D Scatter ─────────────────────────────────────────────────────
    with tab3:
        if seg_col and all(c in _seg.columns for c in ["Recency", "Frequency", "Monetary"]):
            section_header("3D RFM Cluster Scatter")
            st.markdown(
                '<div class="info-box">💡 Each point is a customer. '
                'Colour = cluster segment. Drag to rotate.</div>',
                unsafe_allow_html=True,
            )
            sample = _seg.sample(min(2000, len(_seg)), random_state=42)
            fig = px.scatter_3d(
                sample, x="Recency", y="Frequency", z="Monetary",
                color=seg_col, opacity=0.65,
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={"Recency": "Recency (days)",
                        "Frequency": "Frequency (orders)",
                        "Monetary": "Monetary (£)"},
            )
            fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run segmentation.py to generate cluster labels.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DEMAND FORECAST
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📈 Demand Forecast":
    st.markdown(
        '<div class="page-title">📈 Demand Forecast</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-sub">Prophet-based 30-day revenue forecast with What-If controls</div>',
        unsafe_allow_html=True,
    )

    _daily    = _get("daily")
    _forecast = _get("forecast")

    if (_daily is None or _forecast is None) and not _no_data_banner("Forecast"):
        st.stop()
    if _daily is None:
        _daily    = st.session_state["demo_data"]["daily"]
        _forecast = st.session_state["demo_data"]["forecast"]

    yhat_col    = "yhat_ensemble" if "yhat_ensemble" in _forecast.columns else "yhat"
    last_actual = _daily["ds"].max()
    future_fc   = _forecast[_forecast["ds"] > last_actual].copy()

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_fc_rev = future_fc[yhat_col].sum()
    avg_fc_daily = future_fc[yhat_col].mean()
    actual_avg   = _daily["y"].mean()
    lift         = (avg_fc_daily - actual_avg) / (actual_avg + 1e-8) * 100
    if not future_fc.empty:
        peak_day = future_fc.loc[future_fc[yhat_col].idxmax(), "ds"]
        peak_str = peak_day.strftime("%b %d")
    else:
        peak_str = "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(f"£{total_fc_rev:,.0f}", "30-Day Forecast"),  unsafe_allow_html=True)
    c2.markdown(kpi_card(f"£{avg_fc_daily:,.0f}", "Avg Daily Forecast"), unsafe_allow_html=True)
    c3.markdown(kpi_card(peak_str, "Peak Day"),                         unsafe_allow_html=True)
    c4.markdown(kpi_card(f"{lift:+.1f}%", "vs Historical Avg"),         unsafe_allow_html=True)

    st.markdown("---")

    # ── What-If Controls ──────────────────────────────────────────────────────
    with st.expander("⚙️ What-If Controls", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            demand_adj = st.slider("Demand Adjustment (%)", -30, 50, 0, 5)
        with col2:
            days_ahead = st.slider("Forecast Horizon (days)", 7, 90, 30, 7)

    fc_adj       = _forecast.copy()
    fc_adj[yhat_col] *= (1 + demand_adj / 100)
    if "yhat_lower" in fc_adj.columns:
        fc_adj["yhat_lower"] *= (1 + demand_adj / 100)
        fc_adj["yhat_upper"] *= (1 + demand_adj / 100)
    future_adj   = fc_adj[fc_adj["ds"] > last_actual].head(days_ahead)

    # ── Forecast Chart ────────────────────────────────────────────────────────
    section_header("Demand Forecast vs Actuals")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_daily["ds"], y=_daily["y"],
        name="Actual Revenue", line=dict(color="#0969da", width=1.5),
    ))
    hist_fc = fc_adj[fc_adj["ds"] <= last_actual]
    fig.add_trace(go.Scatter(
        x=hist_fc["ds"], y=hist_fc[yhat_col],
        name="Model Fit", line=dict(color="#d4a72c", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=future_adj["ds"], y=future_adj[yhat_col],
        name=f"Forecast ({days_ahead}d)",
        line=dict(color="#cf222e", width=2.5),
    ))
    if "yhat_upper" in future_adj.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([future_adj["ds"], future_adj["ds"][::-1]]),
            y=pd.concat([future_adj["yhat_upper"], future_adj["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(207,34,46,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI",
        ))
    fig.add_vline(
        x=last_actual.timestamp() * 1000, line_dash="dash",
        line_color="#8c8c8c", annotation_text="Forecast Start",
    )
    fig.update_layout(
        **chart_layout(420),
        hovermode="x unified",
        yaxis_title="Revenue (£)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Weekly + Table ────────────────────────────────────────────────────────
    cola, colb = st.columns(2)

    with cola:
        section_header("Weekly Forecast Summary")
        future_adj["Week"] = future_adj["ds"].dt.to_period("W").astype(str)
        weekly = future_adj.groupby("Week")[yhat_col].sum().reset_index()
        weekly.columns = ["Week", "Revenue"]
        fig = px.bar(
            weekly, x="Week", y="Revenue",
            color="Revenue", color_continuous_scale=["#cfe2ff", "#0969da"],
        )
        fig.update_layout(**chart_layout(280), showlegend=False,
                          coloraxis_showscale=False, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    with colb:
        section_header("Daily Forecast Table")
        display_fc = future_adj[["ds", yhat_col]].copy()
        display_fc.columns = ["Date", "Forecast Revenue (£)"]
        display_fc["Date"]                  = display_fc["Date"].dt.strftime("%Y-%m-%d")
        display_fc["Forecast Revenue (£)"] = display_fc["Forecast Revenue (£)"].round(2)
        st.dataframe(display_fc, use_container_width=True, height=280, hide_index=True)

    # ── Historical Seasonality ────────────────────────────────────────────────
    section_header("Historical Revenue Seasonality (Monthly Avg)")
    _daily_viz = _daily.copy()
    _daily_viz["Month"] = _daily_viz["ds"].dt.strftime("%b")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_avg = (
        _daily_viz.groupby("Month")["y"]
        .mean()
        .reindex(month_order)
        .reset_index()
    )
    monthly_avg.columns = ["Month", "Avg Revenue"]
    fig = px.line(
        monthly_avg, x="Month", y="Avg Revenue", markers=True,
        color_discrete_sequence=["#0969da"],
    )
    fig.update_layout(**chart_layout(260), yaxis_title="Avg Daily Revenue (£)")
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: CHURN PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

elif page == "⚠️ Churn Prediction":
    st.markdown(
        '<div class="page-title">⚠️ Churn Prediction</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-sub">XGBoost model · Customer risk scoring · Retention targeting</div>',
        unsafe_allow_html=True,
    )

    _churn = _get("churn")

    if _churn is None and not _no_data_banner("Churn"):
        st.stop()
    if _churn is None:
        _churn = st.session_state["demo_data"]["churn"]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    churn_rate  = _churn["churned"].mean() * 100 if "churned" in _churn.columns else 0
    high_risk   = (_churn["churn_risk_label"] == "High").sum() if "churn_risk_label" in _churn.columns else 0
    low_risk    = (_churn["churn_risk_label"] == "Low").sum()  if "churn_risk_label" in _churn.columns else 0
    avg_prob    = _churn["churn_probability"].mean() if "churn_probability" in _churn.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(f"{len(_churn):,}", "Total Customers"),       unsafe_allow_html=True)
    c2.markdown(kpi_card(f"{churn_rate:.1f}%", "Historical Churn"),    unsafe_allow_html=True)
    c3.markdown(kpi_card(f"{high_risk:,}", "🔴 High Risk"),            unsafe_allow_html=True)
    c4.markdown(kpi_card(f"{avg_prob:.2f}", "Avg Churn Probability"),  unsafe_allow_html=True)

    st.markdown("---")

    if "churn_risk_label" in _churn.columns and "churn_probability" in _churn.columns:
        cola, colb = st.columns(2)

        with cola:
            section_header("Churn Risk Distribution")
            rc = _churn["churn_risk_label"].value_counts().reset_index()
            rc.columns = ["Risk", "Count"]
            color_map = {"High": "#cf222e", "Medium": "#d4a72c", "Low": "#1a7f37"}
            fig = px.bar(
                rc, x="Risk", y="Count", color="Risk",
                color_discrete_map=color_map,
            )
            fig.update_layout(**chart_layout(300), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with colb:
            section_header("Churn Probability Distribution")
            fig = px.histogram(
                _churn, x="churn_probability", nbins=40,
                color_discrete_sequence=["#cf222e"],
                labels={"churn_probability": "Churn Probability"},
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="#57606a",
                          annotation_text="Default Threshold (0.5)")
            fig.update_layout(**chart_layout(300))
            st.plotly_chart(fig, use_container_width=True)

        # ── Risk Threshold Slider ─────────────────────────────────────────────
        threshold = st.slider("Risk Threshold", 0.10, 0.90, 0.50, 0.05)
        at_risk   = _churn[_churn["churn_probability"] >= threshold]
        st.markdown(
            f'<div class="alert-box">⚠️ At threshold <b>{threshold:.2f}</b>: '
            f'<b>{len(at_risk):,} customers</b> flagged at-risk '
            f'({len(at_risk) / len(_churn) * 100:.1f}%)</div>',
            unsafe_allow_html=True,
        )

        # ── High Risk Table ───────────────────────────────────────────────────
        section_header("High Risk Customers — Top 30")
        top30 = (
            _churn.sort_values("churn_probability", ascending=False)
            .head(30)[["CustomerID", "churn_probability", "churn_risk_label"]]
            .reset_index(drop=True)
        )
        top30["churn_probability"] = top30["churn_probability"].round(4)
        st.dataframe(top30, use_container_width=True, hide_index=True)

        # ── Model Report Images (if available) ───────────────────────────────
        cola, colb = st.columns(2)
        with cola:
            roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
            if os.path.exists(roc_path):
                section_header("ROC Curve")
                st.image(roc_path, caption="ROC Curve – XGBoost Churn Model")
        with colb:
            fi_path = os.path.join(REPORTS_DIR, "feature_importance.png")
            cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
            if os.path.exists(fi_path):
                section_header("Feature Importance")
                st.image(fi_path, caption="Top feature drivers of churn")
            elif os.path.exists(cm_path):
                section_header("Confusion Matrix")
                st.image(cm_path, caption="Confusion matrix on test set")
    else:
        st.info("Run churn_prediction.py to generate churn predictions.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: INVENTORY
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📦 Inventory":
    st.markdown(
        '<div class="page-title">📦 Inventory Optimisation</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-sub">EOQ · Safety stock · Reorder point recommendations</div>',
        unsafe_allow_html=True,
    )

    _inv = _get("inventory")

    if _inv is None and not _no_data_banner("Inventory"):
        st.stop()
    if _inv is None:
        _inv = st.session_state["demo_data"]["inventory"]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    reorder_now = (_inv["stock_status"] == "🔴 Reorder Now").sum()
    monitor     = (_inv["stock_status"] == "🟡 Monitor").sum()
    ok          = (_inv["stock_status"] == "🟢 OK").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card(f"{len(_inv):,}", "Total Products"),   unsafe_allow_html=True)
    c2.markdown(kpi_card(f"{reorder_now}", "🔴 Reorder Now"),  unsafe_allow_html=True)
    c3.markdown(kpi_card(f"{monitor}", "🟡 Monitor"),          unsafe_allow_html=True)
    c4.markdown(kpi_card(f"{ok}", "🟢 OK"),                   unsafe_allow_html=True)

    if reorder_now > 0:
        st.markdown(
            f'<div class="alert-box">⚠️ <b>{reorder_now} products</b> need '
            f'immediate reordering to avoid stockouts.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── What-If Controls ──────────────────────────────────────────────────────
    with st.expander("⚙️ What-If: Adjust Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            demand_change = st.slider("Demand Change (%)", -30, 50, 0, 5)
        with col2:
            lead_time = st.slider("Lead Time (Days)", 1, 30, LEAD_TIME_DAYS)
        with col3:
            svc = st.selectbox(
                "Service Level",
                ["90% (Z=1.28)", "95% (Z=1.65)", "99% (Z=2.33)"],
                index=1,
            )
    z_val = {"90% (Z=1.28)": 1.28, "95% (Z=1.65)": 1.65, "99% (Z=2.33)": 2.33}[svc]

    inv = _inv.copy()
    inv["adj_avg_daily"]     = inv["avg_daily_demand"] * (1 + demand_change / 100)
    inv["adj_safety_stock"]  = (z_val * inv["std_daily_demand"] * np.sqrt(lead_time)).round(0)
    inv["adj_reorder_point"] = (
        inv["adj_avg_daily"] * lead_time + inv["adj_safety_stock"]
    ).round(0)

    # ── Charts ────────────────────────────────────────────────────────────────
    cola, colb = st.columns(2)
    cmap = {"🔴 Reorder Now": "#cf222e", "🟡 Monitor": "#d4a72c", "🟢 OK": "#1a7f37"}

    with cola:
        section_header("Stock Status Breakdown")
        sc = inv["stock_status"].value_counts().reset_index()
        sc.columns = ["Status", "Count"]
        fig = px.pie(
            sc, names="Status", values="Count",
            color="Status", color_discrete_map=cmap, hole=0.4,
        )
        fig.update_traces(textinfo="label+percent+value")
        fig.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with colb:
        section_header("Demand vs Safety Stock (Top 80)")
        top80 = inv.head(80)
        fig = px.scatter(
            top80, x="avg_daily_demand", y="adj_safety_stock",
            size="total_revenue", color="stock_status",
            hover_name="Description", color_discrete_map=cmap,
            labels={"avg_daily_demand": "Avg Daily Demand",
                    "adj_safety_stock": "Safety Stock (units)"},
        )
        fig.update_layout(**chart_layout(340))
        st.plotly_chart(fig, use_container_width=True)

    cola, colb = st.columns(2)
    with cola:
        section_header("Top 15 – Reorder Point")
        top15 = inv.head(15)
        fig = px.bar(
            top15.sort_values("adj_reorder_point"),
            x="adj_reorder_point", y="Description",
            orientation="h", color="adj_reorder_point",
            color_continuous_scale=["#cfe2ff", "#0969da"],
            labels={"adj_reorder_point": "Reorder Point (units)"},
        )
        fig.update_layout(**chart_layout(380), showlegend=False,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with colb:
        section_header("Top 15 – Safety Stock")
        fig = px.bar(
            top15.sort_values("adj_safety_stock"),
            x="adj_safety_stock", y="Description",
            orientation="h", color="adj_safety_stock",
            color_continuous_scale=["#ffd8b2", "#d4a72c"],
            labels={"adj_safety_stock": "Safety Stock (units)"},
        )
        fig.update_layout(**chart_layout(380), showlegend=False,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Full Table with Filter ────────────────────────────────────────────────
    section_header("All Products – Reorder Recommendations")
    status_filter = st.multiselect(
        "Filter by Status",
        options=inv["stock_status"].unique().tolist(),
        default=inv["stock_status"].unique().tolist(),
    )
    filtered = inv[inv["stock_status"].isin(status_filter)]
    show_cols = [
        "StockCode", "Description", "avg_daily_demand",
        "adj_safety_stock", "adj_reorder_point", "eoq", "stock_status",
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]
    st.dataframe(
        filtered[show_cols].round(2).reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

    # Download button
    csv_bytes = filtered[show_cols].round(2).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV",
        csv_bytes,
        file_name="inventory_recommendations.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown('<div class="page-title">ℹ️ About RetailPulse</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Project details, tech stack, and author information</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Project Overview

        **RetailPulse** is an end-to-end data science project built during the
        **Zidio Development Data Science Internship** (June–September 2026).

        It demonstrates a complete ML pipeline from raw data to deployed dashboard:
        - Data ingestion, cleaning, feature engineering
        - Customer segmentation (unsupervised ML)
        - Churn prediction (supervised ML)
        - Demand forecasting (time series)
        - Inventory optimisation (operations research)
        - Interactive Streamlit dashboard

        ---

        ### 📊 Model Performance Targets

        | Model            | Metric           | Target   |
        |------------------|------------------|----------|
        | K-Means          | Silhouette Score | ≥ 0.35   |
        | Prophet          | MAPE             | ≤ 12%    |
        | XGBoost Churn    | AUC-ROC          | ≥ 0.88   |
        """)

    with col2:
        st.markdown("""
        ### 🛠 Tech Stack

        | Category      | Tools                                    |
        |---------------|------------------------------------------|
        | Language      | Python 3.11                              |
        | ML / Stats    | Scikit-learn, XGBoost                    |
        | Forecasting   | Prophet, statsmodels                     |
        | Dashboard     | Streamlit, Plotly                        |
        | Data          | Pandas, NumPy                            |
        | Persistence   | Joblib                                   |
        | Testing       | Pytest                                   |
        | Deployment    | Docker, Streamlit Cloud                  |

        ---

        ### 👤 Author

        **Bhavani Nalajala**
        Data Science & Analytics Intern — Zidio Development (2026)

        - 📧 [LinkedIn](https://www.linkedin.com/in/bhavani-nalajala-586705316/)
        - 🐙 [GitHub](https://github.com/Bhavani2101)

        ---

        ### 📄 License

        MIT License — free to use and modify with attribution.
        """)

    st.markdown("---")
    st.markdown("""
    ### 🚀 Deployment

    **Streamlit Cloud:**
    1. Push to GitHub (public repo)
    2. Go to [share.streamlit.io](https://share.streamlit.io)
    3. Select repo → `app.py` → Deploy

    **Docker:**
    ```bash
    docker build -t retailpulse .
    docker run -p 8501:8501 retailpulse
    # Open http://localhost:8501
    ```
    """)
