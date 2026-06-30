# src/inventory.py
# RetailPulse – Inventory Optimisation (EOQ + Safety Stock)
#
# Computes inventory recommendations per product using standard supply-chain formulas:
#   - Safety Stock  = Z × σ_daily × √lead_time
#   - Reorder Point = μ_daily × lead_time + Safety_Stock
#   - EOQ           = √(2 × D × S / H)   (Economic Order Quantity)
#
# Products are ranked by a priority score (revenue × demand variability) so the
# most business-critical items appear first.

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_PROCESSED, REPORTS_DIR,
    CLEANED_FILE, INVENTORY_FILE,
    LEAD_TIME_DAYS, SAFETY_STOCK_MULTIPLIER,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_cleaned_data(filepath: str = CLEANED_FILE) -> pd.DataFrame:
    """
    Load the cleaned transaction data for inventory analysis.

    Args:
        filepath: Path to cleaned_retail.csv

    Returns:
        Cleaned transactions DataFrame.

    Raises:
        FileNotFoundError: If the cleaned data file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Cleaned data not found: {filepath}\n"
            "Run data_preprocessing.py first."
        )
    df = pd.read_csv(filepath, parse_dates=["InvoiceDate"])
    print(f"[INV] Loaded cleaned data: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE PRODUCT-LEVEL DEMAND STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_product_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily demand statistics for each product (StockCode).

    Metrics per product:
    - avg_daily_demand:    Mean units sold per day
    - std_daily_demand:    Standard deviation of daily units sold
    - max_daily_demand:    Peak units in a single day
    - total_qty_sold:      All-time total units sold
    - days_with_sales:     Days on which at least one unit was sold
    - demand_variability:  Coefficient of variation (std / mean)
    - total_revenue:       All-time revenue generated

    Args:
        df: Cleaned transactions DataFrame.

    Returns:
        DataFrame with one row per product and demand stats.
    """
    print("\n[INV] Computing product demand statistics …")

    # Roll up to daily product-level quantities
    daily_product = (
        df.groupby(
            ["StockCode", "Description", pd.Grouper(key="InvoiceDate", freq="D")]
        )["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "daily_qty"})
    )

    # Aggregate demand statistics per product
    stats = (
        daily_product.groupby(["StockCode", "Description"])
        .agg(
            avg_daily_demand =("daily_qty", "mean"),
            std_daily_demand =("daily_qty", "std"),
            max_daily_demand =("daily_qty", "max"),
            total_qty_sold   =("daily_qty", "sum"),
            days_with_sales  =("daily_qty", lambda x: (x > 0).sum()),
        )
        .reset_index()
        .fillna(0)
    )

    # Coefficient of variation (measures how erratic demand is)
    stats["demand_variability"] = (
        stats["std_daily_demand"] / (stats["avg_daily_demand"] + 1e-8)
    )

    # Merge in total revenue per product
    revenue = (
        df.groupby("StockCode")["TotalAmount"]
        .sum()
        .reset_index()
        .rename(columns={"TotalAmount": "total_revenue"})
    )
    stats = stats.merge(revenue, on="StockCode", how="left")

    print(f"[INV] Product stats computed: {len(stats):,} unique products")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. REORDER RECOMMENDATIONS (EOQ + SAFETY STOCK)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reorder_recommendations(
    stats: pd.DataFrame,
    lead_time: int          = LEAD_TIME_DAYS,
    service_level_z: float  = 1.65,   # 95% service level (Z-score)
    ordering_cost: float    = 10.0,   # Fixed cost per order (£)
    holding_cost: float     = 0.20,   # Holding cost per unit per day (£)
) -> pd.DataFrame:
    """
    Compute reorder recommendations using supply-chain formulas.

    Key formulas:
    - Safety Stock  = Z × σ_daily × √lead_time
      (Buffer stock to cover demand uncertainty during lead time)
    - Reorder Point = μ_daily × lead_time + Safety Stock
      (Trigger a new order when stock falls to this level)
    - EOQ           = √(2 × D × S / H)
      (Optimal order quantity to minimise total inventory cost)

    Stock status:
    - 🔴 Reorder Now  – current stock ≤ reorder point
    - 🟡 Monitor      – stock ≤ 1.5 × reorder point
    - 🟢 OK           – stock is comfortable

    Note: Current stock is simulated for demo purposes (= 30-day demand × 1.2).

    Args:
        stats:          Product demand statistics DataFrame.
        lead_time:      Supplier lead time in days.
        service_level_z: Z-score for the desired service level.
        ordering_cost:  Fixed cost per purchase order (S in EOQ formula).
        holding_cost:   Per-unit-per-day holding cost (H in EOQ formula).

    Returns:
        DataFrame with reorder recommendations, sorted by priority.
    """
    print(f"\n[INV] Computing reorder recommendations …")
    print(f"  Lead time     : {lead_time} days")
    print(f"  Service level : 95%  (Z = {service_level_z})")

    rec = stats.copy()

    # Safety stock (units needed as buffer during lead time)
    rec["safety_stock"] = (
        service_level_z * rec["std_daily_demand"] * np.sqrt(lead_time)
    ).round(0)

    # Reorder point (order more when inventory hits this level)
    rec["reorder_point"] = (
        rec["avg_daily_demand"] * lead_time + rec["safety_stock"]
    ).round(0)

    # Economic Order Quantity (how much to order at a time)
    annual_demand = rec["avg_daily_demand"] * 365
    rec["eoq"] = np.sqrt(
        2 * annual_demand * ordering_cost / (holding_cost + 1e-8)
    ).round(0).clip(lower=1)

    # Simulate current stock level (30-day demand × 1.2 buffer)
    rec["simulated_current_stock"] = (
        rec["avg_daily_demand"] * 30 * 1.2
    ).round(0)

    # Assign stock status
    rec["stock_status"] = rec.apply(
        lambda r: "🔴 Reorder Now"
        if r["simulated_current_stock"] <= r["reorder_point"]
        else (
            "🟡 Monitor"
            if r["simulated_current_stock"] <= r["reorder_point"] * 1.5
            else "🟢 OK"
        ),
        axis=1,
    )

    # Priority: products with high revenue and high demand variability come first
    rec["priority_score"] = (
        rec["total_revenue"] * (1 + rec["demand_variability"])
    ).rank(ascending=False)
    rec = rec.sort_values("priority_score").reset_index(drop=True)

    print(f"\n  Stock Status Summary:")
    print(rec["stock_status"].value_counts().to_string())

    # Save recommendations
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    rec.to_csv(INVENTORY_FILE, index=False)
    print(f"\n[INV] Recommendations saved → {INVENTORY_FILE}")
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_inventory_charts(rec: pd.DataFrame) -> None:
    """
    Generate and save inventory visualisation charts.

    Charts:
    1. Top 20 products – reorder point vs safety stock (horizontal bar)
    2. Stock status distribution (pie chart)

    Args:
        rec: Inventory recommendations DataFrame.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    top20 = rec.head(20)

    # ── Chart 1: Reorder point + safety stock side by side ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(
        top20["Description"].str[:25], top20["reorder_point"], color="steelblue"
    )
    axes[0].set_title("Top 20 Products – Reorder Point")
    axes[0].set_xlabel("Units")
    axes[0].invert_yaxis()

    axes[1].barh(
        top20["Description"].str[:25], top20["safety_stock"], color="coral"
    )
    axes[1].set_title("Top 20 Products – Safety Stock")
    axes[1].set_xlabel("Units")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "inventory_recommendations.png"), dpi=150)
    plt.close()

    # ── Chart 2: Stock status pie chart ────────────────────────────────────
    status_counts = rec["stock_status"].value_counts()
    color_map = {
        "🔴 Reorder Now": "#e74c3c",
        "🟡 Monitor":     "#f39c12",
        "🟢 OK":          "#27ae60",
    }
    colors = [color_map.get(k, "gray") for k in status_counts.index]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        status_counts.values,
        labels=status_counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax.set_title("Inventory Stock Status Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "stock_status_pie.png"), dpi=150)
    plt.close()

    print("[INV] Charts saved → reports/inventory_recommendations.png + stock_status_pie.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. WHAT-IF ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def what_if_analysis(
    rec: pd.DataFrame,
    demand_change_pct: float = 20.0,
    lead_time_change: int    = 3,
    service_level_z: float   = 1.65,
) -> pd.DataFrame:
    """
    Simulate how reorder recommendations change under different scenarios.

    Useful for answering questions like:
    - "What happens if demand increases 20% during a promotion?"
    - "What if our supplier lead time increases by 3 days?"

    Args:
        rec:               Current recommendations DataFrame.
        demand_change_pct: Percentage change in demand (+20 = 20% increase).
        lead_time_change:  Extra lead-time days to simulate.
        service_level_z:   Z-score for safety stock calculation.

    Returns:
        Adjusted recommendations DataFrame.
    """
    adj          = rec.copy()
    new_lead     = LEAD_TIME_DAYS + lead_time_change
    demand_mult  = 1 + demand_change_pct / 100

    adj["adj_avg_daily"]     = adj["avg_daily_demand"] * demand_mult
    adj["adj_safety_stock"]  = (
        service_level_z * adj["std_daily_demand"] * np.sqrt(new_lead)
    ).round(0)
    adj["adj_reorder_point"] = (
        adj["adj_avg_daily"] * new_lead + adj["adj_safety_stock"]
    ).round(0)

    avg_increase = (
        (adj["adj_reorder_point"] - rec["reorder_point"])
        / (rec["reorder_point"] + 1e-8) * 100
    ).mean()

    print(
        f"\n[WHAT-IF] Demand {demand_change_pct:+.0f}%  |  "
        f"Lead time +{lead_time_change}d:"
    )
    print(f"  Avg reorder point increase: {avg_increase:.1f}%")
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cleaned = load_cleaned_data()
    stats   = compute_product_stats(cleaned)
    rec     = compute_reorder_recommendations(stats)
    plot_inventory_charts(rec)
    what_if_analysis(rec, demand_change_pct=20, lead_time_change=3)
    print("\n✅ Inventory optimisation complete!")
