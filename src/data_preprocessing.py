# src/data_preprocessing.py
# RetailPulse – Data Loading, Cleaning & Feature Engineering
#
# This module handles:
#   1. Generating synthetic retail data (so the app works out-of-the-box)
#   2. Loading real data from the UCI Online Retail dataset
#   3. Cleaning the raw data (remove cancellations, nulls, duplicates)
#   4. Building RFM (Recency, Frequency, Monetary) features per customer
#   5. Building rolling time-series features for forecasting

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Allow imports from the project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_RAW, DATA_PROCESSED,
    RAW_FILE, CLEANED_FILE, RFM_FILE, FORECAST_FILE,
    RANDOM_STATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(filepath: str = RAW_FILE) -> pd.DataFrame:
    """
    Load the Online Retail II dataset from a CSV file.

    Download the dataset from:
    https://archive.ics.uci.edu/dataset/502/online+retail+ii
    Save it as: data/raw/online_retail.csv

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        A pandas DataFrame with raw transaction records.
    """
    print(f"[DATA] Loading raw data from: {filepath}")
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    print(f"[DATA] Raw shape: {df.shape}")
    return df


def generate_synthetic_data(
    n_customers: int = 1000,
    n_transactions: int = 50_000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate realistic synthetic retail transaction data.

    This lets the app run without downloading the real dataset.
    The synthetic data mimics the Online Retail II schema including:
    - InvoiceNo, StockCode, Description, Quantity
    - InvoiceDate, UnitPrice, CustomerID, Country
    - ~5% cancellations (InvoiceNo starts with 'C')
    - ~2% missing CustomerIDs

    Args:
        n_customers:    Number of unique customers to simulate.
        n_transactions: Total number of transaction rows.
        seed:           Random seed for reproducibility.

    Returns:
        A pandas DataFrame in Online Retail format.
    """
    print("[DATA] Generating synthetic retail data …")
    np.random.seed(seed)

    date_range = pd.date_range("2022-01-01", "2024-12-31", freq="D")

    invoice_nos  = [f"INV{str(i).zfill(6)}" for i in range(1, n_transactions + 1)]
    customer_ids = np.random.randint(10_000, 10_000 + n_customers, n_transactions)
    stock_codes  = [f"SC{np.random.randint(1000, 9999)}" for _ in range(n_transactions)]
    descriptions = np.random.choice(
        ["Widget A", "Widget B", "Gadget X", "Tool Y",
         "Product Z", "Item Alpha", "Item Beta"],
        n_transactions,
    )
    quantities   = np.random.randint(1, 50, n_transactions)
    unit_prices  = np.round(np.random.uniform(0.5, 50.0, n_transactions), 2)
    invoice_dates = np.random.choice(date_range, n_transactions)
    countries    = np.random.choice(
        ["United Kingdom", "Germany", "France", "Spain", "Australia"],
        n_transactions,
        p=[0.70, 0.10, 0.10, 0.05, 0.05],
    )

    df = pd.DataFrame({
        "InvoiceNo":   invoice_nos,
        "StockCode":   stock_codes,
        "Description": descriptions,
        "Quantity":    quantities,
        "InvoiceDate": pd.to_datetime(invoice_dates),
        "UnitPrice":   unit_prices,
        "CustomerID":  customer_ids.astype(float),
        "Country":     countries,
    })

    # Inject ~5% cancellations (negative qty, InvoiceNo starts with 'C')
    cancel_idx = np.random.choice(df.index, int(n_transactions * 0.05), replace=False)
    df.loc[cancel_idx, "InvoiceNo"] = "C" + df.loc[cancel_idx, "InvoiceNo"]
    df.loc[cancel_idx, "Quantity"]  = -df.loc[cancel_idx, "Quantity"]

    # Inject ~2% missing CustomerIDs (guest checkouts)
    missing_idx = np.random.choice(df.index, int(n_transactions * 0.02), replace=False)
    df.loc[missing_idx, "CustomerID"] = np.nan

    print(f"[DATA] Synthetic data shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw retail transaction data.

    Steps performed:
    - Parse InvoiceDate as datetime
    - Remove cancellation rows (InvoiceNo starts with 'C')
    - Drop rows with missing CustomerID
    - Remove duplicate rows
    - Remove rows with non-positive Quantity or UnitPrice
    - Cast CustomerID to integer
    - Add derived columns: TotalAmount, Year, Month, DayOfWeek, Date

    Args:
        df: Raw transactions DataFrame.

    Returns:
        Cleaned DataFrame with additional derived columns.
    """
    print("\n[CLEAN] Starting cleaning pipeline …")
    original_rows = len(df)

    # Ensure InvoiceDate is datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Remove cancellations
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    print(f"  → After removing cancellations:      {len(df):,} rows")

    # Drop rows with missing CustomerID (guest checkouts)
    df = df.dropna(subset=["CustomerID"])
    print(f"  → After dropping null CustomerID:    {len(df):,} rows")

    # Remove exact duplicates
    df = df.drop_duplicates()
    print(f"  → After dropping duplicates:         {len(df):,} rows")

    # Keep only valid transactions (positive quantity and price)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    print(f"  → After removing invalid Qty/Price:  {len(df):,} rows")

    # Cast CustomerID to integer for cleaner display
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Add useful derived columns
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    df["Year"]        = df["InvoiceDate"].dt.year
    df["Month"]       = df["InvoiceDate"].dt.month
    df["DayOfWeek"]   = df["InvoiceDate"].dt.dayofweek
    df["Date"]        = df["InvoiceDate"].dt.date

    removed = original_rows - len(df)
    print(f"[CLEAN] Done. Removed {removed:,} rows ({removed / original_rows * 100:.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. RFM FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Build RFM (Recency, Frequency, Monetary) features per customer.

    RFM is a classic marketing framework:
    - Recency   – how recently a customer bought (lower = better)
    - Frequency – how often they buy (higher = better)
    - Monetary  – how much they spend (higher = better)

    Each dimension is scored 1–5 and combined into an RFM_Score.
    Customers are then labelled into business segments.

    Args:
        df:            Cleaned transactions DataFrame.
        snapshot_date: Reference date for recency calculation.
                       Defaults to (max date + 1 day).

    Returns:
        DataFrame with one row per customer, including RFM scores and segment.
    """
    print("\n[RFM] Building RFM features …")

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency   =("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency =("InvoiceNo",   "nunique"),
        Monetary  =("TotalAmount", "sum"),
    ).reset_index()

    # Score 1–5: higher is always better (Recency is inverted – lower days = higher score)
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"
    ).astype(int)
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype(int)
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype(int)

    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    # Business segment labels based on RFM scores
    def _assign_segment(row):
        if row["RFM_Score"] >= 13:
            return "Champions"
        elif row["R_Score"] >= 4 and row["F_Score"] >= 3:
            return "Loyal Customers"
        elif row["R_Score"] >= 3 and row["F_Score"] == 1:
            return "Potential Loyalists"
        elif row["R_Score"] <= 2 and row["F_Score"] >= 3:
            return "At-Risk"
        elif row["R_Score"] == 1 and row["F_Score"] == 1:
            return "Lost"
        else:
            return "Others"

    rfm["Segment"] = rfm.apply(_assign_segment, axis=1)

    print(f"[RFM] Segment breakdown:\n{rfm['Segment'].value_counts().to_string()}")
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROLLING TIME-SERIES FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactions into daily total revenue and add rolling features.

    Rolling features help the forecasting model detect trends and seasonality:
    - 7-day rolling mean  (short-term trend)
    - 30-day rolling mean (medium-term trend)
    - Lag features: lag_1, lag_7, lag_30 (yesterday, last week, last month)

    Args:
        df: Cleaned transactions DataFrame.

    Returns:
        Daily sales DataFrame with columns: ds, y, rolling_7d_mean, …
    """
    print("\n[DAILY] Building daily sales features …")

    daily = (
        df.groupby("Date")["TotalAmount"]
        .sum()
        .reset_index()
        .rename(columns={"Date": "ds", "TotalAmount": "y"})
    )
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)

    daily["rolling_7d_mean"]  = daily["y"].rolling(7,  min_periods=1).mean()
    daily["rolling_30d_mean"] = daily["y"].rolling(30, min_periods=1).mean()
    daily["rolling_7d_std"]   = daily["y"].rolling(7,  min_periods=1).std().fillna(0)
    daily["lag_1"]            = daily["y"].shift(1).bfill()
    daily["lag_7"]            = daily["y"].shift(7).bfill()
    daily["lag_30"]           = daily["y"].shift(30).bfill()

    print(f"[DAILY] Daily sales shape: {daily.shape}")
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_processed_data(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    """
    Save all processed files to the data/processed directory.

    Args:
        df:    Cleaned transactions.
        rfm:   RFM scores per customer.
        daily: Daily sales with rolling features.
    """
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    df.to_csv(CLEANED_FILE,  index=False)
    rfm.to_csv(RFM_FILE,     index=False)
    daily.to_csv(FORECAST_FILE, index=False)
    print(f"\n[SAVE] Files saved to: {DATA_PROCESSED}/")
    print(f"  cleaned_retail.csv  → {len(df):,} rows")
    print(f"  rfm_scores.csv      → {len(rfm):,} customers")
    print(f"  daily_sales.csv     → {len(daily):,} days")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run standalone to generate processed data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(DATA_RAW, exist_ok=True)

    # Use synthetic data by default.
    # Replace generate_synthetic_data() with load_raw_data() once you have the CSV.
    raw   = generate_synthetic_data(n_customers=1000, n_transactions=50_000)
    clean = clean_data(raw)
    rfm   = build_rfm(clean)
    daily = build_daily_sales(clean)

    save_processed_data(clean, rfm, daily)
    print("\n✅ Data preprocessing complete!")
