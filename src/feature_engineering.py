# src/feature_engineering.py
# RetailPulse – Feature Engineering for Churn Prediction
#
# Builds customer-level behavioural features used to train the churn model.
# Merges transaction-level aggregates with RFM scores into one feature table.

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_PROCESSED,
    CLEANED_FILE, RFM_FILE, CHURN_FILE,
    CHURN_DAYS,
)


def build_churn_features(
    cleaned_path: str = CLEANED_FILE,
    rfm_path: str     = RFM_FILE,
    churn_days: int   = CHURN_DAYS,
) -> pd.DataFrame:
    """
    Build a feature table for churn prediction.

    Strategy:
    - Split the timeline at (max_date - churn_days)
    - Train features are computed on the earlier window
    - The churn label is 1 if the customer did NOT appear in the later window

    Features created per customer:
    - total_revenue, avg_order_value, order_count, unique_products
    - avg_qty, std_qty, recency_days, tenure_days
    - purchase_freq_30d (orders in last 30 days of the training window)
    - R_Score, F_Score, M_Score, RFM_Score (from RFM table)
    - churned (0 = still active, 1 = churned)

    Args:
        cleaned_path: Path to cleaned_retail.csv
        rfm_path:     Path to rfm_scores.csv
        churn_days:   Days of inactivity that defines a churned customer.

    Returns:
        Feature DataFrame ready for model training.

    Raises:
        FileNotFoundError: If the input CSV files do not exist.
    """
    # Validate input files
    for path in [cleaned_path, rfm_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run data_preprocessing.py first."
            )

    print("[FEAT] Building churn features …")
    df  = pd.read_csv(cleaned_path, parse_dates=["InvoiceDate"])
    rfm = pd.read_csv(rfm_path)

    # Define the train/holdout cutoff
    snapshot = df["InvoiceDate"].max()
    cutoff   = snapshot - pd.Timedelta(days=churn_days)

    train_df   = df[df["InvoiceDate"] <= cutoff]
    holdout_df = df[df["InvoiceDate"] >  cutoff]
    active_customers = set(holdout_df["CustomerID"].unique())

    print(f"  Snapshot : {snapshot.date()}")
    print(f"  Cutoff   : {cutoff.date()} (churn window = {churn_days} days)")
    print(f"  Train rows: {len(train_df):,} | Holdout rows: {len(holdout_df):,}")

    # Aggregate behavioural features from the training window
    features = (
        train_df.groupby("CustomerID").agg(
            total_revenue     =("TotalAmount",  "sum"),
            avg_order_value   =("TotalAmount",  "mean"),
            order_count       =("InvoiceNo",    "nunique"),
            unique_products   =("StockCode",    "nunique"),
            avg_qty           =("Quantity",     "mean"),
            std_qty           =("Quantity",     "std"),
            recency_days      =("InvoiceDate",
                                lambda x: (cutoff - x.max()).days),
            tenure_days       =("InvoiceDate",
                                lambda x: (x.max() - x.min()).days),
            purchase_freq_30d =("InvoiceDate",
                                lambda x: (
                                    (x >= cutoff - pd.Timedelta(days=30)) &
                                    (x <= cutoff)
                                ).sum()),
        )
        .reset_index()
        .fillna(0)
    )

    # Merge in RFM scores
    rfm_cols = ["CustomerID", "R_Score", "F_Score", "M_Score", "RFM_Score"]
    rfm_cols = [c for c in rfm_cols if c in rfm.columns]
    features = features.merge(rfm[rfm_cols], on="CustomerID", how="left")

    # Create churn label
    features["churned"] = (~features["CustomerID"].isin(active_customers)).astype(int)

    churn_rate = features["churned"].mean() * 100
    print(f"  Total customers : {len(features):,}")
    print(f"  Churned         : {features['churned'].sum():,} ({churn_rate:.1f}%)")

    # Save to file
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    features.to_csv(CHURN_FILE, index=False)
    print(f"[FEAT] Saved → {CHURN_FILE}")

    return features


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    features = build_churn_features()
    print("\n✅ Feature engineering complete!")
    print(features.head(3).to_string())
