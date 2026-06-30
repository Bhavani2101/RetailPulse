# run_pipeline.py
# RetailPulse – Master Pipeline Runner
#
# Runs every stage of the RetailPulse pipeline in the correct order:
#   Step 1 → Data preprocessing (load, clean, RFM, daily sales)
#   Step 2 → Customer segmentation (K-Means)
#   Step 3 → Feature engineering (churn features)
#   Step 4 → Churn prediction (XGBoost)
#   Step 5 → Demand forecasting (Prophet)
#   Step 6 → Inventory optimisation (EOQ + safety stock)
#
# Usage:
#   python run_pipeline.py                  # full run with synthetic data
#   python run_pipeline.py --real-data      # use data/raw/online_retail.csv
#   python run_pipeline.py --skip-churn     # skip XGBoost (if xgboost not installed)
#   python run_pipeline.py --skip-forecast  # skip Prophet (if prophet not installed)
#
# After this script completes, launch the dashboard:
#   streamlit run app.py

import argparse
import os
import sys
import time

# ── Make project root importable ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _banner(step: int, title: str) -> None:
    """Print a formatted step header to the console."""
    print(f"\n{'='*60}")
    print(f"  STEP {step}: {title}")
    print(f"{'='*60}")


def run_pipeline(use_real_data: bool, skip_churn: bool, skip_forecast: bool) -> None:
    """
    Execute the full RetailPulse ML pipeline.

    Args:
        use_real_data:   If True, load data/raw/online_retail.csv instead
                         of generating synthetic data.
        skip_churn:      If True, skip churn model training (e.g. xgboost
                         not installed).
        skip_forecast:   If True, skip Prophet training (e.g. prophet not
                         installed; a linear fallback will be used instead).
    """
    start_time = time.time()

    # ── STEP 1: Data Preprocessing ────────────────────────────────────────────
    _banner(1, "Data Preprocessing")
    from src.data_preprocessing import (
        load_raw_data,
        generate_synthetic_data,
        clean_data,
        build_rfm,
        build_daily_sales,
        save_processed_data,
    )
    from config import DATA_RAW, RAW_FILE
    import os

    os.makedirs(DATA_RAW, exist_ok=True)

    if use_real_data:
        if not os.path.exists(RAW_FILE):
            print(
                f"\n[ERROR] Real data not found at: {RAW_FILE}\n"
                "Download the Online Retail II dataset from:\n"
                "  https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
                "Save it as: data/raw/online_retail.csv\n"
                "Then re-run this script."
            )
            sys.exit(1)
        raw = load_raw_data(RAW_FILE)
    else:
        raw = generate_synthetic_data(n_customers=1000, n_transactions=50_000)

    cleaned = clean_data(raw)
    rfm     = build_rfm(cleaned)
    daily   = build_daily_sales(cleaned)
    save_processed_data(cleaned, rfm, daily)
    print("[STEP 1] ✅ Data preprocessing complete")

    # ── STEP 2: Customer Segmentation ─────────────────────────────────────────
    _banner(2, "Customer Segmentation (K-Means)")
    from src.segmentation import (
        load_rfm,
        preprocess_rfm,
        find_optimal_k,
        run_kmeans,
        plot_clusters_pca,
        interpret_clusters,
    )
    from config import DATA_PROCESSED
    import os

    rfm_df            = load_rfm()
    X_scaled, sc, _   = preprocess_rfm(rfm_df)
    best_k            = find_optimal_k(X_scaled, k_range=range(2, 9))
    rfm_df, sil_score = run_kmeans(X_scaled, rfm_df, k=best_k)
    plot_clusters_pca(X_scaled, rfm_df["KMeans_Cluster"].values, "KMeans Clusters")
    rfm_df, cluster_summary = interpret_clusters(rfm_df)

    rfm_df.to_csv(os.path.join(DATA_PROCESSED, "segmented_customers.csv"), index=False)
    cluster_summary.to_csv(os.path.join(DATA_PROCESSED, "cluster_summary.csv"))
    print(f"[STEP 2] ✅ Segmentation complete  |  Silhouette = {sil_score:.4f}")

    # ── STEP 3: Feature Engineering ───────────────────────────────────────────
    _banner(3, "Feature Engineering (Churn Features)")
    from src.feature_engineering import build_churn_features

    churn_features = build_churn_features()
    print("[STEP 3] ✅ Feature engineering complete")

    # ── STEP 4: Churn Prediction ──────────────────────────────────────────────
    _banner(4, "Churn Prediction (XGBoost)")
    if skip_churn:
        print("[STEP 4] ⏭  Skipped (--skip-churn flag set)")
    else:
        try:
            from src.churn_prediction import train_churn_model, score_all_customers
            import pandas as pd
            from config import CHURN_FILE

            model, feature_cols, auc = train_churn_model()
            feats = pd.read_csv(CHURN_FILE)
            score_all_customers(feats, model, feature_cols)
            print(f"[STEP 4] ✅ Churn model complete  |  AUC-ROC = {auc:.4f}")
        except ImportError as e:
            print(f"[STEP 4] ⚠️  Skipped – missing dependency: {e}")
            print("         Install with:  pip install xgboost")

    # ── STEP 5: Demand Forecasting ────────────────────────────────────────────
    _banner(5, "Demand Forecasting (Prophet)")
    if skip_forecast:
        print("[STEP 5] ⏭  Skipped (--skip-forecast flag set)")
    else:
        from src.demand_forecasting import (
            load_daily_sales,
            train_prophet,
            train_linear_forecast,
            save_forecast,
        )

        daily_df = load_daily_sales()

        try:
            from prophet import Prophet  # noqa: F401
            _, fc, mape = train_prophet(daily_df)
            label = "Prophet"
        except ImportError:
            print("[STEP 5] Prophet not found – using linear fallback")
            print("         Install Prophet with:  pip install prophet")
            _, fc, mape = train_linear_forecast(daily_df)
            label = "Linear"

        if fc is not None:
            save_forecast(fc, "prophet_forecast.csv")
        print(f"[STEP 5] ✅ Forecast complete  |  {label} MAPE = {mape:.2f}%")

    # ── STEP 6: Inventory Optimisation ────────────────────────────────────────
    _banner(6, "Inventory Optimisation (EOQ + Safety Stock)")
    from src.inventory import (
        load_cleaned_data,
        compute_product_stats,
        compute_reorder_recommendations,
        plot_inventory_charts,
    )

    cleaned_df = load_cleaned_data()
    stats      = compute_product_stats(cleaned_df)
    rec        = compute_reorder_recommendations(stats)
    plot_inventory_charts(rec)
    print("[STEP 6] ✅ Inventory optimisation complete")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ✅ RetailPulse pipeline complete in {elapsed:.1f}s")
    print(f"{'='*60}")
    print("\nNext step → launch the dashboard:")
    print("  streamlit run app.py\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RetailPulse – End-to-End ML Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use data/raw/online_retail.csv instead of synthetic data",
    )
    parser.add_argument(
        "--skip-churn",
        action="store_true",
        help="Skip XGBoost churn model training",
    )
    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help="Skip Prophet demand forecasting",
    )
    args = parser.parse_args()

    run_pipeline(
        use_real_data  = args.real_data,
        skip_churn     = args.skip_churn,
        skip_forecast  = args.skip_forecast,
    )
