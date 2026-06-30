# tests/test_pipeline.py
# RetailPulse – Unit Tests
#
# Run with:   pytest tests/test_pipeline.py -v
#
# Each test is self-contained: it generates its own tiny synthetic dataset
# rather than relying on files on disk, so the tests pass in any environment.

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Make the project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_data():
    """Small raw transaction dataset (200 rows) for testing."""
    from src.data_preprocessing import generate_synthetic_data
    return generate_synthetic_data(n_customers=50, n_transactions=200, seed=42)


@pytest.fixture(scope="module")
def cleaned_data(raw_data):
    """Cleaned version of the raw fixture."""
    from src.data_preprocessing import clean_data
    return clean_data(raw_data)


@pytest.fixture(scope="module")
def rfm_data(cleaned_data):
    """RFM scores computed from the cleaned fixture."""
    from src.data_preprocessing import build_rfm
    return build_rfm(cleaned_data)


@pytest.fixture(scope="module")
def daily_data(cleaned_data):
    """Daily sales time series from the cleaned fixture."""
    from src.data_preprocessing import build_daily_sales
    return build_daily_sales(cleaned_data)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 – Data Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestDataPreprocessing:

    def test_synthetic_data_shape(self, raw_data):
        """Synthetic data should have the expected number of rows and columns."""
        assert len(raw_data) == 200
        expected_cols = {
            "InvoiceNo", "StockCode", "Description",
            "Quantity", "InvoiceDate", "UnitPrice",
            "CustomerID", "Country",
        }
        assert expected_cols.issubset(set(raw_data.columns))

    def test_synthetic_data_has_cancellations(self, raw_data):
        """About 5% of rows should be cancellations (InvoiceNo starts with C)."""
        cancel_mask = raw_data["InvoiceNo"].astype(str).str.startswith("C")
        assert cancel_mask.sum() > 0, "Expected some cancellation rows"

    def test_clean_removes_cancellations(self, cleaned_data):
        """No InvoiceNo should start with 'C' after cleaning."""
        cancel_mask = cleaned_data["InvoiceNo"].astype(str).str.startswith("C")
        assert cancel_mask.sum() == 0

    def test_clean_removes_null_customer_ids(self, cleaned_data):
        """CustomerID must not contain any NaN values after cleaning."""
        assert cleaned_data["CustomerID"].isnull().sum() == 0

    def test_clean_adds_total_amount(self, cleaned_data):
        """TotalAmount column should be added and always positive."""
        assert "TotalAmount" in cleaned_data.columns
        assert (cleaned_data["TotalAmount"] > 0).all()

    def test_clean_output_smaller_than_raw(self, raw_data, cleaned_data):
        """Cleaning should reduce the row count (cancellations + nulls removed)."""
        assert len(cleaned_data) < len(raw_data)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 – RFM Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

class TestRFMFeatures:

    def test_rfm_one_row_per_customer(self, rfm_data, cleaned_data):
        """RFM table should have exactly one row per unique customer."""
        n_customers = cleaned_data["CustomerID"].nunique()
        assert len(rfm_data) == n_customers

    def test_rfm_required_columns(self, rfm_data):
        """RFM table must contain all expected columns."""
        expected = {
            "CustomerID", "Recency", "Frequency", "Monetary",
            "R_Score", "F_Score", "M_Score", "RFM_Score", "Segment",
        }
        assert expected.issubset(set(rfm_data.columns))

    def test_rfm_scores_in_range(self, rfm_data):
        """R, F, M scores must each be in [1, 5]."""
        for col in ["R_Score", "F_Score", "M_Score"]:
            assert rfm_data[col].between(1, 5).all(), f"{col} out of range [1,5]"

    def test_rfm_monetary_positive(self, rfm_data):
        """All customers should have positive total spend."""
        assert (rfm_data["Monetary"] > 0).all()

    def test_rfm_segment_labels(self, rfm_data):
        """Segment labels should be from the defined set of business labels."""
        valid_segments = {
            "Champions", "Loyal Customers", "Potential Loyalists",
            "At-Risk", "Lost", "Others",
        }
        actual = set(rfm_data["Segment"].unique())
        assert actual.issubset(valid_segments), f"Unexpected segments: {actual - valid_segments}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 – Daily Sales / Rolling Features
# ─────────────────────────────────────────────────────────────────────────────

class TestDailySales:

    def test_daily_has_ds_and_y(self, daily_data):
        """Daily sales must have 'ds' (date) and 'y' (revenue) columns."""
        assert "ds" in daily_data.columns
        assert "y"  in daily_data.columns

    def test_daily_sorted(self, daily_data):
        """Dates should be in ascending order."""
        assert daily_data["ds"].is_monotonic_increasing

    def test_daily_y_positive(self, daily_data):
        """All daily revenue values should be positive."""
        assert (daily_data["y"] > 0).all()

    def test_daily_has_rolling_features(self, daily_data):
        """Rolling mean and lag features must be present."""
        for col in ["rolling_7d_mean", "rolling_30d_mean", "lag_1", "lag_7"]:
            assert col in daily_data.columns, f"Missing column: {col}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 – Customer Segmentation
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentation:

    def test_kmeans_adds_cluster_column(self, rfm_data):
        """Running K-Means should add a KMeans_Cluster column to the RFM table."""
        from src.segmentation import preprocess_rfm, run_kmeans
        X_scaled, _, _ = preprocess_rfm(rfm_data)
        result, sil    = run_kmeans(X_scaled, rfm_data.copy(), k=3)
        assert "KMeans_Cluster" in result.columns

    def test_silhouette_score_valid(self, rfm_data):
        """Silhouette score should be a valid float between -1 and 1."""
        from src.segmentation import preprocess_rfm, run_kmeans
        X_scaled, _, _ = preprocess_rfm(rfm_data)
        _, sil         = run_kmeans(X_scaled, rfm_data.copy(), k=3)
        assert -1.0 <= sil <= 1.0

    def test_business_labels_assigned(self, rfm_data):
        """interpret_clusters should add a Business_Segment column."""
        from src.segmentation import preprocess_rfm, run_kmeans, interpret_clusters
        X_scaled, _, _    = preprocess_rfm(rfm_data)
        rfm_copy, _       = run_kmeans(X_scaled, rfm_data.copy(), k=3)
        labelled, summary = interpret_clusters(rfm_copy)
        assert "Business_Segment" in labelled.columns
        assert not labelled["Business_Segment"].isnull().any()


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 – Churn Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

class TestChurnFeatures:

    def test_churn_features_built(self, cleaned_data, rfm_data, tmp_path):
        """build_churn_features should produce the required columns."""
        from src.data_preprocessing import save_processed_data, build_daily_sales
        from src.feature_engineering import build_churn_features

        # Write temp CSV files that feature_engineering reads
        cleaned_path = str(tmp_path / "cleaned_retail.csv")
        rfm_path     = str(tmp_path / "rfm_scores.csv")
        cleaned_data.to_csv(cleaned_path, index=False)
        rfm_data.to_csv(rfm_path, index=False)

        # Patch config paths
        import src.feature_engineering as fe
        original_cleaned = fe.CLEANED_FILE
        original_rfm     = fe.RFM_FILE
        original_churn   = fe.CHURN_FILE
        fe.CLEANED_FILE  = cleaned_path
        fe.RFM_FILE      = rfm_path
        fe.CHURN_FILE    = str(tmp_path / "churn_features.csv")

        try:
            feats = build_churn_features(
                cleaned_path=cleaned_path,
                rfm_path=rfm_path,
            )
        finally:
            fe.CLEANED_FILE = original_cleaned
            fe.RFM_FILE     = original_rfm
            fe.CHURN_FILE   = original_churn

        required_cols = {"CustomerID", "total_revenue", "order_count", "churned"}
        assert required_cols.issubset(set(feats.columns))

    def test_churn_label_binary(self, cleaned_data, rfm_data, tmp_path):
        """churned column must be 0 or 1 only."""
        from src.feature_engineering import build_churn_features

        cleaned_path = str(tmp_path / "cleaned2.csv")
        rfm_path     = str(tmp_path / "rfm2.csv")
        cleaned_data.to_csv(cleaned_path, index=False)
        rfm_data.to_csv(rfm_path, index=False)

        import src.feature_engineering as fe
        fe.CHURN_FILE = str(tmp_path / "churn2.csv")

        feats = build_churn_features(cleaned_path=cleaned_path, rfm_path=rfm_path)
        assert set(feats["churned"].unique()).issubset({0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 – Inventory Optimisation
# ─────────────────────────────────────────────────────────────────────────────

class TestInventory:

    def test_product_stats_columns(self, cleaned_data):
        """compute_product_stats should return required statistics columns."""
        from src.inventory import compute_product_stats
        stats = compute_product_stats(cleaned_data)
        required = {
            "StockCode", "avg_daily_demand",
            "std_daily_demand", "total_revenue",
        }
        assert required.issubset(set(stats.columns))

    def test_reorder_recommendations_columns(self, cleaned_data, tmp_path):
        """compute_reorder_recommendations should return the key output columns."""
        from src.inventory import compute_product_stats, compute_reorder_recommendations
        import src.inventory as inv_mod

        # Redirect output to tmp_path so we don't pollute the real data dir
        inv_mod.INVENTORY_FILE = str(tmp_path / "inv.csv")

        stats = compute_product_stats(cleaned_data)
        rec   = compute_reorder_recommendations(stats)

        required = {"safety_stock", "reorder_point", "eoq", "stock_status"}
        assert required.issubset(set(rec.columns))

    def test_reorder_point_positive(self, cleaned_data, tmp_path):
        """All reorder points should be non-negative."""
        from src.inventory import compute_product_stats, compute_reorder_recommendations
        import src.inventory as inv_mod

        inv_mod.INVENTORY_FILE = str(tmp_path / "inv2.csv")

        stats = compute_product_stats(cleaned_data)
        rec   = compute_reorder_recommendations(stats)
        assert (rec["reorder_point"] >= 0).all()

    def test_stock_status_values(self, cleaned_data, tmp_path):
        """stock_status must only contain the three defined labels."""
        from src.inventory import compute_product_stats, compute_reorder_recommendations
        import src.inventory as inv_mod

        inv_mod.INVENTORY_FILE = str(tmp_path / "inv3.csv")

        stats = compute_product_stats(cleaned_data)
        rec   = compute_reorder_recommendations(stats)
        valid = {"🔴 Reorder Now", "🟡 Monitor", "🟢 OK"}
        assert set(rec["stock_status"].unique()).issubset(valid)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 – Utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils:

    def test_format_currency(self):
        """format_currency should produce the correct string."""
        from src.utils import format_currency
        assert format_currency(1_234_567.89) == "£1,234,568"
        assert format_currency(999.5, "$")  == "$1,000"

    def test_mape_zero_error(self):
        """MAPE should be 0 when predictions exactly match actuals."""
        from src.utils import mape
        y = np.array([100.0, 200.0, 300.0])
        assert mape(y, y) == pytest.approx(0.0, abs=1e-6)

    def test_mape_nonzero(self):
        """MAPE should be > 0 when predictions differ from actuals."""
        from src.utils import mape
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        assert mape(y_true, y_pred) > 0

    def test_load_csv_safe_missing_file(self, tmp_path):
        """load_csv_safe should raise FileNotFoundError for missing files."""
        from src.utils import load_csv_safe
        with pytest.raises(FileNotFoundError):
            load_csv_safe(str(tmp_path / "nonexistent.csv"))
