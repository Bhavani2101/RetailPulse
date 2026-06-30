# config.py
# RetailPulse – Central Configuration
# All project-wide constants live here so every module stays in sync.

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")

# ── Data Files ─────────────────────────────────────────────────────────────────
RAW_FILE        = os.path.join(DATA_RAW, "online_retail.csv")
CLEANED_FILE    = os.path.join(DATA_PROCESSED, "cleaned_retail.csv")
RFM_FILE        = os.path.join(DATA_PROCESSED, "rfm_scores.csv")
FORECAST_FILE   = os.path.join(DATA_PROCESSED, "daily_sales.csv")
CHURN_FILE      = os.path.join(DATA_PROCESSED, "churn_features.csv")
INVENTORY_FILE  = os.path.join(DATA_PROCESSED, "inventory_recommendations.csv")

# ── Segmentation ───────────────────────────────────────────────────────────────
N_CLUSTERS      = 6       # Default number of K-Means clusters
RANDOM_STATE    = 42      # Reproducibility seed

# ── Forecasting ────────────────────────────────────────────────────────────────
FORECAST_HORIZON = 30     # How many days ahead to forecast
MAPE_TARGET      = 12.0   # Target Mean Absolute Percentage Error (%)

# ── Churn Prediction ───────────────────────────────────────────────────────────
CHURN_DAYS  = 90          # Days of inactivity before a customer is labelled churned
AUC_TARGET  = 0.88        # Target AUC-ROC score

# ── Inventory Optimisation ─────────────────────────────────────────────────────
SAFETY_STOCK_MULTIPLIER = 1.5
LEAD_TIME_DAYS          = 7   # Supplier lead time in days
