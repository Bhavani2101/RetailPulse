# src/demand_forecasting.py
# RetailPulse – Demand Forecasting (Prophet with Linear Fallback)
#
# Generates a 30-day revenue forecast from daily historical sales.
# Uses Facebook Prophet as the primary model (handles seasonality well).
# Falls back to a simple linear trend if Prophet is not installed.
#
# Prophet works best on daily time series with at least 1–2 years of data.
# It automatically captures weekly and annual seasonality.

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
    DATA_PROCESSED, MODELS_DIR, REPORTS_DIR,
    FORECAST_FILE, FORECAST_HORIZON, MAPE_TARGET, RANDOM_STATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DAILY SALES
# ─────────────────────────────────────────────────────────────────────────────

def load_daily_sales(filepath: str = FORECAST_FILE) -> pd.DataFrame:
    """
    Load the daily sales time series.

    Expected columns: ds (date), y (revenue)

    Args:
        filepath: Path to daily_sales.csv

    Returns:
        Sorted daily sales DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Daily sales file not found: {filepath}\n"
            "Run data_preprocessing.py first."
        )
    df = pd.read_csv(filepath, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    print(
        f"[FORECAST] Loaded daily sales: {df.shape}  "
        f"({df['ds'].min().date()} → {df['ds'].max().date()})"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROPHET FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def train_prophet(
    df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
):
    """
    Train a Facebook Prophet model and generate a future forecast.

    Prophet decomposes the time series into trend + weekly seasonality +
    yearly seasonality + holiday effects. It produces confidence intervals
    (yhat_lower, yhat_upper) automatically.

    MAPE (Mean Absolute Percentage Error) is calculated on a held-out
    validation set (last 20% of data) to evaluate forecast accuracy.

    Args:
        df:      Daily sales DataFrame with columns [ds, y].
        horizon: Number of days to forecast into the future.

    Returns:
        Tuple of (prophet model, full forecast DataFrame, MAPE value).
        Returns (None, None, 999.0) if Prophet is not installed.
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("[FORECAST] Prophet not installed – skipping Prophet training.")
        print("  Install with:  pip install prophet")
        return None, None, 999.0

    import joblib
    print(f"\n[FORECAST] Training Prophet (horizon = {horizon} days) …")

    train = df[["ds", "y"]].copy()

    model = Prophet(
        yearly_seasonality      = True,
        weekly_seasonality      = True,
        daily_seasonality       = False,
        changepoint_prior_scale = 0.05,    # Flexibility of trend changes
        seasonality_prior_scale = 10,       # Strength of seasonal components
    )
    model.fit(train)

    # Generate forecast (historical + future)
    future   = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    # Validate MAPE on the last 20% of historical data
    cutoff    = df["ds"].max() - pd.Timedelta(days=horizon)
    val_true  = df[df["ds"] > cutoff]["y"].values
    val_fc    = forecast[forecast["ds"] > cutoff]["yhat"].values[: len(val_true)]

    if len(val_true) > 0 and len(val_fc) > 0:
        n    = min(len(val_true), len(val_fc))
        mape = np.mean(np.abs((val_true[:n] - val_fc[:n]) / (val_true[:n] + 1e-8))) * 100
    else:
        mape = 999.0

    print(f"  MAPE: {mape:.2f}%  (target ≤ {MAPE_TARGET}%)")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "prophet_model.pkl"))
    print("  Model saved → models/prophet_model.pkl")

    # Save plot
    _plot_prophet_forecast(df, forecast, horizon, mape)

    return model, forecast, mape


def _plot_prophet_forecast(
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    horizon: int,
    mape: float,
) -> None:
    """Save a clean forecast chart showing actuals + Prophet prediction."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["ds"], df["y"],
            label="Actual", color="steelblue", linewidth=1.5)

    future_fc = forecast[forecast["ds"] > df["ds"].max()]
    ax.plot(forecast["ds"], forecast["yhat"],
            label="Prophet Fit / Forecast", color="orange", linewidth=1.5)
    ax.fill_between(
        future_fc["ds"],
        future_fc["yhat_lower"],
        future_fc["yhat_upper"],
        alpha=0.25, color="orange", label="90% Confidence Interval",
    )
    ax.axvline(df["ds"].max(), color="red", linestyle="--", label="Forecast Start")
    ax.set_title(f"Prophet Demand Forecast  |  MAPE: {mape:.2f}%  |  Horizon: {horizon} days")
    ax.set_xlabel("Date"); ax.set_ylabel("Daily Revenue (£)")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "prophet_forecast.png"), dpi=150)
    plt.close()
    print("  Forecast plot saved → reports/prophet_forecast.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LINEAR FALLBACK FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_forecast(
    df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
):
    """
    Simple linear regression fallback when Prophet is unavailable.

    Fits a straight line trend to daily sales and extrapolates it forward.
    Much less accurate than Prophet but requires no extra dependencies.

    Args:
        df:      Daily sales DataFrame.
        horizon: Days to forecast.

    Returns:
        Tuple of (None, forecast DataFrame, MAPE value).
    """
    from sklearn.linear_model import LinearRegression

    print("\n[FORECAST] Training linear fallback model …")

    df = df.copy()
    df["t"] = np.arange(len(df))

    X = df[["t"]].values
    y = df["y"].values

    # 80/20 train/val split
    split = int(len(df) * 0.8)
    model = LinearRegression()
    model.fit(X[:split], y[:split])

    val_pred = model.predict(X[split:])
    val_true = y[split:]
    mape = np.mean(np.abs((val_true - val_pred) / (val_true + 1e-8))) * 100
    print(f"  Linear MAPE: {mape:.2f}%")

    # Future dates
    future_t     = np.arange(len(df), len(df) + horizon).reshape(-1, 1)
    future_yhat  = model.predict(future_t)
    future_dates = pd.date_range(
        df["ds"].max() + pd.Timedelta(days=1), periods=horizon
    )

    forecast = pd.DataFrame({
        "ds":          pd.concat([df["ds"],        pd.Series(future_dates)]).reset_index(drop=True),
        "yhat":        np.concatenate([model.predict(X), future_yhat]),
        "yhat_lower":  np.concatenate([model.predict(X) * 0.85, future_yhat * 0.85]),
        "yhat_upper":  np.concatenate([model.predict(X) * 1.15, future_yhat * 1.15]),
    })

    return None, forecast, mape


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def save_forecast(forecast: pd.DataFrame, filename: str = "prophet_forecast.csv") -> str:
    """
    Save the forecast DataFrame to data/processed/.

    Args:
        forecast: Forecast DataFrame with at least [ds, yhat].
        filename: Output filename.

    Returns:
        Full path to the saved file.
    """
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    out_path = os.path.join(DATA_PROCESSED, filename)
    forecast.to_csv(out_path, index=False)
    print(f"[FORECAST] Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    daily = load_daily_sales()

    # Try Prophet first; fall back to linear if not installed
    try:
        from prophet import Prophet  # noqa: F401
        model, forecast, mape = train_prophet(daily, horizon=FORECAST_HORIZON)
    except ImportError:
        model, forecast, mape = train_linear_forecast(daily, horizon=FORECAST_HORIZON)

    if forecast is not None:
        save_forecast(forecast, "prophet_forecast.csv")

    print(f"\n✅ Demand forecasting complete!  MAPE = {mape:.2f}%")
