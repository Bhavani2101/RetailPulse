# src/churn_prediction.py
# RetailPulse – Churn Prediction (XGBoost)
#
# Trains an XGBoost binary classifier to predict which customers will churn.
# Includes:
#   - Train/test split with stratification (preserves class balance)
#   - XGBoost with sensible default hyperparameters
#   - Model evaluation: AUC-ROC, Precision@20%, confusion matrix
#   - Plots: ROC curve, confusion matrix heatmap, feature importance
#   - Saving model + feature list with joblib
#   - Scoring all customers with churn probability + risk label

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_PROCESSED, MODELS_DIR, REPORTS_DIR,
    CHURN_FILE, RANDOM_STATE, AUC_TARGET,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAIN CHURN MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_churn_model(features_path: str = CHURN_FILE):
    """
    Train an XGBoost classifier to predict customer churn.

    Steps:
    1. Load churn feature table
    2. Split into 80% train / 20% test (stratified by churn label)
    3. Train XGBoost with balanced class weights
    4. Evaluate on test set: AUC-ROC + Precision@20%
    5. Save model and feature column list to models/

    Args:
        features_path: Path to churn_features.csv

    Returns:
        Tuple of (trained model, feature column names, AUC-ROC score)

    Raises:
        FileNotFoundError: If churn_features.csv does not exist.
        ImportError:       If xgboost is not installed.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError(
            "xgboost is required for churn prediction.\n"
            "Install it with:  pip install xgboost"
        )

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Churn features not found: {features_path}\n"
            "Run feature_engineering.py first."
        )

    print("[CHURN] Loading churn features …")
    feats = pd.read_csv(features_path)

    # Separate features (X) and target (y)
    target_col   = "churned"
    exclude_cols = ["CustomerID", target_col]
    feature_cols = [c for c in feats.columns if c not in exclude_cols]

    X = feats[feature_cols].fillna(0)
    y = feats[target_col]

    # Stratified train/test split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Churn rate (train): {y_train.mean()*100:.1f}%")

    # XGBoost parameters – beginner-friendly defaults
    params = {
        "n_estimators":     300,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric":      "auc",
        "random_state":     RANDOM_STATE,
        # Balance classes: scale_pos_weight = negatives / positives
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
    }

    print("\n[CHURN] Training XGBoost …")
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluation ─────────────────────────────────────────────────────────
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    auc           = roc_auc_score(y_test, y_pred_proba)
    prec_at_20pct = _precision_at_k(y_test, y_pred_proba, k=0.20)

    print(f"\n[CHURN] Results:")
    print(f"  AUC-ROC        : {auc:.4f}  (target ≥ {AUC_TARGET})")
    print(f"  Precision@20%  : {prec_at_20pct:.4f}  (target ≥ 0.75)")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Active','Churned'])}")

    # ── Save model ─────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,       os.path.join(MODELS_DIR, "churn_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "churn_features.pkl"))
    print("  Model saved → models/churn_model.pkl")

    # ── Plots ──────────────────────────────────────────────────────────────
    _plot_roc_curve(model, X_test, y_test, auc)
    _plot_confusion_matrix(y_test, y_pred)
    _plot_feature_importance(model, feature_cols)

    return model, feature_cols, auc


# ─────────────────────────────────────────────────────────────────────────────
# 2. EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _precision_at_k(y_true: pd.Series, y_proba: np.ndarray, k: float = 0.20) -> float:
    """
    Precision@K: Of the top K% of customers ranked by churn probability,
    what fraction actually churned?

    This is a business-relevant metric: it tells us how accurate the model
    is when we act on only the riskiest customers.

    Args:
        y_true:  True churn labels.
        y_proba: Predicted churn probabilities.
        k:       Top fraction to evaluate (0.20 = top 20%).

    Returns:
        Precision within the top K% of predictions.
    """
    n       = max(1, int(len(y_true) * k))
    top_idx = np.argsort(y_proba)[::-1][:n]
    return y_true.iloc[top_idx].mean()


def _plot_roc_curve(model, X_test, y_test, auc: float) -> None:
    """Plot and save the ROC curve."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve  |  AUC = {auc:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print("  ROC curve saved → reports/roc_curve.png")


def _plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Plot and save the confusion matrix as a heatmap."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Active", "Churned"])
    ax.set_yticklabels(["Active", "Churned"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14,
            )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  Confusion matrix saved → reports/confusion_matrix.png")


def _plot_feature_importance(model, feature_cols: list) -> None:
    """Plot and save XGBoost feature importances (gain-based)."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1][:15]  # Top 15 features

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_cols[i] for i in sorted_idx[::-1]],
        importances[sorted_idx[::-1]],
        color="steelblue",
    )
    ax.set_title("Top 15 Features – XGBoost Gain Importance")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("  Feature importance saved → reports/feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCORE ALL CUSTOMERS
# ─────────────────────────────────────────────────────────────────────────────

def score_all_customers(
    feats: pd.DataFrame,
    model,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Assign churn probability and risk label to every customer.

    Risk labels:
    - High   : churn_probability ≥ 0.6
    - Medium : 0.3 ≤ churn_probability < 0.6
    - Low    : churn_probability < 0.3

    The result is saved as data/processed/churn_scores.csv.

    Args:
        feats:        Churn feature DataFrame (one row per customer).
        model:        Trained XGBoost model.
        feature_cols: List of feature column names used during training.

    Returns:
        DataFrame with churn_probability and churn_risk_label per customer.
    """
    X = feats[feature_cols].fillna(0)
    feats = feats.copy()
    feats["churn_probability"] = model.predict_proba(X)[:, 1]
    feats["churn_risk_label"]  = pd.cut(
        feats["churn_probability"],
        bins=[0.0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    out_cols  = ["CustomerID", "churn_probability", "churn_risk_label", "churned"]
    out_cols  = [c for c in out_cols if c in feats.columns]
    out_path  = os.path.join(DATA_PROCESSED, "churn_scores.csv")
    feats[out_cols].to_csv(out_path, index=False)

    print(f"\n[CHURN] Risk distribution:\n{feats['churn_risk_label'].value_counts().to_string()}")
    print(f"  Churn scores saved → {out_path}")
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    model, feature_cols, auc = train_churn_model()

    # Score all customers
    feats = pd.read_csv(CHURN_FILE)
    score_all_customers(feats, model, feature_cols)

    print(f"\n✅ Churn prediction complete!  AUC-ROC = {auc:.4f}")
