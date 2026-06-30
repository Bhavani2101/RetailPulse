# src/segmentation.py
# RetailPulse – Customer Segmentation (K-Means)
#
# Segments customers into business groups using RFM features + K-Means clustering.
# Includes:
#   - Log-transform + StandardScaler preprocessing
#   - Elbow method + Silhouette score to choose best K
#   - K-Means clustering with joblib model saving
#   - PCA visualisation (2D scatter plot)
#   - Business interpretation of each cluster

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_PROCESSED, MODELS_DIR, REPORTS_DIR,
    RFM_FILE, RANDOM_STATE, N_CLUSTERS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def load_rfm(filepath: str = RFM_FILE) -> pd.DataFrame:
    """
    Load the RFM scores table.

    Args:
        filepath: Path to rfm_scores.csv

    Returns:
        RFM DataFrame with one row per customer.

    Raises:
        FileNotFoundError: If the RFM file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"RFM file not found: {filepath}\n"
            "Run data_preprocessing.py first."
        )
    df = pd.read_csv(filepath)
    print(f"[SEG] Loaded RFM: {df.shape}")
    return df


def preprocess_rfm(rfm: pd.DataFrame):
    """
    Prepare RFM features for clustering.

    Applies log1p transform to reduce skew in Frequency and Monetary,
    then standardises all three features to zero mean and unit variance.

    Args:
        rfm: RFM DataFrame (must contain Recency, Frequency, Monetary columns).

    Returns:
        Tuple of (X_scaled, scaler, feature_names)
    """
    feature_cols = ["Recency", "Frequency", "Monetary"]
    X = rfm[feature_cols].copy()

    # Log-transform skewed distributions before scaling
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"]  = np.log1p(X["Monetary"])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIND OPTIMAL NUMBER OF CLUSTERS (K)
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11)) -> int:
    """
    Use the Elbow method + Silhouette score to find the best K.

    The Elbow method plots inertia (within-cluster sum of squares) vs K.
    The Silhouette score measures how well-separated clusters are (higher = better).
    We pick K with the highest silhouette score.

    A PNG chart is saved to the reports directory.

    Args:
        X_scaled: Scaled feature matrix.
        k_range:  Range of K values to try.

    Returns:
        The best K (integer).
    """
    inertias    = []
    silhouettes = []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Plot elbow + silhouette side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "bo-", linewidth=1.5)
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), silhouettes, "rs-", linewidth=1.5)
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Score")

    plt.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(REPORTS_DIR, "elbow_silhouette.png"), dpi=150)
    plt.close()

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"[SEG] Best K (silhouette): {best_k}  |  Score: {max(silhouettes):.4f}")
    return best_k


# ─────────────────────────────────────────────────────────────────────────────
# 3. K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_kmeans(
    X_scaled: np.ndarray,
    rfm: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """
    Fit K-Means and assign cluster labels to each customer.

    The trained model is saved to models/kmeans.pkl for reuse.

    Args:
        X_scaled: Scaled feature matrix.
        rfm:      RFM DataFrame (will receive a new KMeans_Cluster column).
        k:        Number of clusters to create.

    Returns:
        RFM DataFrame with a KMeans_Cluster column added.
    """
    print(f"\n[SEG] Running K-Means with k={k} …")

    km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)
    print(f"  Silhouette Score : {sil_score:.4f}  (target ≥ 0.35)")

    rfm["KMeans_Cluster"] = labels

    # Save trained model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(km, os.path.join(MODELS_DIR, "kmeans.pkl"))
    print(f"  Model saved → models/kmeans.pkl")

    return rfm, sil_score


# ─────────────────────────────────────────────────────────────────────────────
# 4. PCA VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_clusters_pca(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    title: str = "KMeans Clusters",
) -> None:
    """
    Reduce features to 2D using PCA and plot the cluster scatter.

    PCA (Principal Component Analysis) projects the 3D RFM space onto 2D
    so we can visualise the clusters in a scatter plot.

    Args:
        X_scaled: Scaled feature matrix.
        labels:   Cluster label array.
        title:    Plot title and output filename prefix.
    """
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap="tab10", alpha=0.6, s=20,
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.tight_layout()

    filename = title.replace(" ", "_").lower() + ".png"
    plt.savefig(os.path.join(REPORTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  Cluster plot saved → reports/{filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUSINESS INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def interpret_clusters(rfm: pd.DataFrame) -> tuple:
    """
    Auto-label each K-Means cluster with a business-friendly name.

    Clusters are ranked by average RFM_Score (highest = Champions) and
    assigned descriptive labels such as Loyal Customers, At-Risk, etc.

    Args:
        rfm: RFM DataFrame with KMeans_Cluster and RFM_Score columns.

    Returns:
        Tuple of (updated rfm DataFrame, cluster summary DataFrame).
    """
    summary = rfm.groupby("KMeans_Cluster").agg(
        Count        =("CustomerID", "count"),
        Avg_Recency  =("Recency",    "mean"),
        Avg_Frequency=("Frequency",  "mean"),
        Avg_Monetary =("Monetary",   "mean"),
        Avg_RFM_Score=("RFM_Score",  "mean"),
    ).round(2)

    # Rank clusters by RFM score descending and assign business labels
    business_labels = [
        "Champions",
        "Loyal Customers",
        "At-Risk Customers",
        "Potential Loyalists",
        "Lost Customers",
        "Others",
    ]
    sorted_clusters = summary["Avg_RFM_Score"].sort_values(ascending=False).index
    label_map = {
        cluster: business_labels[i] if i < len(business_labels) else f"Cluster {cluster}"
        for i, cluster in enumerate(sorted_clusters)
    }

    summary["Business_Label"] = summary.index.map(label_map)
    rfm["Business_Segment"]   = rfm["KMeans_Cluster"].map(label_map)

    print("\n[SEG] Cluster Summary:")
    print(summary[["Count", "Avg_Recency", "Avg_Frequency",
                   "Avg_Monetary", "Business_Label"]].to_string())
    return rfm, summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rfm              = load_rfm()
    X_scaled, scaler, _ = preprocess_rfm(rfm)

    best_k           = find_optimal_k(X_scaled)
    rfm, sil_score   = run_kmeans(X_scaled, rfm, k=best_k)

    plot_clusters_pca(X_scaled, rfm["KMeans_Cluster"].values, "KMeans Clusters")

    rfm, cluster_summary = interpret_clusters(rfm)

    # Save segmented customers and cluster summary
    segmented_path = os.path.join(DATA_PROCESSED, "segmented_customers.csv")
    summary_path   = os.path.join(DATA_PROCESSED, "cluster_summary.csv")
    rfm.to_csv(segmented_path, index=False)
    cluster_summary.to_csv(summary_path)

    print(f"\n✅ Segmentation complete!  Silhouette Score = {sil_score:.4f}")
    print(f"   Segmented customers → {segmented_path}")
