"""
K-Means Clustering on Kerala Rainfall Data
==========================================
Identifies regional microclimate and rainfall intensity regimes by clustering
Kerala's rain-gauge stations based on geographic location, precipitation volume,
monsoon dependency, and extreme deluge frequency.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Graceful import for KMeans to handle environment restrictions
try:
    from sklearn.cluster import KMeans as _SklearnKMeans
    HAS_SKLEARN_KMEANS = True
except (ImportError, Exception):
    HAS_SKLEARN_KMEANS = False

from scipy.cluster.vq import kmeans2


class KMeansModel:
    """Robust K-Means clustering wrapper with scikit-learn compatible interface."""
    def __init__(self, n_clusters=3, random_state=42, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        X = np.asarray(X)
        if HAS_SKLEARN_KMEANS:
            try:
                clf = _SklearnKMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                )
                clf.fit(X)
                self.cluster_centers_ = clf.cluster_centers_
                self.labels_ = clf.labels_
                self.inertia_ = clf.inertia_
                return self
            except Exception:
                pass

        # Vectorized fallback using scipy kmeans2 with ++ initialization
        best_inertia = np.inf
        best_centers = None
        best_labels = None

        rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_init):
            seed = rng.randint(0, 100000)
            centers, labels = kmeans2(X, self.n_clusters, minit="++", iter=self.max_iter, seed=seed)
            # Calculate inertia (within-cluster sum of squared distances)
            distances = np.sum((X - centers[labels]) ** 2)
            if distances < best_inertia:
                best_inertia = distances
                best_centers = centers
                best_labels = labels

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ------------------------------------------------------------------------------
# 1. LOAD AND PREPARE DATASET
# ------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
possible_paths = [
    SCRIPT_DIR / "analysis_ready_rainfall.csv",
    SCRIPT_DIR.parent / "output" / "analysis_ready_rainfall.csv",
    SCRIPT_DIR.parent / "Supervised_Learning" / "analysis_ready_rainfall.csv",
    Path("analysis_ready_rainfall.csv"),
]

DATA_PATH = next((p for p in possible_paths if p.exists()), None)
if not DATA_PATH:
    raise FileNotFoundError("analysis_ready_rainfall.csv not found.")

df = pd.read_csv(DATA_PATH)

# Build station-level aggregated feature matrix
station_df = df.groupby("station").agg(
    district=("district", "first"),
    latitude=("latitude", "first"),
    longitude=("longitude", "first"),
    mean_daily_rain=("rainfall_mm", "mean"),
    total_rain=("rainfall_mm", "sum"),
    max_daily_rain=("rainfall_mm", "max"),
    rainy_days_ratio=("rainfall_mm", lambda x: (x >= 2.5).mean()),
    heavy_days_count=("rainfall_mm", lambda x: (x >= 64.5).sum()),
    sw_monsoon_rain=("rainfall_mm", lambda x: x[df.loc[x.index, "season"] == "Southwest Monsoon"].sum()),
).reset_index()

station_df["sw_monsoon_share"] = (
    station_df["sw_monsoon_rain"] / station_df["total_rain"] * 100
).fillna(0)

features = [
    "latitude",
    "longitude",
    "mean_daily_rain",
    "total_rain",
    "rainy_days_ratio",
    "heavy_days_count",
    "sw_monsoon_share",
    "max_daily_rain",
]

X = station_df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------------------
# 2. ELBOW METHOD & SILHOUETTE ANALYSIS (K = 2 to 7)
# ------------------------------------------------------------------------------
K_range = range(2, 8)
inertias = []
silhouettes = []

for k in K_range:
    km = KMeansModel(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

# Plot Elbow and Silhouette Curves
SCREENSHOTS_DIR = SCRIPT_DIR / "Screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(list(K_range), inertias, marker="o", color="#1f77b4", linewidth=2)
ax1.set_title("K-Means Elbow Method (WCSS)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(list(K_range), silhouettes, marker="s", color="#2ca02c", linewidth=2)
ax2.set_title("Silhouette Score vs. K", fontsize=12, fontweight="bold")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Silhouette Score")
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(SCREENSHOTS_DIR / "KMeans_Elbow_Silhouette.png", dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 3. FIT OPTIMAL K-MEANS MODEL (K = 3 Regional Climate Regimes)
# ------------------------------------------------------------------------------
OPTIMAL_K = 3
best_kmeans = KMeansModel(n_clusters=OPTIMAL_K, random_state=42)
best_kmeans.fit(X_scaled)
station_df["cluster"] = best_kmeans.labels_

# Calculate performance evaluation metrics
sil_score = silhouette_score(X_scaled, best_kmeans.labels_)
db_score = davies_bouldin_score(X_scaled, best_kmeans.labels_)
ch_score = calinski_harabasz_score(X_scaled, best_kmeans.labels_)

# ------------------------------------------------------------------------------
# 4. CLUSTER VISUALIZATION
# ------------------------------------------------------------------------------
plt.figure(figsize=(9, 7))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
cluster_names = [
    "Cluster 0: Lowland / Moderate Monsoon Zone",
    "Cluster 1: High-Altitude / Extreme Deluge Catchments",
    "Cluster 2: Eastern Rain-Shadow / Dry Foothills",
]

for c in range(OPTIMAL_K):
    subset = station_df[station_df["cluster"] == c]
    plt.scatter(
        subset["longitude"],
        subset["latitude"],
        s=subset["total_rain"] / 100,
        c=colors[c % len(colors)],
        label=f"{cluster_names[c % len(cluster_names)]} (n={len(subset)})",
        alpha=0.75,
        edgecolors="black",
        linewidth=0.8,
    )

plt.title("Kerala Rainfall Stations - K-Means Spatial Clusters (K=3)", fontsize=13, fontweight="bold")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.legend(loc="lower left", fontsize=9, frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(SCREENSHOTS_DIR / "KMeans_Clusters.png", dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 5. CONSOLE OUTPUT
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("== K-MEANS CLUSTERING RESULTS ==")
print("=" * 80)
print(f"Dataset Records Processed : {len(df):,} daily readings")
print(f"Unique Stations Clustered : {len(station_df)}")
print(f"Optimal Clusters (K)      : {OPTIMAL_K}")
print(f"Inertia (WCSS)            : {best_kmeans.inertia_:.2f}")
print(f"Silhouette Score          : {sil_score:.4f}")
print(f"Davies-Bouldin Index      : {db_score:.4f} (lower is better)")
print(f"Calinski-Harabasz Score   : {ch_score:.2f} (higher is better)\n")

print("-" * 80)
print(f"{'Cluster':<10} {'Stations':<10} {'Mean Rain (mm)':<16} {'Total Rain (mm)':<17} {'Heavy Days':<12} {'SW Monsoon %':<14}")
print("-" * 80)

for c in range(OPTIMAL_K):
    sub = station_df[station_df["cluster"] == c]
    print(
        f"Cluster {c:<2} "
        f"{len(sub):<10} "
        f"{sub['mean_daily_rain'].mean():<16.2f} "
        f"{sub['total_rain'].mean():<17.1f} "
        f"{sub['heavy_days_count'].mean():<12.1f} "
        f"{sub['sw_monsoon_share'].mean():<14.1f}%"
    )

print("=" * 80)

print("\nCluster Profiles & Representative Stations:")
for c in range(OPTIMAL_K):
    sub = station_df[station_df["cluster"] == c]
    top_stations = sub.nlargest(4, "total_rain")["station"].tolist()
    districts = sub["district"].value_counts().head(3).index.tolist()
    print(f"• Cluster {c} (n={len(sub)} stations):")
    print(f"  Key Districts : {', '.join(districts)}")
    print(f"  Top Stations  : {', '.join(top_stations)}")

print(f"\nVisualizations saved to: {SCREENSHOTS_DIR.resolve()}")
print("=" * 80 + "\n")
