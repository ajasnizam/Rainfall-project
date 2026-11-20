"""
Hierarchical (Agglomerative) Clustering on Kerala Rainfall Data
===============================================================
Performs bottom-up agglomerative clustering on Kerala's meteorological stations,
evaluates linkage methods (Ward, Complete, Average), visualizes the Dendrogram,
and identifies hierarchical microclimate regimes.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

# Graceful import for AgglomerativeClustering
try:
    from sklearn.cluster._agglomerative import AgglomerativeClustering
    HAS_SKLEARN_AGG = True
except Exception:
    HAS_SKLEARN_AGG = False


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
# 2. EVALUATE LINKAGE METHODS & COPHENETIC CORRELATION
# ------------------------------------------------------------------------------
distance_matrix = pdist(X_scaled)
linkage_methods = ["ward", "complete", "average"]
linkage_trees = {}
cophenetic_scores = {}

for method in linkage_methods:
    Z = linkage(X_scaled, method=method)
    linkage_trees[method] = Z
    c_score, _ = cophenet(Z, distance_matrix)
    cophenetic_scores[method] = c_score

# ------------------------------------------------------------------------------
# 3. FIT OPTIMAL AGGLOMERATIVE MODEL (Ward Linkage, K = 3)
# ------------------------------------------------------------------------------
OPTIMAL_K = 3
best_linkage = linkage_trees["ward"]

# Obtain cluster labels (0-indexed)
agg_labels = fcluster(best_linkage, t=OPTIMAL_K, criterion="maxclust") - 1
station_df["cluster"] = agg_labels

# Calculate performance evaluation metrics
sil_score = silhouette_score(X_scaled, agg_labels)
db_score = davies_bouldin_score(X_scaled, agg_labels)
ch_score = calinski_harabasz_score(X_scaled, agg_labels)

# ------------------------------------------------------------------------------
# 4. PLOT DENDROGRAM
# ------------------------------------------------------------------------------
SCREENSHOTS_DIR = SCRIPT_DIR / "Screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(14, 6))
dendrogram(
    best_linkage,
    labels=station_df["station"].values,
    leaf_rotation=90,
    leaf_font_size=7,
    color_threshold=best_linkage[-(OPTIMAL_K - 1), 2],
)
plt.title(f"Hierarchical Agglomerative Dendrogram (Ward Linkage, K={OPTIMAL_K})", fontsize=12, fontweight="bold")
plt.xlabel("Rain-Gauge Stations")
plt.ylabel("Distance Threshold (Ward's Variance)")
plt.axhline(y=best_linkage[-(OPTIMAL_K - 1), 2], color="red", linestyle="--", label=f"Cluster Cut (K={OPTIMAL_K})")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(SCREENSHOTS_DIR / "Hierarchical_Dendrogram.png", dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 5. SPATIAL CLUSTER VISUALIZATION
# ------------------------------------------------------------------------------
plt.figure(figsize=(9, 7))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
cluster_names = [
    "Cluster 0: Moderate-Rain Coastal / Foothill Belt",
    "Cluster 1: Heavy Deluge Highland Catchments",
    "Cluster 2: Southern / Rain-Shadow Low Rainfall Belt",
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

plt.title("Kerala Rainfall Stations - Hierarchical Agglomerative Clusters (K=3)", fontsize=13, fontweight="bold")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.legend(loc="lower left", fontsize=9, frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(SCREENSHOTS_DIR / "Hierarchical_Clusters.png", dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 6. CONSOLE OUTPUT
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("== HIERARCHICAL (AGGLOMERATIVE) CLUSTERING RESULTS ==")
print("=" * 80)
print(f"Dataset Records Processed : {len(df):,} daily readings")
print(f"Unique Stations Clustered : {len(station_df)}")
print(f"Selected Clusters (K)     : {OPTIMAL_K}")
print(f"Primary Linkage Method    : Ward (Minimum Variance)")
print(f"Silhouette Score          : {sil_score:.4f}")
print(f"Davies-Bouldin Index      : {db_score:.4f} (lower is better)")
print(f"Calinski-Harabasz Score   : {ch_score:.2f} (higher is better)\n")

print("Linkage Method Comparison (Cophenetic Correlation Coefficient):")
for method, score in cophenetic_scores.items():
    print(f"  • {method.capitalize():<10} Linkage : {score:.4f}")

print("\n" + "-" * 80)
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
