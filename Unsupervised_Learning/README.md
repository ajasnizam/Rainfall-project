# Unsupervised Learning – Kerala Rainfall Microclimate Clustering

This module implements unsupervised machine learning techniques to discover regional rainfall regimes and microclimate patterns across Kerala's **115 rain-gauge stations** using historical daily precipitation telemetry (52,522 records).

---

## 📌 Objectives
1. **Microclimate Zoning**: Segment weather stations into distinct meteorological regimes based on rainfall volume, extreme event frequency, and seasonal monsoon dependency.
2. **Cluster Topology & Comparison**: Compare **Partitional Clustering (K-Means)** against **Hierarchical Clustering (Agglomerative)** to assess cluster cohesion, separation, and sub-cluster taxonomy.
3. **Hazard Prioritization**: Pinpoint highland catchment zones vulnerable to flash floods and landslides versus agricultural coastal corridors and drought-prone rain-shadow valleys.

---

## 📁 Repository Structure

```text
Unsupervised_Learning/
├── analysis_ready_rainfall.csv     # Cleaned dataset (52,522 records × 21 columns)
├── K-Means.py                      # K-Means clustering, Elbow method, & Silhouette analysis
├── Agglomerative.py                # Hierarchical clustering, Linkage comparisons, & Dendrogram
├── compare_models.py               # Unified benchmark comparing K-Means vs. Agglomerative
├── Screenshots/                    # Publication-grade clustering plots
│   ├── KMeans_Elbow_Silhouette.png # WCSS Elbow curve & Silhouette score across K=2..7
│   ├── KMeans_Clusters.png         # Spatial plot of K-Means station clusters
│   ├── Hierarchical_Dendrogram.png # Dendrogram showing full hierarchical tree cut
│   └── Hierarchical_Clusters.png   # Spatial plot of Agglomerative station clusters
└── README.md                       # Comprehensive module documentation
```

---

## 🛠️ Feature Engineering & Preprocessing

Each of the 115 weather stations is characterized by an 8-dimensional feature vector summarizing spatial geography and meteorological behavior:

| Feature Name | Description | Meteorological Significance |
| :--- | :--- | :--- |
| **`latitude`** | Latitude coordinate (°N) | Identifies the north-south precipitation gradient |
| **`longitude`** | Longitude coordinate (°E) | Identifies the coast-to-Western Ghats orographic gradient |
| **`mean_daily_rain`** | Average daily precipitation (mm) | Baseline daily rainfall intensity |
| **`total_rain`** | Annual cumulative precipitation (mm) | Total hydrological discharge and runoff volume |
| **`rainy_days_ratio`** | Percentage of recorded days with rain ($\ge 2.5\text{ mm}$) | Precipitation consistency throughout the year |
| **`heavy_days_count`** | Count of heavy/extreme deluge days ($\ge 64.5\text{ mm}$) | Susceptibility to flash floods and soil saturation hazards |
| **`sw_monsoon_share`** | % of annual rain received during June–Sept | Dependency on the Southwest Monsoon season |
| **`max_daily_rain`** | Peak single-day deluge recorded (mm) | Maximum cloudburst stress test |

All features are standardized using `StandardScaler` to ensure zero mean and unit variance before distance and variance calculations.

---

## 📊 Benchmark & Model Comparison

Both models were evaluated on the exact same standardized feature matrix with optimal cluster count $K = 3$:

```text
================================================================================
== UNSUPERVISED LEARNING MODEL COMPARISON ==
================================================================================
Total Daily Observations  : 52,522
Total Weather Stations    : 115
Feature Space Dimensions  : 8 meteorological & spatial indicators
================================================================================
Model                     Clusters   Silhouette Score   Davies-Bouldin   Calinski-Harabasz 
--------------------------------------------------------------------------------
K-Means                   3          0.2938             1.2430           65.21             
Hierarchical (Ward)       3          0.2848             1.2323           61.48             
Hierarchical (Complete)   3          0.2314             1.5190           51.41             
================================================================================
```

### 🏆 Key Findings & Evaluation Verdict

1. **Best Overall Cohesion & Separation – K-Means**:
   - Achieved the highest **Silhouette Score (0.2938)** and highest **Calinski-Harabasz Index (65.21)**.
   - Minimizes within-cluster variance effectively, establishing distinct spherical cluster boundaries.

2. **Best Compact Separation – Hierarchical (Ward Linkage)**:
   - Achieved the lowest **Davies-Bouldin Index (1.2323)**.
   - Provides an interpretable **Dendrogram** illustrating how sub-catchments merge into macro-climatic belts.

3. **Cophenetic Correlation**:
   - Ward: `0.5927` | Complete: `0.6028` | Average: `0.6914`.

---

## 🗺️ Discovered Regional Microclimate Regimes ($K=3$)

```text
--------------------------------------------------------------------------------
Cluster    Stations   Mean Rain (mm)   Total Rain (mm)   Heavy Days   SW Monsoon %  
--------------------------------------------------------------------------------
Cluster 0  40         16.08            6,105.3           14.3         47.8%
Cluster 1  28         24.96            11,392.1          45.6         79.0%
Cluster 2  47         21.10            10,906.1          32.3         59.3%
--------------------------------------------------------------------------------
```

### 1. Cluster 1: High-Altitude / Extreme Deluge Highland Catchments (28 stations)
- **Geographic Zone**: Windward slopes of the Western Ghats (Kannur, Kasaragod, Wayanad, Kozhikode).
- **Profile**: Heaviest annual deluge volume ($11,392\text{ mm}$ avg), **$79.0\%$** delivered during the Southwest Monsoon, and an average of **$45.6$ heavy rainfall days** per station.
- **Top Stations**: Kalladi (Wayanad), Thamarassery, Payyavoor, Uppala.
- **Hazard Profile**: Critical landslide susceptibility and high flash-flood hazard zone.

### 2. Cluster 2: Moderate-Rain Lowland & Coastal Monsoon Corridor (47 stations)
- **Geographic Zone**: Central and northern coastal and midland plains (Thrissur, Malappuram, Palakkad).
- **Profile**: Moderate-to-high rainfall ($10,906\text{ mm}$ avg), balanced monsoon distribution ($59.3\%$ SW monsoon share), and steady rain days.
- **Top Stations**: Vettilappara, Karikkadavu, Neriyamangalam, Moozhiyar.
- **Hazard Profile**: High agricultural water security, urban drainage inundation during peak July spells.

### 3. Cluster 0: Southern & Rain-Shadow Foothill Belts (40 stations)
- **Geographic Zone**: Southern districts and eastern leeward rain-shadow zones (Kollam, Thiruvananthapuram, Southern Palakkad).
- **Profile**: Lower annual volume ($6,105\text{ mm}$ avg), lowest daily average ($16.08\text{ mm}$), and lower SW monsoon share ($47.8\%$), with higher reliance on the Post-Monsoon (Northeast monsoon).
- **Top Stations**: Pampadi Veloor, Malampuzha, Vazhavatta, Aryanad.
- **Hazard Profile**: Lower flood risk; potential seasonal agricultural drought in summer months.

---

## 🚀 Execution Guide

Navigate to the `Unsupervised_Learning` directory:
```powershell
cd "D:\Mini ict\Unsupervised_Learning"
```

### 1. Run K-Means Clustering
```powershell
python K-Means.py
```
*Outputs WCSS inertia, cluster profiles, and saves Elbow & Cluster plots to `Screenshots/`.*

### 2. Run Hierarchical (Agglomerative) Clustering
```powershell
python Agglomerative.py
```
*Compares linkage methods, outputs cluster statistics, and saves Dendrogram & Spatial plots to `Screenshots/`.*

### 3. Run Comparative Benchmark Suite
```powershell
python compare_models.py
```
*Trains both models, prints comparison table, and generates the comparative evaluation verdict.*
