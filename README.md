# Kerala Daily Rainfall – ETL Pipeline, EDA & Supervised Learning

A comprehensive Data Engineering and Machine Learning project that transforms raw Indian Government meteorological telemetry (53,162 records with non-standard timestamps, missing administrative hierarchies, and duplicates) into a clean, analysis-ready dataset (52,522 rows × 24 columns), conducts in-depth Exploratory Data Analysis (EDA), and trains a suite of Supervised Machine Learning models to classify rainfall intensity and predict precipitation volumes.

---

## 📌 Contents
- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [End-to-End Workflow](#end-to-end-workflow)
- [Project Structure](#project-structure)
- [Team and Responsibilities](#team-and-responsibilities)
- [Part 1: ETL Pipeline](#part-1-etl-pipeline)
- [Part 2: Exploratory Data Analysis (EDA)](#part-2-exploratory-data-analysis-eda)
- [Part 3: Supervised Machine Learning](#part-3-supervised-machine-learning)
  - [Models Implemented](#models-implemented)
  - [Model Benchmark & Performance Comparison](#model-benchmark--performance-comparison)
  - [Key Findings & Winning Model](#key-findings--winning-model)
- [Quick Start & Execution Guide](#quick-start--execution-guide)
- [Requirements](#requirements)

---

## Project Overview

Kerala's tropical monsoon climate features high-volume, erratic deluges leading to severe floods and landslides. This project delivers an end-to-end data science solution:
1. **Automated ETL Pipeline**: Extracts raw meteorological telemetry from the National Water Informatics Centre (NWIC), enforces data quality assertions, eliminates conflicting readings, derives temporal and spatial indicators, and outputs an analysis-ready dataset.
2. **Exploratory Data Analysis (EDA)**: Analyzes precipitation distribution across Kerala's 14 districts, 104+ stations, and river basins through 10 publication-grade visualizations.
3. **Supervised Machine Learning**: Formulates multiclass classification models predicting official India Meteorological Department (IMD) rainfall tiers (*Very Light, Light, Moderate, Heavy, Very Heavy, Extremely Heavy*) without target leakage, establishes baseline regression for precipitation volume, and evaluates all models using a unified comparative benchmark harness (`test_models.py`).

---

## Data Source

| Parameter | Specification |
|---|---|
| **Dataset Name** | Kerala Surface Water Manual Daily Rainfall Station Data |
| **Provider** | National Water Development Agency / National Water Informatics Centre (NWIC), Ministry of Jal Shakti, Government of India |
| **Portal Link** | [NWIC Kerala Daily Rainfall Portal](https://nwdp.nwic.gov.in/dataset/rainfall-kerala-surface-water-manual-daily/resource/d175f1f6-cccf-4980-97ab-877650d05074) |
| **Raw Format** | CSV (20 columns, 53,162 raw records) |
| **Coverage** | All 14 Kerala districts, 104+ rain-gauge stations |
| **Clean Output** | 52,522 validated records × 24 engineered features |

---

## End-to-End Workflow

```
                        Raw Data Source (raw_data/rainfall_1.csv)
                                         │
                                         ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ 1. EXTRACT & VALIDATE                                                    │
    │ • Ingest raw records, assert dimensions (53,162 rows × 20 cols)          │
    │ • Check boundary conditions, null thresholds, non-negative values        │
    │ • Responsibility: Member 2 (Jesvin)                                      │
    └──────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ 2. TRANSFORM & ENRICH                                                    │
    │ • Clean snake_case headers, resolve 640 station-day duplicate conflicts  │
    │ • Parse custom date strings ('DDMMYYYY hh:mm:ss AM'), cast data types    │
    │ • Engineer IMD categories, monsoon seasons, station aggregates & ranks   │
    │ • Responsibilities: Member 3 (Sreejith) & Member 4 (Adharsh)             │
    └──────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ 3. LOAD & REPORT GENERATION                                              │
    │ • Persist analysis-ready dataset (output/analysis_ready_rainfall.csv)    │
    │ • Generate 11 analytical summary CSV reports in reports/                 │
    │ • Responsibilities: Member 4 (Adharsh) & Member 5 (Ajas)                 │
    └──────────────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
    ┌───────────────────────────────┐         ┌───────────────────────────────┐
    │ 4. EXPLORATORY DATA ANALYSIS  │         │ 5. SUPERVISED LEARNING        │
    │ • 10 publication-grade charts │         │ • Leakage-free feature prep   │
    │ • Spatio-temporal analysis    │         │ • 5 Multiclass Classifiers    │
    │ • Statistical report writeup  │         │ • 1 Regression Baseline       │
    │ • All Team Members            │         │ • Unified Comparative Suite   │
    └───────────────────────────────┘         │ • All Team Members            │
                                              └───────────────────────────────┘
```

---

## Project Structure

```
Rainfall-project/
├── raw_data/
│   ├── rainfall_1.csv                      # Raw telemetry from government portal
│   └── rainfall_raw_copy.csv               # Automated extract archive copy
├── reports/                                # Generated analytical reports (11 CSVs)
│   ├── cleaned_data.csv                    # Complete cleaned dataset copy
│   ├── summary_statistics.csv              # Parametric & non-parametric statistics
│   ├── district_wise_report.csv            # District aggregations
│   ├── monthly_report.csv                  # Monthly rainfall progression
│   ├── seasonal_report.csv                 # Monsoon volume breakdown
│   ├── rainfall_category_report.csv        # Distribution across IMD tiers
│   ├── top10_rainiest_days.csv             # Statewide heaviest deluge events
│   ├── top10_stations_by_avg_rainfall.csv  # Top 10 wettest stations
│   ├── bottom10_stations_by_avg_rainfall.csv # Top 10 driest stations
│   ├── highest_lowest_values.csv           # Extreme statistical boundary metrics
│   └── transformation_log.csv              # Row-by-row transformation audit trail
├── output/
│   ├── analysis_ready_rainfall.csv         # Production analysis-ready dataset (52,522 rows × 24 cols)
│   └── eda/                                # Rendered publication-grade EDA chart images (10 PNGs)
├── screenshots/                            # Pipeline verification screenshots
├── EDA/                                    # Exploratory Data Analysis suite
│   ├── eda.ipynb                           # Interactive Jupyter/Colab notebook
│   ├── eda.py                              # Reusable Matplotlib & Seaborn plotting routines
│   ├── EDA_REPORT.md                       # Comprehensive markdown analytical report
│   └── requirements.txt                    # EDA-specific environment dependencies
├── Supervised_Learning/                    # Supervised Machine Learning Module
│   ├── test_models.py                      # Unified benchmark comparing all models side-by-side
│   ├── xgboost_model.py                    # XGBoost Classifier (Top Performer)
│   ├── Decision_Tree.py                    # Decision Tree Classifier with balanced weighting
│   ├── RandomForest.py                     # Random Forest Ensemble Classifier (300 trees)
│   ├── knn_model.py                        # K-Nearest Neighbors Classifier
│   ├── model_svc.py                        # Support Vector Classifier (RBF kernel)
│   ├── Linear_Regression.py                # OLS Linear Regression baseline for rainfall_mm
│   ├── analysis_ready_rainfall.csv         # Local model training dataset
│   ├── README.md                           # Supervised Learning detailed documentation
│   └── Tasks/                              # Individual member task documentation
│       ├── README.md                       # Master index for tasks
│       ├── MEMBER_1_JESWIN_README.md       # Member 1 contributions
│       ├── MEMBER_2_ADHARSH_README.md      # Member 2 contributions
│       ├── MEMBER_3_ASLAF_README.md        # Member 3 contributions
│       ├── MEMBER_4_SREEJITH_README.md     # Member 4 contributions
│       └── MEMBER_5_AJAS_README.md         # Member 5 contributions
├── pipeline.py                             # Complete, modular ETL pipeline script
├── requirements.txt                        # Unified project dependencies
└── README.md                               # Master project documentation
```

---

## Team and Responsibilities

| Member | Name | ETL / EDA Phase Responsibilities | Supervised Learning Phase Responsibilities |
|:---:|:---:|---|---|
| **Member 1** | **Aslaf** | **Documentation & EDA Reporting**<br>Authored project specs, dataset schemas, transformation rationale, and `EDA_REPORT.md`. | **Support Vector Machine (SVC)**<br>Developed `model_svc.py` (RBF kernel, rolling lagged rainfall features, `TimeSeriesSplit`). |
| **Member 2** | **Jesvin** | **Extraction & Validation**<br>Built `extract()`, `validate()`, dimension assertions, and EDA Charts 2 & 4. | **Decision Tree Classification**<br>Developed `Decision_Tree.py` with balanced class weights to maximize minority deluge recall. |
| **Member 3** | **Sreejith** | **Data Cleaning & Structure**<br>Built `transform()` Part 1 (T1–T9), duplicate handling, date parsing, and EDA Charts 5 & 6. | **K-Nearest Neighbors (KNN)**<br>Developed `knn_model.py` with standardized Euclidean distance matching and instant fitting. |
| **Member 4** | **Adharsh** | **Feature Engineering & Reports**<br>Built `transform()` Part 2 (T10–T14), generated 11 CSV reports, and EDA Charts 3 & 8. | **Random Forest Classification**<br>Developed `RandomForest.py` (300 estimators, `VarianceThreshold`, and ANOVA F-score `SelectKBest`). |
| **Member 5** | **Ajas**<br>*(Team Lead)* | **Loading, Pipeline Orchestration & EDA Architecture**<br>Built `load()`, `run_etl_pipeline()`, architected `EDA/eda.ipynb`, repo management, and EDA Charts 1, 7, 9, 10. | **XGBoost, Model Audit & Benchmark Suite**<br>Created `xgboost_model.py` (Top overall accuracy: 43.63%), audited/debugged all team models, built `test_models.py` benchmark, and created baseline `Linear_Regression.py`. |

---

## Part 1: ETL Pipeline

The pipeline is self-contained in `pipeline.py` and executes 14 systematic transformations:
- **T1–T5 (Structural Standardization)**: Renamed headers to snake_case, stripped exact duplicates, dropped 5 sparse columns (>90% missing), normalized district/station names to Title Case, and imputed unknown river/basin tags.
- **T6–T9 (Data Integrity & Deduplication)**: Parsed custom timestamps, cast numerical columns (`float64`, integer LGD codes), eliminated negative rain and out-of-boundary coordinates, and resolved 640 conflicting station-day readings.
- **T10–T14 (Meteorological Enrichment)**: Decomposed dates (`year`, `month`, `day_name`), mapped Indian monsoon seasons, classified IMD intensity categories, calculated cumulative station totals and statewide ranks, and sorted chronologically.

---

## Part 2: Exploratory Data Analysis (EDA)

Documented in [`EDA/eda.ipynb`](EDA/eda.ipynb) and [`EDA/EDA_REPORT.md`](EDA/EDA_REPORT.md). Key findings include:
- **81.6% Rain Days**: Over 80% of recordings qualify as IMD rain days ($\ge 2.5\text{ mm}$), with Light (40.7%) and Moderate (34.5%) bands dominating occurrence.
- **Extreme Deluge Disparity**: Although severe deluges ($\ge 64.5\text{ mm}$) account for only 6.4% of events, they contribute over **42%** of total runoff volume.
- **Peak Deluge Month**: July records the highest cumulative volume (280.6k mm, averaging 30.5 mm/day), driven by the Southwest Monsoon (62.3% of annual precipitation).
- **Western Ghats Orographic Lift**: Stations north of $11^\circ\text{N}$ and along windward slopes average 24.5–37.3 mm/day, compared to < 10 mm/day in eastern rain-shadow zones.

---

## Part 3: Supervised Machine Learning

### Models Implemented
All models reside in `Supervised_Learning/` and are trained to predict IMD rainfall category (`rainfall_category`) without target leakage:
1. **XGBoost Classifier (`xgboost_model.py`)**: Gradient boosted trees with softprob objective, subsample, and column regularization.
2. **Decision Tree Classifier (`Decision_Tree.py`)**: Tree classifier with balanced class weighting and hyperparameter tuning.
3. **Random Forest Classifier (`RandomForest.py`)**: 300-tree ensemble with variance thresholding and ANOVA feature selection.
4. **K-Nearest Neighbors (`knn_model.py`)**: Scaled instance-based classifier (`k=25`).
5. **Support Vector Classifier (`model_svc.py`)**: RBF-kernel margin classifier with rolling lagged rainfall indicators.
6. **Linear Regression (`Linear_Regression.py`)**: Baseline regressor predicting continuous rainfall volume (`rainfall_mm`).

---

### Model Benchmark & Performance Comparison

All models were evaluated on the exact same 80/20 stratified split (42,017 training samples, 10,505 testing samples across 6 IMD categories) using `test_models.py`:

```text
================================================================================
== MODEL PERFORMANCE COMPARISON ==
================================================================================
Model            Accuracy   Macro Prec   Macro Rec   Macro F1   Weighted F1   Train Time
--------------------------------------------------------------------------------
XGBoost          0.4363     0.2191       0.1978      0.1843     0.3922        4.01s
KNN              0.3976     0.1827       0.1787      0.1648     0.3539        0.00s
Random Forest    0.3000     0.1908       0.2129      0.1880     0.3158        14.12s
Decision Tree    0.2970     0.2063       0.2957      0.2101     0.3145        1.84s
SVC              0.2844     0.1971       0.2166      0.1846     0.3069        16.50s
================================================================================
```

### Key Findings & Winning Model

1. **🏆 Best Overall Model – XGBoost (Accuracy: 43.63%, Weighted F1: 0.3922)**:
   - Delivers the highest overall predictive accuracy and highest weighted F1-score across all 6 IMD rainfall categories.
   - Robustly captures non-linear meteorological interactions while maintaining high training efficiency (~4.0s).
2. **Best Balanced Minority Recall – Decision Tree (Macro F1: 0.2101, Macro Recall: 0.2957)**:
   - Achieves the highest recall on severe minority classes (50% on *Extremely Heavy* rainfall) through balanced class weighting.
3. **Fastest Execution – KNN (Instantaneous fit, ~0.00s)**:
   - Competitive baseline accuracy (39.76%) with zero upfront training cost.
4. **Regression Baseline – Linear Regression (`RMSE = 25.56 mm`)**:
   - Quantifies the linear relationship of spatial coordinates and day-of-year seasonal variables on continuous rainfall volume.

---

## Quick Start & Execution Guide

### 1. Set Up Environment
```powershell
# Clone repository
git clone https://github.com/ajasnizam/Rainfall-project.git
cd Rainfall-project

# Install dependencies
pip install -r requirements.txt
```

### 2. Run ETL Pipeline
```powershell
python pipeline.py
```
*Generates the clean dataset in `output/` and 11 summary reports in `reports/` in ~2–3 seconds.*

### 3. Run Supervised Learning Benchmark
```powershell
cd Supervised_Learning
python test_models.py
```
*Trains all models, displays comparison table, and prints the winning model verdict.*

### 4. Run Individual Models
```powershell
python xgboost_model.py
python Decision_Tree.py
python RandomForest.py
python knn_model.py
python model_svc.py
python Linear_Regression.py
```

---

## Requirements

The project uses standard scientific Python packages defined in `requirements.txt`:
```text
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.5
xgboost>=2.0
joblib>=1.3
matplotlib>=3.7
seaborn>=0.12
```
