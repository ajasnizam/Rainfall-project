# Member 5 – Ajas: XGBoost, Pipeline Integration, Model Configuration & Comparative Benchmarking

## 👤 Member Information
- **Name:** Ajas
- **Member ID:** Member 5
- **Assigned Modules & Leadership Tasks:**
  - **XGBoost Classifier Implementation** (`xgboost_model.py`)
  - **End-to-End Model Configuration, Audit & Bug Fixing** (All Models)
  - **Comparative Benchmark Suite Development** (`test_models.py`)
  - **Baseline Linear Regression** (`Linear_Regression.py`)

---

## 📌 Task & Responsibilities

### 1. XGBoost Model Creation (`xgboost_model.py`)
- Designed and trained an advanced gradient boosted tree classifier using `XGBClassifier`:
  - **Architecture**: 300 estimators, max depth 6, learning rate 0.1, multi-class soft probability objective (`multi:softprob`).
  - **Regularization & Sampling**: Subsample ratio 0.8, column subsample by tree 0.8 to prevent overfitting.
  - **Preprocessing**: Robust `ColumnTransformer` with numeric median imputation, standard scaling, and categorical one-hot encoding.
  - **Target Encoding**: `LabelEncoder` mapped to IMD rainfall labels (*Very Light, Light, Moderate, Heavy, Very Heavy, Extremely Heavy*).
- **Result**: **Highest performing model overall** (Accuracy: **43.63%**, Weighted F1: **0.3922**).

---

### 2. Full Pipeline Configuration & Quality Assurance Across All Models
- **Standardized Console Formatting**: Updated every model's terminal output to use clean, consistent double-equals headers (`== ... CLASSIFICATION RESULTS ==`) and 80-character boundary lines matching team presentation standards.
- **Removed Unnecessary File Outputs**: Cleaned up legacy CSV outputs (`member4_rf_result.csv`, `results_svc.csv`) and disabled external file writing in Decision Tree (`results/` folder, JSON, PNG), ensuring all models print cleanly and directly to the console.
- **Dependency & Headless Compatibility Fixes**:
  - Replaced missing `seaborn` in `Decision_Tree.py` with standard `sklearn.metrics.ConfusionMatrixDisplay`.
  - Replaced GUI-blocking `plt.show()` with `plt.close()` to enable uninterrupted batch execution.
- **Path Resolution**: Fixed hardcoded/relative path issues across all models to ensure reliable execution from any terminal directory.
- **Warning Suppression**: Filtered deprecation and `Pandas4Warning` notices to deliver clean, professional console logs.

---

### 3. Unified Comparative Benchmark (`test_models.py`)
- Built a unified testing harness that executes, times, and evaluates all five classification models on the identical 80/20 stratified split (42,017 train / 10,505 test samples).
- Calculated standard comparison metrics across all models:
  - Accuracy
  - Macro Precision
  - Macro Recall
  - Macro F1-score
  - Weighted F1-score
  - Training Duration (seconds)
- **Automated Decision Engine**: Evaluates metrics and automatically outputs the comparative table and final verdict on the best-performing model.

---

### 4. Benchmark Results Summary

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

### 🏆 Final Verdict
- **Winner:** **XGBoost Classifier** is declared the best model for Kerala rainfall classification due to its superior accuracy, weighted F1-score, and rapid training speed.

---

### 5. Linear Regression Baseline (`Linear_Regression.py`)
- Implemented OLS Linear Regression for predicting continuous precipitation amounts (`rainfall_mm`), achieving `RMSE = 25.56 mm` and establishing a baseline for regression tasks.

---

## 🚀 How to Run

```powershell
cd "D:\Mini ict\Supervised_Learning"

# Run the XGBoost model
python xgboost_model.py

# Run the complete model comparison suite & verdict
python test_models.py

# Run the Linear Regression baseline
python Linear_Regression.py
```

