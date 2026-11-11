# Supervised Learning - Kerala Rainfall Prediction

This directory contains supervised machine learning models developed for the Kerala Rainfall Mini Project. The models address two core analytical tasks:

1. **Multiclass Classification**: Predict IMD rainfall intensity categories (`rainfall_category`: *Very Light, Light, Moderate, Heavy, Very Heavy, Extremely Heavy*).
2. **Regression**: Predict continuous daily precipitation amounts (`rainfall_mm`).

---

## 📁 Repository Structure

| File | Type | Algorithm / Description | Target |
| :--- | :--- | :--- | :--- |
| **`test_models.py`** | Benchmark Suite | Automated benchmark that trains, evaluates, and compares all classification models side-by-side | `rainfall_category` |
| **`xgboost_model.py`** | Classification | Extreme Gradient Boosting Classifier (`XGBClassifier`) with regularization and softprob objectives | `rainfall_category` |
| **`Decision_Tree.py`** | Classification | Decision Tree Classifier with balanced class weights and hyperparameter optimization | `rainfall_category` |
| **`RandomForest.py`** | Classification | Random Forest Ensemble Classifier (300 estimators) with variance threshold feature filtering | `rainfall_category` |
| **`knn_model.py`** | Classification | K-Nearest Neighbors Classifier (`n_neighbors=25`, uniform distance) | `rainfall_category` |
| **`model_svc.py`** | Classification | Support Vector Classifier (RBF kernel, `StandardScaler`, `SelectKBest` feature selection) | `rainfall_category` |
| **`Linear_Regression.py`** | Regression | Baseline Ordinary Least Squares (OLS) Linear Regression model | `rainfall_mm` |
| **`analysis_ready_rainfall.csv`**| Dataset | Cleaned, analysis-ready rainfall dataset | Input data |

---

## 🛠️ Feature Engineering & Leakage Prevention

To ensure valid and realistic real-world evaluation:
- **Target Leakage Prevention**: Columns such as `rainfall_mm`, `station_total_rainfall_mm`, and `station_rank` are strictly excluded from input feature matrices for classification because `rainfall_category` is derived directly from rainfall amount.
- **Categorical Encoding**: Administrative identifiers (`state_lgd_code`, `district_lgd_code`), geographical regions (`district`, `station`), and temporal attributes (`day_of_week`, `season`) are processed using missing value imputation and `OneHotEncoder(handle_unknown="ignore")`.
- **Numerical Preprocessing**: Missing numerical attributes (`latitude`, `longitude`, `year`, `month`, lagged values) are imputed via median strategies and standardized using `StandardScaler`.
- **Stratified Evaluation**: All classification splits utilize an 80/20 train-test ratio stratified on the target classes to preserve the distribution of extreme rainfall events.

---

## 📊 Model Comparison & Benchmark Results

All classification models were trained and benchmarked on the identical 80/20 stratified split (42,017 training samples, 10,505 testing samples across 6 IMD categories) using `test_models.py`:

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

### 🏆 Key Findings & Verdict

1. **Best Overall Model – XGBoost (Accuracy: 43.63%, Weighted F1: 0.3922)**
   - Delivers the highest overall classification accuracy and strongest generalization across complex meteorological feature interactions.
   - Fast and efficient training duration (~4.0s).

2. **Best Balanced Minority Recall – Decision Tree (Macro F1: 0.2101, Macro Recall: 0.2957)**
   - Utilizes balanced class weighting to capture underrepresented extreme weather categories (e.g., *Extremely Heavy* and *Very Heavy* rainfall) with 50% recall on severe events.

3. **Fastest Execution – KNN (Instantaneous fit, ~0.00s)**
   - Non-parametric baseline offering competitive accuracy (39.76%) with zero upfront training overhead.

4. **Regression Baseline – Linear Regression (`R² = 0.0081`, `RMSE = 25.56 mm`)**
   - Illustrates linear feature weights for spatial coordinates and day-of-year seasonal variables.

---

## 🚀 How to Run

Navigate to the `Supervised_Learning` directory:
```powershell
cd "D:\Mini ict\Supervised_Learning"
```

### 1. Run the Full Model Benchmark & Comparison
Train and evaluate all models side-by-side with automated ranking and summary:
```powershell
python test_models.py
```

### 2. Run Individual Models
Each script outputs formatted evaluation metrics directly to the console:

```powershell
# Classification Models
python xgboost_model.py
python Decision_Tree.py
python RandomForest.py
python knn_model.py
python model_svc.py

# Regression Model
python Linear_Regression.py
```

### 3. Run All Models Sequentially
```powershell
python Decision_Tree.py; `
python xgboost_model.py; `
python knn_model.py; `
python RandomForest.py; `
python Linear_Regression.py; `
python model_svc.py
```

---

## 📦 Requirements

The scripts require standard scientific Python libraries:
```text
pandas
numpy
scikit-learn
xgboost
```

If not already installed in your Python environment:
```powershell
pip install pandas numpy scikit-learn xgboost
```
