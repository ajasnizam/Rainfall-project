# Supervised Learning – Team Tasks & Contributions Overview

This directory contains the documentation of individual contributions for the **Supervised Learning** module of the Kerala Rainfall Mini Project. All corresponding model implementations reside in the parent `Supervised_Learning` directory.

---

## 👥 Team Members & Task Allocation

| Member | Name | Primary Task / Model | Documentation File | Core Script |
| :---: | :--- | :--- | :--- | :--- |
| **Member 1** | **Jeswin** | Decision Tree Classification | [`MEMBER_1_JESWIN_README.md`](./MEMBER_1_JESWIN_README.md) | `Decision_Tree.py` |
| **Member 2** | **Adharsh** | Random Forest Classification | [`MEMBER_2_ADHARSH_README.md`](./MEMBER_2_ADHARSH_README.md) | `RandomForest.py` |
| **Member 3** | **Aslaf** | Support Vector Machine (SVC) | [`MEMBER_3_ASLAF_README.md`](./MEMBER_3_ASLAF_README.md) | `model_svc.py` |
| **Member 4** | **Sreejith** | K-Nearest Neighbors (KNN) | [`MEMBER_4_SREEJITH_README.md`](./MEMBER_4_SREEJITH_README.md) | `knn_model.py` |
| **Member 5** | **Ajas** | XGBoost, Model Audit & Verification, Unified Benchmark Suite, Linear Regression | [`MEMBER_5_AJAS_README.md`](./MEMBER_5_AJAS_README.md) | `xgboost_model.py`<br>`test_models.py`<br>`Linear_Regression.py` |

---

## 📋 Summary of Contributions

### [Member 1 – Jeswin](./MEMBER_1_JESWIN_README.md)
- Developed and tuned the **Decision Tree Classifier** (`Decision_Tree.py`).
- Prevented target leakage and applied `class_weight="balanced"` to maximize recall on rare severe rainfall classes.
- Achieved highest minority class recall (50% on *Extremely Heavy* rainfall).

### [Member 2 – Adharsh](./MEMBER_2_ADHARSH_README.md)
- Built the **Random Forest Classifier** (`RandomForest.py`) with 300 estimators.
- Integrated `VarianceThreshold` filtering and ANOVA F-score feature selection (`SelectKBest`).
- Reduced variance and improved tree ensemble stability.

### [Member 3 – Aslaf](./MEMBER_3_ASLAF_README.md)
- Constructed the **Support Vector Classifier** (`model_svc.py`) using RBF kernel.
- Formulated rolling 3-reading lagged rainfall metrics to supply prior weather context.
- Optimized hyperplanes using time-series cross-validation.

### [Member 4 – Sreejith](./MEMBER_4_SREEJITH_README.md)
- Implemented the **K-Nearest Neighbors Classifier** (`knn_model.py`) with `n_neighbors=25`.
- Normalized feature vectors with standard scaling for Euclidean distance matching.
- Delivered instantaneous training execution (~0.00s) and strong baseline accuracy (39.76%).

### [Member 5 – Ajas](./MEMBER_5_AJAS_README.md)
- **XGBoost Implementation**: Created the top-performing `xgboost_model.py` (Accuracy: **43.63%**, Weighted F1: **0.3922**).
- **Quality Assurance & Configuration**: Audited and debugged all team models, fixed path issues, eliminated dependency errors, suppressed warnings, removed unnecessary file writes, and standardized `==` header outputs across every script.
- **Comparative Benchmarking**: Engineered and executed `test_models.py` to evaluate all models side-by-side on an identical stratified split and determine the overall winning model.
- **Linear Regression**: Implemented baseline regression model (`Linear_Regression.py`) for continuous precipitation prediction (`RMSE: 25.56 mm`).

---

## 🚀 Running All Tasks

```powershell
cd "D:\Mini ict\Supervised_Learning"

# Run the complete model comparison suite
python test_models.py

# Run individual models
python Decision_Tree.py
python RandomForest.py
python model_svc.py
python knn_model.py
python xgboost_model.py
python Linear_Regression.py
```
