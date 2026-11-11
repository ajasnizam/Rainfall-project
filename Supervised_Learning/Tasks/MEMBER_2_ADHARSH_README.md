# Member 2 – Adharsh: Random Forest Classification

## 👤 Member Information
- **Name:** Adharsh
- **Member ID:** Member 2
- **Assigned Module:** Random Forest Ensemble Classifier (`RandomForest.py`)
- **Target Variable:** `rainfall_category` (Multiclass: 6 IMD categories)

---

## 📌 Task & Responsibilities
1. **Model Development**:
   - Developed an ensemble-based `RandomForestClassifier` to enhance classification stability and capture non-linear interactions across geographical and seasonal predictors.
2. **Feature Engineering & One-Hot Encoding**:
   - Applied `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")` for categorical features (`district`, `season`, `month_name`, `day_of_week`).
   - Passed numerical coordinates (`latitude`, `longitude`, `year`, `month`).
3. **Feature Selection Pipeline**:
   - Utilized `VarianceThreshold(threshold=0)` to remove constant/zero-variance columns.
   - Incorporated `SelectKBest(score_func=f_classif, k=30)` to isolate the top discriminative features using ANOVA F-value ranking.
4. **Ensemble Configuration**:
   - Configured `n_estimators=300`, `class_weight="balanced"`, `random_state=42`, and parallelized execution (`n_jobs=-1`).

---

## 📊 Evaluation & Performance
- **Model Script:** `D:\Mini ict\Supervised_Learning\RandomForest.py`
- **Accuracy:** `30.00%`
- **Macro Precision:** `0.1908`
- **Macro Recall:** `0.2129`
- **Macro F1-Score:** `0.1880`
- **Training Time:** `~14.12s`
- **Key Strength:** Robust against overfitting due to bagging and multi-tree aggregation.

---

## 🚀 How to Run
```powershell
cd "D:\Mini ict\Supervised_Learning"
python RandomForest.py
```
Outputs training duration, accuracy, classification report, and confusion matrix directly to the console.

