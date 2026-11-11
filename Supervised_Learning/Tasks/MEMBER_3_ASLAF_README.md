# Member 3 – Aslaf: Support Vector Classification (SVC)

## 👤 Member Information
- **Name:** Aslaf
- **Member ID:** Member 3
- **Assigned Module:** Support Vector Machine Classifier (`model_svc.py`)
- **Target Variable:** `rainfall_category` (Multiclass: 6 IMD categories)

---

## 📌 Task & Responsibilities
1. **Model Development**:
   - Designed a kernelized `SVC` (Radial Basis Function / RBF kernel) to map high-dimensional meteorological and station attributes into separable hyperplanes.
2. **Lagged Feature Engineering**:
   - Created historical rainfall indicators (`previous_rainfall_mm`, `previous_3_reading_mean`, `has_previous_reading`) grouped by station to incorporate prior temporal weather patterns without introducing target leakage.
3. **Pipeline Optimization**:
   - Integrated `StandardScaler` to ensure zero mean and unit variance for distance-sensitive margin calculations.
   - Applied `VarianceThreshold` and `SelectKBest(f_classif, k=25)` for feature dimension reduction.
4. **Time-Series Cross-Validation**:
   - Tuned penalty parameter `C`, kernel parameter `gamma`, and `class_weight` using `TimeSeriesSplit(3)` to preserve chronological sequence integrity.

---

## 📊 Evaluation & Performance
- **Model Script:** `D:\Mini ict\Supervised_Learning\model_svc.py`
- **Accuracy:** `28.44%` (test sample: `27.32%` on time-split)
- **Macro Precision:** `0.2113`
- **Macro Recall:** `0.2589`
- **Macro F1-Score:** `0.1927`
- **Key Strength:** Maximizes margin boundaries in non-linear decision spaces with balanced class handling.

---

## 🚀 How to Run
```powershell
cd "D:\Mini ict\Supervised_Learning"
python model_svc.py
```
Outputs model hyperparameters, per-class metrics, confusion matrix, and training time directly to the console.

