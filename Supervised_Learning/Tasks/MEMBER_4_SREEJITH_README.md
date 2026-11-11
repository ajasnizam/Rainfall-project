# Member 4 – Sreejith: K-Nearest Neighbors (KNN)

## 👤 Member Information
- **Name:** Sreejith
- **Member ID:** Member 4
- **Assigned Module:** K-Nearest Neighbors Classifier (`knn_model.py`)
- **Target Variable:** `rainfall_category` (Multiclass: 6 IMD categories)

---

## 📌 Task & Responsibilities
1. **Model Development**:
   - Implemented an instance-based `KNeighborsClassifier` to assign rainfall categories based on proximity to localized historical weather patterns.
2. **Feature Preprocessing & Scaling**:
   - Mapped temporal and calendar variables (`season_num`, `day_num`) into numerical scales.
   - Scaled feature vectors using `StandardScaler` to prevent high-magnitude features from dominating Euclidean distance calculations.
3. **Neighbor Tuning & Split**:
   - Selected `n_neighbors=25` with uniform weighting for smooth decision boundaries resistant to isolated anomalies.
   - Employed an 80/20 stratified train-test split (42,017 train, 10,505 test).

---

## 📊 Evaluation & Performance
- **Model Script:** `D:\Mini ict\Supervised_Learning\knn_model.py`
- **Accuracy:** `39.76%` – `41.39%`
- **Macro Precision:** `0.2178`
- **Macro Recall:** `0.1872`
- **Macro F1-Score:** `0.1742`
- **Training Time:** `~0.00s` (Instantaneous non-parametric fitting)
- **Key Strength:** Extremely fast training time with solid baseline accuracy across regional weather patterns.

---

## 🚀 How to Run
```powershell
cd "D:\Mini ict\Supervised_Learning"
python knn_model.py
```
Outputs classification performance and detailed metrics directly to the console.

