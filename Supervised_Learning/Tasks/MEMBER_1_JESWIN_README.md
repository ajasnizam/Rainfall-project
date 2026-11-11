# Member 1 – Jeswin: Decision Tree Classification

## 👤 Member Information
- **Name:** Jeswin
- **Member ID:** Member 1
- **Assigned Module:** Decision Tree Classifier (`Decision_Tree.py`)
- **Target Variable:** `rainfall_category` (Multiclass: 6 IMD categories)

---

## 📌 Task & Responsibilities
1. **Model Development**:
   - Built and optimized the `DecisionTreeClassifier` pipeline to classify daily rainfall into IMD intensity categories (*Very Light, Light, Moderate, Heavy, Very Heavy, Extremely Heavy*).
2. **Preventing Target Leakage**:
   - Deliberately excluded direct rainfall values (`rainfall_mm`, `station_total_rainfall_mm`, `station_rank`) from the predictor feature set since `rainfall_category` is directly derived from them.
3. **Handling Severe Class Imbalance**:
   - Configured `class_weight="balanced"` to ensure underrepresented severe weather events (*Extremely Heavy*, *Very Heavy*) are not ignored in favor of frequent dry/light days.
4. **Hyperparameter Tuning**:
   - Conducted Stratified 3-Fold Cross-Validation (`GridSearchCV`) tuning `criterion`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.

---

## 📊 Evaluation & Performance
- **Model Script:** `D:\Mini ict\Supervised_Learning\Decision_Tree.py`
- **Accuracy:** `29.70%`
- **Macro Precision:** `0.2063`
- **Macro Recall:** `0.2957` (Highest macro recall across models)
- **Macro F1-Score:** `0.2101`
- **Key Strength:** Effectively captured minority classes with a 50% recall rate on *Extremely Heavy* rainfall observations.

---

## 🚀 How to Run
```powershell
cd "D:\Mini ict\Supervised_Learning"
python Decision_Tree.py
```
Outputs performance metrics and confusion matrix directly to the console.

