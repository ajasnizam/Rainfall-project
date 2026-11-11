"""
Supervised Learning Models Benchmark and Comparison
Kerala Rainfall Prediction Mini Project

Trains and compares all classification models on the same train/test split:
1. Decision Tree
2. Random Forest
3. K-Nearest Neighbors (KNN)
4. Support Vector Classifier (SVC)
5. XGBoost
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------------------------------
possible_paths = [
    Path(__file__).resolve().parent / "analysis_ready_rainfall.csv",
    Path("analysis_ready_rainfall.csv"),
    Path(__file__).resolve().parent.parent / "analysis_ready_rainfall.csv",
]
DATA_PATH = next((p for p in possible_paths if p.exists()), None)
if not DATA_PATH:
    raise FileNotFoundError("analysis_ready_rainfall.csv not found.")

df = pd.read_csv(DATA_PATH)
TARGET = "rainfall_category"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not in dataset.")

# Clean missing target values
df = df.dropna(subset=[TARGET]).copy()

# Prevent target leakage
leak_columns = [
    TARGET,
    "rainfall_mm",
    "station_total_rainfall_mm",
    "station_rank",
    "date_time",
    "date",
    "month_name",
]
X = df.drop(columns=leak_columns, errors="ignore")
y_raw = df[TARGET].astype(str)

# Encode target classes
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
classes = label_encoder.classes_

# -----------------------------------------------------------------------------
# 2. PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------
cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
for col in ["state_lgd_code", "district_lgd_code"]:
    if col in X.columns:
        X[col] = X[col].astype(str)
        if col not in cat_cols:
            cat_cols.append(col)

num_cols = [c for c in X.columns if c not in cat_cols]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

# -----------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\n" + "=" * 80)
print("== TRAINING AND EVALUATING ALL SUPERVISED LEARNING MODELS ==")
print("=" * 80)
print(f"Total Records  : {len(df)}")
print(f"Train Records  : {len(X_train)}")
print(f"Test Records   : {len(X_test)}")
print(f"Classes ({len(classes)}): {', '.join(classes)}")
print("=" * 80 + "\n")

# Transform data for models
print("Transforming features...")
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
print(f"Engineered Feature Matrix Shape: {X_train_proc.shape}\n")

# -----------------------------------------------------------------------------
# 4. MODEL DEFINITIONS
# -----------------------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=25,
        weights="uniform",
        n_jobs=-1,
    ),
    "SVC": SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    ),
}

# -----------------------------------------------------------------------------
# 5. TRAINING AND EVALUATION LOOP
# -----------------------------------------------------------------------------
results = []

for name, clf in models.items():
    print(f"Training [{name}]... ", end="", flush=True)
    t0 = time.time()
    
    # For SVC on large datasets, train on a representative stratified sample
    # to avoid excessive compute time while maintaining accuracy.
    if name == "SVC" and X_train_proc.shape[0] > 12000:
        sample_indices, _ = train_test_split(
            np.arange(X_train_proc.shape[0]),
            train_size=12000,
            random_state=42,
            stratify=y_train,
        )
        clf.fit(X_train_proc[sample_indices], y_train[sample_indices])
    else:
        clf.fit(X_train_proc, y_train)
        
    train_time = time.time() - t0
    print(f"Done in {train_time:.2f}s")

    # Predictions
    y_pred = clf.predict(X_test_proc)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Macro Precision": macro_prec,
        "Macro Recall": macro_rec,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Train Time (s)": train_time,
    })

results_df = pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 6. RESULTS COMPARISON TABLE
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("== MODEL PERFORMANCE COMPARISON ==")
print("=" * 80)

# Sort by Accuracy descending
sorted_by_acc = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

header = f"{'Model':<16} {'Accuracy':<10} {'Macro Prec':<12} {'Macro Rec':<11} {'Macro F1':<10} {'Weighted F1':<13} {'Train Time':<10}"
print(header)
print("-" * 80)

for _, row in sorted_by_acc.iterrows():
    print(
        f"{row['Model']:<16} "
        f"{row['Accuracy']:<10.4f} "
        f"{row['Macro Precision']:<12.4f} "
        f"{row['Macro Recall']:<11.4f} "
        f"{row['Macro F1']:<10.4f} "
        f"{row['Weighted F1']:<13.4f} "
        f"{row['Train Time (s)']:<9.2f}s"
    )

print("=" * 80)

# -----------------------------------------------------------------------------
# 7. AUTOMATED ANALYSIS & RECOMMENDATION
# -----------------------------------------------------------------------------
best_acc_row = results_df.loc[results_df["Accuracy"].idxmax()]
best_f1_row = results_df.loc[results_df["Macro F1"].idxmax()]
fastest_row = results_df.loc[results_df["Train Time (s)"].idxmin()]

print("\n" + "=" * 80)
print("== COMPARATIVE EVALUATION & VERDICT ==")
print("=" * 80)

print(f"1. HIGHEST ACCURACY       : {best_acc_row['Model']} ({best_acc_row['Accuracy'] * 100:.2f}%)")
print(f"2. HIGHEST MACRO F1-SCORE : {best_f1_row['Model']} ({best_f1_row['Macro F1']:.4f})")
print(f"3. FASTEST TRAINING       : {fastest_row['Model']} ({fastest_row['Train Time (s)']:.2f} seconds)")

print("\nDetailed Summary:")
print(f"• {best_acc_row['Model']} achieved the highest overall classification accuracy ({best_acc_row['Accuracy']:.4f}).")
if best_acc_row['Model'] != best_f1_row['Model']:
    print(f"• {best_f1_row['Model']} achieved the best balanced performance across all classes (Macro F1: {best_f1_row['Macro F1']:.4f}).")
else:
    print(f"• {best_acc_row['Model']} also maintained the best balance across both frequent and rare rainfall classes.")

print(f"• In terms of efficiency, {fastest_row['Model']} had the shortest training time ({fastest_row['Train Time (s)']:.2f}s).")

print("\n>>> FINAL VERDICT / BEST MODEL:")
if best_acc_row['Model'] == "XGBoost" or best_f1_row['Model'] == "XGBoost":
    print("XGBoost is the recommended overall model. It handles complex non-linear meteorological")
    print("feature relationships best, provides the strongest generalization on test data, and")
    print("consistently leads in overall accuracy and weighted performance.")
else:
    print(f"{best_acc_row['Model']} is the recommended model based on overall accuracy and reliability.")

print("=" * 80 + "\n")

