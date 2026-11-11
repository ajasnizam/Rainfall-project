"""
XGBoost classifier for Kerala rainfall data.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

# ------------------------------------------------------------
# Load data (expects analysis_ready_rainfall.csv in this folder)
# ------------------------------------------------------------
possible_paths = [
    Path(__file__).resolve().parent / "analysis_ready_rainfall.csv",
    Path("analysis_ready_rainfall.csv"),
    Path(__file__).resolve().parent.parent / "analysis_ready_rainfall.csv",
]
DATA_PATH = next((p for p in possible_paths if p.exists()), None)
if not DATA_PATH:
    raise FileNotFoundError("analysis_ready_rainfall.csv not found")

df = pd.read_csv(DATA_PATH)
TARGET = "rainfall_category"
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not in dataset")

df = df.dropna(subset=[TARGET]).copy()
# Drop columns that would leak the target
leak_cols = [TARGET, "rainfall_mm", "station_total_rainfall_mm", "station_rank", "date_time", "date", "month_name"]
X = df.drop(columns=leak_cols, errors="ignore")
y_raw = df[TARGET].astype(str)
# Encode target
le = LabelEncoder()
y = le.fit_transform(y_raw)

# ------------------------------------------------------------
# Feature preprocessing
# ------------------------------------------------------------
cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
# Ensure ID columns are treated as strings
for col in ["state_lgd_code", "district_lgd_code"]:
    if col in X.columns:
        X[col] = X[col].astype(str)
        if col not in cat_cols:
            cat_cols.append(col)
num_cols = [c for c in X.columns if c not in cat_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)
pipeline = Pipeline([("preprocess", preprocess), ("clf", model)])

# ------------------------------------------------------------
# Train / test split and model training
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

# ------------------------------------------------------------
# Evaluation output with double‑equals heading
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("== XGBOOST CLASSIFICATION RESULTS ==")
print("=" * 80)
print(f"Accuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {precision:.4f}")
print(f"Macro Recall      : {recall:.4f}")
print(f"Macro F1-score    : {f1:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
