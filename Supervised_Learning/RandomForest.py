
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Load dataset
possible_paths = [
    Path(__file__).resolve().parent / "analysis_ready_rainfall.csv",
    Path("analysis_ready_rainfall.csv"),
    Path(__file__).resolve().parent.parent / "analysis_ready_rainfall.csv",
]
DATA_PATH = next((p for p in possible_paths if p.exists()), None)
if not DATA_PATH:
    raise FileNotFoundError("analysis_ready_rainfall.csv not found")

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

# STEP 2: Create IMD rainfall category
df = df.dropna(subset=["rainfall_mm", "date"]).copy()

def rainfall_category(mm):
    if mm <= 0:
        return "No Rain"
    elif mm <= 2.4:
        return "Very Light"
    elif mm <= 15.5:
        return "Light"
    elif mm <= 64.4:
        return "Moderate"
    elif mm <= 115.5:
        return "Heavy"
    elif mm <= 204.4:
        return "Very Heavy"
    else:
        return "Extremely Heavy"

df["target"] = df["rainfall_mm"].apply(rainfall_category)

# STEP 3: Select input features
categorical_features = [
    "district", "season", "month_name", "day_of_week"
]

numerical_features = [
    "latitude", "longitude", "year", "month"
]

features = categorical_features + numerical_features

X = df[features]
y = df["target"]

# STEP 4: Split data by date (80:20)
df["date"] = pd.to_datetime(df["date"])

sorted_index = df.sort_values("date").index

split = int(len(sorted_index) * 0.80)

train_index = sorted_index[:split]
test_index = sorted_index[split:]

X_train = X.loc[train_index]
X_test = X.loc[test_index]

y_train = y.loc[train_index]
y_test = y.loc[test_index]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# STEP 5: One-Hot Encoding
encoder = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# STEP 6: Feature Selection
feature_names = encoder.get_feature_names_out()

variance = VarianceThreshold(threshold=0)

X_train = variance.fit_transform(X_train)
X_test = variance.transform(X_test)

feature_names = feature_names[variance.get_support()]

k = min(30, X_train.shape[1])

selector = SelectKBest(score_func=f_classif, k=k)

X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

selected_features = feature_names[selector.get_support()]

# STEP 7: Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# STEP 8: Train model
start_time = time.time()

rf_model.fit(X_train, y_train)

training_time = time.time() - start_time

print("\nRandom Forest Training Completed!")

# STEP 9: Prediction
y_pred = rf_model.predict(X_test)

# STEP 10: Evaluation
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 80)
print("== RANDOM FOREST CLASSIFICATION RESULTS ==")
print("=" * 80)
print(f"Accuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {precision:.4f}")
print(f"Macro Recall      : {recall:.4f}")
print(f"Macro F1-score    : {f1:.4f}")
print(f"Training Time     : {training_time:.2f} seconds\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# STEP 11: Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=rf_model.classes_
)

print("\nConfusion Matrix:")
print(cm)
