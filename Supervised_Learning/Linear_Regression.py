# Linear Regression on cleaned rainfall data
# -------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
possible_paths = [
    Path(__file__).resolve().parent / "analysis_ready_rainfall.csv",
    Path("analysis_ready_rainfall.csv"),
    Path(__file__).resolve().parent.parent / "analysis_ready_rainfall.csv",
    Path("../EDA/output/analysis_ready_rainfall.csv"),
]
DATA_PATH = next((p for p in possible_paths if p.exists()), None)
if not DATA_PATH:
    raise FileNotFoundError("analysis_ready_rainfall.csv not found")

df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------------
# Basic feature engineering
# ------------------------------------------------------------------
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
df["day_of_year"] = df["date_time"].dt.dayofyear
df["year"] = df["date_time"].dt.year
features = ["latitude", "longitude", "day_of_year", "year"]
X = df[features]
y = df["rainfall_mm"]

# ------------------------------------------------------------------
# Train‑test split (80% train / 20% test)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# Fit the model
# ------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# Evaluation on the test set
# ------------------------------------------------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 80)
print("== LINEAR REGRESSION RESULTS ==")
print("=" * 80)
print(f"R² score          : {r2:.4f}")
print(f"RMSE (mm)         : {rmse:.2f}\n")
print("Coefficients (feature -> weight):")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:15s}: {coef:.6f}")
print(f"  {'Intercept':15s}: {model.intercept_:.6f}")

