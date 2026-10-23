"""
Forecasting monthly electricity demand (MW) in Lagos State with XGBoost.

BSc thesis, University of Lagos: "A Comparative Analysis: Forecasting
Electricity Consumption in Lagos State Using Machine Learning Models".

Data: LGMW.csv with columns DATE and LGMW (monthly demand in megawatts).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

color_pal = sns.color_palette()
plt.style.use("fivethirtyeight")

TARGET = "LGMW"
FEATURES = ["month", "quarter", "year"]
TEST_START = "2020-01-28"


def create_features(df):
    """Create calendar features from the datetime index."""
    df = df.copy()
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    return df


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
LGMW = pd.read_csv("LGMW.csv", parse_dates=["DATE"]).set_index("DATE").sort_index()
print(LGMW.describe())

LGMW.plot(style=".", figsize=(15, 5), title="Lagos monthly energy use (MW)")
plt.show()

# ---------------------------------------------------------------------------
# 2. Time-based train/test split (no shuffling)
# ---------------------------------------------------------------------------
train = LGMW.loc[LGMW.index < TEST_START]
test = LGMW.loc[LGMW.index >= TEST_START]

fig, ax = plt.subplots(figsize=(15, 5))
train[TARGET].plot(ax=ax, label="Training set", title="Train/Test Split")
test[TARGET].plot(ax=ax, label="Test set")
ax.axvline(pd.Timestamp(TEST_START), color="black", ls="--")
ax.legend()
plt.show()

# Seasonal pattern by month
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=create_features(LGMW), x="month", y=TARGET, ax=ax)
ax.set_title("MW by month")
plt.show()

# ---------------------------------------------------------------------------
# 3. Features, with a validation set carved from the END of the training data.
#    Early stopping uses the validation set only, so the test set stays unseen.
# ---------------------------------------------------------------------------
train = create_features(train)
test = create_features(test)

fit_part = train.iloc[:-12]
valid_part = train.iloc[-12:]

X_fit, y_fit = fit_part[FEATURES], fit_part[TARGET]
X_valid, y_valid = valid_part[FEATURES], valid_part[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# ---------------------------------------------------------------------------
# 4. Train XGBoost
# ---------------------------------------------------------------------------
reg = xgb.XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=50,
    learning_rate=0.01,
)
reg.fit(
    X_fit,
    y_fit,
    eval_set=[(X_fit, y_fit), (X_valid, y_valid)],
    verbose=100,
)
best_n_trees = reg.best_iteration + 1
print(f"Best number of trees (chosen on validation set): {best_n_trees}")

feature_importance = pd.DataFrame(
    data=reg.feature_importances_,
    index=reg.feature_names_in_,
    columns=["importance"],
)
feature_importance.sort_values("importance").plot(kind="barh", title="Feature Importance")
plt.show()

# ---------------------------------------------------------------------------
# 5. Evaluate on the held-out test set
# ---------------------------------------------------------------------------
test["prediction"] = reg.predict(X_test)

ax = LGMW[TARGET].plot(figsize=(15, 5), label="Actual")
test["prediction"].plot(ax=ax, style="-", label="Prediction")
ax.set_title("Actual demand and test-set prediction")
ax.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(y_test, test["prediction"]))
mae = mean_absolute_error(y_test, test["prediction"])
mape = mean_absolute_percentage_error(y_test, test["prediction"])
print(f"Test RMSE: {rmse:0.2f} MW")
print(f"Test MAE:  {mae:0.2f} MW")
print(f"Test MAPE: {mape * 100:0.1f}%")

# ---------------------------------------------------------------------------
# 6. Retrain on all data and project forward to December 2026.
#    Note: tree models cannot extrapolate beyond years seen in training, so
#    future years largely repeat the learned monthly seasonal pattern.
# ---------------------------------------------------------------------------
full = create_features(LGMW)
reg_full = xgb.XGBRegressor(n_estimators=best_n_trees, learning_rate=0.01)
reg_full.fit(full[FEATURES], full[TARGET])

future_index = pd.date_range(
    start=LGMW.index.max() + pd.DateOffset(months=1),
    end="2026-12-31",
    freq="MS",
)
future = create_features(pd.DataFrame(index=future_index))
future["prediction"] = reg_full.predict(future[FEATURES])

ax = LGMW[TARGET].plot(figsize=(15, 5), label="Actual")
future["prediction"].plot(ax=ax, color=color_pal[4], label="Projection")
ax.set_title("Monthly demand projection to December 2026")
ax.legend()
plt.show()
