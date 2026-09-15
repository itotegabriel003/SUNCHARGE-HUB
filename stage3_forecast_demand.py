"""
STAGE 3: Build a simple demand forecasting model.

Goal (in plain language): predict how much energy (in Wh) the community
will need in a given hour, using patterns from past hours -- what hour of
day it is, what day of week, the season/weather, and how much was used
recently. This lets the hub operator plan ahead (e.g. "expect a big draw
at 6pm today, make sure the battery bank is topped up by then").

Model used: RandomForestRegressor (scikit-learn). It's a good first choice
for a prototype because it handles non-linear daily/seasonal patterns
well, needs little tuning, and gives us a "feature importance" list we
can show to explain WHY it predicts what it predicts (useful for a
non-technical audience).

Run:  python stage3_forecast_demand.py
Output: ./models/demand_forecast_model.joblib
        printed accuracy metrics
        ./outputs/stage3_forecast_vs_actual.png
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
OUT_DIR = Path(__file__).parent / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

print("STAGE 3: Building the demand forecasting model")
print("=" * 60)

hub = pd.read_csv(DATA_DIR / "simulated_hub_energy_hourly.csv", parse_dates=["date"])
hub = hub.sort_values(["date", "hour"]).reset_index(drop=True)

# ---- Feature engineering: turn raw columns into things a model can learn from ----
hub["day_of_week"] = hub["date"].dt.dayofweek       # 0=Monday
hub["month"] = hub["date"].dt.month
hub["is_weekend"] = (hub["day_of_week"] >= 5).astype(int)
weather_map = {"sunny": 0, "cloudy": 1, "rainy": 2}
hub["weather_code"] = hub["weather_condition"].map(weather_map)

# "Lag" features: what was consumption at this same hour yesterday, and
# what was the average consumption over the previous 3 days? This gives
# the model a sense of recent trend/growth, not just hour-of-day pattern.
hub["consumption_same_hour_yesterday"] = hub.groupby("hour")["energy_consumed_wh"].shift(1)
hub["consumption_rolling_3day_avg"] = (
    hub.groupby("hour")["energy_consumed_wh"].transform(lambda s: s.shift(1).rolling(3).mean())
)
hub = hub.dropna(subset=["consumption_same_hour_yesterday", "consumption_rolling_3day_avg"]).reset_index(drop=True)

FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "weather_code",
    "temperature_c", "consumption_same_hour_yesterday", "consumption_rolling_3day_avg",
]
TARGET = "energy_consumed_wh"

X = hub[FEATURES]
y = hub[TARGET]

# Time-based split (train on the first ~80% of days, test on the most recent
# ~20%) -- this is more realistic than a random split for forecasting,
# because in real life you always predict the future from the past.
split_idx = int(len(hub) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
avg_actual = y_test.mean()

print(f"\nTrained on {len(X_train)} hourly records, tested on the most recent {len(X_test)} hours.")
print(f"\n[Model accuracy on unseen (test) data]")
print(f"  Mean Absolute Error : {mae:.0f} Wh  (average prediction is off by about this much)")
print(f"  As a % of typical demand ({avg_actual:.0f} Wh): {100*mae/avg_actual:.1f}%")
print(f"  R^2 score            : {r2:.2f}  (1.0 = perfect, 0 = no better than guessing the average)")

# ---- Feature importance, explained simply ----
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n[What the model relies on most, to predict demand]")
for feat, imp in importances.items():
    print(f"  {feat:35s}: {imp*100:5.1f}%")

# ---- Save chart: predicted vs actual for the test period ----
plt.figure(figsize=(11, 4))
plt.plot(y_test.values[:14*7], label="Actual", color="#34495e")
plt.plot(y_pred[:14*7], label="Predicted", color="#e74c3c", linestyle="--")
plt.title("Forecast vs Actual Hourly Demand -- first week of test period (simulated data)")
plt.ylabel("Wh")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "stage3_forecast_vs_actual.png")
plt.close()

# ---- Save the trained model so the dashboard (Stage 6) can reuse it ----
joblib.dump({"model": model, "features": FEATURES}, MODEL_DIR / "demand_forecast_model.joblib")

print("\n" + "=" * 60)
print("STAGE 3 COMPLETE.")
print(f"Model saved to: {MODEL_DIR / 'demand_forecast_model.joblib'}")
print(f"Chart saved to: {OUT_DIR / 'stage3_forecast_vs_actual.png'}")
print("\nNOTE: accuracy here reflects how well the model learns patterns in the")
print("SIMULATED data. Once real transaction data is collected at the pilot,")
print("this model must be retrained on real data before its numbers are trusted.")
