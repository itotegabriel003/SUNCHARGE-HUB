"""
STAGE 2: Explore energy generation, consumption and charging patterns.

This script reads the simulated data from Stage 1 and prints a plain-language
summary of what's going on, plus saves a few charts as PNG images so the
patterns can be seen at a glance.

Run:  python stage2_explore_data.py
Output: printed summary + PNG charts in ./outputs/
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

hub = pd.read_csv(DATA_DIR / "simulated_hub_energy_hourly.csv", parse_dates=["date"])
txns = pd.read_csv(DATA_DIR / "simulated_charging_transactions.csv", parse_dates=["date"])
households = pd.read_csv(DATA_DIR / "simulated_households.csv", parse_dates=["join_date"])

print("STAGE 2: Exploring the simulated SunCharge Hub data")
print("=" * 60)

# ---- 1. Generation vs consumption, overall ----
total_gen_kwh = hub["solar_generation_wh"].sum() / 1000
total_cons_kwh = hub["energy_consumed_wh"].sum() / 1000
print(f"\n[Generation vs Consumption over {hub['date'].nunique()} days]")
print(f"  Total solar generated : {total_gen_kwh:,.0f} kWh")
print(f"  Total energy consumed : {total_cons_kwh:,.0f} kWh")
print(f"  Self-sufficiency      : {100 * total_gen_kwh / total_cons_kwh:.0f}% of demand met directly by solar+battery")

# ---- 2. Hourly demand pattern (average by hour of day) ----
hourly_avg = hub.groupby("hour")["energy_consumed_wh"].mean().sort_values(ascending=False)
peak_hour = hourly_avg.index[0]
print(f"\n[Peak charging period]")
print(f"  Busiest hour of the day on average: {peak_hour}:00 "
      f"({hourly_avg.iloc[0]:.0f} Wh consumed on average)")

plt.figure(figsize=(8, 4))
hub.groupby("hour")["energy_consumed_wh"].mean().plot(kind="bar", color="#e67e22")
plt.title("Average Energy Consumption by Hour of Day (simulated)")
plt.xlabel("Hour")
plt.ylabel("Avg Wh consumed")
plt.tight_layout()
plt.savefig(OUT_DIR / "stage2_hourly_demand_pattern.png")
plt.close()

# ---- 3. Battery bank level over time ----
plt.figure(figsize=(10, 4))
daily_batt = hub.groupby("date")["battery_bank_level_pct"].mean()
daily_batt.plot(color="#2980b9")
plt.title("Community Battery Bank Level Over Time (simulated, daily average)")
plt.ylabel("% charged")
plt.tight_layout()
plt.savefig(OUT_DIR / "stage2_battery_bank_trend.png")
plt.close()

# ---- 4. Weather impact on generation ----
weather_gen = hub.groupby("weather_condition")["solar_generation_wh"].mean().sort_values(ascending=False)
print(f"\n[Effect of weather on solar generation]")
for w, val in weather_gen.items():
    print(f"  {w:8s}: {val:.0f} Wh/hour average")

# ---- 5. Charging transaction volume over time ----
weekly_txns = txns.set_index("date").resample("W").size()
plt.figure(figsize=(10, 4))
weekly_txns.plot(color="#27ae60")
plt.title("Charging Transactions per Week (simulated)")
plt.ylabel("Number of charges")
plt.tight_layout()
plt.savefig(OUT_DIR / "stage2_weekly_transactions.png")
plt.close()

print(f"\n[Adoption / usage]")
print(f"  Households registered : {len(households)}")
print(f"  Total charging visits : {len(txns):,}")
print(f"  Avg visits/household  : {len(txns) / len(households):.1f} over the {NUM_MONTHS if (NUM_MONTHS := 9) else 9} months")

# ---- 6. Device type mix ----
devices = pd.read_csv(DATA_DIR / "simulated_devices.csv")
mix = devices["device_type"].value_counts()
print(f"\n[Device mix]")
for dtype, count in mix.items():
    print(f"  {dtype:22s}: {count}")

print("\n" + "=" * 60)
print("STAGE 2 COMPLETE.")
print(f"Charts saved to: {OUT_DIR}/")
print(" - stage2_hourly_demand_pattern.png")
print(" - stage2_battery_bank_trend.png")
print(" - stage2_weekly_transactions.png")
