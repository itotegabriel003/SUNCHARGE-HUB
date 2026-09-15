"""
STAGE 1: Create and clean the simulated dataset for SunCharge Hub (Ganze, Kilifi County)

IMPORTANT — READ THIS FIRST:
Every number in the CSV files produced by this script is SIMULATED / SYNTHETIC.
It is built from reasonable assumptions (typical rural solar hub sizes, typical
household lantern/battery charging habits, typical Kilifi seasonal rainfall
patterns) so that the rest of the prototype (forecasting, anomaly detection,
dashboard) has realistic-looking data to work with.

NONE of it is real measured data from an actual SunCharge Hub pilot.
Once the real hub is running, these CSV files should be replaced by data
logged from real charging transactions, a real solar charge controller,
and real household kerosene-use surveys.

Every generated file is prefixed "simulated_" so it can never be confused
with real pilot data later.

Run:  python stage1_generate_dataset.py
Output: ./data/simulated_households.csv
        ./data/simulated_devices.csv
        ./data/simulated_charging_transactions.csv
        ./data/simulated_hub_energy_hourly.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# 0. Setup
# ----------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

START_DATE = datetime(2025, 1, 1)   # simulate 9 months of operation
NUM_MONTHS = 9
END_DATE = START_DATE + timedelta(days=NUM_MONTHS * 30)

N_HOUSEHOLDS = 140          # registered households at the hub
OPERATING_HOURS = list(range(6, 20))  # hub open 6am - 7:59pm (14 hours/day)

print("STAGE 1: Generating simulated SunCharge Hub dataset")
print(f"  Period       : {START_DATE.date()} to {END_DATE.date()} ({NUM_MONTHS} months)")
print(f"  Households   : {N_HOUSEHOLDS}")
print(f"  Random seed  : {SEED} (so results are reproducible)")
print("-" * 60)


# ----------------------------------------------------------------------
# 1. Households
# ----------------------------------------------------------------------
# Households join the hub gradually over the first ~4 months (realistic
# uptake curve rather than everyone joining on day 1).
join_offsets_days = np.clip(
    rng.exponential(scale=35, size=N_HOUSEHOLDS), 0, 120
).astype(int)

household_ids = [f"HH{str(i+1).zfill(3)}" for i in range(N_HOUSEHOLDS)]
household_size = rng.integers(2, 9, size=N_HOUSEHOLDS)  # people per household
distance_km = np.round(rng.uniform(0.3, 6.0, size=N_HOUSEHOLDS), 1)

# Baseline kerosene use BEFORE joining the hub (litres/week), correlated
# loosely with household size (bigger household -> more lighting need)
baseline_kerosene_l_per_week = np.round(
    0.35 * household_size + rng.normal(0, 0.4, size=N_HOUSEHOLDS), 2
).clip(0.3, None)

households = pd.DataFrame({
    "household_id": household_ids,
    "household_size": household_size,
    "distance_to_hub_km": distance_km,
    "join_date": [(START_DATE + timedelta(days=int(d))).date() for d in join_offsets_days],
    "baseline_kerosene_l_per_week": baseline_kerosene_l_per_week,
})
households.to_csv(DATA_DIR / "simulated_households.csv", index=False)
print(f"  -> simulated_households.csv  ({len(households)} rows)")


# ----------------------------------------------------------------------
# 2. Devices (rechargeable batteries & solar lanterns owned per household)
# ----------------------------------------------------------------------
device_rows = []
device_counter = 1
for hh in household_ids:
    n_devices = rng.choice([1, 1, 2, 2, 3], p=[0.35, 0.30, 0.20, 0.10, 0.05])
    for _ in range(n_devices):
        device_type = rng.choice(["solar_lantern", "rechargeable_battery"], p=[0.55, 0.45])
        capacity_wh = (
            round(rng.uniform(3, 8), 1) if device_type == "solar_lantern"
            else round(rng.uniform(20, 60), 1)
        )
        device_rows.append({
            "device_id": f"DEV{str(device_counter).zfill(4)}",
            "household_id": hh,
            "device_type": device_type,
            "rated_capacity_wh": capacity_wh,
            # devices age -> efficiency drifts down a bit over time, used later
            # to make anomaly injection realistic
            "install_month_offset": int(rng.integers(0, NUM_MONTHS)),
        })
        device_counter += 1

devices = pd.DataFrame(device_rows)
devices.to_csv(DATA_DIR / "simulated_devices.csv", index=False)
print(f"  -> simulated_devices.csv  ({len(devices)} rows)")


# ----------------------------------------------------------------------
# 3. Hub-level hourly energy: solar generation, storage, consumption
# ----------------------------------------------------------------------
# Kilifi coastal climate: long rains ~Apr-Jun, short rains ~Oct-Nov,
# otherwise mostly sunny. We simulate this as a seasonal cloud-cover factor.
def seasonal_cloud_factor(date):
    month = date.month
    if month in (4, 5, 6):        # long rains
        return rng.normal(0.55, 0.12)
    elif month in (10, 11):       # short rains
        return rng.normal(0.7, 0.12)
    else:                          # dry season
        return rng.normal(0.9, 0.08)

SYSTEM_PEAK_KW = 5.0                 # size of the hub's solar array (5 kWp)
BATTERY_BANK_CAPACITY_WH = 20000     # 20 kWh community battery bank

hub_rows = []
battery_level_wh = BATTERY_BANK_CAPACITY_WH * 0.6  # starting charge

current_date = START_DATE
while current_date < END_DATE:
    cloud_factor = float(np.clip(seasonal_cloud_factor(current_date), 0.15, 1.0))
    weather = "sunny" if cloud_factor > 0.75 else ("cloudy" if cloud_factor > 0.45 else "rainy")
    temperature_c = round(rng.normal(27 if weather == "sunny" else 24, 1.5), 1)

    for hour in OPERATING_HOURS:
        # Solar generation follows a bell curve peaking at ~midday (hour 12-13)
        solar_shape = max(0, np.sin(np.pi * (hour - 6) / 13))
        solar_gen_wh = SYSTEM_PEAK_KW * 1000 * solar_shape * cloud_factor * rng.uniform(0.9, 1.05)
        solar_gen_wh = max(0, round(solar_gen_wh, 1))

        # Community demand: two peaks -> mid-morning (drop-off) and evening (pickup)
        morning_peak = np.exp(-((hour - 9) ** 2) / 4) * 1.0
        evening_peak = np.exp(-((hour - 18) ** 2) / 3) * 1.4
        base_demand = 300 + 1800 * (morning_peak + evening_peak)
        # demand grows slowly as more households join over the months
        months_elapsed = (current_date - START_DATE).days / 30
        growth_factor = min(1.0, 0.3 + months_elapsed / NUM_MONTHS)
        consumption_wh = round(base_demand * growth_factor * rng.uniform(0.85, 1.15), 1)

        # Update battery bank: charge from solar, discharge to meet consumption
        battery_level_wh = battery_level_wh + solar_gen_wh - consumption_wh
        battery_level_wh = float(np.clip(battery_level_wh, 0, BATTERY_BANK_CAPACITY_WH))

        hub_rows.append({
            "date": current_date.date(),
            "hour": hour,
            "weather_condition": weather,
            "temperature_c": temperature_c,
            "solar_generation_wh": solar_gen_wh,
            "energy_consumed_wh": consumption_wh,
            "battery_bank_level_wh": round(battery_level_wh, 1),
            "battery_bank_level_pct": round(100 * battery_level_wh / BATTERY_BANK_CAPACITY_WH, 1),
        })

    current_date += timedelta(days=1)

hub_energy = pd.DataFrame(hub_rows)
hub_energy.to_csv(DATA_DIR / "simulated_hub_energy_hourly.csv", index=False)
print(f"  -> simulated_hub_energy_hourly.csv  ({len(hub_energy)} rows)")


# ----------------------------------------------------------------------
# 4. Charging transactions (household brings device -> gets charged)
# ----------------------------------------------------------------------
# Each household visits roughly 1-3 times per week, evening-weighted,
# only after they've joined. A handful of devices are given a slow
# "degrading" fault in the second half of the period, to be caught later
# by the anomaly-detection stage.

households_lookup = households.set_index("household_id")
devices_by_household = devices.groupby("household_id")["device_id"].apply(list).to_dict()
device_info = devices.set_index("device_id")

# Pick ~6% of devices to develop a fault partway through (clearly labelled,
# used only to test whether Stage 4's model can find them)
faulty_devices = set(
    devices["device_id"].sample(frac=0.06, random_state=SEED).tolist()
)
fault_start_offset = {d: rng.integers(60, NUM_MONTHS * 30 - 20) for d in faulty_devices}

transactions = []
txn_counter = 1
current_date = START_DATE
while current_date < END_DATE:
    day_offset = (current_date - START_DATE).days
    for hh in household_ids:
        join_date = households_lookup.loc[hh, "join_date"]
        if current_date.date() < join_date:
            continue
        # ~35% chance this household charges something on a given day
        if rng.random() > 0.35:
            continue
        hh_devices = devices_by_household.get(hh, [])
        if not hh_devices:
            continue
        device_id = rng.choice(hh_devices)
        cap = device_info.loc[device_id, "rated_capacity_wh"]

        # evening-weighted visit hour
        hour = int(np.clip(rng.normal(17, 2.5), 6, 19))

        start_pct = round(rng.uniform(5, 40), 1)
        # Normal charging efficiency ~ 0.85-0.95 of rated capacity gained
        efficiency = rng.uniform(0.85, 0.95)

        is_faulty_now = (
            device_id in faulty_devices and day_offset >= fault_start_offset[device_id]
        )
        if is_faulty_now:
            # Faulty device: charges slower / less fully, or drains oddly fast
            efficiency = rng.uniform(0.35, 0.6)

        end_pct = min(100, start_pct + efficiency * 100 * rng.uniform(0.8, 1.0))
        duration_min = round((end_pct - start_pct) / 100 * cap / rng.uniform(8, 14) * 60, 1)
        duration_min = max(5, duration_min)
        energy_delivered_wh = round((end_pct - start_pct) / 100 * cap, 2)

        transactions.append({
            "transaction_id": f"T{str(txn_counter).zfill(6)}",
            "date": current_date.date(),
            "hour": hour,
            "household_id": hh,
            "device_id": device_id,
            "battery_level_start_pct": start_pct,
            "battery_level_end_pct": round(end_pct, 1),
            "energy_delivered_wh": energy_delivered_wh,
            "charge_duration_min": duration_min,
            # weekly kerosene use reported at time of visit -> lets us track
            # the household's kerosene use trending down over time
            "household_kerosene_use_l_this_week": None,  # filled in below
        })
        txn_counter += 1
    current_date += timedelta(days=1)

transactions_df = pd.DataFrame(transactions)

# Simulate kerosene use trending down the longer a household has used the hub
def simulate_kerosene(row):
    baseline = households_lookup.loc[row["household_id"], "baseline_kerosene_l_per_week"]
    join_date = households_lookup.loc[row["household_id"], "join_date"]
    weeks_using_hub = max(0, (row["date"] - join_date).days / 7)
    # Adoption curve: usage drops toward ~15-30% of baseline over ~10 weeks
    floor_fraction = rng.uniform(0.15, 0.30)
    decay = np.exp(-weeks_using_hub / 4)
    current = baseline * (floor_fraction + (1 - floor_fraction) * decay)
    current *= rng.uniform(0.9, 1.1)  # weekly noise
    return round(max(0, current), 2)

transactions_df["household_kerosene_use_l_this_week"] = transactions_df.apply(simulate_kerosene, axis=1)
transactions_df["is_simulated_fault_injected"] = transactions_df["device_id"].isin(faulty_devices) & (
    transactions_df.apply(
        lambda r: (r["date"] - START_DATE.date()).days >= fault_start_offset.get(r["device_id"], 10**9),
        axis=1,
    )
)

transactions_df.to_csv(DATA_DIR / "simulated_charging_transactions.csv", index=False)
print(f"  -> simulated_charging_transactions.csv  ({len(transactions_df)} rows)")
print(f"     ({len(faulty_devices)} devices were given a simulated developing fault,")
print(f"      used later in Stage 4 to test the anomaly-detection model)")

print("-" * 60)
print("STAGE 1 COMPLETE. All files are in ./data/ and prefixed 'simulated_'.")
print("Remember: this is synthetic data for prototyping only.")
