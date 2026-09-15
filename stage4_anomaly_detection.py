"""
STAGE 4: Build a basic battery anomaly / maintenance detection model.

Goal (in plain language): look at every device's charging history and flag
ones that are behaving unusually -- e.g. charging much slower than similar
devices, gaining less charge per minute than they used to, or draining
unusually fast between visits. These are early warning signs a battery or
lantern may need maintenance or replacement, well before it fails
completely.

Model used: IsolationForest (scikit-learn). It's a good fit for a
prototype because it doesn't need labelled "this device is broken"
examples -- it just learns what "normal" charging behaviour looks like
across all devices, and flags the ones that don't fit the pattern. That
matches real life: at pilot launch we won't yet have confirmed maintenance
records to train a supervised model on.

Run:  python stage4_anomaly_detection.py
Output: ./outputs/stage4_flagged_devices.csv
        printed summary, including how well it caught the 12 devices
        that Stage 1 deliberately gave a simulated developing fault
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

print("STAGE 4: Detecting battery / device anomalies")
print("=" * 60)

txns = pd.read_csv(DATA_DIR / "simulated_charging_transactions.csv", parse_dates=["date"])
devices = pd.read_csv(DATA_DIR / "simulated_devices.csv")

# ---- Build per-transaction "charging efficiency" features ----
txns["charge_gained_pct"] = txns["battery_level_end_pct"] - txns["battery_level_start_pct"]
# Wh gained per minute of charging -- the key health signal. A healthy
# device charges at a fairly steady rate; a failing one charges slower
# and slower over time, or the rate becomes erratic.
txns["wh_per_minute"] = txns["energy_delivered_wh"] / txns["charge_duration_min"].clip(lower=1)

# ---- Aggregate to one row per device: its recent charging "fingerprint" ----
# We compare each device's LAST 5 visits against its FIRST 5 visits, since
# a developing fault shows up as a change over time, not just a low number.
def device_features(group):
    group = group.sort_values("date")
    first5 = group.head(5)
    last5 = group.tail(5)
    return pd.Series({
        "n_visits": len(group),
        "avg_wh_per_minute_recent": last5["wh_per_minute"].mean(),
        "avg_wh_per_minute_early": first5["wh_per_minute"].mean(),
        "avg_charge_gained_pct_recent": last5["charge_gained_pct"].mean(),
        "std_wh_per_minute_recent": last5["wh_per_minute"].std(),
    })

device_stats = txns.groupby("device_id").apply(device_features, include_groups=False).reset_index()
device_stats = device_stats.dropna()

# The core signal: how much has charging speed dropped from early visits
# to recent visits? A healthy device stays roughly flat; a failing one drops.
device_stats["pct_change_in_rate"] = (
    (device_stats["avg_wh_per_minute_recent"] - device_stats["avg_wh_per_minute_early"])
    / device_stats["avg_wh_per_minute_early"].replace(0, np.nan)
)
device_stats = device_stats.dropna()

FEATURES = [
    "avg_wh_per_minute_recent", "avg_charge_gained_pct_recent",
    "std_wh_per_minute_recent", "pct_change_in_rate",
]

model = IsolationForest(
    n_estimators=200, contamination=0.08, random_state=42
)
device_stats["anomaly_flag"] = model.fit_predict(device_stats[FEATURES])
# IsolationForest returns -1 for anomalies, 1 for normal -- convert to plain labels
device_stats["maintenance_alert"] = device_stats["anomaly_flag"].map({-1: "NEEDS CHECK", 1: "normal"})
device_stats["anomaly_score"] = model.decision_function(device_stats[FEATURES])  # lower = more unusual

device_stats = device_stats.merge(devices[["device_id", "household_id", "device_type"]], on="device_id", how="left")
device_stats = device_stats.sort_values("anomaly_score")

flagged = device_stats[device_stats["maintenance_alert"] == "NEEDS CHECK"]
print(f"\nAnalyzed {len(device_stats)} devices with enough charging history.")
print(f"Flagged {len(flagged)} devices as 'NEEDS CHECK' (unusual charging pattern):\n")
print(flagged[["device_id", "household_id", "device_type", "pct_change_in_rate", "avg_wh_per_minute_recent"]]
      .to_string(index=False))

# ---- Sanity check against the faults we deliberately planted in Stage 1 ----
known_faulty = set(txns.loc[txns["is_simulated_fault_injected"], "device_id"].unique())
flagged_ids = set(flagged["device_id"])
caught = known_faulty & flagged_ids
print(f"\n[Sanity check against Stage 1's planted faults]")
print(f"  Devices deliberately given a developing fault in the simulation : {len(known_faulty)}")
print(f"  Of those, correctly flagged by the model                       : {len(caught)}")
if known_faulty:
    print(f"  Detection rate on this simulated test                          : {100*len(caught)/len(known_faulty):.0f}%")

device_stats.to_csv(OUT_DIR / "stage4_flagged_devices.csv", index=False)
print(f"\nFull results saved to: {OUT_DIR / 'stage4_flagged_devices.csv'}")

print("\n" + "=" * 60)
print("STAGE 4 COMPLETE.")
print("NOTE: 'NEEDS CHECK' means 'worth a technician looking at it' -- it is")
print("a triage tool, not a diagnosis. And this detection rate is against")
print("SIMULATED faults; on real hardware, real failure patterns should be")
print("used to validate and retrain this model once available.")
