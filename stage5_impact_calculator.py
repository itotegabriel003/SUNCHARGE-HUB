"""
STAGE 5: Calculate kerosene displacement and estimated CO2e avoided.

Goal (in plain language): for each household, compare how much kerosene
they used BEFORE joining the hub (baseline) to how much they're using NOW
(most recent reported figure). The difference is litres of kerosene
"avoided" per week. We then convert litres avoided into an estimated
kilograms of CO2-equivalent (CO2e) avoided, using a published emission
factor for kerosene combustion.

IMPORTANT ON THE EMISSION FACTOR:
This script uses 2.52 kg CO2e per litre of kerosene burned. This is a
commonly cited figure for kerosene/paraffin combustion emissions (based on
kerosene's carbon content and standard combustion assumptions), NOT a
number invented for this prototype. Different sources (IPCC guidelines,
national inventories, specific lamp studies) give slightly different
values (often 2.5-3.1 kg CO2e/litre depending on what's included, e.g.
black carbon). Before this number is used in any official impact report,
the SunCharge Hub team should confirm which factor a funder or regulator
expects, and cite it explicitly.

Also note: this script converts BASELINE and CURRENT kerosene use, which
in this prototype are SIMULATED numbers (see Stage 1). At the real pilot,
these two numbers must come from actual household surveys
(a "before" survey at sign-up, and periodic "current use" check-ins).

Run:  python stage5_impact_calculator.py
Output: ./outputs/stage5_household_impact.csv
        ./outputs/stage5_impact_summary.csv
        printed plain-language summary
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Published-style emission factor for kerosene combustion (kg CO2e per litre).
# See docstring above -- confirm/replace with your funder's preferred source
# before using in a real report.
CO2E_PER_LITRE_KEROSENE = 2.52

print("STAGE 5: Calculating kerosene displacement and CO2e avoided")
print("=" * 60)

households = pd.read_csv(DATA_DIR / "simulated_households.csv", parse_dates=["join_date"])
txns = pd.read_csv(DATA_DIR / "simulated_charging_transactions.csv", parse_dates=["date"])

# Most recent reported kerosene use per household (their latest visit)
latest_kerosene = (
    txns.sort_values("date")
    .groupby("household_id")["household_kerosene_use_l_this_week"]
    .last()
    .rename("current_kerosene_l_per_week")
)

impact = households.set_index("household_id").join(latest_kerosene, how="left")
impact = impact.reset_index()

# Households with no transactions yet (very recent joiners) have no "current"
# figure -- for them we conservatively assume no change yet.
impact["current_kerosene_l_per_week"] = impact["current_kerosene_l_per_week"].fillna(
    impact["baseline_kerosene_l_per_week"]
)

impact["kerosene_avoided_l_per_week"] = (
    impact["baseline_kerosene_l_per_week"] - impact["current_kerosene_l_per_week"]
).clip(lower=0)

impact["co2e_avoided_kg_per_week"] = impact["kerosene_avoided_l_per_week"] * CO2E_PER_LITRE_KEROSENE

# Also express as an annualised estimate, clearly labelled as an extrapolation
impact["kerosene_avoided_l_annualised_estimate"] = impact["kerosene_avoided_l_per_week"] * 52
impact["co2e_avoided_kg_annualised_estimate"] = impact["co2e_avoided_kg_per_week"] * 52

impact.to_csv(OUT_DIR / "stage5_household_impact.csv", index=False)

# ---- Community-wide summary ----
total_l_week = impact["kerosene_avoided_l_per_week"].sum()
total_co2e_week = impact["co2e_avoided_kg_per_week"].sum()
avg_pct_reduction = (
    (impact["kerosene_avoided_l_per_week"] / impact["baseline_kerosene_l_per_week"].replace(0, pd.NA))
    .mean() * 100
)

summary = pd.DataFrame([{
    "households_reporting": len(impact),
    "total_kerosene_avoided_l_per_week": round(total_l_week, 1),
    "total_co2e_avoided_kg_per_week": round(total_co2e_week, 1),
    "avg_pct_reduction_per_household": round(avg_pct_reduction, 1),
    "total_kerosene_avoided_l_annualised_estimate": round(total_l_week * 52, 1),
    "total_co2e_avoided_kg_annualised_estimate": round(total_co2e_week * 52, 1),
    "emission_factor_used_kg_co2e_per_litre": CO2E_PER_LITRE_KEROSENE,
}])
summary.to_csv(OUT_DIR / "stage5_impact_summary.csv", index=False)

print(f"\nHouseholds included         : {len(impact)}")
print(f"Avg weekly reduction        : {avg_pct_reduction:.0f}% less kerosene per household (vs their own baseline)")
print(f"Community kerosene avoided  : {total_l_week:.1f} litres/week")
print(f"Community CO2e avoided      : {total_co2e_week:.1f} kg CO2e/week")
print(f"\nAnnualised ESTIMATE (simple extrapolation, not a measured annual figure):")
print(f"  Kerosene avoided  : {total_l_week*52:,.0f} litres/year")
print(f"  CO2e avoided      : {total_co2e_week*52:,.0f} kg CO2e/year  (~{total_co2e_week*52/1000:.1f} tonnes CO2e/year)")

print("\n" + "=" * 60)
print("STAGE 5 COMPLETE.")
print(f"Files saved: {OUT_DIR / 'stage5_household_impact.csv'}")
print(f"             {OUT_DIR / 'stage5_impact_summary.csv'}")
print("\nREMINDER: baseline & current kerosene figures are SIMULATED in this")
print("prototype. Real impact reporting requires real household survey data.")
