"""
STAGE 6: SunCharge Hub -- Streamlit prototype dashboard.

This pulls together everything from Stages 1-5 into one screen an operator
could look at. It is deliberately simple: no logins, no database -- it
just reads the CSV files already generated and displays them.

Run:   streamlit run stage6_dashboard.py

If you haven't run Stages 1-5 yet, this dashboard will tell you what's
missing and how to generate it.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"

st.set_page_config(page_title="SunCharge Hub -- Prototype Dashboard", layout="wide")

st.title("SunCharge Hub — Ganze, Kilifi County")
st.caption("AI-powered energy management prototype")

st.warning(
    "**PROTOTYPE NOTICE:** Every number on this dashboard comes from a "
    "*simulated* dataset (see Stage 1), built to test how this system would "
    "work -- it is **not** real data from an operating SunCharge Hub. "
    "Once the pilot is running, the data files behind this dashboard should "
    "be replaced with real logged transactions, real solar-controller "
    "readings, and real household kerosene surveys.",
    icon="⚠️",
)

# ---------------------------------------------------------------------
# Load data (with friendly errors if a stage hasn't been run yet)
# ---------------------------------------------------------------------
required_files = {
    "Hub energy":       DATA_DIR / "simulated_hub_energy_hourly.csv",
    "Transactions":     DATA_DIR / "simulated_charging_transactions.csv",
    "Households":       DATA_DIR / "simulated_households.csv",
    "Devices":          DATA_DIR / "simulated_devices.csv",
    "Anomaly results":  OUT_DIR / "stage4_flagged_devices.csv",
    "Impact summary":   OUT_DIR / "stage5_impact_summary.csv",
    "Household impact": OUT_DIR / "stage5_household_impact.csv",
}
missing = [name for name, path in required_files.items() if not path.exists()]
if missing:
    st.error(
        "Missing data for: " + ", ".join(missing) +
        ". Run stage1_generate_dataset.py through stage5_impact_calculator.py first."
    )
    st.stop()

hub = pd.read_csv(required_files["Hub energy"], parse_dates=["date"])
txns = pd.read_csv(required_files["Transactions"], parse_dates=["date"])
households = pd.read_csv(required_files["Households"])
devices = pd.read_csv(required_files["Devices"])
flagged = pd.read_csv(required_files["Anomaly results"])
impact_summary = pd.read_csv(required_files["Impact summary"]).iloc[0]
household_impact = pd.read_csv(required_files["Household impact"])

latest_date = hub["date"].max()
latest_day_data = hub[hub["date"] == latest_date].sort_values("hour")

# ---------------------------------------------------------------------
# Top KPI row
# ---------------------------------------------------------------------
st.subheader(f"Snapshot -- most recent simulated day ({latest_date.date()})")

current_battery_pct = latest_day_data["battery_bank_level_pct"].iloc[-1]
solar_today_wh = latest_day_data["solar_generation_wh"].sum()
consumed_today_wh = latest_day_data["energy_consumed_wh"].sum()
peak_hour_today = latest_day_data.loc[latest_day_data["energy_consumed_wh"].idxmax(), "hour"]
users_served_today = txns[txns["date"] == latest_date]["household_id"].nunique()

kpi_cols = st.columns(5)
kpi_cols[0].metric("Battery bank level", f"{current_battery_pct:.0f}%")
kpi_cols[1].metric("Solar generated today", f"{solar_today_wh/1000:.1f} kWh")
kpi_cols[2].metric("Energy consumed today", f"{consumed_today_wh/1000:.1f} kWh")
kpi_cols[3].metric("Peak charging hour", f"{int(peak_hour_today)}:00")
kpi_cols[4].metric("Households served today", f"{users_served_today}")

st.divider()

# ---------------------------------------------------------------------
# Demand forecast
# ---------------------------------------------------------------------
st.subheader("Predicted demand (next-hour style forecast)")
model_path = BASE / "models" / "demand_forecast_model.joblib"
if model_path.exists():
    import joblib
    bundle = joblib.load(model_path)
    model, feat_cols = bundle["model"], bundle["features"]

    # Build one feature row per hour of the latest day, using the same
    # feature logic as Stage 3, to show a "predicted vs actual" strip.
    hub_sorted = hub.sort_values(["date", "hour"]).copy()
    hub_sorted["day_of_week"] = hub_sorted["date"].dt.dayofweek
    hub_sorted["month"] = hub_sorted["date"].dt.month
    hub_sorted["is_weekend"] = (hub_sorted["day_of_week"] >= 5).astype(int)
    hub_sorted["weather_code"] = hub_sorted["weather_condition"].map({"sunny": 0, "cloudy": 1, "rainy": 2})
    hub_sorted["consumption_same_hour_yesterday"] = hub_sorted.groupby("hour")["energy_consumed_wh"].shift(1)
    hub_sorted["consumption_rolling_3day_avg"] = (
        hub_sorted.groupby("hour")["energy_consumed_wh"].transform(lambda s: s.shift(1).rolling(3).mean())
    )
    recent = hub_sorted.dropna(subset=feat_cols).tail(14)  # last 14 hourly records
    recent = recent.copy()
    recent["predicted_wh"] = model.predict(recent[feat_cols])

    chart_df = recent.set_index(recent["date"].astype(str) + " " + recent["hour"].astype(str) + ":00")[
        ["energy_consumed_wh", "predicted_wh"]
    ].rename(columns={"energy_consumed_wh": "Actual", "predicted_wh": "Predicted"})
    st.line_chart(chart_df)
    st.caption(
        "Model: Random Forest trained on simulated history (hour of day, day of week, "
        "weather, recent trend). Retrain on real transaction data once the pilot is live."
    )
else:
    st.info("Run stage3_forecast_demand.py to generate the forecasting model.")

st.divider()

# ---------------------------------------------------------------------
# Two-column layout: allocation guidance + maintenance alerts
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Energy allocation guidance")
    # Simple rule-based recommendation for the prototype (Stage 3's forecast
    # feeds this -- a more advanced optimizer could replace this rule later).
    if current_battery_pct < 30:
        st.error(
            f"Battery bank at {current_battery_pct:.0f}%. Recommend: "
            "prioritize essential charging only (lanterns over larger batteries) "
            "until solar generation recovers."
        )
    elif current_battery_pct < 60:
        st.warning(
            f"Battery bank at {current_battery_pct:.0f}%. Recommend: "
            "serve all queued households, but monitor closely if today is cloudy/rainy."
        )
    else:
        st.success(
            f"Battery bank at {current_battery_pct:.0f}%. Recommend: "
            "normal operations -- capacity available for all expected demand."
        )
    st.caption(
        "This is a simple rule-based first pass for the prototype. A future "
        "version could optimize allocation using the demand forecast directly."
    )

with col2:
    st.subheader("Battery / device maintenance alerts")
    needs_check = flagged[flagged["maintenance_alert"] == "NEEDS CHECK"]
    st.metric("Devices flagged for check", len(needs_check))
    if len(needs_check):
        st.dataframe(
            needs_check[["device_id", "household_id", "device_type", "pct_change_in_rate"]]
            .rename(columns={"pct_change_in_rate": "change in charge rate"})
            .style.format({"change in charge rate": "{:.0%}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No devices currently flagged.")
    st.caption("Flag = unusual charging pattern -- worth a technician's look, not a confirmed fault.")

st.divider()

# ---------------------------------------------------------------------
# Environmental impact
# ---------------------------------------------------------------------
st.subheader("Environmental impact (kerosene displacement)")
imp_cols = st.columns(4)
imp_cols[0].metric("Kerosene avoided", f"{impact_summary['total_kerosene_avoided_l_per_week']:.0f} L/week")
imp_cols[1].metric("CO2e avoided", f"{impact_summary['total_co2e_avoided_kg_per_week']:.0f} kg/week")
imp_cols[2].metric("Avg reduction per household", f"{impact_summary['avg_pct_reduction_per_household']:.0f}%")
imp_cols[3].metric(
    "Est. annual CO2e avoided",
    f"{impact_summary['total_co2e_avoided_kg_annualised_estimate']/1000:.1f} t",
)
st.caption(
    f"CO2e estimated using {impact_summary['emission_factor_used_kg_co2e_per_litre']} kg CO2e per litre "
    "of kerosene (standard published combustion factor -- confirm against your preferred source before "
    "using in an official report). Annual figures are a simple extrapolation from current weekly rates."
)

with st.expander("See household-level impact detail"):
    st.dataframe(
        household_impact[
            ["household_id", "household_size", "baseline_kerosene_l_per_week",
             "current_kerosene_l_per_week", "kerosene_avoided_l_per_week", "co2e_avoided_kg_per_week"]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "SunCharge Hub AI prototype -- built for exploration and demonstration purposes. "
    "All figures above are derived from simulated data (see stage1_generate_dataset.py) "
    "and must not be quoted as real-world pilot results."
)
