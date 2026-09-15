# SunCharge Hub — AI-Powered Energy Management Prototype

A prototype for a community solar charging hub in Ganze, Kilifi County,
Kenya. Households bring rechargeable batteries and solar lanterns to the
hub to be charged, instead of buying individual solar systems.

## ⚠️ Everything here is simulated

There are **no physical sensors or real hub data** behind this prototype yet.
Stage 1 generates a realistic-looking but entirely **synthetic** dataset —
every file is prefixed `simulated_` so it can never be mistaken for real
pilot data. The point of this prototype is to prove the *approach* works
(the data pipeline, the models, the dashboard) so that when the real hub
starts collecting real transactions, you can plug that real data in with
minimal changes.

**Do not use any number from this prototype in a funding report, pitch, or
public claim about real-world impact.** Once the pilot launches, replace
the simulated CSVs with real logged data and rerun everything.

## What it does

Four core functions, built in six stages:

| # | Function | Stage(s) |
|---|----------|----------|
| 1 | Forecast daily/hourly community energy demand | Stage 3 |
| 2 | Monitor battery performance & flag maintenance needs | Stage 4 |
| 3 | Recommend solar/battery energy allocation | Stage 6 (rule-based, uses Stage 3's forecast) |
| 4 | Track kerosene avoided & estimated CO2e avoided | Stage 5 |

## The six stages, explained simply

**Stage 1 — `stage1_generate_dataset.py`**
Creates a fake but realistic 9-month history for the hub: 140 households
joining gradually, their lanterns/batteries, hourly solar generation and
battery-bank level (shaped by Kilifi's wet/dry seasons), and ~11,600
individual charging visits. It also secretly gives 12 devices a slowly
worsening fault, so we can test in Stage 4 whether our detection model
actually catches real problems.

**Stage 2 — `stage2_explore_data.py`**
Looks at the data and prints/plots things like: when is the hub busiest?
How much more solar do we generate on sunny vs rainy days? Is usage
growing over time? This is just "getting to know the data" before
building anything predictive.

**Stage 3 — `stage3_forecast_demand.py`**
Trains a model (Random Forest) that predicts how much energy will be
needed in a given hour, based on the hour of day, day of week, recent
weather, and recent usage trend. On the simulated test data it explains
about 97% of the variation in demand — a useful signal, but remember:
*trained on fake data, so it must be retrained once real data exists.*

**Stage 4 — `stage4_anomaly_detection.py`**
Looks at each device's charging speed over time. If a device is charging
noticeably slower/more erratically than it used to (or than similar
devices), it gets flagged "NEEDS CHECK" — a hint for the hub operator to
have a technician look at it before it fails completely. In this test it
caught 9 of the 12 deliberately-broken devices (75%).

**Stage 5 — `stage5_impact_calculator.py`**
For each household, compares kerosene use *before* the hub (baseline) to
kerosene use *now* (their latest visit). The difference = litres avoided.
Multiplied by a standard emission factor (2.52 kg CO2e per litre of
kerosene — a commonly published figure, not invented for this project),
this gives estimated CO2e avoided. **Confirm the emission factor with your
funder/regulator's preferred source before quoting it externally.**

**Stage 6 — `stage6_dashboard.py`**
A Streamlit web dashboard pulling everything together into one screen:
current battery level, solar generated, energy consumed, predicted
demand, peak charging period, households served, kerosene avoided, CO2e
avoided, and maintenance alerts.

## How to run it

```bash
pip install -r requirements.txt

# Run Stages 1-5 in order (generates data, trains models, computes impact)
python run_all_stages.py

# Then launch the dashboard
streamlit run stage6_dashboard.py
```

Or run any single stage on its own, e.g. `python stage4_anomaly_detection.py`
— each one prints a plain-language explanation of what it's doing and why.

## Folder structure after running

```
suncharge_hub/
├── data/                              <- Stage 1 output (simulated CSVs)
├── models/                            <- Stage 3's trained forecast model
├── outputs/                           <- Charts + Stage 4/5 result CSVs
├── stage1_generate_dataset.py
├── stage2_explore_data.py
├── stage3_forecast_demand.py
├── stage4_anomaly_detection.py
├── stage5_impact_calculator.py
├── stage6_dashboard.py
├── run_all_stages.py
├── requirements.txt
└── README.md   <- you are here
```

## Moving from prototype to real pilot

To use this with a real SunCharge Hub, you'd need to replace each simulated
input with a real source, roughly:

| Simulated file | Real-world replacement |
|---|---|
| `simulated_households.csv` | Household registration form + a baseline kerosene-use survey at sign-up |
| `simulated_devices.csv` | Device registry (barcode/ID tags on batteries & lanterns at hand-out) |
| `simulated_hub_energy_hourly.csv` | Readings from the solar charge controller / battery bank monitor |
| `simulated_charging_transactions.csv` | Logbook or simple app at the hub recording each charge (device ID, start/end battery %, time) |
| Kerosene "current use" figures | Periodic follow-up surveys (e.g. monthly) with households |

Once real data is flowing, retrain Stage 3's model and re-validate Stage
4's anomaly thresholds on real maintenance outcomes before trusting the
alerts.
