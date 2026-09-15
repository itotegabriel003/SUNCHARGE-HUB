"""
Runs Stages 1-5 in order (everything except the dashboard, which you launch
separately since it needs 'streamlit run', not 'python').

Usage:  python run_all_stages.py
Then:   streamlit run stage6_dashboard.py
"""

import subprocess
import sys

stages = [
    "stage1_generate_dataset.py",
    "stage2_explore_data.py",
    "stage3_forecast_demand.py",
    "stage4_anomaly_detection.py",
    "stage5_impact_calculator.py",
]

for stage in stages:
    print(f"\n{'#' * 70}\n# Running {stage}\n{'#' * 70}\n")
    result = subprocess.run([sys.executable, stage])
    if result.returncode != 0:
        print(f"\n!!! {stage} failed. Stopping here so you can fix it before continuing.")
        sys.exit(1)

print("\nAll stages 1-5 complete. Now run:  streamlit run stage6_dashboard.py")
