import json

import pandas as pd

from src.config import FEATURE_DATA_PATH, REPORT_DIR, ensure_directories


def main():
    ensure_directories()
    df = pd.read_csv(FEATURE_DATA_PATH, parse_dates=["timestamp_hour"])

    hour_avg = df.groupby(df["timestamp_hour"].dt.hour)["demand_count"].mean().sort_values()
    best_hours = hour_avg.reset_index(name="avg_hourly_demand")
    best_hours["rank"] = best_hours["avg_hourly_demand"].rank(method="dense")
    best_hours.to_csv(REPORT_DIR / "optimal_charging_hours.csv", index=False)

    station_summary = df.groupby("station_id").agg(
        avg_demand=("demand_count", "mean"),
        avg_energy_kwh=("total_energy_kwh", "mean"),
        peak_demand=("demand_count", "max"),
    ).reset_index()
    station_summary.to_csv(REPORT_DIR / "station_usage_summary.csv", index=False)

    payload = {
        "lowest_demand_hours": best_hours.head(3)["timestamp_hour"].tolist(),
        "highest_demand_hours": best_hours.tail(3)["timestamp_hour"].tolist(),
        "stations_analyzed": int(station_summary["station_id"].nunique()),
    }
    out_path = REPORT_DIR / "outcomes_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
