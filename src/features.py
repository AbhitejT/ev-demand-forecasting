import pandas as pd
from sqlalchemy import text

from src.config import FEATURE_DATA_PATH, ensure_directories, get_engine


def build_features(df):
    base = df.copy()
    base["charging_start_time"] = pd.to_datetime(base["charging_start_time"], utc=True)
    base["timestamp_hour"] = base["charging_start_time"].dt.floor("h")

    observed = base.groupby(["charging_station_id", "timestamp_hour"], as_index=False).agg(
        demand_count=("charging_station_id", "size"),
        total_energy_kwh=("energy_consumed_kwh", "sum"),
        avg_duration_minutes=("charging_duration_minutes", "mean"),
        avg_temperature_c=("temperature_c", "mean"),
    )
    observed = observed.rename(columns={"charging_station_id": "station_id"})

    stations = sorted(base["charging_station_id"].dropna().unique().tolist())
    min_hour = base["timestamp_hour"].min()
    max_hour = base["timestamp_hour"].max()
    hours = pd.date_range(min_hour, max_hour, freq="h")

    grid_rows = []
    for s in stations:
        for h in hours:
            grid_rows.append({"station_id": s, "timestamp_hour": h})
    grid = pd.DataFrame(grid_rows)

    grouped = grid.merge(observed, on=["station_id", "timestamp_hour"], how="left")
    grouped["demand_count"] = grouped["demand_count"].fillna(0).astype(int)
    grouped["total_energy_kwh"] = grouped["total_energy_kwh"].fillna(0.0)
    grouped["avg_duration_minutes"] = grouped["avg_duration_minutes"].fillna(0.0)

    station_temp = grouped.groupby("station_id")["avg_temperature_c"].transform("mean")
    global_temp = grouped["avg_temperature_c"].mean()
    grouped["avg_temperature_c"] = grouped["avg_temperature_c"].fillna(station_temp)
    grouped["avg_temperature_c"] = grouped["avg_temperature_c"].fillna(global_temp)

    grouped["hour_of_day"] = grouped["timestamp_hour"].dt.hour
    grouped["day_of_week"] = grouped["timestamp_hour"].dt.dayofweek
    grouped["is_weekend"] = grouped["day_of_week"].isin([5, 6])

    grouped = grouped.sort_values(["station_id", "timestamp_hour"])

    grouped["lag_1h_demand"] = grouped.groupby("station_id")["demand_count"].shift(1).fillna(0)
    grouped["lag_3h_demand"] = grouped.groupby("station_id")["demand_count"].shift(3).fillna(0)
    grouped["lag_24h_demand"] = grouped.groupby("station_id")["demand_count"].shift(24).fillna(0)

    rolling_3h = grouped.groupby("station_id")["demand_count"].shift(1).rolling(3, min_periods=1).mean()
    grouped["rolling_mean_3h_demand"] = rolling_3h.fillna(0.0)

    grouped["lag_1h_energy"] = grouped.groupby("station_id")["total_energy_kwh"].shift(1).fillna(0.0)
    grouped["lag_24h_energy"] = grouped.groupby("station_id")["total_energy_kwh"].shift(24).fillna(0.0)

    return grouped


def main():
    ensure_directories()
    engine = get_engine()
    with engine.connect() as conn:
        clean = pd.read_sql(text("SELECT * FROM charging_sessions_clean"), conn)

    feats = build_features(clean)
    feats.to_csv(FEATURE_DATA_PATH, index=False)

    with engine.begin() as conn:
        feats.to_sql("station_hourly_features", conn, if_exists="replace", index=False)

    print("Built " + str(len(feats)) + " hourly feature rows at " + str(FEATURE_DATA_PATH))


if __name__ == "__main__":
    main()
