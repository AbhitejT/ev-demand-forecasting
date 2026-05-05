import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from acnportal.acndata import DataClient
from sqlalchemy import text

from src.config import (
    ACN_API_TOKEN,
    ACN_RAW_DATA_PATH,
    ACN_SITES,
    CLEAN_DATA_PATH,
    ensure_directories,
    get_engine,
    write_df,
)


NUMERIC_COLS = [
    "battery_capacity_kwh",
    "energy_consumed_kwh",
    "charging_duration_minutes",
    "charging_cost",
    "state_of_charge_start",
    "state_of_charge_end",
    "distance_driven_km",
    "temperature_c",
    "vehicle_age_years",
]

STRING_COLS = ["vehicle_model", "charger_type", "user_type", "day_of_week", "time_of_day"]


def parse_dt(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", ""))


def time_of_day(dt_series):
    hours = dt_series.dt.hour
    labels = []
    for h in hours:
        if h >= 6 and h < 12:
            labels.append("Morning")
        elif h >= 12 and h < 18:
            labels.append("Afternoon")
        else:
            labels.append("Night")
    return pd.Series(labels, index=dt_series.index)


def fetch_acn():
    ensure_directories()
    if not ACN_API_TOKEN:
        raise ValueError("ACN_API_TOKEN is required to fetch ACN-Data sessions.")

    start = parse_dt(os.getenv("ACN_START"))
    end = parse_dt(os.getenv("ACN_END"))
    max_sessions = int(os.getenv("ACN_MAX_SESSIONS", "20000"))

    client = DataClient(api_token=ACN_API_TOKEN, url="https://ev.caltech.edu/api/v1/")
    records = []

    for site in ACN_SITES:
        if start or end:
            iterator = client.get_sessions_by_time(site=site, start=start, end=end, timeseries=False)
        else:
            iterator = client.get_sessions(site=site, timeseries=False)

        for session in iterator:
            row = {
                "sessionID": session.get("sessionID"),
                "stationID": session.get("stationID"),
                "siteID": session.get("siteID", site),
                "clusterID": session.get("clusterID"),
                "userID": session.get("userID"),
                "connectionTime": session.get("connectionTime"),
                "disconnectTime": session.get("disconnectTime"),
                "doneChargingTime": session.get("doneChargingTime"),
                "kWhDelivered": session.get("kWhDelivered"),
                "timezone": session.get("timezone"),
            }
            records.append(row)

            if len(records) % 1000 == 0:
                print("Fetched " + str(len(records)) + " sessions so far...")
            if len(records) >= max_sessions:
                print("Reached ACN_MAX_SESSIONS=" + str(max_sessions) + ", stopping fetch early.")
                break

    if len(records) == 0:
        raise ValueError("No ACN sessions were fetched. Check token/site permissions.")

    df = pd.DataFrame(records)
    df.to_csv(ACN_RAW_DATA_PATH, index=False)
    print(json.dumps({"rows": len(df), "sites": ACN_SITES, "output": str(ACN_RAW_DATA_PATH)}))


def ingest_acn():
    ensure_directories()
    if not ACN_RAW_DATA_PATH.exists():
        raise FileNotFoundError("ACN raw data not found at " + str(ACN_RAW_DATA_PATH) + ". Run fetch_acn first.")

    df = pd.read_csv(ACN_RAW_DATA_PATH)

    required = ["stationID", "connectionTime", "disconnectTime", "kWhDelivered"]
    for col in required:
        if col not in df.columns:
            raise ValueError("ACN file missing required field: " + col)

    start = pd.to_datetime(df["connectionTime"], errors="coerce", utc=True)
    end = pd.to_datetime(df["disconnectTime"], errors="coerce", utc=True)
    duration_minutes = (end - start).dt.total_seconds() / 60
    duration_minutes = duration_minutes.clip(lower=0)

    n = len(df)

    if "userID" in df.columns:
        user_ids = df["userID"].astype(str)
    else:
        user_ids = pd.Series(["unknown"] * n)

    if "siteID" in df.columns:
        site_ids = df["siteID"].astype(str)
    else:
        site_ids = pd.Series(["unknown"] * n)

    if "clusterID" in df.columns:
        cluster_ids = df["clusterID"].astype(str)
    else:
        cluster_ids = pd.Series(["Unknown"] * n)

    staged = pd.DataFrame({
        "user_id": user_ids,
        "vehicle_model": "Unknown",
        "battery_capacity_kwh": np.nan,
        "charging_station_id": df["stationID"].astype(str),
        "charging_station_location": site_ids,
        "charging_start_time": start,
        "charging_end_time": end,
        "energy_consumed_kwh": pd.to_numeric(df["kWhDelivered"], errors="coerce"),
        "charging_duration_minutes": duration_minutes,
        "charging_cost": np.nan,
        "time_of_day": time_of_day(start),
        "day_of_week": start.dt.day_name(),
        "state_of_charge_start": np.nan,
        "state_of_charge_end": np.nan,
        "distance_driven_km": np.nan,
        "temperature_c": np.nan,
        "vehicle_age_years": np.nan,
        "charger_type": cluster_ids,
        "user_type": "Unknown",
        "source_file": ACN_RAW_DATA_PATH.name,
    })

    staged = staged.dropna(subset=["charging_station_id", "charging_start_time", "charging_end_time"])

    stations = staged[["charging_station_id", "charging_station_location"]].drop_duplicates(subset=["charging_station_id"])
    stations = stations.rename(columns={
        "charging_station_id": "station_id",
        "charging_station_location": "station_location",
    })

    engine = get_engine()
    station_records = stations.to_dict(orient="records")
    insert_sql = text(
        "INSERT INTO stations (station_id, station_location) "
        "VALUES (:station_id, :station_location) "
        "ON CONFLICT (station_id) DO UPDATE SET station_location = EXCLUDED.station_location"
    )
    with engine.begin() as conn:
        conn.execute(insert_sql, station_records)

    write_df(staged, "charging_sessions_raw", if_exists="append")
    print("Ingested " + str(len(staged)) + " ACN sessions and " + str(len(stations)) + " stations")


def impute_and_clean(df):
    out = df.copy()
    out = out.drop_duplicates()

    out["charging_start_time"] = pd.to_datetime(out["charging_start_time"], errors="coerce", utc=True)
    out["charging_end_time"] = pd.to_datetime(out["charging_end_time"], errors="coerce", utc=True)

    out = out.dropna(subset=["charging_station_id", "charging_start_time", "charging_end_time"])

    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        median = out[col].median()
        out[col] = out[col].fillna(median)

    computed_duration = (out["charging_end_time"] - out["charging_start_time"]).dt.total_seconds() / 60.0
    out["charging_duration_minutes"] = computed_duration.where(computed_duration > 0, out["charging_duration_minutes"])

    out = out[(out["charging_duration_minutes"] >= 5) & (out["charging_duration_minutes"] <= 24 * 60)]
    out = out[(out["energy_consumed_kwh"] >= 0.1) & (out["energy_consumed_kwh"] <= 200)]

    for col in STRING_COLS:
        out[col] = out[col].astype(str).str.strip().str.title()

    return out.reset_index(drop=True)


def clean_sessions():
    ensure_directories()
    engine = get_engine()
    with engine.connect() as conn:
        raw = pd.read_sql(text("SELECT * FROM charging_sessions_raw"), conn)

    clean = impute_and_clean(raw)
    clean.to_csv(CLEAN_DATA_PATH, index=False)

    with engine.begin() as conn:
        clean.to_sql("charging_sessions_clean", conn, if_exists="replace", index=False)

    print("Cleaned sessions: " + str(len(clean)) + " written to " + str(CLEAN_DATA_PATH))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "fetch":
        fetch_acn()
    elif step == "ingest":
        ingest_acn()
    elif step == "clean":
        clean_sessions()
    else:
        fetch_acn()
        ingest_acn()
        clean_sessions()
