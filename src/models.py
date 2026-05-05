import json
import sys
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import (
    CLEAN_DATA_PATH,
    FEATURE_DATA_PATH,
    FORECAST_DATA_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    ensure_directories,
    get_engine,
)


FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "lag_1h_demand",
    "lag_3h_demand",
    "lag_24h_demand",
    "rolling_mean_3h_demand",
    "lag_1h_energy",
    "lag_24h_energy",
]

CLUSTER_COLS = [
    "charging_duration_minutes",
    "energy_consumed_kwh",
    "state_of_charge_start",
    "state_of_charge_end",
    "hour_of_day",
]


def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": float(mae), "rmse": float(rmse)}


def run_cv_model(df, target_col):
    data = df.sort_values("timestamp_hour").copy()
    X = data[FEATURE_COLUMNS].astype(float)
    y = data[target_col].astype(float)

    splitter = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    last_model = None

    for train_idx, test_idx in splitter.split(X):
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        fold_metrics.append(get_metrics(y.iloc[test_idx], preds))
        last_model = model

    mae_list = [m["mae"] for m in fold_metrics]
    rmse_list = [m["rmse"] for m in fold_metrics]
    mean_mae = float(sum(mae_list) / len(mae_list))
    mean_rmse = float(sum(rmse_list) / len(rmse_list))
    return last_model, {"mae": mean_mae, "rmse": mean_rmse}


def train_xgb():
    ensure_directories()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ev_demand_predictor")

    df = pd.read_csv(FEATURE_DATA_PATH, parse_dates=["timestamp_hour"])

    with mlflow.start_run(run_name="xgboost_station_hourly"):
        demand_model, demand_metrics = run_cv_model(df, "demand_count")
        energy_model, energy_metrics = run_cv_model(df, "total_energy_kwh")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(demand_model, MODEL_DIR / "xgb_demand.joblib")
        joblib.dump(energy_model, MODEL_DIR / "xgb_energy.joblib")

        metrics = {
            "demand_mae": demand_metrics["mae"],
            "demand_rmse": demand_metrics["rmse"],
            "energy_mae": energy_metrics["mae"],
            "energy_rmse": energy_metrics["rmse"],
        }
        mlflow.log_metrics(metrics)
        mlflow.log_text(json.dumps(metrics, indent=2), "xgb_metrics.json")
        print(json.dumps(metrics, indent=2))


def train_kmeans():
    ensure_directories()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ev_demand_predictor")

    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["charging_start_time"])
    df["hour_of_day"] = pd.to_datetime(df["charging_start_time"], utc=True).dt.hour

    X = df[CLUSTER_COLS].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=4, random_state=42, n_init="auto")
    labels = model.fit_predict(X_scaled)
    score = float(silhouette_score(X_scaled, labels))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "kmeans_sessions.joblib")
    joblib.dump(scaler, MODEL_DIR / "kmeans_scaler.joblib")

    with mlflow.start_run(run_name="kmeans_behavior_profiles"):
        mlflow.log_metric("silhouette_score", score)
        mlflow.log_text(json.dumps({"silhouette_score": score}, indent=2), "kmeans_metrics.json")

    df_out = df.copy()
    df_out["behavior_cluster"] = labels
    df_out.to_csv(MODEL_DIR / "clustered_sessions.csv", index=False)
    print(json.dumps({"silhouette_score": score}, indent=2))


def batch_predict():
    ensure_directories()
    demand_model = joblib.load(MODEL_DIR / "xgb_demand.joblib")
    energy_model = joblib.load(MODEL_DIR / "xgb_energy.joblib")

    df = pd.read_csv(FEATURE_DATA_PATH, parse_dates=["timestamp_hour"])
    X = df[FEATURE_COLUMNS].astype(float)

    out = df[["station_id", "timestamp_hour"]].copy()
    out["predicted_demand"] = demand_model.predict(X).clip(min=0)
    out["predicted_energy_kwh"] = energy_model.predict(X).clip(min=0)
    out["model_name"] = "xgboost"
    out["model_version"] = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    out.to_csv(FORECAST_DATA_PATH, index=False)

    engine = get_engine()
    with engine.begin() as conn:
        out.to_sql("station_hourly_forecasts", conn, if_exists="append", index=False)

    print("Wrote " + str(len(out)) + " forecast rows to " + str(FORECAST_DATA_PATH))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "xgb":
        train_xgb()
    elif step == "kmeans":
        train_kmeans()
    elif step == "predict":
        batch_predict()
    else:
        train_xgb()
        train_kmeans()
        batch_predict()
