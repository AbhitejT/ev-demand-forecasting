import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import FEATURE_DATA_PATH, MODEL_DIR

app = FastAPI(title="EV Charging Demand Predictor API", version="0.1.0")


class ForecastRequest(BaseModel):
    station_id: str
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    is_weekend: bool
    lag_1h_demand: float
    lag_3h_demand: float
    lag_24h_demand: float
    rolling_mean_3h_demand: float
    lag_1h_energy: float
    lag_24h_energy: float


def load_models():
    demand_path = MODEL_DIR / "xgb_demand.joblib"
    energy_path = MODEL_DIR / "xgb_energy.joblib"
    if not demand_path.exists() or not energy_path.exists():
        raise FileNotFoundError("Trained XGBoost models are not available in artifacts/models")
    demand_model = joblib.load(demand_path)
    energy_model = joblib.load(energy_path)
    return demand_model, energy_model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stations")
def stations():
    if not FEATURE_DATA_PATH.exists():
        return []
    df = pd.read_csv(FEATURE_DATA_PATH, usecols=["station_id"])
    station_list = sorted(df["station_id"].dropna().unique().tolist())
    return station_list


@app.post("/predict")
def predict(payload: ForecastRequest):
    try:
        demand_model, energy_model = load_models()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    row = {
        "hour_of_day": payload.hour_of_day,
        "day_of_week": payload.day_of_week,
        "is_weekend": int(payload.is_weekend),
        "lag_1h_demand": payload.lag_1h_demand,
        "lag_3h_demand": payload.lag_3h_demand,
        "lag_24h_demand": payload.lag_24h_demand,
        "rolling_mean_3h_demand": payload.rolling_mean_3h_demand,
        "lag_1h_energy": payload.lag_1h_energy,
        "lag_24h_energy": payload.lag_24h_energy,
    }
    X = pd.DataFrame([row])

    demand = float(demand_model.predict(X)[0])
    energy = float(energy_model.predict(X)[0])

    if demand < 0:
        demand = 0.0
    if energy < 0:
        energy = 0.0

    return {
        "station_id": payload.station_id,
        "predicted_demand": demand,
        "predicted_energy_kwh": energy,
    }
