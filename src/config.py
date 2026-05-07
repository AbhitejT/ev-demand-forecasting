import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

ACN_RAW_DATA_PATH = Path(os.getenv("ACN_RAW_DATA_PATH", str(ROOT / "data/raw/acn_sessions.csv")))
CLEAN_DATA_PATH = Path(os.getenv("CLEAN_DATA_PATH", str(ROOT / "data/processed/charging_sessions_clean.csv")))
FEATURE_DATA_PATH = Path(os.getenv("FEATURE_DATA_PATH", str(ROOT / "data/processed/station_hourly_features.csv")))
FORECAST_DATA_PATH = Path(os.getenv("FORECAST_DATA_PATH", str(ROOT / "data/processed/station_hourly_forecasts.csv")))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT / "artifacts/models")))
REPORT_DIR = Path(os.getenv("REPORT_DIR", str(ROOT / "artifacts/reports")))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://evdp_user:evdp_pass@localhost:5432/evdp")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
ACN_API_TOKEN = os.getenv("ACN_API_TOKEN", "")

ACN_SITES = []
for s in os.getenv("ACN_SITES", "caltech,jpl").split(","):
    s = s.strip().lower()
    if s:
        ACN_SITES.append(s)


def ensure_directories():
    paths = [
        ACN_RAW_DATA_PATH.parent,
        CLEAN_DATA_PATH.parent,
        FEATURE_DATA_PATH.parent,
        FORECAST_DATA_PATH.parent,
        MODEL_DIR,
        REPORT_DIR,
    ]
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def get_engine():
    return create_engine(DATABASE_URL, future=True)


def write_df(df, table_name, if_exists="append"):
    engine = get_engine()
    with engine.begin() as conn:
        df.to_sql(
            table_name,
            conn,
            if_exists=if_exists,
            index=False,
            method="multi",
            chunksize=1000,
        )
