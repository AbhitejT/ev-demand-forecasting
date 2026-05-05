# EV Charging Demand Predictor

Predicts EV charging demand and energy use per station per hour using historical
ACN charging session data.

## What it does
- Pulls charging session data from ACN-Data
- Loads it into Postgres and cleans it up
- Builds station-hour features (lags, rolling averages, time-of-day, etc.)
- Trains an XGBoost model for demand and energy
- Trains a KMeans model for charging-behavior clusters
- Serves predictions through a FastAPI endpoint and a Streamlit dashboard

## File layout
```
src/
  config.py     # paths, env vars, db helpers
  etl.py        # fetch_acn, ingest_acn, clean_sessions
  features.py   # build_features
  models.py     # train_xgb, train_kmeans, batch_predict
  report.py     # summary csv/json reports
  api.py        # FastAPI service
  pipeline.py   # runs everything in order
dashboard/
  app.py        # Streamlit dashboard
infra/          # docker-compose + sql migration
```

## Setup
1. Make a virtual environment and install:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -e .`
2. Start Postgres + MLflow:
   - `docker compose -f infra/docker-compose.yml up -d`
3. Set the ACN token:
   - `export ACN_API_TOKEN=<your_token>`
   - optional: `export ACN_SITES=caltech,jpl`

## Running the pipeline
Run the whole thing at once:
- `python -m src.pipeline`

Or run an individual step:
- `python -m src.etl fetch`
- `python -m src.etl ingest`
- `python -m src.etl clean`
- `python -m src.features`
- `python -m src.models xgb`
- `python -m src.models kmeans`
- `python -m src.models predict`
- `python -m src.report`

## Using it
- API: `uvicorn src.api:app --reload`
- Dashboard: `streamlit run dashboard/app.py`

In the dashboard you pick a station and see average demand by hour, the
recommended low-demand hours for charging, and the recent station activity.

## Environment variables
- `DATABASE_URL` (default points at the docker-compose Postgres)
- `ACN_RAW_DATA_PATH` (default `data/raw/acn_sessions.csv`)
- `ACN_API_TOKEN` (required for the ACN fetch)
- `ACN_SITES` (default `caltech,jpl`)
- `CLEAN_DATA_PATH` (default `data/processed/charging_sessions_clean.csv`)
- `FEATURE_DATA_PATH` (default `data/processed/station_hourly_features.csv`)
- `MODEL_DIR` (default `artifacts/models`)
