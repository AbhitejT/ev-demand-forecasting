CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS stations (
    station_id TEXT PRIMARY KEY,
    station_location TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS charging_sessions_raw (
    session_id BIGSERIAL,
    user_id TEXT,
    vehicle_model TEXT,
    battery_capacity_kwh DOUBLE PRECISION,
    charging_station_id TEXT REFERENCES stations(station_id),
    charging_station_location TEXT,
    charging_start_time TIMESTAMPTZ,
    charging_end_time TIMESTAMPTZ,
    energy_consumed_kwh DOUBLE PRECISION,
    charging_duration_minutes DOUBLE PRECISION,
    charging_cost DOUBLE PRECISION,
    time_of_day TEXT,
    day_of_week TEXT,
    state_of_charge_start DOUBLE PRECISION,
    state_of_charge_end DOUBLE PRECISION,
    distance_driven_km DOUBLE PRECISION,
    temperature_c DOUBLE PRECISION,
    vehicle_age_years DOUBLE PRECISION,
    charger_type TEXT,
    user_type TEXT,
    source_file TEXT,
    inserted_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable(
    'charging_sessions_raw',
    'charging_start_time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '7 days'
);

CREATE INDEX IF NOT EXISTS idx_sessions_station_time
    ON charging_sessions_raw (charging_station_id, charging_start_time DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_id
    ON charging_sessions_raw (session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_location
    ON charging_sessions_raw (charging_station_location);

CREATE TABLE IF NOT EXISTS station_hourly_features (
    station_id TEXT NOT NULL,
    timestamp_hour TIMESTAMPTZ NOT NULL,
    demand_count INTEGER NOT NULL,
    total_energy_kwh DOUBLE PRECISION NOT NULL,
    avg_duration_minutes DOUBLE PRECISION,
    avg_temperature_c DOUBLE PRECISION,
    hour_of_day INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    station_utilization_rate DOUBLE PRECISION,
    peak_frequency_flag BOOLEAN,
    PRIMARY KEY (station_id, timestamp_hour)
);

SELECT create_hypertable(
    'station_hourly_features',
    'timestamp_hour',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '14 days'
);

CREATE TABLE IF NOT EXISTS station_hourly_forecasts (
    station_id TEXT NOT NULL,
    timestamp_hour TIMESTAMPTZ NOT NULL,
    predicted_demand DOUBLE PRECISION,
    predicted_energy_kwh DOUBLE PRECISION,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (station_id, timestamp_hour, model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_station_time
    ON station_hourly_forecasts (station_id, timestamp_hour DESC);
