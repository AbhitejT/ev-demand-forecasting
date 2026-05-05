import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import FEATURE_DATA_PATH, MODEL_DIR

st.set_page_config(page_title="EV Charging Demand Dashboard", layout="wide")
st.title("EV Charging Demand Predictor")
st.caption("Simple presentation view for predicted station demand and best charging hours.")

if not FEATURE_DATA_PATH.exists():
    st.warning("Feature dataset not found. Run ETL and feature build scripts first.")
    st.stop()

df = pd.read_csv(FEATURE_DATA_PATH, parse_dates=["timestamp_hour"])
stations = sorted(df["station_id"].dropna().unique().tolist())

selected_station = st.selectbox("Station", stations)
station_df = df[df["station_id"] == selected_station].sort_values("timestamp_hour")

hourly_profile = station_df.groupby(station_df["timestamp_hour"].dt.hour)["demand_count"].mean()
hourly_profile = hourly_profile.reset_index(name="avg_demand")
hourly_profile = hourly_profile.rename(columns={"timestamp_hour": "hour_of_day"})

best_hours = hourly_profile.sort_values("avg_demand").head(3)["hour_of_day"].tolist()
best_hours_str = ", ".join([str(h) for h in best_hours])

total_sessions = int(station_df["demand_count"].sum())
avg_hourly = round(float(station_df["demand_count"].mean()), 3)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Sessions", total_sessions, border=True)
with col2:
    st.metric("Avg Hourly Demand", avg_hourly, border=True)
with col3:
    st.metric("Recommended Low-Demand Hours", best_hours_str, border=True)

fig_demand = px.bar(
    hourly_profile,
    x="hour_of_day",
    y="avg_demand",
    title="Average Demand by Hour (Selected Station)",
    labels={"hour_of_day": "Hour of Day", "avg_demand": "Average Demand"},
)
st.plotly_chart(fig_demand, use_container_width=True)

st.subheader("Recent Station Activity")
recent = station_df[["timestamp_hour", "demand_count", "total_energy_kwh"]].tail(24)
st.dataframe(recent, use_container_width=True)

clusters_path = MODEL_DIR / "clustered_sessions.csv"
if clusters_path.exists():
    st.subheader("Behavior Clusters (Optional)")
    cdf = pd.read_csv(clusters_path)
    if "behavior_cluster" in cdf.columns:
        summary = cdf["behavior_cluster"].value_counts().rename_axis("cluster").reset_index(name="sessions")
        st.bar_chart(summary.set_index("cluster"))
