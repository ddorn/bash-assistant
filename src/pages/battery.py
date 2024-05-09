import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


battery_logs = "~/logs/battery_log.csv"

data = pd.read_csv(battery_logs)

st.title("Battery Logs")
st.write(data.head())

# Plot "Battery Level" vs combined columns "Time" + "Date"
data["DateTime"] = pd.to_datetime(data["Date"] + " " + data["Time"])

# Add a None entry when there is more than N minutes between the logs
times_diff = data["DateTime"].diff()
data["DateTime"] = data["DateTime"].where(times_diff < pd.Timedelta("5 min"), None)

# Plot last 4 days
last_4_days = data["Date"].unique()[-4:]
fig = px.line(
    data[data["Date"].isin(last_4_days)],
    x="DateTime",
    y="Battery Level",
    color="Date",
    title="Battery Level vs Time",
)
st.write(fig)

# A plot per day, without considering the Date
data["Time"] = pd.to_datetime(data["Time"])
data["Time"] = data["Time"].dt.time

fig = go.Figure()
for date in data["Date"].unique():
    df = data[data["Date"] == date]
    # Make Time a datetime object, all on the same date
    times = pd.to_datetime("2021-01-01 " + df["Time"].astype(str))
    # Add a None entry when there is more than N minutes between the logs
    times_diff = times.diff()
    times = times.where(times_diff < pd.Timedelta("5 min"), None)

    fig.add_trace(
        go.Scatter(
            x=times,
            y=df["Battery Level"],
            mode="lines",
            name=date,
        )
    )

fig.update_layout(title="Battery Level vs Time per Day")
st.plotly_chart(fig)


# Scatter "Estimated Time Remaining" (fmt: hh:mm:ss) vs "Battery Level"
# Convert "Estimated Time Remaining" to minutes
data["Estimated Time Remaining"] = pd.to_timedelta(data["Estimated Time Remaining"])
data["Estimated Time Remaining"] = data["Estimated Time Remaining"].dt.total_seconds() / 60

# Ignore charging
data = data[data["Battery Status"] == "Discharging"]


fig = px.scatter(
    data,
    x="Battery Level",
    y="Estimated Time Remaining",
    color="Date",
    title="Estimated Time Remaining vs Battery Level",
)
# X axis from 100 to 0
fig.update_xaxes(autorange="reversed")
# Y axis from 0 to 4h
fig.update_yaxes(range=[0, 240])
st.write(fig)
