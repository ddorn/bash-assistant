import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(layout="wide")

battery_logs = "~/logs/battery.csv"

data = pd.read_csv(battery_logs)
data["DateTime"] = pd.to_datetime(data["DateTime"])
data["Date"] = data["DateTime"].dt.date
data["Weekday"] = data["DateTime"].dt.strftime("%A")

st.title("Battery Logs")
st.write(data.head())

# %% Filter for the last N days, as prompted
first_day = data["DateTime"].min().date()
today = pd.Timestamp.now().date()
days = (today - first_day).days
last_days = st.slider("Last N days", 1, days + 1, 14)
last_days = today - pd.Timedelta(days=last_days)

data = data[data["DateTime"].dt.date >= last_days]

# %% Plot of the time I spend on my computer each day
# For each log, find the time until the next log, but capped by 5 minutes
times_diff = data["DateTime"].diff()
times_diff = times_diff.dt.total_seconds() / 60
times_diff = times_diff.clip(upper=2)


data["Time"] = pd.to_datetime(data["DateTime"].dt.strftime("%H:%M:%S"))
data["TimeHuman"] = data["DateTime"].dt.strftime("%H:%M")

fig = px.scatter(
    data,
    x="Date",
    y="Time",
    color="Weekday",
    # Set the correct order for the weekdays
    category_orders={
        "Weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    },
    # Set a nice ordered cycling color scale
    color_discrete_sequence=px.colors.sequential.Plasma,
    title="Time Spent on Computer per Day",
)
# Add trace for total time spent per day
data["Time Spent"] = times_diff
data["Time Spent"] = data["Time Spent"].fillna(0)
time_spent = data.groupby("Date")["Time Spent"].sum().reset_index()
# Convert to hours
time_spent["Time Spent"] = time_spent["Time Spent"] / 60
time_spent["Weekday"] = pd.to_datetime(time_spent["Date"]).dt.strftime("%A")

fig.add_trace(
    go.Line(
        x=time_spent["Date"],
        y=time_spent["Time Spent"],
        # mode="markers+text",
        text=time_spent["Time Spent"].apply(lambda x: f"{x:.2f}h"),
        textposition="top center",
        marker=dict(color="black"),
        name="Total Time Spent",
        # Secondary y axis
        yaxis="y2",
        # Fill the area under the line with light color
        fill="tozeroy",
        fillcolor="rgba(0,0,0,0.1)",
    )
)
st.plotly_chart(fig, use_container_width=True)

# %% Bar plot of number of hours each week
data["Week"] = data["DateTime"].dt.strftime("%Y-%U")
time_spent = data.groupby("Week")["Time Spent"].sum().reset_index()
time_spent["Time Spent"] = time_spent["Time Spent"] / 60
time_spent["Week"] = pd.to_datetime(time_spent["Week"] + "-0", format="%Y-%U-%w")

fig = px.bar(
    time_spent,
    x="Week",
    y="Time Spent",
    title="Time Spent on Computer per Week",
)
fig.update_yaxes(title_text="Time Spent (hours)")
st.plotly_chart(fig, use_container_width=True)


# %%     Plot "Battery Level" vs combined columns "Time" + "Date"

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

# %% A plot per day, without considering the Date
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
            # name=date,
        )
    )

fig.update_layout(title="Battery Level vs Time per Day")
st.plotly_chart(fig)


# %% Scatter "Estimated Time Remaining" (fmt: hh:mm:ss) vs "Battery Level"
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
