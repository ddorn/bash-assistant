#!/usr/bin/env python3
"""
Weekly Habit Analysis Dashboard
Analyzes Habit Loop data to show weekly progress and trends
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
import zipfile
import tempfile
import shutil
import os
import numpy as np


def load_data():
    """Load and process habit data"""
    # Load checkmarks data
    df = pd.read_csv("habits_data/Checkmarks.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # Load habits metadata
    habits_meta = pd.read_csv("habits_data/Habits.csv")

    return df, habits_meta


def identify_active_habits(df, days_back=30):
    """Identify habits that have been tracked in the last N days"""
    most_recent = df["Date"].max()
    cutoff_date = most_recent - timedelta(days=days_back)

    recent_df = df[df["Date"] >= cutoff_date].copy()
    habit_cols = [col for col in df.columns if col != "Date"]

    active_habits = []
    for habit in habit_cols:
        recent_values = recent_df[habit].dropna()
        non_negative_ones = recent_values[recent_values != -1]
        if len(non_negative_ones) > 0:
            active_habits.append(habit)

    return active_habits


def prepare_weekly_data(df, active_habits, start_date):
    """Prepare weekly aggregated data for active habits"""
    # Filter data from start_date onwards (using Monday of that week)
    df = filter_by_start_date(df, start_date)

    # Add week information
    df["Year"] = df["Date"].dt.year
    df["Week"] = df["Date"].dt.isocalendar().week
    df["YearWeek"] = df["Year"].astype(str) + "-W" + df["Week"].astype(str).str.zfill(2)

    # Get all weeks and remove the current incomplete week
    unique_weeks = sorted(df["YearWeek"].unique())

    # Remove the most recent week (current week) as it's likely incomplete
    if len(unique_weeks) > 1:
        complete_weeks = unique_weeks[:-1]  # Drop the last (current) week
    else:
        complete_weeks = unique_weeks

    # Use all complete weeks from the start date
    target_weeks = complete_weeks

    # Filter to target weeks and active habits
    weekly_df = df[df["YearWeek"].isin(target_weeks)].copy()

    # Calculate weekly completion counts for active habits
    weekly_stats = []

    for week in target_weeks:
        week_data = weekly_df[weekly_df["YearWeek"] == week]
        week_stats = {"YearWeek": week}

        for habit in active_habits:
            if habit in week_data.columns:
                # Count completions (2) and attempts (0), ignore -1 (not tracked)
                habit_values = week_data[habit].dropna()
                tracked_values = habit_values[habit_values != -1]

                if len(tracked_values) > 0:
                    completions = len(tracked_values[tracked_values == 2])
                    total_days = len(tracked_values)
                    completion_rate = completions / total_days if total_days > 0 else 0
                else:
                    completions = 0
                    total_days = 0
                    completion_rate = 0

                week_stats[f"{habit}_completions"] = completions
                week_stats[f"{habit}_total_days"] = total_days
                week_stats[f"{habit}_rate"] = completion_rate
            else:
                week_stats[f"{habit}_completions"] = 0
                week_stats[f"{habit}_total_days"] = 0
                week_stats[f"{habit}_rate"] = 0

        weekly_stats.append(week_stats)

    return pd.DataFrame(weekly_stats), target_weeks


def create_habit_heatmap(weekly_df, active_habits):
    """Create a heatmap showing habit completion rates (rows=habits, cols=weeks)"""

    # Prepare data for heatmap
    heatmap_data = []
    weeks = weekly_df["YearWeek"].tolist()

    # Add individual habits
    for habit in active_habits:
        habit_row = []
        for week in weeks:
            week_data = weekly_df[weekly_df["YearWeek"] == week]
            if not week_data.empty:
                rate = week_data.iloc[0][f"{habit}_rate"] * 100  # Convert to percentage
                habit_row.append(rate)
            else:
                habit_row.append(0)
        heatmap_data.append(habit_row)

    # Add overall average row
    overall_row = []
    for week in weeks:
        week_data = weekly_df[weekly_df["YearWeek"] == week]
        if not week_data.empty:
            week_rates = [week_data.iloc[0][f"{habit}_rate"] for habit in active_habits]
            avg_rate = sum(week_rates) / len(week_rates) * 100
            overall_row.append(avg_rate)
        else:
            overall_row.append(0)
    heatmap_data.append(overall_row)

    # Create labels
    y_labels = active_habits + ["📊 OVERALL"]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=weeks,
            y=y_labels,
            colorscale="RdYlGn",  # Red-Yellow-Green scale
            zmin=0,
            zmax=100,
            text=[[f"{val:.0f}%" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            colorbar=dict(title="Completion Rate (%)"),
        )
    )

    fig.update_layout(
        title="Habit Completion Heatmap",
        xaxis_title="Week",
        yaxis_title="Habits",
        height=200 + len(y_labels) * 30,
        yaxis=dict(tickmode="array", tickvals=list(range(len(y_labels))), ticktext=y_labels),
        font=dict(size=14),
    )

    return fig


def create_changes_heatmap(weekly_df, active_habits):
    """Create a heatmap showing week-over-week changes"""
    if len(weekly_df) < 2:
        return None

    # Calculate changes between consecutive weeks
    changes_data = []
    change_weeks = []  # Week pairs for x-axis labels

    for i in range(1, len(weekly_df)):
        current_week = weekly_df.iloc[i]
        previous_week = weekly_df.iloc[i - 1]
        change_weeks.append(f"{previous_week['YearWeek']} → {current_week['YearWeek']}")

        week_changes = []
        overall_changes = []

        for habit in active_habits:
            current_rate = current_week[f"{habit}_rate"] * 100  # Convert to percentage points
            previous_rate = previous_week[f"{habit}_rate"] * 100

            # Calculate absolute difference in percentage points
            change = current_rate - previous_rate

            # Cap changes at -100 to +100 points for better visualization
            change = max(-100, min(100, change))
            week_changes.append(change)
            overall_changes.append(change)

        # Add overall average change
        avg_change = sum(overall_changes) / len(overall_changes) if overall_changes else 0
        week_changes.append(avg_change)

        changes_data.append(week_changes)

    # Transpose so habits are rows and week transitions are columns
    z_data = list(map(list, zip(*changes_data)))

    # Create labels
    y_labels = active_habits + ["📊 OVERALL"]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=change_weeks,
            y=y_labels,
            colorscale="RdBu_r",  # Red-Blue scale (red=negative, blue=positive)
            zmin=-100,
            zmax=100,
            text=[[f"{val:+.0f}" for val in row] for row in z_data],
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            colorbar=dict(title="Change (points)", tickvals=[-100, -50, 0, 50, 100]),
        )
    )

    fig.update_layout(
        title="Week-over-Week Changes Heatmap",
        xaxis_title="Week Transition",
        yaxis_title="Habits",
        height=200 + len(y_labels) * 30,
        font=dict(size=14),
        xaxis=dict(tickangle=45),
    )

    return fig


def calculate_week_over_week_changes(weekly_df, active_habits):
    """Calculate week-over-week changes"""
    changes = []

    if len(weekly_df) >= 2:
        current_week = weekly_df.iloc[-1]
        previous_week = weekly_df.iloc[-2]

        for habit in active_habits:
            current_rate = current_week[f"{habit}_rate"]
            previous_rate = previous_week[f"{habit}_rate"]

            if previous_rate > 0:
                change = ((current_rate - previous_rate) / previous_rate) * 100
            else:
                change = 100 if current_rate > 0 else 0

            changes.append(
                {
                    "Habit": habit,
                    "Current_Week_Rate": current_rate * 100,
                    "Previous_Week_Rate": previous_rate * 100,
                    "Change_Percent": change,
                    "Current_Completions": current_week[f"{habit}_completions"],
                    "Previous_Completions": previous_week[f"{habit}_completions"],
                }
            )

    return pd.DataFrame(changes)


def process_uploaded_zip(uploaded_file):
    """Process uploaded zip file and extract habit data"""
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            temp_zip_path = os.path.join(temp_dir, "habits.zip")
            with open(temp_zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract zip file
            extract_dir = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # Remove old habits_data if it exists
            if os.path.exists("habits_data"):
                shutil.rmtree("habits_data")

            # Move extracted data to habits_data
            shutil.move(extract_dir, "habits_data")

            return True, "✅ Data updated successfully!"

    except Exception as e:
        return False, f"❌ Error processing zip file: {str(e)}"


def calculate_performance_summary(weekly_df, active_habits):
    """Calculate best/worst performing habits and trends"""
    if len(weekly_df) < 1:
        return {}

    # Get latest week performance
    latest_week = weekly_df.iloc[-1]
    latest_rates = [(habit, latest_week[f"{habit}_rate"] * 100) for habit in active_habits]
    latest_rates.sort(key=lambda x: x[1], reverse=True)

    # Calculate changes if we have multiple weeks
    changes = []
    if len(weekly_df) >= 2:
        current_week = weekly_df.iloc[-1]
        previous_week = weekly_df.iloc[-2]

        for habit in active_habits:
            current_rate = current_week[f"{habit}_rate"] * 100
            previous_rate = previous_week[f"{habit}_rate"] * 100
            change = current_rate - previous_rate
            changes.append((habit, change))

        changes.sort(key=lambda x: x[1], reverse=True)

    return {
        "top_performers": latest_rates[:3],
        "bottom_performers": latest_rates[-3:],
        "most_improved": changes[:3] if changes else [],
        "needs_attention": changes[-3:] if changes else [],
    }


def calculate_habit_correlations(df_filtered, active_habits):
    """Calculate correlations between habits"""

    # Filter to active habits and convert to numeric (2=done, 0=not done, ignore -1)
    habit_data = df_filtered[active_habits].copy()

    # Convert to binary (1 for completed, 0 for not completed, NaN for not tracked)
    habit_binary = habit_data.replace({2: 1, 0: 0, -1: np.nan})

    # Calculate correlation matrix
    corr_matrix = habit_binary.corr()

    # Extract strongest correlations (excluding self-correlations)
    correlations = []
    for i, habit1 in enumerate(active_habits):
        for j, habit2 in enumerate(active_habits):
            if i < j:  # Avoid duplicates and self-correlations
                corr_val = corr_matrix.loc[habit1, habit2]
                if not np.isnan(corr_val):
                    correlations.append((habit1, habit2, corr_val))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: x[2])

    return corr_matrix, correlations[:5] + correlations[-5:]


def calculate_weekday_patterns(df_filtered, active_habits):
    """Calculate performance by weekday"""

    df_filtered["Weekday"] = df_filtered["Date"].dt.day_name()

    weekday_stats = {}
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for weekday in weekdays:
        day_data = df_filtered[df_filtered["Weekday"] == weekday]
        day_rates = []

        for habit in active_habits:
            if habit in day_data.columns:
                completions, total_days, completion_rate = calculate_habit_completion_stats(
                    day_data[habit]
                )
                day_rates.append(completion_rate)

        weekday_stats[weekday] = day_rates

    return weekday_stats, weekdays


def create_correlation_heatmap(corr_matrix, active_habits):
    """Create correlation matrix heatmap"""
    # Remove diagonal (set to NaN so it shows as blank)
    corr_matrix_display = corr_matrix.copy()
    np.fill_diagonal(corr_matrix_display.values, np.nan)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix_display.values,
            x=active_habits,
            y=active_habits,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=[
                [f"{val:.2f}" if not np.isnan(val) else "" for val in row]
                for row in corr_matrix_display.values
            ],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Correlation", tickvals=[-1, -0.5, 0, 0.5, 1]),
        )
    )

    fig.update_layout(
        title="Habit Correlation Matrix",
        xaxis_title="Habits",
        yaxis_title="Habits",
        height=400 + len(active_habits) * 30,
        font=dict(size=14),
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0),
    )

    return fig


def sort_habits_by_position(active_habits, habits_meta):
    """Sort habits by their position in the original habit order"""
    # Create position mapping
    position_map = {}
    for _, row in habits_meta.iterrows():
        if row["Name"] in active_habits:
            position_map[row["Name"]] = int(row["Position"])

    # Sort habits by position, put any without position at the end
    sorted_habits = sorted(active_habits, key=lambda h: position_map.get(h, -999), reverse=True)
    return sorted_habits


def create_weekday_heatmap(weekday_stats, weekdays, active_habits):
    """Create weekday patterns heatmap"""
    # Prepare data (habits as rows, weekdays as columns)
    heatmap_data = []

    # Add individual habits
    for habit in active_habits:
        habit_row = []
        for weekday in weekdays:
            if weekday in weekday_stats:
                habit_idx = active_habits.index(habit)
                if habit_idx < len(weekday_stats[weekday]):
                    rate = weekday_stats[weekday][habit_idx] * 100
                else:
                    rate = 0
            else:
                rate = 0
            habit_row.append(rate)
        heatmap_data.append(habit_row)

    # Add overall average row
    overall_row = []
    for weekday in weekdays:
        if weekday in weekday_stats and weekday_stats[weekday]:
            avg_rate = sum(weekday_stats[weekday]) / len(weekday_stats[weekday]) * 100
        else:
            avg_rate = 0
        overall_row.append(avg_rate)
    heatmap_data.append(overall_row)

    # Create labels
    y_labels = active_habits + ["📊 OVERALL"]

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=weekdays,
            y=y_labels,
            colorscale="RdYlGn",
            zmin=0,
            zmax=100,
            text=[[f"{val:.0f}%" for val in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            colorbar=dict(title="Completion Rate (%)"),
        )
    )

    fig.update_layout(
        title="Weekday Performance Patterns",
        xaxis_title="Day of Week",
        yaxis_title="Habits",
        height=200 + len(y_labels) * 30,
        font=dict(size=14),
    )

    return fig


def calculate_daily_completion_rates(df_filtered, active_habits):
    """Calculate daily completion rates over time"""

    # Sort by date
    df_filtered = df_filtered.sort_values("Date")

    daily_rates = []

    for _, row in df_filtered.iterrows():
        date = row["Date"]
        total_completions = 0
        total_tracked = 0

        for habit in active_habits:
            if habit in row and pd.notna(row[habit]) and row[habit] != -1:
                total_tracked += 1
                if row[habit] == 2:  # Completed
                    total_completions += 1

        completion_rate = (total_completions / total_tracked * 100) if total_tracked > 0 else 0

        daily_rates.append(
            {
                "Date": date,
                "Completion_Rate": completion_rate,
                "Total_Completions": total_completions,
                "Total_Tracked": total_tracked,
            }
        )

    return pd.DataFrame(daily_rates)


def create_daily_completion_chart(daily_df):
    """Create line chart showing daily completion rates over time"""
    fig = go.Figure()

    # Add the main line
    fig.add_trace(
        go.Scatter(
            x=daily_df["Date"],
            y=daily_df["Completion_Rate"],
            mode="lines+markers",
            name="Daily Completion Rate",
            line=dict(width=2, color="#2E86AB"),
            marker=dict(size=4),
            hovertemplate="<b>%{x}</b><br>" + "Completion Rate: %{y:.1f}%<br>" + "<extra></extra>",
        )
    )

    # Add trend line
    if len(daily_df) > 1:
        # Calculate 7-day rolling average
        daily_df["Rolling_7"] = daily_df["Completion_Rate"].rolling(window=7, center=True).mean()

        fig.add_trace(
            go.Scatter(
                x=daily_df["Date"],
                y=daily_df["Rolling_7"],
                mode="lines",
                name="7-day Average",
                line=dict(width=3, color="#F18F01", dash="dash"),
                hovertemplate="<b>%{x}</b><br>"
                + "7-day Average: %{y:.1f}%<br>"
                + "<extra></extra>",
            )
        )

        # Add week boundary markers using add_shape (more reliable than add_vline)
    if len(daily_df) > 0:
        # Find all Mondays (start of weeks) in the date range
        start_date = daily_df["Date"].min()
        end_date = daily_df["Date"].max()

        # Convert to datetime for arithmetic operations
        start_dt = start_date.to_pydatetime()
        end_dt = end_date.to_pydatetime()

        # Get the first Monday on or after start_date
        days_since_monday = start_dt.weekday()
        first_monday = start_dt - timedelta(days=days_since_monday)
        if first_monday < start_dt:
            first_monday += timedelta(days=7)

        # Generate all Mondays in the range
        current_monday = first_monday
        week_starts = []

        while current_monday <= end_dt:
            week_starts.append(current_monday)
            current_monday += timedelta(days=7)

        # Add vertical lines using shapes (more reliable)
        for week_start in week_starts:
            fig.add_shape(
                type="line",
                x0=week_start,
                x1=week_start,
                y0=0,
                y1=100,
                line=dict(color="rgba(128,128,128,0.3)", width=1, dash="dot"),
                xref="x",
                yref="y",
            )

            # Add text annotation separately
            fig.add_annotation(
                x=week_start,
                y=95,
                text=f"Week of {week_start.strftime('%b %d')}",
                showarrow=False,
                font=dict(size=10, color="gray"),
                xanchor="center",
            )

    # Add horizontal line at 75% (good target)
    fig.add_hline(
        y=75,
        line_dash="dot",
        line_color="green",
        annotation_text="75% Target",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Daily Habit Completion Over Time",
        xaxis_title="Date",
        yaxis_title="Completion Rate (%)",
        height=400,
        font=dict(size=14),
        hovermode="x unified",
        yaxis=dict(range=[0, 100]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def calculate_total_completions(df_filtered, active_habits):
    """Calculate total completions for each habit over the analysis period"""

    completions = []

    for habit in active_habits:
        if habit in df_filtered.columns:
            # Count completions (2) and ignore -1 (not tracked)
            total_completions, _, _ = calculate_habit_completion_stats(df_filtered[habit])

            completions.append({"Habit": habit, "Total_Completions": total_completions})

    return pd.DataFrame(completions)


def create_completions_histogram(completions_df):
    """Create histogram showing total completions per habit"""
    # Sort by completion count for better visualization
    completions_df = completions_df.sort_values("Total_Completions", ascending=True)

    fig = go.Figure()

    # Create bar chart
    fig.add_trace(
        go.Bar(
            x=completions_df["Habit"],
            y=completions_df["Total_Completions"],
            name="Total Completions",
            marker=dict(
                color=completions_df["Total_Completions"],
                colorscale="RdYlGn",  # Red to Green scale
                showscale=True,
                colorbar=dict(title="Completions"),
            ),
            text=completions_df["Total_Completions"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>" + "Total Completions: %{y}<br>" + "<extra></extra>",
        )
    )

    fig.update_layout(
        title="Total Habit Completions",
        xaxis_title="Habits",
        yaxis_title="Number of Completions",
        height=400,
        font=dict(size=14),
        xaxis=dict(tickangle=45),
        showlegend=False,
    )

    return fig


def calculate_habit_timelines(df):
    """Calculate timeline data for all habits (first day to last day tracked)"""
    # Get all habit columns (exclude Date column)
    habit_cols = [col for col in df.columns if col != "Date"]

    timelines = []

    for habit in habit_cols:
        # Get all non-null values that are not -1 (not tracked)
        habit_data = df[df[habit].notna() & (df[habit] != -1)]

        if len(habit_data) > 0:
            first_date = habit_data["Date"].min()
            last_date = habit_data["Date"].max()

            # Calculate some stats
            total_days_tracked = len(habit_data)
            total_completions = len(habit_data[habit_data[habit] == 2])
            completion_rate = (
                (total_completions / total_days_tracked * 100) if total_days_tracked > 0 else 0
            )

            # Calculate duration in days
            duration_days = (last_date - first_date).days + 1

            # Determine if habit is currently active (tracked in last 30 days)
            most_recent = df["Date"].max()
            cutoff_date = most_recent - timedelta(days=30)
            is_active = last_date >= cutoff_date

            timelines.append(
                {
                    "Habit": habit,
                    "Start_Date": first_date,
                    "End_Date": last_date,
                    "Duration_Days": duration_days,
                    "Days_Tracked": total_days_tracked,
                    "Total_Completions": total_completions,
                    "Completion_Rate": completion_rate,
                    "Is_Active": is_active,
                }
            )

    return pd.DataFrame(timelines)


def create_habit_gantt_chart(timelines_df):
    """Create Gantt chart showing habit timelines"""
    if len(timelines_df) == 0:
        return None

    # Sort by start date for better visualization
    timelines_df = timelines_df.sort_values("Start_Date")

    fig = go.Figure()

    # Add bars for each habit using a different approach
    for i, row in timelines_df.iterrows():
        # Create hover text with detailed info
        hover_text = (
            f"<b>{row['Habit']}</b><br>"
            f"Period: {row['Start_Date'].strftime('%Y-%m-%d')} to {row['End_Date'].strftime('%Y-%m-%d')}<br>"
            f"Duration: {row['Duration_Days']} days<br>"
            f"Days tracked: {row['Days_Tracked']}<br>"
            f"Completions: {row['Total_Completions']}<br>"
            f"Success rate: {row['Completion_Rate']:.1f}%<br>"
            f"Status: {'🟢 Active' if row['Is_Active'] else '🔴 Inactive'}"
        )

        # Use Scatter with thick lines to create Gantt bars
        color = "#2E86AB" if row["Is_Active"] else "#A8A8A8"

        fig.add_trace(
            go.Scatter(
                x=[row["Start_Date"], row["End_Date"]],
                y=[row["Habit"], row["Habit"]],
                mode="lines",
                line=dict(color=color, width=20),
                name=row["Habit"],
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
            )
        )

    # Add legend manually
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="#2E86AB"),
            name="🟢 Active (last 30 days)",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="#A8A8A8"),
            name="🔴 Inactive",
            showlegend=True,
        )
    )

    # Add vertical lines for month/quarter markers
    if len(timelines_df) > 0:
        start_date = timelines_df["Start_Date"].min()
        end_date = timelines_df["End_Date"].max()
        total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        # Determine marker settings based on timeline length
        use_quarterly = total_months > 24
        quarter_months = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
        line_opacity = "0.4" if use_quarterly else "0.3"

        # Add time markers
        current_date = start_date.replace(day=1)
        while current_date <= end_date:
            should_add_marker = not use_quarterly or current_date.month in quarter_months

            if should_add_marker:
                fig.add_vline(
                    x=current_date,
                    line_dash="dot",
                    line_color=f"rgba(128,128,128,{line_opacity})",
                    line_width=1,
                )

            # Add year label for January
            if current_date.month == 1:
                fig.add_annotation(
                    x=current_date,
                    y=1.02,
                    yref="paper",
                    text=str(current_date.year),
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    xanchor="center",
                )

            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)

    fig.update_layout(
        title="Habit Timeline (Gantt Chart)",
        xaxis_title="Timeline",
        yaxis_title="Habits",
        height=max(400, len(timelines_df) * 30 + 200),
        font=dict(size=12),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(type="date", tickformat="%Y-%m-%d"),
        yaxis=dict(categoryorder="array", categoryarray=timelines_df["Habit"].tolist()),
    )

    return fig


def filter_by_start_date(df, start_date):
    """Helper function to filter dataframe from start_date onwards, adjusted to Monday of that week"""
    # Convert to datetime if it's a date
    if hasattr(start_date, "weekday"):
        start_datetime = pd.to_datetime(start_date)
    else:
        start_datetime = start_date

    # Get the Monday of the week containing start_date
    # weekday() returns 0=Monday, 1=Tuesday, etc.
    days_since_monday = start_datetime.weekday()
    monday_of_week = start_datetime - timedelta(days=days_since_monday)

    return df[df["Date"] >= monday_of_week].copy()


def calculate_habit_completion_stats(habit_data):
    """Helper function to calculate completion stats from habit data"""
    habit_values = habit_data.dropna()
    tracked_values = habit_values[habit_values != -1]

    if len(tracked_values) > 0:
        completions = len(tracked_values[tracked_values == 2])
        total_days = len(tracked_values)
        completion_rate = completions / total_days if total_days > 0 else 0
    else:
        completions = 0
        total_days = 0
        completion_rate = 0

    return completions, total_days, completion_rate


def main():
    st.set_page_config(page_title="Weekly Habit Analysis", page_icon="📊", layout="wide")

    st.title("📊 Weekly Habit Analysis Dashboard")
    st.markdown("*Analyze your habit patterns and weekly progress*")

    # Sidebar for data upload and info
    with st.sidebar:
        st.header("📁 Update Data")
        uploaded_file = st.file_uploader(
            "Upload new Habit Loop export (ZIP file)",
            type=["zip"],
            help="Select the ZIP file exported from Habit Loop app",
        )

        if uploaded_file is not None:
            if st.button("🔄 Update Data", type="primary"):
                with st.spinner("Processing zip file..."):
                    success, message = process_uploaded_zip(uploaded_file)
                    if success:
                        st.success(message)
                        st.rerun()  # Refresh the app with new data
                    else:
                        st.error(message)

        # Load data
    try:
        df, habits_meta = load_data()

        # Analysis period selector (after data is loaded)
        st.sidebar.header("⏰ Analysis Period")

        # Get date range from data
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()

        # Default to 6 weeks back from most recent date
        default_start = max_date - timedelta(weeks=6)
        if default_start < min_date:
            default_start = min_date

        start_date = st.sidebar.date_input(
            "Start date for analysis",
            value=default_start,
            min_value=min_date,
            max_value=max_date,
            help="Select the starting date for your analysis period. Analysis will start from the Monday of the selected week.",
        )

        # Calculate and show the actual Monday being used
        start_datetime = pd.to_datetime(start_date)
        days_since_monday = start_datetime.weekday()
        monday_of_week = start_datetime - timedelta(days=days_since_monday)

        if days_since_monday > 0:
            st.sidebar.info(
                f"📅 Analysis starts from Monday: {monday_of_week.strftime('%Y-%m-%d')}"
            )
        else:
            st.sidebar.info(
                f"📅 Analysis starts from: {monday_of_week.strftime('%Y-%m-%d')} (Monday)"
            )

        # Identify active habits
        active_habits = identify_active_habits(df)

        # Sort habits by their original position/priority
        active_habits = sort_habits_by_position(active_habits, habits_meta)

        st.sidebar.header("📋 Active Habits")
        st.sidebar.write(f"Found **{len(active_habits)}** active habits")

        # Show non-daily habits warning
        non_daily = habits_meta[habits_meta["Interval"] != 1]
        if len(non_daily) > 0:
            st.sidebar.warning("⚠️ Non-daily habits detected:")
            for _, row in non_daily.iterrows():
                if row["Name"] in active_habits:
                    st.sidebar.write(f"• {row['Name']} (every {row['Interval']} days)")

        # Prepare weekly data with user-selected period
        weekly_df, target_weeks = prepare_weekly_data(df, active_habits, start_date)

        # Most Recent Week Metric at the top
        if len(weekly_df) >= 1:
            most_recent_complete_week = weekly_df.iloc[-1]
            total_completions_recent = sum(
                [most_recent_complete_week[f"{h}_completions"] for h in active_habits]
            )
            total_possible_recent = sum(
                [most_recent_complete_week[f"{h}_total_days"] for h in active_habits]
            )
            recent_rate = (
                (total_completions_recent / total_possible_recent * 100)
                if total_possible_recent > 0
                else 0
            )

            # Calculate change from previous week
            if len(weekly_df) >= 2:
                prev_week = weekly_df.iloc[-2]
                total_completions_prev = sum([prev_week[f"{h}_completions"] for h in active_habits])
                total_possible_prev = sum([prev_week[f"{h}_total_days"] for h in active_habits])
                prev_rate = (
                    (total_completions_prev / total_possible_prev * 100)
                    if total_possible_prev > 0
                    else 0
                )
                delta = f"{recent_rate - prev_rate:+.1f}%" if prev_rate > 0 else None
            else:
                delta = None

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric(
                    f"📊 Most Recent Week ({most_recent_complete_week['YearWeek']})",
                    f"{recent_rate:.1f}%",
                    delta=delta,
                    help=f"{total_completions_recent}/{total_possible_recent} habits completed",
                )

        # Performance Summary
        st.subheader("🎯 Performance Overview")
        summary = calculate_performance_summary(weekly_df, active_habits)

        if summary:
            col1, col2 = st.columns(2)

            with col1:
                # Top Performers Box
                top_performers_text = "##### 🏆 Top Performers\n\n"
                for habit, rate in summary["top_performers"]:
                    top_performers_text += f"1. {habit}: {rate:.0f}%\n"
                st.success(top_performers_text)

                # Most Improved Box
                if summary["most_improved"]:
                    improved_text = "##### 🚀 Most Improved\n\n"
                    for habit, change in summary["most_improved"]:
                        if change > 0:
                            improved_text += f"1. {habit}: +{change:.0f}pts\n"
                    st.info(improved_text)

            with col2:
                # Focus Areas Box
                focus_text = "##### ⚠️ Focus Areas\n\n"
                for habit, rate in reversed(summary["bottom_performers"]):
                    focus_text += f"1. {habit}: {rate:.0f}%\n"
                st.error(focus_text)

                # Needs Attention Box
                if summary["needs_attention"]:
                    attention_text = "##### 📉 Needs Attention\n\n"
                    for habit, change in reversed(summary["needs_attention"]):
                        if change < 0:
                            attention_text += f"1. {habit}: {change:.0f}pts\n"
                    st.warning(attention_text)

        # Main content (removed analysis period display)
        # Main visualizations
        st.subheader("🔥 Habit Completion Heatmap")
        st.markdown("*Red = Low completion, Yellow = Medium, Green = High completion*")
        heatmap_chart = create_habit_heatmap(weekly_df, active_habits)
        st.plotly_chart(heatmap_chart, use_container_width=True)

        # Week-over-week changes heatmap
        if len(weekly_df) >= 2:
            st.subheader("📊 Week-over-Week Changes")
            st.markdown("*Red = Decline, White = No change, Blue = Improvement*")
            changes_heatmap = create_changes_heatmap(weekly_df, active_habits)
            if changes_heatmap:
                st.plotly_chart(changes_heatmap, use_container_width=True)

        # Filter data for the selected time period
        df_filtered = filter_by_start_date(df, pd.to_datetime(start_date))

        # Weekday patterns
        st.subheader("📅 Weekday Performance Patterns")
        st.markdown("*Discover which days of the week work best for your habits*")
        weekday_stats, weekdays = calculate_weekday_patterns(df_filtered, active_habits)
        weekday_heatmap = create_weekday_heatmap(weekday_stats, weekdays, active_habits)
        st.plotly_chart(weekday_heatmap, use_container_width=True)

        # Daily completion over time
        st.subheader("📈 Daily Completion Trends")
        st.markdown("*Track your overall habit completion percentage day by day*")
        daily_df = calculate_daily_completion_rates(df_filtered, active_habits)
        if len(daily_df) > 0:
            daily_chart = create_daily_completion_chart(daily_df)
            st.plotly_chart(daily_chart, use_container_width=True)
        else:
            st.info("Not enough data to show daily trends")

        # Correlation analysis
        st.subheader("🔗 Habit Correlation Analysis")
        st.markdown("*Discover which habits tend to succeed or fail together*")

        corr_matrix, top_correlations = calculate_habit_correlations(df_filtered, active_habits)

        # Show correlation heatmap
        corr_heatmap = create_correlation_heatmap(corr_matrix, active_habits)
        st.plotly_chart(corr_heatmap, use_container_width=True)

        # Show top insights with improved layout
        if top_correlations:
            st.markdown("#### 🔍 **Key Insights:**")

            # Separate positive and negative correlations
            positive_correlations = [
                (h1, h2, corr) for h1, h2, corr in top_correlations if corr > 0
            ]
            negative_correlations = [
                (h1, h2, corr) for h1, h2, corr in top_correlations if corr < 0
            ]

            # Sort them by absolute value
            positive_correlations.sort(key=lambda x: x[2], reverse=True)
            negative_correlations.sort(key=lambda x: x[2])  # Most negative first

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🤝 Strongest Positive Correlations:**")
                st.markdown("*When you do one, you tend to do the other*")

                pos_shown = 0
                for habit1, habit2, corr in positive_correlations:
                    # Create a cleaner display with truncated habit names
                    h1_short = habit1[:20] + "..." if len(habit1) > 20 else habit1
                    h2_short = habit2[:20] + "..." if len(habit2) > 20 else habit2

                    # Use info box for better formatting
                    st.info(f"**{corr:.2f}** correlation\n\n{h1_short} ↔ {h2_short}")
                    pos_shown += 1

            with col2:
                st.markdown("**⚡ Strongest Negative Correlations:**")
                st.markdown("*These habits tend to compete with each other*")

                neg_shown = 0
                for habit1, habit2, corr in negative_correlations:
                    # Create a cleaner display with truncated habit names
                    h1_short = habit1[:20] + "..." if len(habit1) > 20 else habit1
                    h2_short = habit2[:20] + "..." if len(habit2) > 20 else habit2

                    # Use warning box for negative correlations
                    st.warning(f"**{corr:.2f}** correlation\n\n{h1_short} ↔ {h2_short}")
                    neg_shown += 1

        # Raw data (expandable)
        with st.expander("🔍 View Raw Weekly Data"):
            st.dataframe(weekly_df, use_container_width=True)

        # Total completions for each habit
        st.subheader("📊 Total Habit Completions")
        use_all_data = st.checkbox(
            "📅 Use all historical data (instead of selected time period)",
            value=False,
            help="Check this to see total completions across all your data, not just the selected weeks",
        )

        # Choose data based on checkbox
        data_for_completions = df if use_all_data else df_filtered
        completions_df = calculate_total_completions(data_for_completions, active_habits)
        completions_histogram = create_completions_histogram(completions_df)
        st.plotly_chart(completions_histogram, use_container_width=True)

        # Habit timelines
        st.subheader("📅 Habit Timelines")
        st.markdown(
            "*Complete history of all your habits - blue bars are currently active, gray bars are discontinued*"
        )
        timelines_df = calculate_habit_timelines(df)
        habit_gantt_chart = create_habit_gantt_chart(timelines_df)
        if habit_gantt_chart:
            st.plotly_chart(habit_gantt_chart, use_container_width=True)
        else:
            st.info("No timeline data available to display")

    except FileNotFoundError as e:
        st.error(
            "Could not find habit data files. Make sure you're running this from the correct directory and that 'habits_data/' folder exists."
        )
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)


main()
