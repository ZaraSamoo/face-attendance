# utils/plots.py
"""
Matplotlib plotting helpers for face-attendance Streamlit app.

Each function returns (fig, ax). Call `st.pyplot(fig)` then `plt.close(fig)` in the Streamlit render function.
"""
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np

def plot_status_pie(df_display: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """
    Categorical Distribution: Pie Chart (Status Breakdown)
    Data: df_display['Status']
    """
    counts = df_display['Status'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = cm.Set3(np.linspace(0, 1, len(counts)))
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title("Attendance Status Breakdown")
    ax.axis('equal')
    return fig, ax

def plot_total_attendance_barh(df_display: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """
    Individual Comparison: Horizontal Bar Chart (Total Attendance per Person)
    Data: df_display.groupby('Name').size()
    """
    counts = df_display.groupby('Name').size().sort_values()
    height = max(4, 0.35 * len(counts))
    fig, ax = plt.subplots(figsize=(8, height))
    colors = cm.tab20(np.linspace(0, 1, len(counts)))
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Total Attendance Records")
    ax.set_title("Total Attendance per Person")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.1, i, str(v), va='center')
    return fig, ax

def plot_arrival_histogram(df_display: pd.DataFrame, start_time: str = "09:00:00") -> Tuple[plt.Figure, plt.Axes]:
    """
    Quantitative Distribution: Histogram (Arrival Time Distribution)
    Calculates arrival delay in minutes from class start_time (default 09:00:00).
    Expects df_display['Time'] as HH:MM:SS or a Timedelta-like string.
    """
    arrival_td = pd.to_timedelta(df_display['Time'], errors='coerce')
    arrival_min = arrival_td.dt.total_seconds() / 60
    start_min = pd.to_timedelta(start_time).total_seconds() / 60
    arrival_from_start = arrival_min - start_min
    arrival_from_start = arrival_from_start.dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(arrival_from_start, bins=30, color='C0', edgecolor='black')
    ax.set_xlabel(f"Arrival delay (minutes) from {start_time}")
    ax.set_ylabel("Count")
    ax.set_title("Arrival Time Distribution (delay from start)")
    return fig, ax

def plot_weekly_trend(df: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """
    Temporal Trend: Line Plot (Weekly Attendance Trend)
    Data: df.groupby('YearWeek').size()
    Expects df to contain a 'Date' column parseable by pd.to_datetime.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    isocal = df['Date'].dt.isocalendar()
    df['YearWeek'] = isocal['year'].astype(str) + "-W" + isocal['week'].astype(str).str.zfill(2)
    weekly = df.groupby('YearWeek').size().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weekly.index, weekly.values, marker='o', linestyle='-')
    ax.set_xlabel("Year-Week")
    ax.set_ylabel("Attendance Count")
    ax.set_title("Weekly Attendance Trend")
    ax.set_xticks(range(len(weekly.index)))
    ax.set_xticklabels(weekly.index, rotation=45, ha='right')
    fig.tight_layout()
    return fig, ax

def plot_morning_evening_scatter(df_display: pd.DataFrame, morning_cutoff: str = "12:00:00") -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter Plot: Morning vs Evening attendance counts per person.
    Purpose: For each person, plot (morning_count, evening_count) so the ratio is visible.
    - morning_cutoff: time string (HH:MM:SS). Records with Time < cutoff are 'morning', >= cutoff are 'evening'.
    Expects df_display to have 'Name' and 'Time'.
    """
    df = df_display.copy()
    df['Time'] = df['Time'].astype(str)
    arrival_td = pd.to_timedelta(df['Time'], errors='coerce')
    cutoff_min = pd.to_timedelta(morning_cutoff).total_seconds() / 60
    arrival_min = arrival_td.dt.total_seconds() / 60

    # Label morning/evening
    df['Shift'] = np.where(arrival_min < cutoff_min, 'Morning', 'Evening')
    counts = df.groupby(['Name', 'Shift']).size().unstack(fill_value=0)

    # Ensure both columns exist
    if 'Morning' not in counts.columns:
        counts['Morning'] = 0
    if 'Evening' not in counts.columns:
        counts['Evening'] = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    x = counts['Morning'].values
    y = counts['Evening'].values
    labels = counts.index.tolist()
    scatter = ax.scatter(x, y, s=80, alpha=0.7, c=cm.tab10(np.linspace(0, 1, len(x))))
    ax.plot([0, max(x.max(), y.max(), 1)], [0, max(x.max(), y.max(), 1)], color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Morning Attendance Count")
    ax.set_ylabel("Evening Attendance Count")
    ax.set_title("Morning vs Evening Attendance per Person (ratio visualization)")
    # Annotate points with names if not too many
    if len(labels) <= 40:
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)
    return fig, ax

def plot_course_attendance_bar(df_display: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """
    Course Attendance: Vertical Bar Chart showing attendance counts per Course.
    Data: df_display['Course']
    """
    if 'Course' not in df_display.columns:
        # return empty figure with message
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No 'Course' column found", ha='center', va='center')
        ax.axis('off')
        return fig, ax

    counts = df_display['Course'].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = cm.tab20(np.linspace(0, 1, len(counts)))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_xlabel("Course")
    ax.set_ylabel("Attendance Count")
    ax.set_title("Attendance by Course")
    ax.set_xticklabels(counts.index, rotation=45, ha='right')
    fig.tight_layout()
    return fig, ax

def plot_day_of_week_bars(df_display: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """
    Subplots and Day-of-Week Analysis (Bar Chart)
    Data: df_display['Day'] (uses Date -> day if Day not present)
    """
    df = df_display.copy()
    if 'Day' not in df.columns or df['Day'].isna().all():
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Day'] = df['Date'].dt.day_name()
        else:
            df['Day'] = df.get('Day', pd.Series(dtype=object))

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['Day'].value_counts().reindex(day_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = cm.Pastel1(np.linspace(0, 1, len(day_counts)))
    ax.bar(day_counts.index, day_counts.values, color=colors)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Attendance Volume")
    ax.set_title("Attendance Volume by Day of Week")
    return fig, ax