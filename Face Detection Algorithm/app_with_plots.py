"""
Face Attendance System with Matplotlib Plotting Integration.

This file extends app.py with visualization charts for attendance analytics.
Run with: python -m streamlit run "Face Detection Algorithm/app_with_plots.py"
"""
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import csv

import cv2
import streamlit as st
from PIL import Image


# ========== Plotting Functions ==========
def plot_status_pie(df_display: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """
    Categorical Distribution: Pie Chart (Status Breakdown)
    Data: df_display['Status']
    
    Groups statuses into:
    - Present/Late (attended successfully)
    - Absent - Non-attendance (registered but never appeared)
    - Absent - Failed verification (appeared but low confidence)
    """
    if 'Status' not in df_display.columns or df_display.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No 'Status' data for pie chart", ha='center', va='center')
        ax.axis('off')
        return fig, ax

    # Group statuses into categories
    status_counts = df_display['Status'].value_counts()
    
    # Define categories
    present_late = ['Present', 'Late']
    absent_non_attendance = ['Absent - Non-attendance']
    absent_failed_verification = ['Absent - Failed verification']
    rejected = ['Rejected - cooldown', 'Rejected - duplicate', 'Rejected - too late']
    
    # Aggregate counts by category
    category_counts = {
        'Present/Late': sum(status_counts.get(s, 0) for s in present_late),
        'Absent - Non-attendance': sum(status_counts.get(s, 0) for s in absent_non_attendance),
        'Absent - Failed verification': sum(status_counts.get(s, 0) for s in absent_failed_verification),
       # 'Rejected': sum(status_counts.get(s, 0) for s in rejected),
    }
    
    # Remove zero categories
    category_counts = {k: v for k, v in category_counts.items() if v > 0}
    
    if not category_counts:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "No valid status data for pie chart", ha='center', va='center')
        ax.axis('off')
        return fig, ax
    
    # Define colors for each category
    color_map = {
        'Present/Late': '#2ecc71',  # Green for present/late
        'Absent - Non-attendance': '#e74c3c',  # Red for non-attendance
        'Absent - Failed verification': '#f39c12',  # Orange for failed verification
        #'Rejected': '#95a5a6',  # Gray for rejected
    }
    
    labels = list(category_counts.keys())
    sizes = list(category_counts.values())
    colors = [color_map.get(label, '#3498db') for label in labels]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors,
        textprops={'fontsize': 10}
    )
    
    # Enhance text visibility
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title("Attendance Status Breakdown\n(Present/Late vs Absent)", fontsize=12, fontweight='bold', pad=20)
    ax.axis('equal')
    
    # Add legend with counts
    legend_labels = [f"{label}: {count}" for label, count in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    return fig, ax


def plot_total_attendance_barh(df_display: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """
    Individual Comparison: Horizontal Bar Chart (Total Attendance per Person)
    Data: df_display.groupby('Name').size()
    """
    if 'Name' not in df_display.columns or df_display.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No 'Name' data for bar chart", ha='center', va='center')
        ax.axis('off')
        return fig, ax

    counts = df_display.groupby('Name').size().sort_values()
    height = max(4, 0.35 * len(counts))
    fig, ax = plt.subplots(figsize=(8, height))
    colors = cm.tab20(np.linspace(0, 1, len(counts)))
    ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Total Attendance Records")
    ax.set_title("Total Attendance per Person")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.1, i, str(v), va='center')
    fig.tight_layout()
    return fig, ax


def plot_arrival_histogram(df_display: pd.DataFrame, start_time: str = "09:00:00", grace_minutes: int = 15) -> Tuple[plt.Figure, plt.Axes]:
    """
    Student Arrival Time Distribution with Status Color-Coding and Confidence Overlay
    
    Creates a visually clear plot showing student arrival times and attendance status for each day.
    Groups arrival times into 5-minute intervals relative to class start time on the x-axis 
    (e.g., "0-5 min", "5-10 min", "10-15 min" from class start).
    Shows the number of students arriving in each interval on the y-axis using stacked bars
    color-coded by attendance status:
    - Green: Present
    - Yellow/Orange: Late  
    - Red: Absent
    
    Features:
    - Stacked bars grouped by 5-minute intervals (relative to class start)
    - X-axis shows time intervals (e.g., "0-5 min", "5-10 min") not actual clock times
    - Vertical line at class start time (0 minutes)
    - Shaded grace period region (green) and late period (orange)
    - Optional overlay of median recognition confidence per interval (purple line)
    - Separate subplots for each day
    - Presentation-ready styling with enhanced visuals
    - Dual y-axis when confidence data is available
    
    Args:
        df_display: DataFrame with Time, Status, Date, and optionally Confidence columns
        start_time: Class start time in HH:MM:SS format (default "09:00:00")
        grace_minutes: Grace period in minutes (default 15)
    
    Returns:
        Tuple of (figure, axes) for Streamlit display
    """
    if 'Time' not in df_display.columns or df_display.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No 'Time' data for histogram", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, ax

    df = df_display.copy()
    
    # Ensure Date column exists and is datetime
    if 'Date' not in df.columns:
        if 'DateStr' in df.columns:
            df['Date'] = pd.to_datetime(df['DateStr'], errors='coerce')
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "No 'Date' column found for day-based analysis", ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig, ax
    else:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    df = df.dropna(subset=['Date', 'Time'])
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Insufficient valid time data for histogram", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, ax
    
    # Parse start time
    try:
        start_td = pd.to_timedelta(start_time)
        start_minutes = start_td.total_seconds() / 60
    except ValueError:
        start_td = pd.to_timedelta("09:00:00")
        start_minutes = 540  # 9:00 AM in minutes
    
    # Convert Time to minutes from midnight
    df['Time'] = df['Time'].astype(str)
    arrival_td = pd.to_timedelta(df['Time'], errors='coerce')
    df['ArrivalMinutes'] = arrival_td.dt.total_seconds() / 60
    df = df.dropna(subset=['ArrivalMinutes'])
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Insufficient valid time data for histogram", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, ax
    
    # Get day information
    df['Day'] = df['Date'].dt.day_name()
    df['DateStr'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Group by day
    unique_days = sorted(df['DateStr'].unique())
    n_days = len(unique_days)
    
    if n_days == 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No valid day data", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, ax
    
    # Create subplots - one per day
    if n_days == 1:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        axes = [ax]
    elif n_days == 2:
        fig, axes_arr = plt.subplots(1, 2, figsize=(14 * 2, 6))
        axes = [axes_arr[0], axes_arr[1]]
    elif n_days <= 4:
        fig, axes_arr = plt.subplots(2, 2, figsize=(14, 12))
        axes = list(axes_arr.flatten())
    else:
        # For more than 4 days, create a grid
        cols = 2
        rows = (n_days + cols - 1) // cols
        fig, axes_arr = plt.subplots(rows, cols, figsize=(14, 6 * rows))
        axes = list(axes_arr.flatten())
    
    # Color mapping for status
    status_colors = {
        'Present': '#2ecc71',  # Green
        'Late': '#f39c12',     # Yellow/Orange
        'Absent - Non-attendance': '#e74c3c',  # Red
        'Absent - Failed verification': '#c0392b',  # Dark Red
    }
    
    # Default color for unknown statuses
    default_color = '#95a5a6'  # Gray
    
    for idx, date_str in enumerate(unique_days):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        day_data = df[df['DateStr'] == date_str].copy()
        
        if day_data.empty:
            ax.text(0.5, 0.5, f"No data for {date_str}", ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue
        
        # Calculate time relative to class start (in minutes from start)
        day_data['MinutesFromStart'] = day_data['ArrivalMinutes'] - start_minutes
        
        # Create time bins (5-minute intervals relative to class start)
        min_offset = day_data['MinutesFromStart'].min()
        max_offset = day_data['MinutesFromStart'].max()
        
        # Round to nearest 5 minutes for binning
        bin_start = (min_offset // 5) * 5
        bin_end = ((max_offset // 5) + 1) * 5
        
        # Create bins every 5 minutes
        bins = np.arange(bin_start, bin_end + 5, 5)
        
        # Group by status and time bin (relative to start)
        day_data['TimeBin'] = (day_data['MinutesFromStart'] // 5) * 5
        
        # Count arrivals by status and time bin
        status_counts = {}
        for status in day_data['Status'].unique():
            status_data = day_data[day_data['Status'] == status]
            counts = status_data.groupby('TimeBin').size()
            status_counts[status] = counts
        
        # Calculate median confidence per time bin (if Confidence column exists)
        confidence_by_bin = {}
        if 'Confidence' in day_data.columns:
            # Convert Confidence to numeric if needed
            day_data['Confidence'] = pd.to_numeric(day_data['Confidence'], errors='coerce')
            confidence_data = day_data[day_data['Confidence'].notna()]
            if not confidence_data.empty:
                confidence_by_bin = confidence_data.groupby('TimeBin')['Confidence'].median().to_dict()
        
        # Create bar positions (using intervals relative to class start)
        bin_centers = bins[:-1] + 2.5
        bar_width = 4
        
        # Plot bars for each status
        bottom = np.zeros(len(bin_centers))
        for status, counts in status_counts.items():
            values = [counts.get(bin_center - 2.5, 0) for bin_center in bin_centers]
            color = status_colors.get(status, default_color)
            ax.bar(bin_centers, values, width=bar_width, bottom=bottom, 
                  label=status, color=color, alpha=0.85, edgecolor='white', linewidth=1.2)
            bottom += values
        
        # Create interval labels (e.g., "0-5 min", "5-10 min", "10-15 min")
        # Handle negative values for early arrivals
        interval_labels = []
        for i in range(len(bins) - 1):
            start_min = int(bins[i])
            end_min = int(bins[i + 1])
            if start_min < 0:
                interval_labels.append(f"{start_min} to {end_min} min")
            else:
                interval_labels.append(f"{start_min}-{end_min} min")
        
        # Set x-axis with interval labels
        ax.set_xticks(bin_centers)
        if len(bin_centers) <= 25:
            ax.set_xticklabels(interval_labels, rotation=45, ha='right', fontsize=9)
        else:
            # Show every nth label to prevent overlap
            step = max(1, len(bin_centers) // 20)
            ax.set_xticks(bin_centers[::step])
            ax.set_xticklabels([interval_labels[i] for i in range(0, len(interval_labels), step)], 
                              rotation=45, ha='right', fontsize=9)
        
        # Add vertical line at class start time (0 minutes from start)
        ax.axvline(x=0, color='#3498db', linestyle='--', linewidth=2.5, 
                  label=f'Class Start ({start_time})', zorder=5, alpha=0.9)
        
        # Shade grace period (0 to grace_minutes)
        ax.axvspan(0, grace_minutes, alpha=0.15, color='#2ecc71', 
                  label=f'Grace Period ({grace_minutes} min)', zorder=0)
        
        # Shade late period (after grace, before 1 hour)
        late_end = 60
        if grace_minutes < late_end:
            ax.axvspan(grace_minutes, late_end, alpha=0.1, color='#f39c12', 
                      label='Late Period', zorder=0)
        
        # Add confidence overlay on secondary y-axis (if confidence data exists)
        has_confidence_overlay = False
        if confidence_by_bin:
            ax2 = ax.twinx()
            confidence_values = [confidence_by_bin.get(bin_center - 2.5, np.nan) 
                               for bin_center in bin_centers]
            # Only plot where we have valid confidence values
            valid_indices = [i for i, v in enumerate(confidence_values) if not np.isnan(v)]
            if valid_indices:
                has_confidence_overlay = True
                valid_bins = [bin_centers[i] for i in valid_indices]
                valid_conf = [confidence_values[i] for i in valid_indices]
                ax2.plot(valid_bins, valid_conf, color='#9b59b6', marker='o', 
                        linewidth=3, markersize=8, label='Median Recognition Confidence', 
                        zorder=6, alpha=0.9, markerfacecolor='white', 
                        markeredgewidth=2, markeredgecolor='#9b59b6')
                ax2.set_ylabel('Recognition Confidence (0-1)', fontsize=11, 
                             fontweight='bold', color='#9b59b6', labelpad=10)
                ax2.set_ylim(0, 1)
                ax2.tick_params(axis='y', labelcolor='#9b59b6', labelsize=10)
                ax2.grid(False)  # Don't show grid on secondary axis
        
        # Enhanced labels and title for presentations
        day_name = day_data['Day'].iloc[0] if 'Day' in day_data.columns else date_str
        ax.set_xlabel('Time Interval from Class Start (minutes)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Number of Students', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title(f'Student Arrival Times & Attendance Status - {day_name} ({date_str})', 
                    fontsize=14, fontweight='bold', pad=15, color='#2c3e50')
        
        # Enhanced grid and styling
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.4, axis='y', color='gray')
        ax.set_axisbelow(True)
        ax.set_facecolor('#fafafa')  # Light gray background for better contrast
        
        # Enhanced legend with better positioning
        handles1, labels1 = ax.get_legend_handles_labels()
        if has_confidence_overlay:
            handles2, labels2 = ax2.get_legend_handles_labels()
            all_handles = handles1 + handles2
            all_labels = labels1 + labels2
        else:
            all_handles = handles1
            all_labels = labels1
        
        # Position legend to avoid overlap
        ax.legend(all_handles, all_labels, loc='upper left', fontsize=9, 
                 framealpha=0.95, frameon=True, fancybox=True, shadow=True,
                 edgecolor='gray', facecolor='white')
        
        ax.set_ylim(bottom=0)
        
        # Add subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor('#bdc3c7')
            spine.set_linewidth(1.2)
    
    # Hide unused subplots
    for idx in range(len(unique_days), len(axes)):
        axes[idx].axis('off')
    
    # Enhanced figure-level styling for presentations
    fig.patch.set_facecolor('white')
    
    # Add overall title only if multiple days
    if n_days > 1:
        fig.suptitle('Student Arrival Times & Attendance Analysis', 
                    fontsize=16, fontweight='bold', y=0.98, color='#2c3e50')
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    else:
        fig.tight_layout()
    
    # Return the first axes for consistency with other plotting functions
    return fig, axes[0] if len(axes) > 0 else fig


def plot_simple_arrival_intervals(df_display: pd.DataFrame, start_time: str = "09:00:00", class_duration: int = 50) -> Tuple[plt.Figure, plt.Axes]:
    """
    Simple Bar Plot: Student Arrivals in 4 Equal Intervals for a 50-Minute Class
    
    Creates a clean, simple bar plot showing student arrivals divided into 4 equal time intervals.
    Only counts students who actually arrived (Present or Late status), excluding absences.
    
    Features:
    - 4 equal intervals (e.g., 0-12.5 min, 12.5-25 min, 25-37.5 min, 37.5-50 min)
    - Only shows actual arrivals (no absences)
    - Clear bar labels showing count in each interval
    - Clean, presentation-ready styling
    
    Args:
        df_display: DataFrame with Time, Status, and Date columns
        start_time: Class start time in HH:MM:SS format (default "09:00:00")
        class_duration: Total class duration in minutes (default 50)
    
    Returns:
        Tuple of (figure, axes) for Streamlit display
    """
    if 'Time' not in df_display.columns or df_display.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No 'Time' data for arrival intervals", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, ax

    df = df_display.copy()
    
    # Filter only actual arrivals (Present or Late), exclude absences
    if 'Status' in df.columns:
        df = df[df['Status'].isin(['Present', 'Late'])]
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No arrival data available", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, ax
    
    # Parse start time
    try:
        start_td = pd.to_timedelta(start_time)
        start_minutes = start_td.total_seconds() / 60
    except ValueError:
        start_td = pd.to_timedelta("09:00:00")
        start_minutes = 540  # 9:00 AM in minutes
    
    # Convert Time to minutes from midnight
    df['Time'] = df['Time'].astype(str)
    arrival_td = pd.to_timedelta(df['Time'], errors='coerce')
    df['ArrivalMinutes'] = arrival_td.dt.total_seconds() / 60
    df = df.dropna(subset=['ArrivalMinutes'])
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Insufficient valid time data", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, ax
    
    # Calculate time relative to class start (in minutes from start)
    df['MinutesFromStart'] = df['ArrivalMinutes'] - start_minutes
    
    # Filter arrivals within class duration (0 to class_duration minutes)
    df = df[(df['MinutesFromStart'] >= 0) & (df['MinutesFromStart'] <= class_duration)]
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No arrivals within class duration", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, ax
    
    # Divide into 4 equal intervals
    interval_size = class_duration / 4  # 12.5 minutes for 50-minute class
    
    # Create interval labels (clean formatting)
    intervals = []
    interval_labels = []
    for i in range(4):
        start = i * interval_size
        end = (i + 1) * interval_size
        intervals.append((start, end))
        # Format: remove .0 for whole numbers, keep .1 for decimals
        start_str = f"{int(start)}" if start == int(start) else f"{start:.1f}"
        end_str = f"{int(end)}" if end == int(end) else f"{end:.1f}"
        interval_labels.append(f"{start_str}-{end_str} min")
    
    # Count arrivals in each interval
    counts = []
    for start, end in intervals:
        # Count arrivals in this interval (start inclusive, end exclusive, except last interval)
        if end == class_duration:
            count = len(df[(df['MinutesFromStart'] >= start) & (df['MinutesFromStart'] <= end)])
        else:
            count = len(df[(df['MinutesFromStart'] >= start) & (df['MinutesFromStart'] < end)])
        counts.append(count)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create bars with clean styling
    bars = ax.bar(interval_labels, counts, width=0.7, color='#3498db', 
                  edgecolor='white', linewidth=2, alpha=0.85)
    
    # Add value labels on top of each bar
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}',
               ha='center', va='bottom', fontsize=14, fontweight='bold', color='#2c3e50')
    
    # Styling
    ax.set_xlabel('Time Interval from Class Start', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_ylabel('Number of Students', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_title(f'Student Arrival Distribution - {class_duration} Minute Class', 
                fontsize=15, fontweight='bold', pad=15, color='#2c3e50')
    
    # Grid and background
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.4, axis='y', color='gray')
    ax.set_axisbelow(True)
    ax.set_facecolor('#fafafa')
    
    # Y-axis starts at 0
    ax.set_ylim(bottom=0)
    # Add some padding at top for labels
    if max(counts) > 0:
        ax.set_ylim(top=max(counts) * 1.15)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    # X-axis label rotation if needed
    ax.tick_params(axis='x', labelsize=11, rotation=0)
    ax.tick_params(axis='y', labelsize=11)
    
    # Make sure y-axis shows integers
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    fig.tight_layout()
    return fig, ax


def plot_weekly_trend(df: pd.DataFrame, similarity_threshold: float = 0.93) -> Tuple[plt.Figure, plt.Axes]:
    """
    Enhanced Weekly Attendance Trend with Dual Analysis:
    1. Attendance Behavior: Daily proportions of Present, Late, and Absent students
    2. Recognition Quality: Median confidence with interquartile ranges (IQR)
    
    Features:
    - Separates attendance behavior from recognition quality
    - Shows daily attendance status proportions (stacked bars)
    - Displays median confidence with IQR (line plot with shaded region)
    - Overlays similarity threshold to highlight risk zones
    - Color-codes to reveal relationship between low confidence and absences
    
    Args:
        df: DataFrame with Date, Status, and Confidence columns
        similarity_threshold: Threshold for face recognition (default 0.93)
    """
    if 'Date' not in df.columns or df.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, "No 'Date' data for weekly trend", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, ax
        
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, "No valid 'Date' data for weekly trend", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, ax

    # Calculate Year-Week and Day
    isocal = df['Date'].dt.isocalendar()
    df['YearWeek'] = isocal['year'].astype(str) + "-W" + isocal['week'].astype(str).str.zfill(2)
    df['DayOfWeek'] = df['Date'].dt.day_name()
    
    # Prepare data for visualization
    # Group by YearWeek and Date to get daily proportions
    df['DateStr'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Handle Confidence column - convert to numeric if it exists
    if 'Confidence' in df.columns:
        df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
    
    # Aggregate by day
    agg_dict = {'Status': lambda x: x.value_counts().to_dict()}
    if 'Confidence' in df.columns:
        agg_dict['Confidence'] = lambda x: x.dropna().tolist() if x.notna().any() else []
    
    daily_data = df.groupby(['YearWeek', 'DateStr']).agg(agg_dict).reset_index()
    
    # Calculate daily proportions and confidence metrics
    daily_stats = []
    for _, row in daily_data.iterrows():
        week = row['YearWeek']
        date_str = row['DateStr']
        status_counts = row['Status'] if isinstance(row['Status'], dict) else {}
        confidences = row.get('Confidence', []) if isinstance(row.get('Confidence', []), list) else []
        
        total = sum(status_counts.values())
        if total == 0:
            continue
            
        # Calculate proportions
        present_pct = (status_counts.get('Present', 0) / total) * 100
        late_pct = (status_counts.get('Late', 0) / total) * 100
        absent_pct = ((status_counts.get('Absent - Non-attendance', 0) + 
                       status_counts.get('Absent - Failed verification', 0)) / total) * 100
        
        # Calculate confidence metrics
        if confidences:
            try:
                conf_array = np.array([float(c) for c in confidences if c != '' and pd.notna(c) and str(c).strip() != ''])
                if len(conf_array) > 0:
                    median_conf = np.median(conf_array)
                    q25 = np.percentile(conf_array, 25)
                    q75 = np.percentile(conf_array, 75)
                else:
                    median_conf = np.nan
                    q25 = np.nan
                    q75 = np.nan
            except (ValueError, TypeError):
                median_conf = np.nan
                q25 = np.nan
                q75 = np.nan
        else:
            median_conf = np.nan
            q25 = np.nan
            q75 = np.nan
        
        daily_stats.append({
            'YearWeek': week,
            'Date': date_str,
            'Present_Pct': present_pct,
            'Late_Pct': late_pct,
            'Absent_Pct': absent_pct,
            'Median_Confidence': median_conf,
            'Q25_Confidence': q25,
            'Q75_Confidence': q75,
            'Total': total
        })
    
    if not daily_stats:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, "No weekly data points for trend analysis", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, ax
    
    stats_df = pd.DataFrame(daily_stats)
    stats_df = stats_df.sort_values(['YearWeek', 'Date'])
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(16, 9))
    
    # Prepare x-axis positions
    x_positions = np.arange(len(stats_df))
    x_labels = [f"{row['YearWeek']}\n{row['Date']}" for _, row in stats_df.iterrows()]
    
    # === LEFT Y-AXIS: Attendance Proportions (Stacked Bars) ===
    present_values = stats_df['Present_Pct'].values
    late_values = stats_df['Late_Pct'].values
    absent_values = stats_df['Absent_Pct'].values
    
    # Stacked bar chart for attendance proportions
    bars1 = ax1.bar(x_positions, present_values, width=0.6, label='Present', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x_positions, late_values, width=0.6, bottom=present_values, label='Late', color='#f39c12', alpha=0.8)
    bars3 = ax1.bar(x_positions, absent_values, width=0.6, bottom=present_values + late_values, 
                    label='Absent', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Week and Date', fontsize=13, fontweight='bold', labelpad=12)
    ax1.set_ylabel('Attendance Proportion (%)', fontsize=13, fontweight='bold', labelpad=12, color='#2c3e50')
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor='#2c3e50', labelsize=11)
    ax1.grid(True, linestyle='--', linewidth=0.7, alpha=0.4, axis='y')
    ax1.set_axisbelow(True)
    
    # === RIGHT Y-AXIS: Recognition Quality (Confidence Metrics) ===
    ax2 = ax1.twinx()
    
    # Plot median confidence with IQR
    median_conf = stats_df['Median_Confidence'].dropna()
    q25_conf = stats_df['Q25_Confidence'].dropna()
    q75_conf = stats_df['Q75_Confidence'].dropna()
    
    valid_indices = median_conf.index
    valid_x = [x_positions[i] for i in range(len(x_positions)) if i in valid_indices]
    
    if len(valid_x) > 0:
        # Plot IQR shaded region
        ax2.fill_between(
            valid_x,
            q25_conf.values,
            q75_conf.values,
            alpha=0.3,
            color='#3498db',
            label='Confidence IQR (25th-75th percentile)'
        )
        
        # Plot median confidence line
        line = ax2.plot(
            valid_x,
            median_conf.values,
            marker='o',
            linestyle='-',
            linewidth=2.5,
            markersize=7,
            color='#2980b9',
            markerfacecolor='#3498db',
            markeredgecolor='white',
            markeredgewidth=1.5,
            label='Median Confidence',
            zorder=5
        )
        
        # Overlay similarity threshold line
        ax2.axhline(
            y=similarity_threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label=f'Similarity Threshold ({similarity_threshold})',
            zorder=4
        )
        
        # Highlight risk zones (below threshold)
        ax2.fill_between(
            valid_x,
            0,
            similarity_threshold,
            alpha=0.15,
            color='red',
            label='Risk Zone (Below Threshold)'
        )
    
    ax2.set_ylabel('Face Recognition Confidence', fontsize=13, fontweight='bold', labelpad=12, color='#2980b9')
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='y', labelcolor='#2980b9', labelsize=11)
    
    # X-axis formatting
    ax1.set_xticks(x_positions)
    if len(x_positions) <= 15:
        ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    else:
        step = max(1, len(x_positions) // 15)
        ticks = list(range(0, len(x_positions), step))
        if ticks[-1] != len(x_positions) - 1:
            ticks.append(len(x_positions) - 1)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([x_labels[i] for i in ticks], rotation=45, ha='right', fontsize=9)
    
    # Title with comprehensive information
    total_days = len(stats_df)
    avg_present = stats_df['Present_Pct'].mean()
    avg_absent = stats_df['Absent_Pct'].mean()
    avg_confidence = stats_df['Median_Confidence'].mean()
    
    title_text = (
        f"Weekly Attendance Trend Analysis: Behavior vs Recognition Quality\n"
        f"Days Analyzed: {total_days} | Avg Present: {avg_present:.1f}% | "
        f"Avg Absent: {avg_absent:.1f}% | Avg Median Confidence: {avg_confidence:.3f}"
    )
    ax1.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add correlation annotation
    if len(median_conf) > 1 and len(absent_values) > 1:
        # Calculate correlation between confidence and absence rate
        valid_absent = [absent_values[i] for i in range(len(absent_values)) if i in valid_indices]
        if len(valid_absent) == len(median_conf) and len(valid_absent) > 1:
            correlation = np.corrcoef(median_conf.values, valid_absent)[0, 1]
            corr_text = f"Confidence-Absence Correlation: {correlation:.3f}"
            if correlation < -0.3:
                corr_text += " (Strong negative: Low confidence → High absence)"
            elif correlation < -0.1:
                corr_text += " (Moderate negative relationship)"
            else:
                corr_text += " (Weak relationship)"
            
            ax1.text(
                0.02, 0.02, corr_text,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange'),
                family='monospace'
            )
    
    # Add statistics box
    stats_text = (
        f"Statistics:\n"
        f"Present Avg: {avg_present:.1f}%\n"
        f"Absent Avg: {avg_absent:.1f}%\n"
        f"Confidence Median: {avg_confidence:.3f}\n"
        f"Threshold: {similarity_threshold}"
    )
    ax1.text(
        0.98, 0.98, stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
        family='monospace'
    )
    
    fig.tight_layout()
    return fig, ax1


def plot_morning_evening_scatter(df_display: pd.DataFrame, morning_cutoff: str = "12:00:00") -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter Plot: Morning vs Evening attendance counts per person.
    Purpose: For each person, plot (morning_count, evening_count) so the ratio is visible.
    - morning_cutoff: time string (HH:MM:SS). Records with Time < cutoff are 'morning', >= cutoff are 'evening'.
    Expects df_display to have 'Name' and 'Time'.
    """
    if 'Name' not in df_display.columns or 'Time' not in df_display.columns or df_display.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Missing 'Name' or 'Time' data for scatter plot", ha='center', va='center')
        ax.axis('off')
        return fig, ax
        
    df = df_display.copy()
    df['Time'] = df['Time'].astype(str)
    arrival_td = pd.to_timedelta(df['Time'], errors='coerce')
    
    try:
        cutoff_min = pd.to_timedelta(morning_cutoff).total_seconds() / 60
    except ValueError:
        cutoff_min = pd.to_timedelta("12:00:00").total_seconds() / 60  # Default fallback

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
    
    if len(x) > 0:
        scatter = ax.scatter(x, y, s=80, alpha=0.7, c=cm.tab10(np.linspace(0, 1, len(x))))
        max_val = max(x.max(), y.max(), 1)
        ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel("Morning Attendance Count")
        ax.set_ylabel("Evening Attendance Count")
        ax.set_title("Morning vs Evening Attendance per Person (ratio visualization)")
        # Annotate points with names if not too many
        if len(labels) <= 40:
            for i, txt in enumerate(labels):
                ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data points for morning/evening analysis", ha='center', va='center')
        ax.axis('off')

    return fig, ax


def plot_course_attendance_bar(df_display: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """
    Course Attendance: Vertical Bar Chart showing attendance counts per Course.
    Data: df_display['Course']
    """
    if 'Course' not in df_display.columns or df_display.empty or df_display['Course'].dropna().empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No 'Course' column or data found", ha='center', va='center')
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

    df = df.dropna(subset=['Day'])

    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No valid 'Day' data for bar chart", ha='center', va='center')
        ax.axis('off')
        return fig, ax
        
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['Day'].value_counts().reindex(day_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = cm.Pastel1(np.linspace(0, 1, len(day_counts)))
    ax.bar(day_counts.index, day_counts.values, color=colors)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Attendance Volume")
    ax.set_title("Attendance Volume by Day of Week")
    fig.tight_layout()
    return fig, ax


# Face Attendance App Class 
class FaceAttendanceApp:
    """OOP Streamlit app using OpenCV-only face features for matching."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.images_path = self.base_dir / "Images"
        self.attendance_file = self.base_dir / "Attendance.csv"
        self.rejected_file = self.base_dir / "Attendance_Rejected.csv"
        self.master_registration_file = self.base_dir / "Master_Registration.csv"
        self.images_path.mkdir(exist_ok=True)

        # Prevent spamming attendance within a short window (in-memory only)
        self.attendance_tracker: dict[str, datetime] = {}
        
        # Track students who appeared during current session (for absent marking)
        # Key: (course, session, date) -> set of student names who appeared
        self.session_appearances: dict[tuple[str, str, str], set[str]] = {}
        # Track failed verification attempts (low confidence) per session
        # Key: (course, session, date) -> dict of {student_name: max_confidence_seen}
        self.session_failed_verifications: dict[tuple[str, str, str], dict[str, float]] = {}

        # Simple classification / metadata defaults
        self.system_version = "1.1"
        # Cosine similarity threshold for OpenCV-based face vectors (0–1)
        self.similarity_threshold = 0.93
        self.class_start_hour = 9
        self.class_start_minute = 0
        self.present_grace_minutes = 15  # <= 15 mins late -> Present; <=60 -> Late; else Rejected
        self.session_duration_minutes = 90  # Default session duration for absent marking

        # Use OpenCV Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # In-memory embeddings (simple grayscale vectors for known faces)
        self.known_names: list[str] = []
        self.known_vectors: list[np.ndarray] = []

        self.load_known_faces()

    # ---------- Utility helpers ----------
    @staticmethod
    def decode_image(uploaded_file) -> np.ndarray:
        """Convert Streamlit camera input to an RGB numpy array."""
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)

    def _face_vector(self, bgr_image: np.ndarray) -> np.ndarray | None:
        """Compute a simple embedding: grayscale face region resized to 64x64 and flattened.

        Returns None if no face is detected.
        """
        try:
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                return None

            x, y, w, h = faces[0]
            face_roi = gray[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi, (64, 64))
            vec = face_resized.flatten().astype("float32")
            vec /= (np.linalg.norm(vec) + 1e-8)
            return vec
        except Exception:
            # Any failure means we simply couldn't compute a usable face vector
            return None

    def load_known_faces(self) -> None:
        """Load stored face images and compute simple vectors for recognition."""
        self.known_names = []
        self.known_vectors = []

        for image_path in self.images_path.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                vec = self._face_vector(img)
                if vec is None:
                    continue
                self.known_names.append(image_path.stem)
                self.known_vectors.append(vec)
            except Exception:
                pass

    def get_master_registration(self, course: str, session: str) -> set[str]:
        """Get the master list of registered students for a course and session."""
        if not self.master_registration_file.exists():
            return set()
        
        try:
            df = pd.read_csv(self.master_registration_file, comment="#")
            if df.empty or "Course" not in df.columns or "Session" not in df.columns or "StudentName" not in df.columns:
                return set()
            mask = (df["Course"] == course) & (df["Session"] == session)
            return set(df[mask]["StudentName"].dropna().unique().tolist())
        except Exception:
            return set()

    def add_to_master_registration(self, course: str, session: str, student_name: str) -> bool:
        """Add a student to the master registration list for a course and session."""
        self.master_registration_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.master_registration_file.exists()
        write_header = (not file_exists) or self.master_registration_file.stat().st_size == 0

        # Check if already exists
        if file_exists:
            try:
                df = pd.read_csv(self.master_registration_file, comment="#")
                if not df.empty and "Course" in df.columns and "Session" in df.columns and "StudentName" in df.columns:
                    mask = (df["Course"] == course) & (df["Session"] == session) & (df["StudentName"] == student_name)
                    if mask.any():
                        return False  # Already registered
            except Exception:
                pass

        with self.master_registration_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Course", "Session", "StudentName", "RegisteredAt"])
            writer.writerow([course, session, student_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        return True

    def mark_absent_students(self, course: str, session: str, date_str: str) -> tuple[int, int]:
        """
        Mark absent students for a session:
        - Non-attendance: registered but never appeared
        - Failed verification: appeared but confidence below threshold
        
        Returns (non_attendance_count, failed_verification_count)
        """
        master_list = self.get_master_registration(course, session)
        if not master_list:
            return 0, 0

        session_key = (course, session, date_str)
        appeared_students = self.session_appearances.get(session_key, set())
        failed_verifications = self.session_failed_verifications.get(session_key, {})

        # Check existing attendance to avoid duplicates
        existing_attendance = set()
        if self.attendance_file.exists():
            try:
                df_existing = pd.read_csv(self.attendance_file, comment="#")
                if not df_existing.empty and "Date" in df_existing.columns and "Course" in df_existing.columns and "Session" in df_existing.columns and "Name" in df_existing.columns:
                    mask = (df_existing["Date"] == date_str) & (df_existing["Course"] == course) & (df_existing["Session"] == session)
                    existing_attendance = set(df_existing[mask]["Name"].dropna().unique().tolist())
            except Exception:
                pass

        non_attendance_count = 0
        failed_verification_count = 0
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        day_str = now.strftime("%A")

        with self.attendance_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for student in master_list:
                if student in existing_attendance:
                    continue  # Already has attendance record

                if student not in appeared_students:
                    # Non-attendance: registered but never appeared
                    writer.writerow([
                        student,
                        date_str,
                        time_str,
                        day_str,
                        "Absent - Non-attendance",
                        course,
                        session,
                        "",
                        "System",
                        now.strftime("%Y-%m-%d %H:%M:%S"),
                        self.system_version,
                        self.similarity_threshold,
                    ])
                    non_attendance_count += 1
                elif student in failed_verifications:
                    # Failed verification: appeared but confidence too low
                    max_confidence = failed_verifications[student]
                    writer.writerow([
                        student,
                        date_str,
                        time_str,
                        day_str,
                        "Absent - Failed verification",
                        course,
                        session,
                        f"{max_confidence:.4f}",
                        "System",
                        now.strftime("%Y-%m-%d %H:%M:%S"),
                        self.system_version,
                        self.similarity_threshold,
                    ])
                    failed_verification_count += 1

        return non_attendance_count, failed_verification_count

    def _log_reject(
        self,
        name: str | None,
        date_str: str,
        time_str: str,
        day_str: str,
        reason: str,
        confidence: float | None,
        source: str,
    ) -> None:
        """Log rejected or failed attempts to a separate CSV."""
        self.rejected_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.rejected_file.exists()
        write_header = (not file_exists) or self.rejected_file.stat().st_size == 0

        with self.rejected_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "Name",
                        "Date",
                        "Time",
                        "Day",
                        "Reason",
                        "Confidence",
                        "Source",
                        "CreatedAt",
                        "SystemVersion",
                        "SimilarityThreshold",
                    ]
                )
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(
                [
                    name or "",
                    date_str,
                    time_str,
                    day_str,
                    reason,
                    f"{confidence:.4f}" if confidence is not None else "",
                    source,
                    created_at,
                    self.system_version,
                    self.similarity_threshold,
                ]
            )

    def mark_attendance(self, name: str, confidence: float, source: str = "Camera") -> tuple[str, bool]:
        """Append attendance with structured schema and classification.

        Returns (status, recorded) where status is Present/Late/Rejected* and recorded
        indicates whether the row was written to the main attendance CSV.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        day_str = now.strftime("%A")

        # Short-term cooldown (in-memory only)
        last_seen = self.attendance_tracker.get(name)
        if last_seen and (now - last_seen) < timedelta(minutes=30):
            self._log_reject(
                name,
                date_str,
                time_str,
                day_str,
                reason="Cooldown active (recent attendance already recorded)",
                confidence=confidence,
                source=source,
            )
            return "Rejected - cooldown", False

        # One attendance per person per day (check persisted CSV)
        if self.attendance_file.exists():
            try:
                df_existing = pd.read_csv(self.attendance_file, comment="#")
                if "Date" in df_existing.columns and "Name" in df_existing.columns and "Status" in df_existing.columns:
                    mask = (df_existing["Name"].str.lower() == name.lower()) & (df_existing["Date"] == date_str)
                    mask = mask & df_existing["Status"].isin(["Present", "Late"])
                    if mask.any():
                        self._log_reject(
                            name,
                            date_str,
                            time_str,
                            day_str,
                            reason="Duplicate attendance for this date",
                            confidence=confidence,
                            source=source,
                        )
                        return "Rejected - duplicate", False
            except Exception:
                # On read failure, fall back to just in-memory cooldown
                pass

        # Time-based status classification
        class_start = now.replace(
            hour=self.class_start_hour, minute=self.class_start_minute, second=0, microsecond=0
        )
        delta_minutes = (now - class_start).total_seconds() / 60.0
        if delta_minutes <= self.present_grace_minutes:
            status = "Present"
        elif delta_minutes <= 60:
            status = "Late"
        else:
            status = "Rejected - too late"
            self._log_reject(
                name,
                date_str,
                time_str,
                day_str,
                reason="Arrived too late for session",
                confidence=confidence,
                source=source,
            )
            return status, False

        # Persist accepted attendance
        self.attendance_tracker[name] = now
        self.attendance_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.attendance_file.exists()
        write_header = (not file_exists) or self.attendance_file.stat().st_size == 0

        with self.attendance_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "Name",
                        "Date",
                        "Time",
                        "Day",
                        "Status",
                        "Course",
                        "Session",
                        "Confidence",
                        "Source",
                        "CreatedAt",
                        "SystemVersion",
                        "SimilarityThreshold",
                    ]
                )

            # Session / course metadata from sidebar (if set)
            course = st.session_state.get("course", "")
            session = st.session_state.get("session", "")
            created_at = now.strftime("%Y-%m-%d %H:%M:%S")

            writer.writerow(
                [
                    name,
                    date_str,
                    time_str,
                    day_str,
                    status,
                    course,
                    session,
                    f"{confidence:.4f}",
                    source,
                    created_at,
                    self.system_version,
                    self.similarity_threshold,
                ]
            )

        return status, True

    # ---------- Page renderers ----------
    def render_home(self) -> None:
        st.markdown('<div class="big-title">Face Attendance System</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Minimal, camera-based attendance using Streamlit and OpenCV.</div>',
            unsafe_allow_html=True,
        )
        with st.container():
            st.markdown("### 👣 Quick steps")
            st.markdown(
                "- **Register**: capture a clear photo with your name.\n"
                "- **Mark Attendance**: capture again to log a timestamp.\n"
                "- **View Attendance**: review and download previous records."
            )

    def render_register(self) -> None:
        st.markdown('<div class="big-title">👤 Register</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Capture a clear, front-facing photo to register your face.</div>',
            unsafe_allow_html=True,
        )

        # Keep multiple registration samples in session for more robust matching
        if "reg_samples" not in st.session_state:
            st.session_state["reg_samples"] = []
        if "last_reg_image_bgr" not in st.session_state:
            st.session_state["last_reg_image_bgr"] = None

        name_input = st.text_input("Name", placeholder="Type your full name")
        register_capture = st.camera_input("Camera")

        col_add, col_clear, col_save = st.columns(3)

        with col_add:
            if st.button("Add sample"):
                if register_capture is None:
                    st.error("Capture a photo before adding a sample.")
                else:
                    try:
                        image_rgb = self.decode_image(register_capture)
                        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    except Exception as exc:
                        st.error(f"Could not process camera image: {exc}")
                    else:
                        vec = self._face_vector(image_bgr)
                        if vec is None:
                            st.error("No face detected in this sample. Please try again.")
                        else:
                            st.session_state["reg_samples"].append(vec)
                            st.session_state["last_reg_image_bgr"] = image_bgr
                            st.info(f"Samples captured: {len(st.session_state['reg_samples'])}")

        with col_clear:
            if st.button("Clear samples"):
                st.session_state["reg_samples"] = []
                st.session_state["last_reg_image_bgr"] = None
                st.info("Cleared stored samples.")

        with col_save:
            if st.button("➕ Save face"):
                name = (name_input or "").strip()
                if not name:
                    st.error("Please provide a valid name (not empty).")
                    return
                if any(existing.lower() == name.lower() for existing in self.known_names):
                    st.error("This name is already registered. Please use a different name or update the existing one.")
                    return

                samples: list[np.ndarray] = st.session_state.get("reg_samples", [])
                if len(samples) < 3:
                    st.error("Please add at least 3 good samples before saving.")
                    return

                # Average all collected face vectors to get a more robust representation
                stacked = np.stack(samples, axis=0)
                avg_vec = stacked.mean(axis=0).astype("float32")
                avg_vec /= (np.linalg.norm(avg_vec) + 1e-8)

                image_bgr = st.session_state.get("last_reg_image_bgr")
                if image_bgr is None:
                    if register_capture is None:
                        st.error("No camera image available to save. Capture a photo again and add as sample.")
                        return
                    try:
                        image_rgb = self.decode_image(register_capture)
                        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    except Exception as exc:
                        st.error(f"Could not process camera image for saving: {exc}")
                        return

                save_path = self.images_path / f"{name}.jpg"
                cv2.imwrite(str(save_path), image_bgr)

                # Update in-memory database
                self.known_names.append(name)
                self.known_vectors.append(avg_vec)

                # Reset samples after successful registration
                st.session_state["reg_samples"] = []
                st.session_state["last_reg_image_bgr"] = None

                st.success(f"Registration successful for {name}! (samples used: {len(samples)})")

    def _match_face(self, bgr_image: np.ndarray) -> tuple[str, float] | None:
        """Match captured face against known vectors using cosine similarity.

        Returns (name, confidence) if a strong enough match is found,
        otherwise None. Confidence is the cosine similarity (0–1).
        """
        if not self.known_vectors:
            return None

        vec = self._face_vector(bgr_image)
        if vec is None:
            return None

        sims = [float(np.dot(vec, known_vec)) for known_vec in self.known_vectors]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= self.similarity_threshold:
            return self.known_names[best_idx], best_sim
        return None

    def render_attendance(self) -> None:
        st.markdown('<div class="big-title">✅ Mark Attendance</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Stand in front of the camera and capture a frame to record attendance.</div>',
            unsafe_allow_html=True,
        )
        if not self.known_names:
            st.info("No registered faces yet. Please register first.")
            return

        attendance_capture = st.camera_input("Take a photo to mark attendance")
        if not attendance_capture:
            return

        try:
            image_rgb = self.decode_image(attendance_capture)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            st.error(f"Could not process camera image: {exc}")
            return

        # Quick visual feedback: draw rectangle if face detected
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        match = self._match_face(image_bgr)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        course = st.session_state.get("course", "")
        session = st.session_state.get("session", "")
        session_key = (course, session, date_str)

        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Captured frame")
        
        if match:
            matched_name, confidence = match
            # Track appearance for this session
            if session_key not in self.session_appearances:
                self.session_appearances[session_key] = set()
            self.session_appearances[session_key].add(matched_name)
            
            status, recorded = self.mark_attendance(matched_name, confidence, source="Camera")
            if recorded:
                st.success(f"Attendance marked for {matched_name} ({status}, confidence={confidence:.3f}).")
            else:
                st.info(f"Attendance not recorded for {matched_name} ({status}).")
        else:
            # Check if any registered student might be present but with low confidence
            master_list = self.get_master_registration(course, session)
            if master_list:
                # Try to find best match even if below threshold
                vec = self._face_vector(image_bgr)
                if vec is not None:
                    sims = [float(np.dot(vec, known_vec)) for known_vec in self.known_vectors]
                    if sims:
                        best_idx = int(np.argmax(sims))
                        best_sim = sims[best_idx]
                        best_name = self.known_names[best_idx]
                        
                        if best_name in master_list:
                            # Track failed verification attempt
                            if session_key not in self.session_failed_verifications:
                                self.session_failed_verifications[session_key] = {}
                            if best_name not in self.session_failed_verifications[session_key] or best_sim > self.session_failed_verifications[session_key][best_name]:
                                self.session_failed_verifications[session_key][best_name] = best_sim
                            
                            st.warning(
                                f"Low confidence match ({best_sim:.3f} < {self.similarity_threshold}) for {best_name}. "
                                f"This will be marked as 'Absent - Failed verification' at session end."
                            )
                            return
            
            time_str = now.strftime("%H:%M:%S")
            day_str = now.strftime("%A")
            # Log as rejected attempt with no recognized name
            self._log_reject(
                name=None,
                date_str=date_str,
                time_str=time_str,
                day_str=day_str,
                reason="No reliable face match",
                confidence=None,
                source="Camera",
            )
            st.error(
                "No reliable match found. Make sure the person is registered and the face is clearly visible."
            )

    def render_history(self) -> None:
        """Show previous attendance records using pandas + numpy and display plots."""
        st.markdown('<div class="big-title">📊 Attendance History</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Browse and export all previously marked attendance entries.</div>',
            unsafe_allow_html=True,
        )

        if not self.attendance_file.exists():
            st.info("No attendance has been recorded yet.")
            return

        # Load CSV into a DataFrame (handle both legacy and new schemas).
        try:
            df_raw = pd.read_csv(
                self.attendance_file,
                comment="#",
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as exc:
            st.error(f"Could not read attendance file: {exc}")
            return

        if df_raw.empty:
            st.info("Attendance file is empty.")
            return

        # Legacy format conversion (as per original code)
        if list(df_raw.columns) == ["Name", "Timestamp"] or df_raw.shape[1] == 2:
            df = df_raw.copy()
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp"])
            df["Date"] = df["Timestamp"].dt.date.astype(str)
            df["Time"] = df["Timestamp"].dt.time.astype(str)
            df["Day"] = df["Timestamp"].dt.day_name()
            df["Status"] = "Present"
            df["Course"] = ""
            df["Session"] = ""
            df["Confidence"] = np.nan
            df["Source"] = "Camera"
            df["CreatedAt"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df["SystemVersion"] = ""
            df["SimilarityThreshold"] = np.nan
        else:
            df = df_raw.copy()
            if "Time" in df.columns:
                df["Time"] = df["Time"].astype(str)

        # Ensure Date column is proper datetime for any schema
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])

        # Optional filtering by name, date and session (as per original code)
        all_names = sorted(df["Name"].dropna().unique().tolist())
        selected_name = st.selectbox("Filter by name (optional)", options=["All"] + all_names)

        if "Date" in df.columns:
            unique_dates = sorted({str(d.date()) for d in df["Date"].dropna()})
        else:
            unique_dates = []
        selected_date = st.selectbox("Filter by date (optional)", options=["All"] + unique_dates)

        if "Session" in df.columns:
            existing_sessions = set(df["Session"].dropna().astype(str).unique().tolist())
            base_sessions = ["Morning", "Evening"]
            all_sessions = [s for s in base_sessions if s in existing_sessions] or base_sessions
            selected_session = st.selectbox(
                "Filter by session (optional)", options=["All"] + all_sessions
            )
        else:
            selected_session = st.selectbox("Filter by session (optional)", options=["All"])

        df_display = df
        if selected_name != "All":
            df_display = df_display[df_display["Name"] == selected_name]
        if selected_date != "All" and "Date" in df_display.columns:
            df_display = df_display[df_display["Date"].dt.date.astype(str) == selected_date]
        if selected_session != "All" and "Session" in df_display.columns:
            df_display = df_display[df_display["Session"] == selected_session]

        # Stats (as per original code)
        name_array = df_display["Name"].to_numpy()
        total_records = int(name_array.size)
        unique_people = int(np.unique(name_array).size)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total records (filtered)", total_records)
        col2.metric("Unique people (filtered)", unique_people)
        if "Date" in df_display.columns:
            total_days = int(df_display["Date"].dt.date.nunique())
            col3.metric("Distinct days (filtered)", total_days)

        # Dataframe Display (as per original code)
        hide_cols = {"Confidence", "SystemVersion", "SimilarityThreshold", "CreatedAt"}
        cols_to_show = [c for c in df_display.columns if c not in hide_cols]

        st.dataframe(
            df_display[cols_to_show].sort_values(
                ["Date", "Time"] if "Date" in df_display.columns and "Time" in df_display.columns else cols_to_show,
                ascending=False,
                ignore_index=True,
            ),
            use_container_width=True,
        )

        # Per-person attendance percentage (as per original code)
        if not df_display.empty:
            counts = df_display.groupby("Name").size().rename("Count")
            total = counts.sum()
            perc = (counts / total * 100.0).round(1).rename("Percent")
            summary_df = pd.concat([counts, perc], axis=1).reset_index()
            st.markdown("#### Attendance summary (filtered)")
            st.dataframe(summary_df, use_container_width=True)

        # --- New: Plotting Section Integration ---
        if not df_display.empty:
            st.markdown("#### 📈 Attendance Charts (Filtered Data)")
            
            # Row 1: Status Pie Chart and Day of Week Bars
            col_pie, col_day = st.columns(2)
            with col_pie:
                fig_pie, _ = plot_status_pie(df_display)
                st.pyplot(fig_pie)
                plt.close(fig_pie)
            with col_day:
                fig_day, _ = plot_day_of_week_bars(df_display)
                st.pyplot(fig_day)
                plt.close(fig_day)
            
            # Row 2: Total Attendance Bar Chart (Horizontal)
            fig_barh, _ = plot_total_attendance_barh(df_display)
            st.pyplot(fig_barh)
            plt.close(fig_barh)
            
            # Row 3: Simple 4-Interval Arrival Distribution
            fig_simple, _ = plot_simple_arrival_intervals(
                df_display,
                start_time=f"{self.class_start_hour:02d}:{self.class_start_minute:02d}:00",
                class_duration=50
            )
            st.markdown("##### Arrival Distribution: 4 Equal Intervals (50-Minute Class)")
            st.pyplot(fig_simple)
            plt.close(fig_simple)
            
            # Row 4: Weekly Trend Line Plot
            if "Date" in df.columns:
                fig_trend, _ = plot_weekly_trend(df, similarity_threshold=self.similarity_threshold)  # Use UNFILTERED data for trend
                st.markdown("##### Weekly Trend Analysis: Attendance Behavior vs Recognition Quality (All Data)")
                st.pyplot(fig_trend)
                plt.close(fig_trend)
            
            # Row 5: Morning/Evening Scatter Plot and Course Bar Plot
            col_scatter, col_course = st.columns(2)
            with col_scatter:
                fig_scatter, _ = plot_morning_evening_scatter(df_display)
                st.pyplot(fig_scatter)
                plt.close(fig_scatter)
            with col_course:
                fig_course, _ = plot_course_attendance_bar(df_display)
                st.pyplot(fig_course)
                plt.close(fig_course)
        # --- End: Plotting Section Integration ---

        # Weekly / monthly summaries as downloadable CSVs (as per original code)
        if "Date" in df.columns and not df.empty:
            df_dates = df.copy()
            df_dates["YearWeek"] = df_dates["Date"].dt.strftime("%Y-W%U")
            df_dates["YearMonth"] = df_dates["Date"].dt.strftime("%Y-%m")

            weekly = (
                df_dates.groupby(["Name", "YearWeek"])
                .size()
                .rename("Count")
                .reset_index()
                .sort_values(["YearWeek", "Name"])
            )
            monthly = (
                df_dates.groupby(["Name", "YearMonth"])
                .size()
                .rename("Count")
                .reset_index()
                .sort_values(["YearMonth", "Name"])
            )

            st.markdown("#### Export summaries")
            w_csv = weekly.to_csv(index=False)
            m_csv = monthly.to_csv(index=False)
            col_w, col_m = st.columns(2)
            col_w.download_button(
                "Download weekly summary CSV",
                data=w_csv,
                file_name="attendance_weekly_summary.csv",
                mime="text/csv",
            )
            col_m.download_button(
                "Download monthly summary CSV",
                data=m_csv,
                file_name="attendance_monthly_summary.csv",
                mime="text/csv",
            )

        # Download filtered base data as CSV (as per original code)
        csv_content = df_display.to_csv(index=False)
        st.download_button(
            "Download filtered records as CSV",
            data=csv_content,
            file_name="attendance_history.csv",
            mime="text/csv",
        )

        # Show recent rejection log entries (audit trail) (as per original code)
        if self.rejected_file.exists():
            try:
                rej_df = pd.read_csv(self.rejected_file, comment="#")
                if not rej_df.empty:
                    st.markdown("#### Recent rejected attempts")
                    st.dataframe(
                        rej_df.sort_values("CreatedAt", ascending=False).head(20),
                        use_container_width=True,
                    )
            except Exception:
                pass

    def render_manage_registration(self) -> None:
        """Manage master registration list for courses and sessions."""
        st.markdown('<div class="big-title">📋 Manage Registration</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Add students to master registration list for courses and sessions.</div>',
            unsafe_allow_html=True,
        )

        course = st.session_state.get("course", "")
        session = st.session_state.get("session", "")

        st.markdown("### Add Student to Master List")
        col1, col2 = st.columns(2)
        with col1:
            selected_course = st.selectbox(
                "Course",
                options=[
                    "CS231",
                    "CS221",
                    "ES205",
                    "CE221",
                    "CS202",
                    "AI201 LAB",
                    "CS221 LAB",
                    "CE221 LAB",
                ],
                index=0 if course else 0,
            )
        with col2:
            selected_session = st.selectbox(
                "Session",
                options=["Morning", "Evening"],
                index=0 if session == "Morning" else 1,
            )

        student_name = st.text_input("Student Name", placeholder="Enter student name")
        
        if st.button("➕ Add to Master List"):
            if not student_name.strip():
                st.error("Please enter a student name.")
            elif student_name.strip() not in self.known_names:
                st.error(f"Student '{student_name}' is not registered in the face recognition system. Please register their face first.")
            else:
                success = self.add_to_master_registration(selected_course, selected_session, student_name.strip())
                if success:
                    st.success(f"Added {student_name.strip()} to {selected_course} - {selected_session} master list.")
                else:
                    st.info(f"{student_name.strip()} is already in the master list for {selected_course} - {selected_session}.")

        st.markdown("### Current Master Registration")
        if self.master_registration_file.exists():
            try:
                df = pd.read_csv(self.master_registration_file, comment="#")
                if not df.empty and "Course" in df.columns and "Session" in df.columns and "StudentName" in df.columns:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Master registration file is empty.")
            except Exception as exc:
                st.error(f"Could not read master registration file: {exc}")
        else:
            st.info("No master registration file found. Add students to create one.")

    def render_end_session(self) -> None:
        """End a session and mark absent students."""
        st.markdown('<div class="big-title">🔚 End Session</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Mark absent students for a completed session.</div>',
            unsafe_allow_html=True,
        )

        course = st.session_state.get("course", "")
        session = st.session_state.get("session", "")
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        st.markdown("### Session Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_course = st.selectbox(
                "Course",
                options=[
                    "CS231",
                    "CS221",
                    "ES205",
                    "CE221",
                    "CS202",
                    "AI201 LAB",
                    "CS221 LAB",
                    "CE221 LAB",
                ],
                index=0 if course else 0,
            )
        with col2:
            selected_session = st.selectbox(
                "Session",
                options=["Morning", "Evening"],
                index=0 if session == "Morning" else 1,
            )
        with col3:
            selected_date = st.date_input("Date", value=now.date())

        date_str = selected_date.strftime("%Y-%m-%d")
        master_list = self.get_master_registration(selected_course, selected_session)
        
        if not master_list:
            st.warning(f"No students registered in master list for {selected_course} - {selected_session}.")
            return

        st.info(f"Master list for {selected_course} - {selected_session} on {date_str}: {len(master_list)} students")
        st.markdown("**Registered students:** " + ", ".join(sorted(master_list)))

        session_key = (selected_course, selected_session, date_str)
        appeared_students = self.session_appearances.get(session_key, set())
        failed_verifications = self.session_failed_verifications.get(session_key, {})

        if appeared_students:
            st.success(f"Students who appeared: {len(appeared_students)} - {', '.join(sorted(appeared_students))}")
        if failed_verifications:
            st.warning(f"Failed verifications: {len(failed_verifications)} - {', '.join(sorted(failed_verifications.keys()))}")

        if st.button("✅ Mark Absent Students", type="primary"):
            non_attendance, failed_verification = self.mark_absent_students(selected_course, selected_session, date_str)
            st.success(
                f"Session ended. Marked {non_attendance} students as 'Absent - Non-attendance' "
                f"and {failed_verification} students as 'Absent - Failed verification'."
            )
            # Clear session tracking
            if session_key in self.session_appearances:
                del self.session_appearances[session_key]
            if session_key in self.session_failed_verifications:
                del self.session_failed_verifications[session_key]

    # ---------- App runner ----------
    def run(self) -> None:
        st.set_page_config(page_title="Face Attendance (Streamlit)",  layout="centered")

        # Global minimal styling (as per original code)
        st.markdown(
            """
            <style>
            body {
                background: radial-gradient(circle at top left, #0f172a 0, #020617 55%, #020617 100%);
            }
            /* Reduce top padding */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2.5rem;
                max-width: 900px;
            }
            /* Simple card look */
            .card {
                padding: 1.5rem 1.75rem;
                border-radius: 0.9rem;
                background: linear-gradient(145deg, #020617, #020617);
                border: 1px solid #1e293b;
                box-shadow: 0 18px 40px -24px rgba(15, 23, 42, 0.9);
            }
            .big-title {
                font-size: 2rem;
                font-weight: 600;
                margin-bottom: 0.3rem;
                color: #e5e7eb;
            }
            .subtitle {
                color: #9ca3af;
                font-size: 0.95rem;
                margin-bottom: 1.25rem;
            }
            /* Softer buttons */
            .stButton>button {
                border-radius: 999px;
                border: 1px solid #1d4ed8;
                padding: 0.5rem 1.4rem;
                font-weight: 500;
                background: linear-gradient(135deg, #1d4ed8, #38bdf8);
                color: #f9fafb;
            }
            .stDownloadButton>button {
                border-radius: 999px;
                padding: 0.4rem 1.1rem;
                border: 1px solid #1f2937;
                background: #020617;
                color: #e5e7eb;
            }
            .sidebar .sidebar-content, [data-testid="stSidebar"] {
                background: #020617;
            }
            .stSelectbox label, .stTextInput label {
                color: #e5e7eb !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.sidebar.markdown("### Face Attendance")
        st.sidebar.caption("Minimal Streamlit demo")

        # Session / course metadata controls (as per original code)
        st.sidebar.markdown("#### Session settings")
        course_options = [
            "CS231",
            "CS221",
            "ES205",
            "CE221",
            "CS202",
            "AI201 LAB",
            "CS221 LAB",
            "CE221 LAB",
        ]
        default_course = st.session_state.get("course", course_options[0])
        if default_course not in course_options:
            default_course = course_options[0]
        st.session_state["course"] = st.sidebar.selectbox(
            "Course",
            options=course_options,
            index=course_options.index(default_course),
        )
        st.session_state["session"] = st.sidebar.selectbox(
            "Session",
            options=["Morning", "Evening"],
            index=0 if st.session_state.get("session", "Morning") == "Morning" else 1,
        )

        page = st.sidebar.radio(
            "Navigate",
            options=["Home", "Register", "Mark Attendance", "View Attendance", "Manage Registration", "End Session"],
            index=0,
        )

        if page == "Home":
            self.render_home()
        elif page == "Register":
            self.render_register()
        elif page == "Mark Attendance":
            self.render_attendance()
        elif page == "Manage Registration":
            self.render_manage_registration()
        elif page == "End Session":
            self.render_end_session()
        else:
            self.render_history()


if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.run()

