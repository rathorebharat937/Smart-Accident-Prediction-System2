import pandas as pd
import numpy as np
from datetime import datetime

def detect_spatial_hotspots(df, grid_size=0.05, density_threshold=5):
    """
    Detect spatial hotspots using grid-based density estimation.
    Parameters:
        df (DataFrame): Must include 'Start_Lat' and 'Start_Lng' columns.
        grid_size (float): Size of grid cell in degrees.
        density_threshold (int): Minimum count to classify as hotspot.
    Returns:
        hotspots (DataFrame), df (DataFrame)
    """
    df = df.copy()
    if "Start_Lat" not in df.columns or "Start_Lng" not in df.columns:
        raise KeyError("Dataset must include 'Start_Lat' and 'Start_Lng' columns.")

    # Create spatial bins
    df["Lat_bin"] = (df["Start_Lat"] / grid_size).round(0)
    df["Lng_bin"] = (df["Start_Lng"] / grid_size).round(0)
    df["Grid_ID"] = df["Lat_bin"].astype(str) + "_" + df["Lng_bin"].astype(str)

    # Count per grid
    grid_counts = df.groupby("Grid_ID").size().reset_index(name="Count")
    grid_counts["Hotspot"] = grid_counts["Count"] >= density_threshold

    # Merge back
    df = df.merge(grid_counts, on="Grid_ID", how="left")
    df["Hotspot"] = df["Hotspot"].fillna(False)

    # Create hotspot summary (average location of grids)
    hotspots = (
        df[df["Hotspot"]]
        .groupby(["Grid_ID", "Lat_bin", "Lng_bin"])
        .agg({"Start_Lat": "mean", "Start_Lng": "mean", "Count": "first"})
        .reset_index(drop=True)
    )

    return hotspots, df


def analyze_temporal_evolution(df, time_col="Start_Time", grid_size=0.05, density_threshold=5, time_unit="M"):
    """
    Analyze temporal evolution of hotspots — new, persistent, and disappeared.
    """
    df = df.copy()
    if "Start_Lat" not in df.columns or "Start_Lng" not in df.columns:
        raise KeyError("Dataset must include 'Start_Lat' and 'Start_Lng' columns.")
    if time_col not in df.columns:
        raise KeyError(f"Dataset must include '{time_col}' column.")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # Create time period & spatial bins
    df["Period"] = df[time_col].dt.to_period(time_unit)
    df["Lat_bin"] = (df["Start_Lat"] / grid_size).round(0)
    df["Lng_bin"] = (df["Start_Lng"] / grid_size).round(0)
    df["Grid_ID"] = df["Lat_bin"].astype(str) + "_" + df["Lng_bin"].astype(str)

    periods = sorted(df["Period"].unique())
    evolution_records = []
    prev_hotspots = set()

    for p in periods:
        subset = df[df["Period"] == p]
        grid_counts = subset.groupby("Grid_ID").size().reset_index(name="Count")
        curr_hotspots = set(grid_counts[grid_counts["Count"] >= density_threshold]["Grid_ID"])

        new = curr_hotspots - prev_hotspots
        disappeared = prev_hotspots - curr_hotspots
        persistent = curr_hotspots & prev_hotspots

        for g in curr_hotspots:
            evolution_records.append({"Period": str(p), "Grid_ID": g, "Status": "Persistent" if g in persistent else "New"})
        for g in disappeared:
            evolution_records.append({"Period": str(p), "Grid_ID": g, "Status": "Disappeared"})

        prev_hotspots = curr_hotspots

    evolution_df = pd.DataFrame(evolution_records)

    # Attach average lat/lng for each grid
    if not evolution_df.empty:
        coord_map = df.groupby("Grid_ID")[["Start_Lat", "Start_Lng"]].mean().reset_index()
        evolution_df = evolution_df.merge(coord_map, on="Grid_ID", how="left")

    return evolution_df
