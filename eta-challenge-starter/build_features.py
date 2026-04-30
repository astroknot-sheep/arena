#!/usr/bin/env python
"""
build_features.py
Run ONCE after data/download_data.py to build lookup tables.
Saves data/lookups.pkl — used by train.py and predict.py.
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

DATA_DIR = Path("data")


def build_zone_pair_lookup(train: pd.DataFrame) -> dict:
    """Core signal: mean duration per (pickup_zone, dropoff_zone) pair."""
    grp = (
        train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    pair_mean = {}
    pair_median = {}
    pair_count = {}
    for row in grp.itertuples():
        key = (row.pickup_zone, row.dropoff_zone)
        pair_mean[key] = row.mean
        pair_median[key] = row.median
        pair_count[key] = row.count

    global_mean = float(train["duration_seconds"].mean())
    return {
        "pair_mean": pair_mean,
        "pair_median": pair_median,
        "pair_count": pair_count,
        "global_mean": global_mean,
    }


def build_zone_hour_lookup(train: pd.DataFrame) -> dict:
    """(pickup_zone, dropoff_zone, hour_bin) → mean duration.
    hour_bin: 0=midnight-4am, 1=4-8am, 2=8am-noon, 3=noon-4pm, 4=4-8pm, 5=8pm-midnight"""
    ts = pd.to_datetime(train["requested_at"])
    df = train.copy()
    df["hour_bin"] = (ts.dt.hour // 4).astype("int8")

    grp = (
        df.groupby(["pickup_zone", "dropoff_zone", "hour_bin"])["duration_seconds"]
        .mean()
        .reset_index()
    )
    lookup = {}
    for row in grp.itertuples():
        lookup[(row.pickup_zone, row.dropoff_zone, row.hour_bin)] = row.duration_seconds
    return lookup


def build_zone_dow_hour_lookup(train: pd.DataFrame) -> dict:
    """Fine-grained: (pu_zone, do_zone, is_weekend, hour) → mean duration."""
    ts = pd.to_datetime(train["requested_at"])
    df = train.copy()
    df["hour"] = ts.dt.hour.astype("int8")
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")

    grp = (
        df.groupby(["pickup_zone", "dropoff_zone", "is_weekend", "hour"])
        ["duration_seconds"].mean().reset_index()
    )
    lookup = {}
    for row in grp.itertuples():
        lookup[(row.pickup_zone, row.dropoff_zone, row.is_weekend, row.hour)] = row.duration_seconds
    return lookup


def build_zone_centroids() -> pd.DataFrame:
    """Compute lat/lon centroid for each NYC taxi zone from shapefile."""
    gdf = gpd.read_file("data/geo/taxi_zones.shp").to_crs(epsg=4326)
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x
    centroids = gdf[["LocationID", "lat", "lon"]].rename(
        columns={"LocationID": "zone_id"}
    )
    # Fill any zones missing from shapefile with NYC center
    all_zones = pd.DataFrame({"zone_id": range(1, 266)})
    centroids = all_zones.merge(centroids, on="zone_id", how="left")
    centroids["lat"] = centroids["lat"].fillna(40.7128)
    centroids["lon"] = centroids["lon"].fillna(-74.0060)
    return centroids


if __name__ == "__main__":
    print("Loading full train data (~37M rows, may take 30s)...")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    print(f"  {len(train):,} rows")

    print("Building zone-pair lookup (most important feature)...")
    pair_lookup = build_zone_pair_lookup(train)
    print(f"  {len(pair_lookup['pair_mean']):,} zone pairs")

    print("Building zone-hour-bin lookup...")
    zh_lookup = build_zone_hour_lookup(train)

    print("Building zone-weekend-hour lookup...")
    zdh_lookup = build_zone_dow_hour_lookup(train)

    print("Building zone centroids...")
    centroids = build_zone_centroids()

    lookups = {
        "pair_lookup": pair_lookup,
        "zh_lookup": zh_lookup,
        "zdh_lookup": zdh_lookup,
        "centroids": centroids,
    }
    with open(DATA_DIR / "lookups.pkl", "wb") as f:
        pickle.dump(lookups, f)
    print("✅ Saved data/lookups.pkl")