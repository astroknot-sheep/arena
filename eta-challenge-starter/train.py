#!/usr/bin/env python
"""
train.py — LightGBM with lookup + geo + temporal features.
Run: python train.py
Produces: model.pkl (loaded by predict.py at inference)
"""
from __future__ import annotations
import pickle, time
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_DIR   = Path("data")
MODEL_PATH = Path("model.pkl")
USE_SAMPLE = False

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def engineer_features(df, centroids, pair_lookup, zh_lookup, zdh_lookup, zmh_lookup, global_mean):
    ts       = pd.to_datetime(df["requested_at"])
    hour     = ts.dt.hour.values.astype("int8")
    dow      = ts.dt.dayofweek.values.astype("int8")
    month    = ts.dt.month.values.astype("int8")
    hour_bin = (hour // 4).astype("int8")
    is_wknd  = (dow >= 5).astype("int8")
    pu = df["pickup_zone"].values
    do = df["dropoff_zone"].values

    pair_mean = pair_lookup["pair_mean"]
    pair_count = pair_lookup["pair_count"]

    pair_mean_feat = np.array([pair_mean.get((p, d), global_mean) for p, d in zip(pu, do)])
    pair_count_feat = np.array([pair_count.get((p, d), 0) for p, d in zip(pu, do)])
    zh_feat = np.array([zh_lookup.get((p, d, hb), global_mean) for p, d, hb in zip(pu, do, hour_bin)])
    zdh_feat = np.array([zdh_lookup.get((p, d, iw, h), global_mean) for p, d, iw, h in zip(pu, do, is_wknd, hour)])
    zmh_feat = np.array([zmh_lookup.get((p, d, m, hb), global_mean) for p, d, m, hb in zip(pu, do, month, hour_bin)])

    cent = centroids.set_index("zone_id")
    pu_lat = cent.loc[np.clip(pu, 1, 265), "lat"].values
    pu_lon = cent.loc[np.clip(pu, 1, 265), "lon"].values
    do_lat = cent.loc[np.clip(do, 1, 265), "lat"].values
    do_lon = cent.loc[np.clip(do, 1, 265), "lon"].values
    dist_km = np.array([haversine_km(a, b, c, d) for a, b, c, d in zip(pu_lat, pu_lon, do_lat, do_lon)])

    return pd.DataFrame({
        "pickup_zone":     pu,
        "dropoff_zone":    do,
        "hour":            hour,
        "dow":             dow,
        "month":           month,
        "is_weekend":      is_wknd,
        "hour_bin":        hour_bin,
        "hour_sin":        np.sin(2*np.pi*hour/24),
        "hour_cos":        np.cos(2*np.pi*hour/24),
        "dow_sin":         np.sin(2*np.pi*dow/7),
        "dow_cos":         np.cos(2*np.pi*dow/7),
        "month_sin":       np.sin(2*np.pi*month/12),
        "month_cos":       np.cos(2*np.pi*month/12),
        "is_morning_rush": ((hour>=7)&(hour<=9)&(is_wknd==0)).astype("int8"),
        "is_evening_rush": ((hour>=16)&(hour<=19)&(is_wknd==0)).astype("int8"),
        "is_night":        ((hour>=22)|(hour<=5)).astype("int8"),
        "pair_mean":       pair_mean_feat,
        "pair_count":      pair_count_feat,
        "zh_mean":         zh_feat,
        "zdh_mean":        zdh_feat,
        "zmh_mean":        zmh_feat,
        "dist_km":         dist_km,
        "pu_lat":          pu_lat,
        "pu_lon":          pu_lon,
        "do_lat":          do_lat,
        "do_lon":          do_lon,
        "is_same_zone":    (pu == do).astype("int8"),
        "passenger_count": df["passenger_count"].astype("int8").values,
    })

def main():
    with open(DATA_DIR / "lookups.pkl", "rb") as f:
        lookups = pickle.load(f)

    pair_lookup  = lookups["pair_lookup"]
    zh_lookup    = lookups["zh_lookup"]
    zdh_lookup   = lookups["zdh_lookup"]
    zmh_lookup   = lookups["zmh_lookup"]
    global_mean  = lookups["pair_lookup"]["global_mean"]
    centroids    = lookups["centroids"]

    train_file = DATA_DIR / ("sample_1M.parquet" if USE_SAMPLE else "train.parquet")
    print(f"Loading {train_file.name}...")
    train = pd.read_parquet(train_file)
    dev   = pd.read_parquet(DATA_DIR / "dev.parquet")
    print(f"  train={len(train):,}  dev={len(dev):,}")

    fe_args = (centroids, pair_lookup, zh_lookup, zdh_lookup, zmh_lookup, global_mean)

    print("Engineering features...")
    X_train = engineer_features(train, *fe_args)
    y_train = train["duration_seconds"].values
    X_dev   = engineer_features(dev, *fe_args)
    y_dev   = dev["duration_seconds"].values

    print("Training LightGBM (objective=mae)...")
    model = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.01, num_leaves=255, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        objective="mae", metric="mae", n_jobs=-1, random_state=42, verbose=-1,
    )
    t0 = time.time()
    model.fit(
        X_train, y_train, eval_set=[(X_dev, y_dev)],
        categorical_feature=["pickup_zone", "dropoff_zone"],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True), lgb.log_evaluation(period=100)],
    )
    print(f"  Trained in {time.time()-t0:.0f}s")

    preds = model.predict(X_dev)
    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\n✅ Dev MAE: {mae:.1f}s")

    payload = {
        "model":       model,
        "pair_lookup": pair_lookup,
        "zh_lookup":   zh_lookup,
        "zdh_lookup":  zdh_lookup,
        "zmh_lookup":  zmh_lookup,
        "global_mean": global_mean,
        "centroids":   centroids,
        "feat_names":  list(X_train.columns),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved {MODEL_PATH}")

if __name__ == "__main__":
    main()