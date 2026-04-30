"""predict.py — Gobblecube grader interface.
The grader calls predict() once per held-out request.
Signature is FIXED. Internals are yours.
Constraint: inference ≤ 200ms on CPU.
"""
from __future__ import annotations
import pickle
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import numpy as np
import pandas as pd

_MODEL_PATH = Path(__file__).parent / "model.pkl"

# ── Load once at import time (not per call) ───────────────────────────────
with open(_MODEL_PATH, "rb") as _f:
    _P = pickle.load(_f)

_MODEL      = _P["model"]
_PAIR       = _P["pair_lookup"]
_ZH         = _P["zh_lookup"]
_ZDH        = _P["zdh_lookup"]
_GMEAN      = _P["global_mean"]
_CENT       = _P["centroids"].set_index("zone_id")


def _hav(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def predict(request: dict) -> float:
    """Predict trip duration in seconds.
    Input: {"pickup_zone": int, "dropoff_zone": int,
            "requested_at": str (ISO 8601), "passenger_count": int}
    """
    ts  = datetime.fromisoformat(request["requested_at"])
    pu  = int(request["pickup_zone"])
    do  = int(request["dropoff_zone"])
    pc  = int(request["passenger_count"])

    h   = ts.hour
    dow = ts.weekday()
    mon = ts.month
    iw  = int(dow >= 5)
    hb  = h // 4

    pm  = _PAIR.get((pu, do), _GMEAN)
    zh  = _ZH.get((pu, do, hb), _GMEAN)
    zdh = _ZDH.get((pu, do, iw, h), _GMEAN)

    pz  = min(max(pu, 1), 265)
    dz  = min(max(do, 1), 265)
    pc_row = _CENT.loc[pz]
    dc_row = _CENT.loc[dz]
    dk  = _hav(pc_row["lat"], pc_row["lon"], dc_row["lat"], dc_row["lon"])

    x = pd.DataFrame([{
        "pickup_zone":     pu,
        "dropoff_zone":    do,
        "hour":            h,
        "dow":             dow,
        "month":           mon,
        "is_weekend":      iw,
        "hour_bin":        hb,
        "hour_sin":        np.sin(2*np.pi*h/24),
        "hour_cos":        np.cos(2*np.pi*h/24),
        "dow_sin":         np.sin(2*np.pi*dow/7),
        "dow_cos":         np.cos(2*np.pi*dow/7),
        "month_sin":       np.sin(2*np.pi*mon/12),
        "month_cos":       np.cos(2*np.pi*mon/12),
        "is_morning_rush": int(7<=h<=9 and not iw),
        "is_evening_rush": int(16<=h<=19 and not iw),
        "is_night":        int(h>=22 or h<=5),
        "pair_mean":       pm,
        "zh_mean":         zh,
        "zdh_mean":        zdh,
        "dist_km":         dk,
        "pu_lat":          pc_row["lat"],
        "pu_lon":          pc_row["lon"],
        "do_lat":          dc_row["lat"],
        "do_lon":          dc_row["lon"],
        "is_same_zone":    int(pu == do),
        "passenger_count": pc,
    }])

    return float(_MODEL.predict(x)[0])