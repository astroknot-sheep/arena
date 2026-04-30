"""predict.py — Gobblecube grader interface.
The grader calls predict() once per held-out request.
Constraint: inference ≤ 200ms per request.
"""
from __future__ import annotations
import pickle
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)

_MODEL_PATH = Path(__file__).parent / "model.pkl"
with open(_MODEL_PATH, "rb") as _f: _P = pickle.load(_f)

_MODEL      = _P["model"]
_PAIR_DATA  = _P["pair_lookup"]
_PAIR_MEAN  = _PAIR_DATA["pair_mean"]
_PAIR_COUNT = _PAIR_DATA["pair_count"]
_ZH         = _P["zh_lookup"]
_ZDH        = _P["zdh_lookup"]
_ZMH        = _P["zmh_lookup"]
_GMEAN      = _P["global_mean"]
_CENT       = _P["centroids"].set_index("zone_id")
_FEAT_NAMES = _P["feat_names"]

def _hav(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def predict(request: dict) -> float:
    """Predict trip duration in seconds."""
    ts  = datetime.fromisoformat(request["requested_at"])
    pu  = int(request["pickup_zone"])
    do  = int(request["dropoff_zone"])
    pc  = int(request["passenger_count"])
    h   = ts.hour
    dow = ts.weekday()
    mon = ts.month
    iw  = int(dow >= 5)
    hb  = h // 4

    pm  = _PAIR_MEAN.get((pu, do), _GMEAN)
    p_cnt = _PAIR_COUNT.get((pu, do), 0)
    zh  = _ZH.get((pu, do, hb), _GMEAN)
    zdh = _ZDH.get((pu, do, iw, h), _GMEAN)
    zmh = _ZMH.get((pu, do, mon, hb), _GMEAN)

    pz, dz = min(max(pu, 1), 265), min(max(do, 1), 265)
    pc_row, dc_row = _CENT.loc[pz], _CENT.loc[dz]
    dk = _hav(pc_row["lat"], pc_row["lon"], dc_row["lat"], dc_row["lon"])

    # Build feature array in EXACT column order from train.py
    x = np.array([[
        pu, do, h, dow, mon, iw, hb,
        np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
        np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
        np.sin(2*np.pi*mon/12), np.cos(2*np.pi*mon/12),
        int(7<=h<=9 and not iw), int(16<=h<=19 and not iw), int(h>=22 or h<=5),
        pm, p_cnt, zh, zdh, zmh,
        dk, pc_row["lat"], pc_row["lon"], dc_row["lat"], dc_row["lon"],
        int(pu == do), pc
    ]], dtype=np.float64)

    return float(_MODEL.predict(x)[0])