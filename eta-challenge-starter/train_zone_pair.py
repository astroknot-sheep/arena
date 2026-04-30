#!/usr/bin/env python
"""Train zone-pair average model with smart fallbacks for sparse pairs."""
import pandas as pd
import pickle
from pathlib import Path

DATA_DIR = Path("data")
MODEL_PATH = Path("zone_pair_model.pkl")

print("Loading training data...")
# Use sample_1M for fast iteration; switch to train.parquet for final model
train = pd.read_parquet(DATA_DIR / "sample_1M.parquet")

print("Computing zone-pair statistics...")
zone_pair_stats = train.groupby(['pickup_zone', 'dropoff_zone'])['duration_seconds'].agg([
    'mean',      # Average duration for this pair
    'std',       # Uncertainty estimate (optional)
    'count'      # How many trips observed
]).reset_index()
zone_pair_stats.columns = ['pickup_zone', 'dropoff_zone', 'mean_duration', 'std_duration', 'count']

# Also compute pickup-zone-only averages (fallback level 2)
pickup_stats = train.groupby('pickup_zone')['duration_seconds'].agg(['mean', 'count']).reset_index()
pickup_stats.columns = ['pickup_zone', 'picpickup_stats.columns = ['pickup_obal mean (fallback level 3)
global_mean = train['duration_seconds'].mean()

print(f"Zone pairs: {lprint(f"Zone pairs):,}")
print(f"Pprint(f"Pprint(f"Pps: {(zoprint(f"Pprint(f"Pprin >print(f"Pprint(f"Pprint(f"lobal mean: {gloprint(f"Pprint(f"Pprint(f"Ppel bunprint(f"Pprint(f"Pprint(f"Pps: {(zoprint(f"Pprint(f"Pprin >print(f"Pprint(ne', 'dropoff_zone']),
    'pickup_stats': pickup_stats.set_index('pickup_zone'),
    'global_mean': global_mean,
    'min_pair_count': 10,      # Only trust pairs with >=10 samples
    'min_pickup_count': 50,    # Only trust pickup-zone avg w    'min_pickup_count': 50,    # Only trust pi) as f:
    pickle.dump(model, f)
print(f"\n✅ Saved model to {MODEL_PATH}")
