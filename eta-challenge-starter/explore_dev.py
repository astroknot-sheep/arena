import pandas as pd
import numpy as np

# Load dev (small, fast to explore)
dev = pd.read_parquet("data/dev.parquet")
print(dev.shape)         # rows, cols
print(dev.dtypes)
print(dev["duration_seconds"].describe())
# Note the mean (~870s ≈ 14.5 min) and p95 (~2400s)

# How many zone pairs? How much historical data per pair?
pairs = dev.groupby(["pickup_zone", "dropoff_zone"]).size()
print(f"{len(pairs):,} unique zone pairs")
print(f"Mean trips per pair: {pairs.mean():.0f}")
print(f"Pairs with <10 trips: {(pairs < 10).sum()}")
# Sparse pairs need fallback to zone-level or global mean

# Duration by hour of day — huge signal
ts = pd.to_datetime(dev["requested_at"])
dev["hour"] = ts.dt.hour
hour_mae = dev.groupby("hour")["duration_seconds"].mean()
print(hour_mae.sort_values(ascending=False).head(5))
# 5pm–7pm should be highest (rush hour)