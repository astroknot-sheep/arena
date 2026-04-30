# Your Submission: Writeup Template

*Replace this file's contents when you submit. A good writeup is ~1 page.
We read every one.*

---

## Your final score

Dev MAE: **265.7 s**

---

## Your approach, in one paragraph

Built a LightGBM regressor optimizing MAE directly (`objective="mae"`) on 28 features. The core insight was that the baseline GBT treats zone IDs as continuous integers and misses the dominant signal: historical average trip duration per pickup→dropoff zone pair. I constructed four lookup tables from the full 37M training trips: (1) zone-pair mean & count, (2) zone-pair × 4-hour-bin mean, (3) zone-pair × weekend/weekday × hour mean, and (4) zone-pair × month × hour-bin mean. These serve as primary features, while LightGBM learns marginal adjustments from haversine distance (using EPSG:2263-projected centroids for geometric accuracy), cyclical time encodings (sin/cos for hour/dow/month), rush-hour flags, and raw zone coordinates. Full training on 37M rows with early stopping took ~11 minutes on Apple M4 Air.

## What you tried that didn't work

- **Aggressive initial hyperparameters** (`learning_rate=0.03`, `num_leaves=255`): Early stopping fired prematurely at round 91, yielding a worse MAE (269.2s). Switching to finer steps (`lr=0.008`, `num_leaves=350`, `min_child_samples=15`) pushed early stopping to round 326 and dropped MAE to 265.7s.
- **Passenger count as a predictive signal**: Consistently showed near-zero feature importance across all iterations. NYC trip duration is driven by route, time, and traffic, not passenger volume.
- **Raw numpy array inference in `predict.py`**: While slightly faster, it triggered repetitive sklearn warnings about missing feature names. Reverted to a lightweight pandas DataFrame with explicit column ordering to maintain clean logs without sacrificing the <200ms constraint.

## Where AI tooling sped you up most

AI (primarily Claude/Cursor) was instrumental in: (1) **Architecture design** — immediately identifying that the baseline GBT was missing the zone-pair historical mean signal before I ran any experiments. (2) **Vectorized feature engineering** — generating efficient numpy/pandas code for haversine distance, cyclical encodings, and multi-key lookup fallbacks. (3) **Debugging** — tracing the `AttributeError: 'dict' object has no attribute 'predict'` payload mismatch and resolving the geopandas CRS projection warning by projecting to EPSG:2263 before centroid calculation. It fell short on hyperparameter tuning, where actual `grade.py` validation was irreplaceable.

## Next experiments

1. **OSRM road-network distance**: Haversine underestimates NYC travel distance due to one-way grids, bridges, and tunnels. OSRM routing distance would provide a stronger physical prior.
2. **NOAA weather join**: Hourly temperature, precipitation, and wind data from JFK/LGA. Rain/snow correlates strongly with ~15% longer trip durations.
3. **Neural tabular models**: Implementing TabNet or a deep MLP with learned zone embeddings on a free Kaggle GPU. Neural architectures typically outperform GBTs by 15–25s MAE on high-cardinality categorical features given sufficient compute.

## How to reproduce

```bash
python data/download_data.py     # one-time, ~500MB
python build_features.py         # builds data/lookups.pkl (~30s)
# Ensure USE_SAMPLE=False in train.py for full training
python train.py                  # ~11 min on M4 Air, writes model.pkl
python grade.py                  # verify MAE
python -m pytest tests/ -v       # smoke tests
docker build -t my-eta .         # package submission

_Total time spent on this challenge: 6 hours._
