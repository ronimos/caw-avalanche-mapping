# caw-avalanche-mapping

Avalanche forecasting support for data-sparse Central Asia. The tool links NWP-driven [SNOWPACK](https://models.slf.ch/p/snowpack/) simulations to opportunistically-collected avalanche observations, trains per-station and pooled classifiers, and produces interactive stability plots and a forecast map. Developed for the ISSW 2026 paper *"Avalanche Forecasting in Data-Sparse Central Asia: A Weak-Supervision Framework."*

## Overview

The pipeline:

1. Load avalanche observations from `data/observations/avalanches_<SEASON>.csv` (all seasons combined).
2. Match each observation to the nearest SNOWPACK station of the same aspect, using `snowpack_stations_locations.csv`.
3. Parse `.pro` (layer stratigraphy, Sn38) and `.smet` (meteorology) files into daily features.
4. Train classifiers on the training season and evaluate them on the held-out test season:
   - **per-station** logistic regression
   - **regional** pooled logistic regression (CV-tuned regularization)
   - **blended** confidence-weighted combination
   - **hierarchical Bayesian** partial-pooling model (the headline model; also yields per-station uncertainty)
5. Evaluate with temporal leave-one-out CV, event-level precision-recall, and an operational false-alarm analysis.
6. Generate per-station interactive stability plots (Plotly) and an interactive Folium forecast map.

See `docs/methods_and_results.md` for the full methods and results writeup.

## Project Structure

```
caw-avalanche-mapping/
├── src/
│   ├── main.py            # Entry point — orchestrates the full pipeline
│   ├── snowpack_io.py     # .pro / .smet parsing, aspect-aware station matching
│   ├── features.py        # daily feature engineering (reduced 5-feature set)
│   ├── classifier.py      # per-station / regional / blended logistic regression + evaluation
│   ├── hierarchical.py    # hierarchical Bayesian (PyMC) partial-pooling model
│   └── visualization.py   # Plotly stability plots + Folium map
├── scripts/
│   └── sync_pro_files.py  # download season .pro/.smet from the remote server
├── data/
│   ├── simulations/<SEASON>/*.pro,*.smet   # SNOWPACK outputs, one set per season
│   ├── observations/avalanches_<SEASON>.csv # field/social-media observations per season
│   └── snowpack_stations_locations.csv      # station id → lat/lon/elevation
├── docs/
│   ├── methods_and_results.md      # paper methods & results
│   ├── ISSW2026_paper_outline.md   # paper outline
│   └── map_design_discussion.md    # open map-design questions
├── output/                # generated artifacts (git-ignored)
├── models/                # saved classifiers (git-ignored)
├── pyproject.toml
└── uv.lock
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `joblib`, `pymc`, `arviz`, `plotly`, `folium`, `matplotlib`.

## Running

```bash
uv run python src/main.py              # full pipeline (trains/loads, evaluates, plots)
uv run python src/main.py --retrain    # force retraining of all classifiers
uv run python src/main.py --no-hierarchical   # skip the PyMC model / MCMC sampling
uv run python src/main.py --forecast-date 2026-01-23
```

Outputs are written to `output/`. Open `output/avalanche_map.html` and click any station marker to open its stability plot.

### Adding a new season

1. Drop `.pro`/`.smet` files into `data/simulations/<new-season>/` (or run `scripts/sync_pro_files.py`).
2. Add `data/observations/avalanches_<new-season>.csv`.
3. Set `SEASON` / `TRAIN_SEASON` in `src/main.py` and rerun with `--retrain`.

### Downloading simulation files

`scripts/sync_pro_files.py` downloads only the (station, aspect) `.pro`/`.smet` pairs needed for the observations, via rsync over SSH. Configure `.env` (copy from `.env.example`):

```
SSH_PORT=<port>
SSH_KEY=~/.ssh/your_key
```

```bash
uv run python scripts/sync_pro_files.py --dry-run   # preview
uv run python scripts/sync_pro_files.py
```

## Data Formats

### Observation CSV (`avalanches_<SEASON>.csv`)

Columns: `Place`, `Lat`, `Long`, `Slope`, `Aspect` (N/E/S/W), `Elevation (M)`, `Date`, `Size`, `Remarks`.

### SNOWPACK `.pro` files

Station outputs named `{station_id}{aspect_suffix}_res.pro`, where the aspect suffix is `1`=N, `2`=E, `3`=S, `4`=W, and no suffix = flat. The parser reads three record codes:

| Code | Content |
|------|---------|
| `0500` | Timestep date |
| `0501` | Layer heights from ground (cm), bottom→top |
| `0532` | Sn38 (natural stability index) per layer, same order as `0501` |

Layers shallower than 20 cm burial depth are excluded.

## Model Features

The classifiers use a reduced five-feature daily set (selected to limit overfitting given sparse events):

| Feature | Role |
|---|---|
| `HS` | Total snow depth — size / valley-reach proxy |
| `HN24` | 24-hour new snow — loading trigger |
| `TA_max` | Daily max air temperature — wet vs dry mechanism |
| `sn38_min` | Whole-profile minimum stability index — weakest-layer strength |
| `depth_lower_wl` | Burial depth of the weakest lower-zone layer |

## Output Files (git-ignored under `output/`)

| File | Contents |
|---|---|
| `avalanche_map.html` | Folium forecast map; click markers for stability plots. Includes a date slider — scrub through the season to watch each station's probability evolve |
| `assets/<SEASON>/stability_*.html` | Per-station Plotly stability charts |
| `evaluation_{regional,blended,hierarchical}.csv` | Aggregate model metrics |
| `evaluation_loo.csv` | Temporal leave-one-out per-fold results |
| `evaluation_operational.csv` | Per-station recall-maximizing threshold + false-alarm rate |
| `hierarchical_uncertainty.csv` | Per-station mean probability + posterior std |
| `pr_curves.png` | Event-level precision-recall curves (all models) |
| `loo_performance_by_events.png` | Learning curve: performance vs. training-event count |
| `operational_thresholds.png` | False-alarm rate per station, by tier |

## Sn38 Interpretation

| Sn38 | Stability |
|------|-----------|
| < 1.0 | Unstable |
| 1.0 – 1.5 | Very low |
| 1.5 – 2.5 | Low |
| 2.5 – 3.5 | Moderate |
| > 3.5 | Stable |
