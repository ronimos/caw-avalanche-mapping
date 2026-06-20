# caw-avalanche-mapping

Avalanche forecasting support for data-sparse Central Asia. The tool links NWP-driven [SNOWPACK](https://models.slf.ch/p/snowpack/) simulations to opportunistically-collected avalanche observations, trains per-station and pooled classifiers, and produces interactive stability plots and a forecast map. Developed for the ISSW 2026 paper *"Avalanche Forecasting in Data-Sparse Central Asia: A Weak-Supervision Framework."*

Live map: served via GitHub Pages from `index.html` at the repository root.

## Overview

The pipeline:

1. Load avalanche observations from `data/observations/avalanches_<SEASON>.csv` (all seasons combined).
2. Match each observation to the nearest SNOWPACK station of the same aspect, using `snowpack_stations_locations.csv`.
3. Parse `.pro` (layer stratigraphy, Sn38) and `.smet` (meteorology) files into daily features.
4. Train classifiers on the training season and evaluate on the held-out test season:
   - **per-station** logistic regression
   - **regional** pooled logistic regression (CV-tuned regularization)
   - **blended** confidence-weighted combination
   - **hierarchical Bayesian** partial-pooling model (also yields per-station uncertainty)
5. Evaluate with temporal leave-one-out CV, event-level precision-recall, and an operational false-alarm analysis.
6. Fit an **operational blended model** on all available data (both seasons combined) for the forecast map.
7. Generate per-station interactive stability plots (Plotly) and an interactive Folium forecast map (`index.html`).

See `docs/methods_and_results.md` for the full methods and results writeup.  
See `docs/map_user_guide.md` for a complete guide to the interactive map.

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
│   ├── observations/avalanches_<SEASON>.csv # field observations per season
│   └── snowpack_stations_locations.csv      # station id → lat/lon/elevation
├── docs/
│   ├── map_user_guide.md           # interactive map feature guide
│   ├── methods_and_results.md      # paper methods & results
│   ├── ISSW2026_paper_outline.md   # paper outline
│   └── map_design_discussion.md    # map design notes
├── assets/<SEASON>/
│   └── stability_*.html   # per-station Plotly stability charts (served by GitHub Pages)
├── index.html             # Folium forecast map (GitHub Pages entry point)
├── output/                # evaluation CSVs and figures (git-ignored)
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
uv run python src/main.py                          # full pipeline
uv run python src/main.py --retrain                # force retraining of all classifiers
uv run python src/main.py --no-hierarchical        # skip PyMC / MCMC sampling
uv run python src/main.py --forecast-date 2026-01-23
```

Outputs: `index.html` (map) and `assets/<SEASON>/stability_*.html` (stability plots) are written directly to the repository root so GitHub Pages can serve them. Evaluation CSVs and figures go to `output/`.

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

## Map Features

The forecast map (`index.html`) shows simulation station circles and field observation triangles, both coloured by the six-level probability scale below. See `docs/map_user_guide.md` for full details.

**Probability colour scale:**

| Colour | Probability |
|--------|-------------|
| Gray | No classifier output |
| Green | < 33 % |
| Gold | 33 – 50 % |
| Orange | 50 – 65 % |
| Red | 65 – 80 % |
| Dark red | ≥ 80 % |

**Key interactions:**
- **Click** any marker → opens the station's interactive stability chart (probability panel on top, 2025-26 season only).
- **Date Navigator** (bottom-centre) → filters observations by date and recolours all markers by the maximum blended probability within ±3 days of the selected date.
- **Time Series Player** (bottom) → scrubs through daily blended probability for each station across the full 2025-26 season.
- **Aspect filter** (bottom-right) → shows one aspect at a time, or "All" (maximum probability per unique location).
- **Regional model overlay** (layer toggle) → shows probabilities from the pooled model for all stations.

**Station confidence tiers** (from operational false-alarm analysis):

| Tier | Fill opacity | False alarm rate | Stations |
|------|-------------|-----------------|----------|
| Ready | 0.82 (solid) | ≤ 10 % | 160942 E, 153203 S, 180343 S, 164801 N |
| Marginal | 0.42 (faded) | 10 – 25 % | 272401 N, 176522 E, 250224 S |
| Not ready | 0.12 (ghost) | > 25 % | All others |

## Output Files

| File | Contents |
|---|---|
| `index.html` | Folium forecast map (GitHub Pages entry point) |
| `assets/<SEASON>/stability_*.html` | Per-station Plotly stability charts |
| `output/evaluation_{regional,blended,hierarchical}.csv` | Aggregate model metrics |
| `output/evaluation_loo.csv` | Temporal leave-one-out per-fold results |
| `output/evaluation_operational.csv` | Per-station recall-maximizing threshold + false-alarm rate |
| `output/hierarchical_uncertainty.csv` | Per-station mean probability + posterior std |
| `output/pr_curves.png` | Event-level precision-recall curves (all models) |
| `output/loo_performance_by_events.png` | Learning curve: performance vs. training-event count |
| `output/operational_thresholds.png` | False-alarm rate per station, by tier |

## Sn38 Interpretation

| Sn38 | Stability |
|------|-----------|
| < 1.0 | Unstable |
| 1.0 – 1.5 | Very low |
| 1.5 – 2.5 | Low |
| 2.5 – 3.5 | Moderate |
| > 3.5 | Stable |
