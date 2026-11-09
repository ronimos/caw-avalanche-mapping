# caw-avalanche-mapping

Avalanche forecasting support for data-sparse Central Asia. The tool links NWP-driven [SNOWPACK](https://models.slf.ch/p/snowpack/) simulations to opportunistically collected avalanche observations, trains per-station and pooled classifiers, and produces interactive stability charts and a forecast map. Developed for the ISSW 2026 paper *"Avalanche Forecasting in Data-Sparse Central Asia: A Weak-Supervision Framework."*

The code runs in two modes:

- Evaluation (default). Trains on 2024-25, evaluates on the held-out 2025-26 season and builds the paper's figures and the 2025-26 map. The map is served by GitHub Pages from `index.html` at the repository root.
- Forecast-only (`--forecast-only`). Trains on every past season and predicts the current one (2026-27). No evaluation. Runs daily from cron and publishes to https://caw.wac3.ch/nwp/caw/test/forecast/index.html.

See `docs/methods_and_results.md` for methods and results, and `docs/map_user_guide.md` for the 2025-26 map.

## Pipeline

1. Load avalanche observations from `data/observations/avalanches_<SEASON>.csv` (all seasons combined).
2. Match each observation to the nearest SNOWPACK station of the same aspect, using `snowpack_stations_locations.csv`.
3. Parse `.pro` (layer stratigraphy, Sn38) and `.smet` (meteorology) files into daily features.
4. Train the classifiers:
   - per-station logistic regression
   - regional pooled logistic regression (CV-tuned regularization)
   - blended, a confidence-weighted combination of the two
   - hierarchical Bayesian partial-pooling model (random intercepts, shared slopes; also yields per-station uncertainty)
5. Evaluation mode only: temporal leave-one-out CV, event-level precision-recall and an operational false-alarm analysis, which sets each station's confidence tier.
6. Fit the operational models on all available data and pick one per station (blended for "ready" stations, hierarchical otherwise unless it fails its reliability check).
7. Write per-station Plotly stability charts and a Folium forecast map.

## Project Structure

```
caw-avalanche-mapping/
├── src/
│   ├── main.py            # entry point; evaluation pipeline and shared helpers
│   ├── forecast.py        # forecast-only mode (main.py --forecast-only)
│   ├── snowpack_io.py     # .pro / .smet parsing, aspect-aware station matching
│   ├── features.py        # daily feature engineering (reduced 5-feature set)
│   ├── classifier.py      # per-station / regional / blended logistic regression + evaluation
│   ├── hierarchical.py    # hierarchical Bayesian (PyMC) partial-pooling model
│   ├── terrain.py         # DEM download and avalanche-path tracing
│   ├── select_pro_files.py # remote-side helper: copy aspect-matched .pro files
│   └── visualization.py   # Plotly stability charts + Folium map
├── scripts/
│   ├── sync_pro_files.py         # copy a season's .pro/.smet files (local run dir or SSH)
│   ├── focus_map.py              # WMP-point / station map with the model-location editor
│   ├── focus_map_editor.js       # editor UI, embedded into the map by focus_map.py
│   ├── save_model_locations.php  # save endpoint for the editor (copied next to the map)
│   ├── daily_forecast.sh         # cron job: sync + forecast + publish
│   └── generate_hierarchical_comparison.py  # presentation figures
├── data/
│   ├── simulations/<SEASON>/*.pro,*.smet   # SNOWPACK outputs, one set per season
│   ├── observations/avalanches_<SEASON>.csv # field observations per season
│   ├── FOCUS_Ava.kml                        # WMP point locations (forecast station set)
│   ├── WMP_Locations-2024.csv               # WMP village list (2024), added to the forecast station set
│   ├── geo/pamir_highway.geojson            # M41 route, OSM relation 180104 (ODbL)
│   └── snowpack_stations_locations.csv      # station id → lat/lon/elevation (older runs)
├── docs/                  # methods, map guide, ISSW paper
├── assets/<SEASON>/stability_*.html  # 2025-26 charts served by GitHub Pages
├── index.html             # 2025-26 Folium map (GitHub Pages entry point)
├── output/                # evaluation CSVs, figures, forecast site (git-ignored)
├── models/                # saved classifiers (git-ignored)
├── pyproject.toml
└── uv.lock
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Main dependencies: `pandas`, `numpy`, `scikit-learn`, `joblib`, `pymc`, `arviz`, `plotly`, `folium`, `matplotlib`, and `xsnow` (pinned from GitLab, not on PyPI).

The paper's case-study PNG (`fig4_stability_160942.png`) is rendered by kaleido, which needs Chrome. Without it the run prints a warning and skips that one figure. Install Chrome with `uv run kaleido_get_chrome`.

## Evaluation Mode

```bash
uv run python src/main.py                          # full pipeline
uv run python src/main.py --retrain                # force retraining of all classifiers
uv run python src/main.py --no-hierarchical        # skip PyMC / MCMC sampling
uv run python src/main.py --forecast-date 2026-01-23
```

`SEASON` (test) and `TRAIN_SEASON` are set at the top of `src/main.py`. The run writes `index.html` and `assets/<SEASON>/stability_*.html` to the repository root, which changes the live GitHub Pages site once committed. Evaluation CSVs and figures go to `output/`.

## Forecast-Only Mode (2026-27)

```bash
uv run python src/main.py --forecast-only [--site-dir DIR] [--forecast-date YYYY-MM-DD] [--retrain] [--no-hierarchical]
```

- Season: `FORECAST_SEASON` in `src/main.py` (`2026-2027`). Every other folder under `data/simulations/` is training data.
- Labels: observations from the training seasons, each matched within its own season's folder. Observations from the forecast season are not used for training, because the forecast folder only holds the selected stations and nearest-station matching would snap them onto an unrelated one.
- Stations: every `.pro`/`.smet` pair in `data/simulations/2026-2027/`.
- Forecast date: the last simulated day unless `--forecast-date` is given.
- Confidence tiers and hierarchical reliability are read from the last evaluation run (`output/evaluation_operational.csv`, `output/hierarchical_uncertainty.csv`). Without them every station is "not ready".
- Stations with no avalanche history of their own get their hierarchical probability shown relative to a regional avalanche day: the population-level prior's mean probability on all known training event days (p = 0.094 for the current model).
- The hierarchical model is cached in `models/forecast/hierarchical.joblib` and resampled only with `--retrain`.
- Output: `index.html` plus `assets/2026-2027/stability_*.html` in `--site-dir` (default `output/forecast/2026-2027/`). The repository-root `index.html` is never touched.

Each forecast chart has an "Open SNOWPACK viewer" button. It opens the niViz page from the sibling `snowpack-mapping` repo (`/nwp/caw/conditions/profiles.html?year=2026&id=<station>`), which stacks the timelines of all five aspects.

### Daily cron job

`scripts/daily_forecast.sh` runs at 11:30 UTC, after the SNOWPACK output lands (about 10:10 UTC) and after the snowpack-mapping conditions map (11:00 UTC):

```
30 11 * * * /home/ops/post-processing/caw-avalanche-mapping/scripts/daily_forecast.sh > /dev/null 2>&1
```

It syncs the station files (applying the saved editor selection with `--prune`), runs forecast mode and publishes to `/var/www/html/nwp/caw/test/forecast/`. It logs to `/home/ops/rtsys/log/caw_forecast.log` and a `flock` stops overlapping runs. Cron runs whatever branch is checked out in this working copy. It does not check that the day's SNOWPACK output arrived, so a late run publishes the previous day.

## Station Selection

### Syncing simulation files

`scripts/sync_pro_files.py` copies `.pro`/`.smet` pairs into `data/simulations/<SEASON>/`.

```bash
# 2026-27: nearest station to every WMP point and the Pamir Highway stations, all aspects, plus the editor's edits
uv run python scripts/sync_pro_files.py --season 2026-2027 \
    --source /mnt/HC_Volume_106760790/model/snowpack/runs/output.caw.2026 \
    --focus data/FOCUS_Ava.kml --focus data/WMP_Locations-2024.csv --highway \
    --selection /var/www/html/nwp/caw/test/model_locations/model_locations.json --prune

# Older seasons: one pair per (station, aspect) found in the observations, over SSH
uv run python scripts/sync_pro_files.py --dry-run
```

- `--source DIR` copies from a local run directory. Without it the script pulls over SSH (`SSH_PORT`, `SSH_KEY` in `.env`).
- `--focus KML` matches every Point placemark to its nearest station by the `.pro` header coordinates of the source run and copies all five aspect files. It also writes `focus_stations.csv` (point, station, distance) next to the files. The 63 WMP points map to 59 stations, all within 4.9 km.
- `--selection JSON` applies the Add/Remove edits from the map editor on top of that baseline.
- `--prune` deletes files in the season folder that are no longer selected. `--dry-run` previews everything.

### Model-location editor map

```bash
uv run python scripts/focus_map.py [--season 2026-2027] [--out DIR] [--source RUN_DIR]
```

Builds `focus_stations_map.html` (default `/home/ops/www/nwp/caw/test/`, served at https://caw.wac3.ch/nwp/caw/test/focus_stations_map.html). Layers: WMP points, their stations, the Pamir Highway, the 49 stations within 8 km of the highway (one grid spacing), all 1,688 run stations (off by default) and the edits.

Clicking a marker or any spot on the map opens the nearest station in a side panel. Mark it Add or Remove for any of Flat, N, E, S, W, then Apply. Save posts the edits to `save_model_locations.php`, which writes `model_locations/model_locations.json` and copies the previous version to `model_locations/backups/model_locations_<UTC timestamp>.json` when the content changed. Unsaved edits stay as a draft in the browser. The search box takes WMP names, station ids, place names (OpenStreetMap Nominatim) or `lat, lon`.

The endpoint has no authentication: anyone who can reach the page can overwrite the file. PHP runs as `apache`, so the save folder needs a one-time ACL:

```bash
mkdir /var/www/html/nwp/caw/test/model_locations
setfacl -m u:apache:rwx,d:u:apache:rwx,d:u:ops:rwx,d:m::rwx /var/www/html/nwp/caw/test/model_locations
```

`model_locations.json` (schema `caw-model-locations/1`):

```json
{
  "schema": "caw-model-locations/1",
  "season": "2026-2027",
  "saved": "2026-11-01T19:14:52Z",
  "selections": [
    {"station": "23158", "action": "add", "aspects": ["N", "S"], "note": "", "source": "...",
     "lat": 37.4963, "lon": 71.5469, "elevation": 2641, "updated": "..."}
  ]
}
```

`add` adds the listed aspects of a station, `remove` drops them. Aspect names map to file suffixes: Flat → none, N → `1`, E → `2`, S → `3`, W → `4`.

## Data Formats

### Observation CSV (`avalanches_<SEASON>.csv`)

Columns: `Place`, `Lat`, `Long`, `Slope`, `Aspect` (N/E/S/W), `Elevation (M)`, `Date`, `Size`, `Remarks`.

### SNOWPACK files

Station outputs are named `{station_id}{aspect_suffix}_res.pro` (and `.smet`). Station ids are 5 digits; the aspect suffix is `1`=N, `2`=E, `3`=S, `4`=W and absent for flat, so `231581_res` is station 23158 north and `13874_res` is station 13874 flat. Inclined aspects are 38°. Parsing goes by stem length, not the last digit. A run with ids of another length breaks this.

The `.pro` parser (via xsnow) reads layer heights from ground and Sn38 per layer. Layers buried less than 20 cm are excluded, so early in the season a station can have no layers at all. The charts then draw the snow surface from daily HS and say so.

## Model Features

The classifiers use a reduced five-feature daily set, selected to limit overfitting with sparse events:

| Feature | Role |
|---|---|
| `HS` | Total snow depth (size / valley-reach proxy) |
| `HN24` | 24-hour new snow (loading trigger) |
| `TA_max` | Daily max air temperature (wet vs dry mechanism) |
| `sn38_min` | Whole-profile minimum stability index (weakest-layer strength) |
| `depth_lower_wl` | Burial depth of the weakest lower-zone layer |

## Map Features

The forecast maps show simulation station circles and field observation triangles. Each is coloured by a signal level, and every marker has a white halo and dark outline so it stays visible on topographic and satellite tiles. See `docs/map_user_guide.md` for details.

Each model reaches a level on its own scale. Blended stations use probability. Hierarchical stations use probability divided by their chart's reference day (a known avalanche day at the station, or the regional one), because their raw probabilities sit near the ~1 % base rate and would all read as very low. Hovering a marker gives the number behind its level.

| Colour | Level | Blended (probability) | Hierarchical (× avalanche day) |
|--------|-------|----------------------|-------------------------------|
| Dark red | Very high | ≥ 80 % | ≥ 2× |
| Red | High | 65 – 80 % | 1 – 2× |
| Orange | Moderate | 50 – 65 % | 0.5 – 1× |
| Gold | Low | 33 – 50 % | 0.25 – 0.5× |
| Green | Very low | < 33 % | < 0.25× |
| Gray | No forecast | | |

The cut points are `MAP_PROB_LEVELS` and `MAP_RATIO_LEVELS` in `src/visualization.py`. 1× means as strong as a real avalanche day; 0.5× matches `HIER_RELIABILITY_THRESHOLD` in `src/main.py`. The regional-model overlay always uses the probability scale.

Interactions:
- Click any station marker to open its stability chart.
- Date Navigator (bottom centre) filters observations by date and recolours markers by the maximum probability within ±3 days.
- Time Series Player (bottom) scrubs through daily probability per station.
- Aspect filter (bottom right) shows one aspect, or "All" (maximum per location).
- Regional model overlay (layer toggle) shows the pooled model for all stations.

Station confidence tiers, from the operational false-alarm analysis of the evaluation run:

| Tier | Marker | False alarm rate | Stations (2025-26 evaluation) |
|------|-------------|-----------------|----------|
| Ready | strong fill (0.95), thick outline | ≤ 10 % | 160942 E, 153203 S, 180343 S, 164801 N |
| Marginal | medium fill (0.75) | 10 – 25 % | 272401 N, 176522 E, 250224 S |
| Not ready | light fill (0.55), thin outline | > 25 % | All others |

A dashed outline marks a hierarchical-model station, a dotted one a blended fallback.

## Output Files

| File | Contents |
|---|---|
| `index.html` | 2025-26 Folium map (GitHub Pages entry point) |
| `assets/<SEASON>/stability_*.html` | Per-station Plotly stability charts |
| `output/forecast/<SEASON>/` | Forecast-only site (map + charts) when `--site-dir` is not given |
| `output/evaluation_{regional,blended,hierarchical}.csv` | Aggregate model metrics |
| `output/evaluation_loo.csv` | Temporal leave-one-out per-fold results |
| `output/evaluation_operational.csv` | Per-station recall-maximizing threshold + false-alarm rate (sets the tiers) |
| `output/hierarchical_uncertainty.csv` | Per-station mean probability, posterior std, reliability flag |
| `output/pr_curves.png` | Event-level precision-recall curves (all models) |
| `output/loo_performance_by_events.png` | Learning curve: performance vs. training-event count |
| `output/operational_thresholds.png` | False-alarm rate per station, by tier |
| `data/simulations/<SEASON>/focus_stations.csv` | WMP point → station manifest from `--focus` |
| `data/geo/pamir_highway_stations.csv` | Stations within 8 km of the Pamir Highway |

## Sn38 Interpretation

| Sn38 | Stability |
|------|-----------|
| < 1.0 | Unstable |
| 1.0 – 1.5 | Very low |
| 1.5 – 2.5 | Low |
| 2.5 – 3.5 | Moderate |
| > 3.5 | Stable |
