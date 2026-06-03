# CLAUDE.md

## Project Summary

Python avalanche stability analysis tool. Parses SNOWPACK `.pro` simulation output, matches field avalanche observations to nearest stations, and generates interactive HTML plots and maps.

## Running the Pipeline

```bash
cd src && uv run python main.py
# or
uv run python src/main.py
```

Outputs go to `plots/`. Entry point is `src/main.py`; all logic lives in `src/snow_utils.py`.

## Key Design Notes

- **Dominant layer algorithm** (`_get_dominant_layers`): walks timesteps in order, tracks which layer has the minimum Sn38 at each step, clusters nearby layers by z-height (±3 cm tolerance), and only surfaces each cluster once it first becomes the minimum. This avoids noisy layer-switching while preserving the history of structurally distinct weak layers.

- **`.pro` parsing**: only codes `0500` (timestamp), `0501` (layer heights), `0532` (Sn38) are used. Layers with burial depth < 20 cm are dropped. Heights in `.pro` files are in **cm** — `total_height` is the max layer height at a given timestep.

- **Station matching**: haversine distance to `.pro` header coordinates. Each unique matched station gets one stability HTML; the map links each observation marker to its station's file.

- **CSV date parsing**: filename is parsed for dates (`2026_02_15`, `20260215`, `2026-02`, etc.). Year+month only → defaults to the 15th.

## Data Layout

```
data/
  simulations/<SEASON>/*.pro   — SNOWPACK station outputs (one file per station)
  observations/                — field observation CSVs (all seasons together)
```

To add a new season, drop `.pro` files into `data/simulations/<new-season>/` and set `SEASON` in `main.py`.

- Current season: `2025-2026` (Tajikistan/Darvoz region stations)
- `Avalanche Information.csv` — multi-year AKAH observation log (Pakistan + Tajikistan)

## Dependencies

Managed with `uv`. Python 3.12 required.
- `pandas`, `plotly`, `folium`, `matplotlib`
- `scikit-learn`, `joblib` — frequentist classifiers (per-station, regional, blended)
- `pymc`, `arviz` — hierarchical Bayesian model (`src/hierarchical.py`); skip with `main.py --no-hierarchical`

## File Outputs (git-ignored via output/)

- `output/assets/<SEASON>/stability/stability_{station_id}.html` — per-station Plotly interactive chart
- `output/assets/<SEASON>/avalanche_map.html` — Folium map; click markers to open stability plots
