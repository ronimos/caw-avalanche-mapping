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

- **`.pro` parsing**: delegates to `xsnow` (`add_surface_sh_as_layer=False` — no synthetic 1 mm surface-hoar layer from record 0514), reshaped to the project's long-format layer DataFrame. Layers with burial depth < 20 cm are dropped (0.001 cm tolerance for float32 rounding). xsnow returns metres; values are converted back to **cm** — `total_height` is the max layer height at a given timestep. `.smet` parsing stays hand-rolled: xsnow's unit registry rejects our files (`dIntEnergySnow` in kJ/m2 vs. expected W m-2).

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
- `rasterio`, `scipy` — DEM read + path tracing (`src/terrain.py`)
- `requests`, `python-dotenv` — OpenTopography DEM download; needs `OPENTOPO_API_KEY` in `.env` (see `.env.example`)
- `xsnow` — SNOWPACK `.pro` parsing (`src/snowpack_io.py`); not on PyPI, pinned from GitLab (`git+https://gitlab.com/avacollabra/postprocessing/xsnow`)

## File Outputs (git-ignored via output/)

- `output/assets/<SEASON>/stability/stability_{station_id}.html` — per-station Plotly interactive chart
- `output/assets/<SEASON>/avalanche_map.html` — Folium map; click markers to open stability plots
