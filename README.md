# snowpack

Python tool for analyzing [SNOWPACK](https://models.slf.ch/p/snowpack/) model output to assess avalanche weak-layer stability and visualize field observations on an interactive map.

## Overview

The workflow:
1. Load avalanche observation points from a CSV file (lat/lon/aspect)
2. Match each observation to the nearest SNOWPACK simulation station (`.pro` file)
3. Parse the `.pro` files to extract layer heights and Sn38 stability indices
4. Generate per-station interactive stability plots (Plotly HTML)
5. Generate an interactive Folium map linking each observation to its station's stability plot

## Project Structure

```
snowpack/
├── src/
│   ├── main.py          # Entry point — orchestrates the full pipeline
│   └── snow_utils.py    # Core parsing, analysis, and plotting functions
├── data/
│   ├── *.pro            # SNOWPACK simulation output files (one per station)
│   ├── 2026_02_avalanches.csv    # Avalanche observations (main input)
│   └── Avalanche Information.csv # Broader observation log (reference)
├── plots/               # Generated HTML outputs (git-ignored)
│   ├── stability_*.html # Per-station stability plots
│   └── avalanche_map.html        # Interactive observation map
├── pyproject.toml
└── uv.lock
```

## Setup

Requires Python 3.12. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

Dependencies: `pandas`, `matplotlib`, `plotly`, `folium`

## Running

```bash
cd src
uv run python main.py
```

Or from the repo root:

```bash
uv run python src/main.py
```

Outputs are written to `plots/`. Open `plots/avalanche_map.html` in a browser, then click any marker to open the linked station's stability plot.

## Data Formats

### Avalanche Observations CSV

Expected columns: `File Name`, `Placemark Name`, `Latitude`, `Longitude`, and optionally an aspect column (may appear as `Unnamed: 4`).

The filename is parsed for a date (e.g. `2026_02_avalanches.csv` → February 15, 2026) to set the initial view window on plots.

### SNOWPACK `.pro` Files

Standard SNOWPACK profile output. The parser reads three record codes:

| Code | Content |
|------|---------|
| `0500` | Timestep date |
| `0501` | Layer heights from ground (m), bottom→top |
| `0532` | Sn38 (natural stability index) per layer, same order as 0501 |

Layers shallower than 20 cm burial depth are excluded from analysis.

## Key Functions (`snow_utils.py`)

| Function | Purpose |
|----------|---------|
| `parse_snow_data(file_path)` | Parse `.pro` → long-format DataFrame (`timestamp`, `layer_z`, `burial_depth`, `sn38`) |
| `plot_interactive_stability(df, output_path, ...)` | Generate 2-panel Plotly HTML: dominant weak layers (top) + min Sn38 timeseries (bottom) |
| `create_avalanche_map(df, output_path, ...)` | Generate Folium map with clickable red markers |
| `find_nearest_pro(lat, lon, data_dir)` | Match a coordinate to the closest `.pro` file by haversine distance |
| `extract_pro_coordinates(pro_file)` | Read lat/lon from `.pro` header |
| `parse_date_from_csv_filename(csv_path)` | Extract event date from filename patterns like `2026_02_15` |

## Stability Plot Details

The upper panel shows "dominant weak layers" — layers that were, at any point in the simulation, the minimum-Sn38 layer. Each such layer gets a scatter trace colored by Sn38 (red = unstable, green = stable), with the snow surface plotted as a grey reference line.

The lower panel shows the minimum Sn38 across all layers at each timestep, providing an overall stability signal.

If an event date is found in the CSV filename, the initial view zooms to ±7 days around that date; the full timeseries is accessible via the range slider.

## Sn38 Interpretation

| Sn38 | Stability |
|------|-----------|
| < 1.0 | Unstable |
| 1.0 – 1.5 | Very low |
| 1.5 – 2.5 | Low |
| 2.5 – 3.5 | Moderate |
| > 3.5 | Stable |
