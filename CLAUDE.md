# CLAUDE.md

## Project Summary

Avalanche forecasting for data-sparse Central Asia. Parses SNOWPACK `.pro`/`.smet` output, matches field avalanche observations to stations, trains per-station, regional, blended and hierarchical Bayesian classifiers, and writes Plotly stability charts and Folium maps. README.md has the user-facing detail; this file covers what to know before changing code.

## Running

```bash
uv run python src/main.py                     # evaluation mode: train 2024-25, test 2025-26, paper figures
uv run python src/main.py --forecast-only     # 2026-27 forecast (src/forecast.py), no evaluation
uv run python scripts/sync_pro_files.py ...   # copy station files into data/simulations/<SEASON>/
uv run python scripts/focus_map.py            # WMP map + model-location editor
scripts/daily_forecast.sh                     # what cron runs at 11:30 UTC
```

Flags: `--retrain` (otherwise models load from `models/`), `--no-hierarchical` (skips MCMC), `--forecast-date`, `--site-dir` (forecast output folder). There is no test suite. Verify chart or map changes by rendering them and looking (see "Checking HTML output").

## Code Layout

- `src/main.py`: evaluation pipeline in one long `main()`, plus helpers shared with forecast mode: `_choose_models` (model switch), `_render_stability_plots`, `_build_sim_stations`, `_split_stem`, `_hier_known_event_ref`. Season constants `SEASON`, `TRAIN_SEASON`, `FORECAST_SEASON` live at the top.
- `src/forecast.py`: forecast-only mode. Imports helpers from `main` (main imports it lazily inside `main()` to avoid a cycle).
- `src/snowpack_io.py`: `.pro` via xsnow, `.smet` hand-rolled, nearest-station matching.
- `src/features.py`: daily features (`HS`, `HN24`, `TA_max`, `sn38_min`, `depth_lower_wl`, plus zone mins).
- `src/classifier.py`, `src/hierarchical.py`: models. `HierModel.predict` with an id not in `station_list` uses the population-level prior.
- `src/visualization.py`: `plot_interactive_stability` (charts) and `create_avalanche_map` (map with large inline JS).
- `scripts/focus_map.py` builds the editor map shell and embeds `focus_map_editor.js` plus a data payload (`window.FOCUS_EDITOR_DATA`). `save_model_locations.php` is copied next to the map.

## Key Design Notes

- Station ids are 5 digits (`STATION_ID_DIGITS`). An aspect file appends one digit: `1`=N, `2`=E, `3`=S, `4`=W, none for flat. Use `_split_stem` for stem → (station, aspect). Never infer the aspect from the last digit alone: 674 flat stations in the 2026 run have ids ending in 1-4.
- `.pro` parsing delegates to xsnow (`add_surface_sh_as_layer=False`, so no synthetic 1 mm surface-hoar layer from record 0514), reshaped to a long-format layer DataFrame. Layers buried < 20 cm are dropped (0.001 cm tolerance for float32 rounding). xsnow returns metres; values are converted to cm. `.smet` parsing stays hand-rolled because xsnow's unit registry rejects our files (`dIntEnergySnow` in kJ/m2 vs W m-2). Daily `HS`/`HN24` from `.smet` are in metres.
- Early season, a `.pro` file can have no snow layers at all (no `sn38` variable) or none deep enough. `parse_snow_data`, `_layer_zone_features` and `_get_dominant_layers` return empty frames with the expected columns in that case. Charts take their x range from the daily features and draw HS from `.smet`. Keep these paths working when touching parsing or plotting.
- Dominant layer algorithm (`_get_dominant_layers`): walks timesteps in order, finds the minimum-Sn38 layer at each, clusters layers by height (±3 cm) and shows each cluster from the time it first becomes the minimum.
- Model switch (`_choose_models`): "ready" stations use blended; others use hierarchical unless `hier_reliable` is False, then `blended_fallback`. Tiers come from the evaluation run's `output/evaluation_operational.csv`; forecast mode only reads them.
- Hierarchical charts plot probability relative to a reference day on a log axis (`_hier_known_event_axis`): the station's own known event days, or in forecast mode a regional reference for stations without history (`forecast._population_event_ref`). Plotly annotations on a log axis need `log10(value)` for y.
- Chart fonts: `WEB_*` sizes for HTML, `PLOTLY_*` sizes only for the paper PNG (`for_print=True`). The print sizes overlap badly on screen.
- Charts load plotly.js from the CDN (`include_plotlyjs='cdn'`), about 4.6 MB less per file. They need internet access to display.
- Folium loads Bootstrap. Do not use Bootstrap class names (`row`, `col`, `btn`, ...) in custom map HTML. `.row > *` forces children to full width; the editor uses `mle-row` for that reason.
- Observation matching uses `snowpack_stations_locations.csv` (1,155 stations, older domain). The 2026 run has 1,688 stations and only 965 are in that CSV, so new-season code reads coordinates from `.pro` headers (`sync_pro_files.read_run_stations`).

## Data and Deployment

```
data/simulations/<SEASON>/*.pro,*.smet   SNOWPACK output per season (2026-2027 not committed: 48 MB, changes daily)
data/observations/avalanches_<SEASON>.csv observations, all seasons loaded together
data/FOCUS_Ava.kml                       WMP points; nearest station to each = forecast baseline
data/WMP_Locations-2024.csv              WMP village list (2024); adds villages not in the KML (out-of-domain ones get their nearest station, with a warning)
data/geo/pamir_highway.geojson           M41 route (OSM relation 180104, ODbL); stations within 8 km join the forecast set (--highway)
```

- 2026-27 source run: `/mnt/HC_Volume_106760790/model/snowpack/runs/output.caw.2026/<station>/`, updated daily around 10:10 UTC.
- Web root is `/var/www/html` (= `/home/ops/www`), served at https://caw.wac3.ch. Test pages: `/var/www/html/nwp/caw/test/` (editor map, `forecast/` site, `model_locations/` save folder).
- `/home/ops` is private and PHP runs as `apache`, so web-written data lives under the web root. The save folder has an ACL for `apache` and default ACLs for `ops`.
- `model_locations/model_locations.json` is live shared data written by the map's Save button, with timestamped backups. Do not edit or delete it as part of code changes. The endpoint has no authentication by the user's choice.
- The repository-root `index.html` and `assets/2025-2026/` are the live GitHub Pages site. Evaluation runs overwrite them; restore with `git checkout -- index.html assets/` unless a regenerated site is wanted. Forecast mode never writes there.
- Cron (`crontab -l`) runs `scripts/daily_forecast.sh` from whatever branch is checked out here. Keep `main` working.
- The SNOWPACK viewer button links to `/nwp/caw/conditions/profiles.html?year=<season start>&id=<station>` from the sibling `../snowpack-mapping` repo. It only has `.pro` files for the current run.

## Checking HTML output

No browser is installed system-wide. A Playwright Chromium in a scratch virtualenv works (`uv pip install playwright`, `playwright install chromium`). Load the page over https or `file://`, collect `pageerror` events and take a screenshot. For JS syntax alone, `uvx --from esprima python -c "import esprima; esprima.parseScript(...)"`.

## Dependencies

Managed with `uv`, Python 3.12.
- `pandas`, `numpy`, `plotly`, `folium`, `matplotlib`
- `scikit-learn`, `joblib` for the frequentist classifiers
- `pymc`, `arviz` for the hierarchical model; skip with `--no-hierarchical`
- `rasterio`, `scipy` for DEM read and path tracing (`src/terrain.py`)
- `requests`, `python-dotenv` for the OpenTopography DEM download (`OPENTOPO_API_KEY` in `.env`)
- `xsnow` for `.pro` parsing; not on PyPI, pinned from `git+https://gitlab.com/avacollabra/postprocessing/xsnow`
- `kaleido` for the paper PNG; needs Chrome (`uv run kaleido_get_chrome`), otherwise skipped with a warning
