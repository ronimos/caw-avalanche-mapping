# Path-Specific Terrain Data — Build Plan

## Motivation

SNOWPACK virtual stations sit on a fixed grid and do not represent the actual
elevation or terrain of the avalanche path. The snow depth (HS) needed for an
avalanche to reach the valley floor is set by the **terrain**: at one path 1 m
of HS is enough, at another you need 2 m. The model currently has no knowledge
of this, so HS means different things at different stations — and this gap is
worst exactly where observations are sparse.

Path geometry is **constant per path**. It is not a daily release predictor; it
is a per-path *consequence threshold* — how much snow/instability is required to
produce an avalanche large enough to reach the valley floor. Because it is a
fixed-effect covariate rather than a per-path random intercept, it **transfers
to paths with little or no observation history** — the sparse-data fix we want.

## What we compute

Walk **up-slope from the debris location** (the known observation coordinate) to
the **start zone**, on a DEM. From the traced path, retrieve:

| Feature            | Definition                                             |
|--------------------|--------------------------------------------------------|
| `vertical_drop` H  | start-zone elevation − debris elevation (m)            |
| `travel_distance` L| planimetric length debris → start zone (m)             |
| `alpha`            | runout angle `atan(H / L)` (deg) — mobility descriptor |
| `startzone_elev`   | start-zone elevation (m) — also drives HS lapse-correction |
| `startzone_aspect` | start-zone aspect (deg)                                |
| `startzone_slope`  | mean slope of the start zone (deg)                     |

## Tracing algorithm (debris → start zone)

Avalanches run in **concavities** (gullies), where flow accumulation is high.
Steepest-*ascent* does the opposite — its flow paths converge onto **convex
ridges** — so a single-thread up-trace can leave the gully. **Downstream**
routing always converges into gullies, so the primary path is built downstream
from a watershed-chosen start zone:

1. **Fill depressions** (priority-flood) so every cell drains to the boundary.
2. **D8 flow direction** + **flow accumulation**.
3. **Snap** the debris onto the nearest channel (highest accumulation within a
   small radius).
4. **Reverse watershed:** delineate the debris' full upslope catchment (every
   cell that drains to it). Its connected steep (≥ STEEP_DEG) clusters are the
   start zones; the **governing** one is the highest (biggest drop → most
   exposed / lowest required HS).
5. **Route downstream** from the governing start zone to the **valley floor**
   (mean slope over VALLEY_RUN cells < VALLEY_DEG). This is the primary path and
   the source of the model geometry (H, L, α, start-zone slope/aspect/elevation).

The single-thread **up-trace** (fill → accumulation → walk upstream along the
main stem, stopping at the channel head or start-zone rollover) is retained only
as a faint QA **reference line** on the map — field review found it places the
start zone lower and sometimes wanders out of the gully, so it is no longer the
primary path.

### Decisions baked in (tunable)
- **STEEP_DEG = 30°, SLOPE_WINDOW = 4** — windowed start-zone stop (robust to a
  single steep-flat-steep undulation that fooled the earlier per-cell test).
- **MIN_CHANNEL_ACCUM = 4** — channel-head stop; halts headwater wandering.
- **VALLEY_DEG = 15°, VALLEY_RUN = 4** — valley-floor stop for the down-trace.
- **Main-stem selection** (max accumulation) keeps the trace on the channel axis
  rather than a steeper side wall.
- **Reverse-watershed:** the debris' full upslope catchment is delineated (every
  cell that drains to it). Its connected steep (≥ STEEP_DEG) clusters are the
  **start zones**; the **governing** one is the highest (biggest drop → most
  exposed / lowest required HS), and its full path to the valley floor supplies
  the path-level model covariate (`gov_vertical_drop`, `gov_alpha`, …). The
  catchment polygon is drawn on the QA map (toggleable layer). The single
  through-debris path is kept for per-event attribution.

## DEM source

OpenTopography global DEM API (`/API/globaldem`), `demtype=COP30` by default
(Copernicus GLO-30, 30 m). Credentials + dataset in `.env`
(`OPENTOPO_API_KEY`, `OPENTOPO_DEMTYPE`; see `.env.example`). A `radius_km`
padded box is requested per debris point and cached to `data/dem/`.

**Resolution:** OpenTopography's only *global* datasets are ~30 m (COP30,
SRTMGL1, AW3D30, NASADEM). The 10 m USGS3DEP set is CONUS-only, so no free 10 m
exists for the Central Asia study area. Switch datasets via `OPENTOPO_DEMTYPE`.

**Caveat:** 30 m under-resolves start zones and gullies. H is reliable; L and
confinement are coarse. Smooth the profile before applying angle thresholds.

## Module layout — `src/terrain.py`

```
fetch_dem(lon, lat, radius_km)      → cached GeoTIFF window (Copernicus GLO-30)
load_dem(path)                      → (elevation array, affine transform, crs)
trace_path(dem, transform, lon, lat)→ PathGeometry (polyline + H, L, alpha)
path_features(geom)                 → dict of the features above
build_path_atlas(observations_df)   → DataFrame keyed by observation; cached CSV
```

Decoupled from the daily pipeline: the atlas is a static, cached table. Nothing
touches the model until the traces are validated.

## Integration (later, after validation)

- Join atlas features into the feature table by observation / station.
- Feed `alpha` (and `startzone_elev` for HS lapse-correction) as **group-level
  covariates on the intercept** in `hierarchical.py` — not as daily features.
- Hold in reserve: graded runout label (debris reached valley floor vs. stopped
  on slope) as a richer target than binary occurrence.

## Validation first

Overlay traced polylines on the Folium map and sanity-check 5–10 known paths
before any of this reaches the model. Auto-selection at 30 m will have failures.

## New dependencies

- `rasterio` — read Copernicus COG tiles (windowed remote reads via `/vsicurl/`).
- `scipy` — profile smoothing / neighbourhood ops (already transitively present
  via scikit-learn; pin explicitly).

## Result so far (leave-one-station-out, 13 training stations)

Terrain as `alpha` on the intercept does **not** improve LOO cross-station
prediction: baseline AP 0.045 / AUC 0.701 vs +terrain AP 0.046 / AUC 0.696
(no-skill AP 0.014). The integration is correct (synthetic test recovers a
planted `gamma` and shows unseen-station transfer), but the signal isn't there
at 13 stations / ~20 events.

Two reasons it may be the wrong test, not just a dead end:
- **Intercept-only shift** says "steeper-runout paths are more avalanche-prone
  regardless of weather" — a *weaker* claim than the original hypothesis, which
  is that terrain sets *how much HS is needed* (a **terrain × HS interaction**,
  not a base-rate shift). The interaction is the real experiment; it needs more
  data before it can be fit without overfitting.
- AP/AUC are ranking metrics; a per-station intercept shift mostly affects
  calibration/level, so it is a weak lever on these scores.

## Status

- [x] `terrain.py`: DEM fetch + cache (Copernicus GLO-30 via `/vsicurl/`, verified live)
- [x] `terrain.py`: path tracing + features (verified on synthetic + real DEM)
- [x] `terrain.py`: `build_path_atlas` (written; not yet run over full observation set)
- [x] Folium overlay of traced paths (`create_trace_validation_map` in `visualization.py`)
- [x] Trace-quality flagging (start-zone slope / travel / drop heuristics → good/check/fail)
- [x] Flow-routed (concave) tracing; full path start-zone → valley floor
- [x] Reverse-watershed: catchment, start-zone enumeration, governing path + polygon overlay
- [ ] Clear the DEM-quota failures (region-batched fetch or quota reset)
- [ ] Human review of traces on satellite imagery; tune thresholds
- [ ] Hierarchical model integration (feed `gov_alpha` + start-zone elev lapse-correction)
