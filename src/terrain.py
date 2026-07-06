"""
terrain.py — Path-specific terrain features from a DEM.

Walks up-slope from each avalanche debris coordinate to the start zone on a DEM
(fetched from OpenTopography), and derives per-path geometry (vertical drop,
travel distance, runout angle). These are *constant per path* — a
consequence-threshold covariate for the model, not a daily release feature. See
`docs/path_terrain_plan.md`.

Public API:
  fetch_dem          (lon, lat, radius_km)      → cached DEM GeoTIFF Path
  trace_path         (dem_path, lon, lat)       → PathGeometry (debris → start zone)
  path_features      (PathGeometry)             → dict of path features
  build_path_atlas   (observations DataFrame)   → per-observation feature table (cached)
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
import rasterio
import requests
from scipy.ndimage import gaussian_filter

dotenv.load_dotenv()

# ── configuration ─────────────────────────────────────────────────────────────

# OpenTopography global DEM API. Credentials + dataset come from .env
# (OPENTOPO_API_KEY, OPENTOPO_DEMTYPE); see .env.example.
_OPENTOPO_ENDPOINT = "https://portal.opentopography.org/API/globaldem"
_REQUEST_TIMEOUT = 120  # seconds

DEM_CACHE = Path(__file__).resolve().parents[1] / "data" / "dem"

# Slope at/above which terrain is "steep enough to release". The upstream walk
# stops once the *mean* slope over the last SLOPE_WINDOW channel cells rolls
# below this (the gully topping out into the start-zone headwall).
STEEP_DEG = 30.0
SLOPE_WINDOW = 4
# Upstream walk also stops when the inflow's contributing area drops below this
# many cells — the channel head, beyond which routing is headwater noise (this
# is what kept the old trace wandering up onto saddles/ridges past the start zone).
MIN_CHANNEL_ACCUM = 4
# Downstream walk stops at the valley floor: mean slope over VALLEY_RUN cells
# below VALLEY_DEG. Kept low + long so a mid-track bench doesn't stop it early —
# only a persistent flat (the actual valley bottom) does.
VALLEY_DEG = 10.0
VALLEY_RUN = 10
# Reverse-watershed: connected steep clusters smaller than this many cells are
# dropped as noise when enumerating start zones.
MIN_STARTZONE_CELLS = 3
# Snap the debris point onto the nearest channel within this cell radius before
# tracing, so the walk starts on the thalweg rather than the fan edge.
CHANNEL_SNAP_RADIUS = 2
# Safety caps so a runaway trace terminates.
MAX_STEPS = 5000
MAX_VERTICAL_M = 3000.0

# 8-connected neighbour offsets and, for each, the index of the opposite offset
# (used to test "does neighbour k flow back into me?").
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_OPP = [7, 6, 5, 4, 3, 2, 1, 0]

_ATLAS_CSV = DEM_CACHE / "path_atlas.csv"

# Region-batched fetch: a few large regional DEMs (one per spatial cluster of
# observations) are downloaded up front, then per-path windows are sliced from
# them locally — turning ~1 API call per observation into ~1 per cluster.
REGION_CACHE = DEM_CACHE / "regions"
_REGIONS: list[tuple[float, float, float, float, Path]] = []  # (w, s, e, n, path)


# ── DEM fetch / cache ─────────────────────────────────────────────────────────

def fetch_dem(lon: float, lat: float, radius_km: float = 4.0,
              demtype: str | None = None) -> Path:
    """
    Download and cache a DEM window centred on (lon, lat) from OpenTopography.

    Requests a `radius_km` padded box from the global DEM API and writes a small
    local GeoTIFF, cached by demtype + rounded bbox so repeated calls near the
    same point reuse one file.

    `demtype` defaults to $OPENTOPO_DEMTYPE (or COP30). Note: OpenTopography's
    only global datasets are ~30 m (COP30, SRTMGL1, AW3D30, NASADEM); the 10 m
    USGS3DEP set is CONUS-only and unavailable for Central Asia.

    Requires OPENTOPO_API_KEY in the environment / .env.
    """
    demtype = demtype or os.getenv("OPENTOPO_DEMTYPE", "COP30")
    DEM_CACHE.mkdir(parents=True, exist_ok=True)

    out, (w, s, e, n) = _tile_path(lon, lat, radius_km, demtype)
    if out.exists():
        return out

    # Prefer slicing from a cached regional DEM (no API call); else download.
    region = _region_covering(w, s, e, n)
    if region is not None:
        _window_region(region, out, w, s, e, n)
    else:
        _download_globaldem(w, s, e, n, demtype, out)
    return out


def _tile_path(lon: float, lat: float, radius_km: float,
               demtype: str) -> tuple[Path, tuple[float, float, float, float]]:
    """Cache path + bbox (w, s, e, n) for a per-path DEM window."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)) or 1e-6)
    w, s, e, n = lon - dlon, lat - dlat, lon + dlon, lat + dlat
    key = f"dem_{demtype}_{s:.3f}_{w:.3f}_{n:.3f}_{e:.3f}.tif".replace("-", "m")
    return DEM_CACHE / key, (w, s, e, n)


def dem_is_cached(lon: float, lat: float, radius_km: float = 4.0,
                  demtype: str | None = None) -> bool:
    """True if the per-path DEM window for (lon, lat) is already on disk."""
    demtype = demtype or os.getenv("OPENTOPO_DEMTYPE", "COP30")
    return _tile_path(lon, lat, radius_km, demtype)[0].exists()


def prepare_regions(observations: pd.DataFrame, radius_km: float = 4.0,
                    cluster_km: float = 50.0, demtype: str | None = None) -> list:
    """
    Download one large DEM per spatial cluster of observations, so subsequent
    `fetch_dem` calls slice their windows locally instead of hitting the API.

    Clusters coordinates with DBSCAN (points within `cluster_km` grouped), pads
    each cluster's bounding box by `radius_km`, and downloads/caches one regional
    DEM per cluster to `data/dem/regions/`. Registers them so `fetch_dem` uses
    them. Returns the list of (w, s, e, n, path). Turns ~1 API call per
    observation into ~1 per cluster.
    """
    demtype = demtype or os.getenv("OPENTOPO_DEMTYPE", "COP30")
    df = observations[["Latitude", "Longitude"]].dropna().drop_duplicates()
    labels = _cluster_coords(df, cluster_km)

    regions = []
    for lab in sorted(set(labels)):
        pts = df[labels == lab]
        lat_m = float(pts["Latitude"].mean())
        pad_lat = radius_km / 111.0
        pad_lon = radius_km / (111.0 * math.cos(math.radians(lat_m)) or 1e-6)
        w = float(pts["Longitude"].min()) - pad_lon
        e = float(pts["Longitude"].max()) + pad_lon
        s = float(pts["Latitude"].min()) - pad_lat
        n = float(pts["Latitude"].max()) + pad_lat
        path = _fetch_region_dem(w, s, e, n, demtype)
        regions.append((w, s, e, n, path))

    _REGIONS[:] = regions
    return regions


def _cluster_coords(df: pd.DataFrame, cluster_km: float) -> np.ndarray:
    """DBSCAN cluster labels for the coordinates (haversine, every point kept)."""
    from sklearn.cluster import DBSCAN

    X = np.radians(df[["Latitude", "Longitude"]].to_numpy())
    return DBSCAN(eps=cluster_km / 6371.0, min_samples=1,
                  metric="haversine").fit(X).labels_


def _fetch_region_dem(w: float, s: float, e: float, n: float, demtype: str) -> Path:
    """Download and cache one regional DEM covering the bbox (w, s, e, n)."""
    REGION_CACHE.mkdir(parents=True, exist_ok=True)
    key = f"region_{demtype}_{s:.3f}_{w:.3f}_{n:.3f}_{e:.3f}.tif".replace("-", "m")
    out = REGION_CACHE / key
    if not out.exists():
        _download_globaldem(w, s, e, n, demtype, out)
    return out


def _download_globaldem(w: float, s: float, e: float, n: float,
                        demtype: str, out: Path) -> None:
    """Fetch a GeoTIFF for bbox (w, s, e, n) from the OpenTopography global DEM API."""
    api_key = os.getenv("OPENTOPO_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENTOPO_API_KEY is not set. Add it to .env (see .env.example)."
        )
    params = {
        "demtype": demtype,
        "south": s, "north": n, "west": w, "east": e,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    resp = requests.get(_OPENTOPO_ENDPOINT, params=params, timeout=_REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenTopography {demtype} request failed "
            f"({resp.status_code}): {resp.text[:200]}"
        )
    out.write_bytes(resp.content)


def _region_covering(w: float, s: float, e: float, n: float) -> Path | None:
    """Return a registered regional DEM whose bounds contain the bbox, else None."""
    for rw, rs, re_, rn, path in _REGIONS:
        if rw <= w and rs <= s and re_ >= e and rn >= n:
            return path
    return None


def _window_region(region_path: Path, out: Path,
                   w: float, s: float, e: float, n: float) -> None:
    """Slice the (w, s, e, n) window out of a regional DEM into a small GeoTIFF."""
    from rasterio.windows import from_bounds

    with rasterio.open(region_path) as src:
        win = from_bounds(w, s, e, n, src.transform)
        data = src.read(1, window=win)
        meta = src.meta.copy()
        meta.update(height=data.shape[0], width=data.shape[1],
                    transform=src.window_transform(win))
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(data, 1)


# ── path tracing ──────────────────────────────────────────────────────────────

@dataclass
class PathGeometry:
    """
    A traced avalanche path and its derived geometry.

    The primary path is the **reverse-watershed governing path**: the debris'
    upslope catchment is delineated, its highest steep cluster is taken as the
    governing (most-exposed) start zone, and the path is routed *downstream* from
    there to the valley floor. Downstream routing converges into the concave
    gully, so it places the start zone and path more reliably than a single-thread
    up-trace — which is kept only as a faint QA reference (`uptrace_lonlat`).
    """
    lonlat: list[tuple[float, float]] = field(default_factory=list)  # start→valley
    elev:   list[float]               = field(default_factory=list)  # elevation at each vertex (m)
    vertical_drop:    float = math.nan   # H: start elev − valley-floor elev (m)
    travel_distance:  float = math.nan   # L: planimetric start → valley floor (m)
    alpha:            float = math.nan   # runout angle atan(H/L) (deg)
    startzone_elev:   float = math.nan
    startzone_aspect: float = math.nan
    startzone_slope:  float = math.nan
    reached_valley:   bool  = False      # down-trace reached a valley floor vs. a cap

    # Reverse-watershed context.
    catchment_area_km2: float = math.nan
    num_start_zones:    int   = 0
    catchment_rings:    list  = field(default_factory=list)   # rings of (lon, lat)
    debris_lonlat:      tuple[float, float] | None = None      # snapped observation point
    uptrace_lonlat:     list  = field(default_factory=list)    # step-up reference (debris→its start)

    @property
    def startzone_lonlat(self) -> tuple[float, float]:
        return self.lonlat[0]

    @property
    def valleyfloor_lonlat(self) -> tuple[float, float]:
        return self.lonlat[-1]


def trace_path(dem_path: Path, lon: float, lat: float) -> PathGeometry:
    """
    Trace the governing avalanche path for a debris coordinate.

    Routes by hydrology: fill depressions → D8 flow direction → accumulation, snap
    the debris onto the nearest channel, then delineate its upslope catchment
    (reverse watershed). The highest steep cluster in the catchment is the
    governing (most-exposed) start zone; the path is routed **downstream** from it
    to the valley floor, which stays in the concave gully. A single-thread
    up-trace is also computed as a faint QA reference. See
    `docs/path_terrain_plan.md`.
    """
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype("float64")
        transform = ds.transform
        nodata = ds.nodata

    if nodata is not None:
        dem[dem == nodata] = np.nan
    # Light smoothing: enough to suppress 30 m pixel noise without merging gullies.
    dem = _smooth(dem, sigma=0.8)

    cr, cc = dem.shape[0] // 2, dem.shape[1] // 2
    dx = _cell_distance_m(transform, cr, cc, cr, cc + 1)
    dy = _cell_distance_m(transform, cr, cc, cr + 1, cc)

    filled = _fill_depressions(dem)
    fdir   = _d8_flowdir(filled, dx, dy)
    accum  = _flow_accum(filled, fdir)
    slope  = _slope_raster(dem, dx, dy)

    r, c = _rowcol(transform, lon, lat)
    r, c = _snap_to_valid(dem, r, c)
    r, c = _snap_to_channel(accum, r, c)

    mask   = _delineate_catchment(fdir, r, c)
    zones  = _start_zones(mask, slope, dem)
    rings  = _catchment_polygon(mask, transform)
    area   = float(mask.sum()) * (dx * dy) / 1e6
    debris_ll = _xy(transform, r, c)

    # Step-up reference (single-thread; can leave the gully) — QA display only.
    up, _ = _trace_upstream(dem, fdir, accum, slope, r, c)
    uptrace = [_xy(transform, rr, cc) for rr, cc in up]

    base = PathGeometry(
        catchment_area_km2=area, num_start_zones=len(zones), catchment_rings=rings,
        debris_lonlat=debris_ll, uptrace_lonlat=uptrace,
    )

    # Governing start zone = the highest steep cell available, from either the
    # watershed clusters (best when debris is low, with tributaries above) or the
    # up-trace (best when debris is mapped high, near its own start zone). We take
    # only the start *cell* from the up-trace, then route downstream — so even if
    # the up-trace wandered, the final path stays in the gully.
    candidates: list[tuple[float, int, int]] = []
    if zones:
        gr, gc, gelev, _ = max(zones, key=lambda z: z[2])
        candidates.append((gelev, gr, gc))
    hs = _highest_steep(up, slope, dem)
    if hs is not None:
        candidates.append(hs)
    if not candidates:
        return base                        # no steep release terrain found

    _, gr, gc = max(candidates)
    gpath, reached_valley = _trace_downstream(dem, fdir, slope, gr, gc)
    geom = _build_geometry(dem, transform, gpath, reached_valley)
    geom.catchment_area_km2 = area
    geom.num_start_zones = len(zones)
    geom.catchment_rings = rings
    geom.debris_lonlat = debris_ll
    geom.uptrace_lonlat = uptrace
    return geom


def _highest_steep(cells, slope, dem) -> tuple[float, int, int] | None:
    """Highest (elev, r, c) among `cells` with slope ≥ STEEP_DEG, or None."""
    steep = [(float(dem[r, c]), r, c) for r, c in cells if slope[r, c] >= STEEP_DEG]
    return max(steep) if steep else None


def _delineate_catchment(fdir: np.ndarray, r0: int, c0: int) -> np.ndarray:
    """Boolean mask of every cell that drains to (r0, c0) — the reverse watershed."""
    ny, nx = fdir.shape
    mask = np.zeros(fdir.shape, dtype=bool)
    mask[r0, c0] = True
    stack = [(r0, c0)]
    while stack:
        r, c = stack.pop()
        for k, (dr, dc) in enumerate(_NB):
            nr, nc = r + dr, c + dc
            if 0 <= nr < ny and 0 <= nc < nx and not mask[nr, nc] \
                    and fdir[nr, nc] == _OPP[k]:   # neighbour flows into (r, c)
                mask[nr, nc] = True
                stack.append((nr, nc))
    return mask


def _start_zones(mask, slope, dem) -> list[tuple[int, int, float, int]]:
    """
    Connected steep (≥ STEEP_DEG) clusters within the catchment.

    Returns (top_row, top_col, top_elev, n_cells) per cluster, dropping clusters
    smaller than MIN_STARTZONE_CELLS. The 'top' cell is the cluster's highest.
    """
    from scipy.ndimage import label

    steep = mask & (slope >= STEEP_DEG)
    lbl, n = label(steep, structure=np.ones((3, 3)))  # type: ignore[misc]
    zones = []
    for i in range(1, n + 1):
        cells = np.argwhere(lbl == i)
        if len(cells) < MIN_STARTZONE_CELLS:
            continue
        elevs = dem[cells[:, 0], cells[:, 1]]
        top = cells[int(np.argmax(elevs))]
        zones.append((int(top[0]), int(top[1]), float(np.max(elevs)), len(cells)))
    return zones


def _catchment_polygon(mask: np.ndarray, transform) -> list[list[tuple[float, float]]]:
    """Polygonise the catchment mask into rings of (lon, lat) for mapping."""
    from rasterio import features

    rings = []
    for shape in features.shapes(mask.astype("uint8"), mask=mask, transform=transform):
        poly, val = shape
        if val != 1:
            continue
        for ring in poly["coordinates"]:
            rings.append([(float(x), float(y)) for x, y in ring])
    return rings


# ── flow routing ──────────────────────────────────────────────────────────────

def _fill_depressions(dem: np.ndarray) -> np.ndarray:
    """
    Priority-flood depression filling (Barnes et al. 2014) so every valid cell
    drains to the DEM boundary — required for well-defined flow routing.
    """
    import heapq

    ny, nx = dem.shape
    valid = np.isfinite(dem)
    filled = np.full(dem.shape, np.inf)
    visited = ~valid

    heap: list[tuple[float, int, int]] = []
    for r in range(ny):
        for c in range(nx):
            if not valid[r, c]:
                continue
            on_edge = r in (0, ny - 1) or c in (0, nx - 1)
            adj_nan = any(
                not valid[r + dr, c + dc]
                for dr, dc in _NB
                if 0 <= r + dr < ny and 0 <= c + dc < nx
            )
            if on_edge or adj_nan:
                filled[r, c] = dem[r, c]
                visited[r, c] = True
                heapq.heappush(heap, (dem[r, c], r, c))

    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in _NB:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ny and 0 <= nc < nx and not visited[nr, nc]:
                nz = max(dem[nr, nc], z)
                filled[nr, nc] = nz
                visited[nr, nc] = True
                heapq.heappush(heap, (nz, nr, nc))
    return filled


def _d8_flowdir(filled: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """D8 flow direction: index into _NB of each cell's steepest-descent neighbour (-1 = none)."""
    diag = math.hypot(dx, dy)
    dists = [diag, dy, diag, dx, dx, diag, dy, diag]  # per _NB entry

    best_slope = np.zeros(filled.shape)
    fdir = np.full(filled.shape, -1, dtype=np.int8)
    for k, (dr, dc) in enumerate(_NB):
        nb = _shift(filled, dr, dc)
        slope = (filled - nb) / dists[k]
        better = (slope > best_slope) & np.isfinite(nb) & np.isfinite(filled)
        best_slope = np.where(better, slope, best_slope)
        fdir = np.where(better, k, fdir)
    fdir[~np.isfinite(filled)] = -1
    return fdir


def _flow_accum(filled: np.ndarray, fdir: np.ndarray) -> np.ndarray:
    """Flow accumulation (cells drained through each cell), high→low push order."""
    ny, nx = filled.shape
    accum = np.ones(filled.shape)
    order = np.argsort(filled, axis=None)[::-1]  # descending; NaN lands first, skipped below
    for i in order:
        r, c = divmod(int(i), nx)
        if not np.isfinite(filled[r, c]):
            continue
        k = int(fdir[r, c])
        if k < 0:
            continue
        dr, dc = _NB[k]
        nr, nc = r + dr, c + dc
        if 0 <= nr < ny and 0 <= nc < nx:
            accum[nr, nc] += accum[r, c]
    return accum


def _snap_to_channel(accum: np.ndarray, r: int, c: int,
                     radius: int = CHANNEL_SNAP_RADIUS) -> tuple[int, int]:
    """Move the seed to the highest-accumulation (most channelised) cell nearby."""
    ny, nx = accum.shape
    best, best_a = (r, c), accum[r, c]
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < ny and 0 <= nc < nx and accum[nr, nc] > best_a:
                best_a, best = accum[nr, nc], (nr, nc)
    return best


def _trace_upstream(dem, fdir, accum, slope, r0, c0) -> tuple[list[tuple[int, int]], bool]:
    """
    Walk upstream from (r0, c0) along the main channel to the start zone.

    At each cell the candidate steps are the higher neighbours that flow *into*
    it; we take the one with the largest accumulation — the main stem, which
    keeps us on the concave thalweg rather than a side wall. The walk stops at:
      - the **channel head** (no uphill inflow, or contributing area <
        MIN_CHANNEL_ACCUM — beyond which routing is headwater noise), or
      - the **start-zone rollover**, where the mean slope over the last
        SLOPE_WINDOW cells drops below STEEP_DEG after passing through steep
        terrain.
    In both cases the sub-steep tail is trimmed so the terminus is the last
    steep cell — the top of the start zone.
    """
    ny, nx = dem.shape
    path = [(r0, c0)]
    visited = {(r0, c0)}
    r, c = r0, c0
    seen_steep = bool(slope[r0, c0] >= STEEP_DEG)
    recent = [float(slope[r0, c0])]

    for _ in range(MAX_STEPS):
        best, best_a = None, -1.0
        for k, (dr, dc) in enumerate(_NB):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ny and 0 <= nc < nx):
                continue
            if fdir[nr, nc] != _OPP[k]:          # neighbour must flow into (r, c)
                continue
            if not (dem[nr, nc] > dem[r, c]):     # and be uphill
                continue
            if (nr, nc) in visited:
                continue
            if accum[nr, nc] > best_a:
                best_a, best = accum[nr, nc], (nr, nc)

        if best is None or best_a < MIN_CHANNEL_ACCUM:
            return _trim_to_steep(path, slope, seen_steep), True   # channel head

        nr, nc = best
        s = float(slope[nr, nc])
        if s >= STEEP_DEG:
            seen_steep = True
        if abs(dem[nr, nc] - dem[r0, c0]) > MAX_VERTICAL_M:
            return _trim_to_steep(path, slope, seen_steep), False

        path.append((nr, nc))
        visited.add((nr, nc))
        recent.append(s)
        r, c = nr, nc

        if (seen_steep and len(recent) >= SLOPE_WINDOW
                and sum(recent[-SLOPE_WINDOW:]) / SLOPE_WINDOW < STEEP_DEG):
            return _trim_to_steep(path, slope, True), True          # start-zone rollover

    return path, False


def _trace_downstream(dem, fdir, slope, r0, c0) -> tuple[list[tuple[int, int]], bool]:
    """
    Walk downstream from (r0, c0) along the flow direction to the valley floor.

    Follows D8 flow to the point where the mean slope over VALLEY_RUN cells drops
    below VALLEY_DEG (the runout / valley floor), then trims the flat tail back
    to the first flattening cell. Also stops at a flow outlet / DEM edge or cap.
    This makes the geometry independent of whether the debris was mapped
    mid-track or already at the runout.
    """
    ny, nx = dem.shape
    path = [(r0, c0)]
    visited = {(r0, c0)}
    r, c = r0, c0
    recent: list[float] = [float(slope[r0, c0])]

    for _ in range(MAX_STEPS):
        k = int(fdir[r, c])
        if k < 0:
            return path, True                     # flow outlet / DEM edge
        dr, dc = _NB[k]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < ny and 0 <= nc < nx) or (nr, nc) in visited:
            return path, True

        path.append((nr, nc))
        visited.add((nr, nc))
        recent.append(float(slope[nr, nc]))
        r, c = nr, nc

        if (len(recent) >= VALLEY_RUN
                and sum(recent[-VALLEY_RUN:]) / VALLEY_RUN < VALLEY_DEG):
            # Trim to the first cell of the flat run: the valley floor.
            trim = 0
            while trim < VALLEY_RUN - 1 and slope[path[-1]] < VALLEY_DEG:
                path = path[:-1]
                trim += 1
            return path, True

    return path, False


def _trim_to_steep(path, slope, seen_steep):
    """Trim trailing sub-STEEP_DEG cells so the terminus is the last steep cell."""
    if not seen_steep:
        return path
    i = len(path) - 1
    while i > 0 and slope[path[i]] < STEEP_DEG:
        i -= 1
    return path[:i + 1]


def _build_geometry(dem, transform, path_rc, reached_valley) -> PathGeometry:
    """Assemble a PathGeometry (H, L, alpha, start-zone descriptors) from cells.

    `path_rc` runs start→valley; the start zone is the first cell, the valley
    floor the last.
    """
    lonlat = [_xy(transform, r, c) for r, c in path_rc]
    elev = [_elev(dem, r, c) for r, c in path_rc]

    start_ll, valley_ll = lonlat[0], lonlat[-1]
    h = elev[0] - elev[-1]
    l = _haversine_m(start_ll[1], start_ll[0], valley_ll[1], valley_ll[0])
    alpha = math.degrees(math.atan2(h, l)) if l > 0 else math.nan

    sr, sc = path_rc[0]
    sl, asp = _slope_aspect(dem, transform, sr, sc)

    return PathGeometry(
        lonlat=lonlat, elev=elev,
        vertical_drop=h, travel_distance=l, alpha=alpha,
        startzone_elev=elev[0], startzone_aspect=asp, startzone_slope=sl,
        reached_valley=reached_valley,
    )


def _slope_raster(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Per-cell terrain slope (degrees) from the DEM gradient."""
    gy, gx = np.gradient(dem, dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def _shift(z: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Return an array where out[r, c] = z[r+dr, c+dc], out-of-bounds → NaN."""
    out = np.full_like(z, np.nan)
    ny, nx = z.shape
    r0s, r1s = max(0, dr), ny + min(0, dr)
    c0s, c1s = max(0, dc), nx + min(0, dc)
    r0d, r1d = max(0, -dr), ny + min(0, -dr)
    c0d, c1d = max(0, -dc), nx + min(0, -dc)
    out[r0d:r1d, c0d:c1d] = z[r0s:r1s, c0s:c1s]
    return out


# ── features / atlas ──────────────────────────────────────────────────────────

def path_features(geom: PathGeometry) -> dict[str, float]:
    """
    Flatten a PathGeometry into the model-facing feature dict.

    Geometry is the reverse-watershed governing path (most-exposed start zone →
    valley floor) — the intended path-level covariate.
    """
    return {
        "vertical_drop":    geom.vertical_drop,
        "travel_distance":  geom.travel_distance,
        "alpha":            geom.alpha,
        "startzone_elev":   geom.startzone_elev,
        "startzone_aspect": geom.startzone_aspect,
        "startzone_slope":  geom.startzone_slope,
        "catchment_area_km2": geom.catchment_area_km2,
        "num_start_zones":    float(geom.num_start_zones),
    }


def build_path_atlas(observations: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """
    Trace a path for each observation and return a per-observation feature table.

    `observations` must have 'Latitude' and 'Longitude' columns. Unique
    coordinates are traced once and reused. The result is cached to
    `data/dem/path_atlas.csv`; pass use_cache=False to rebuild.
    """
    if use_cache and _ATLAS_CSV.exists():
        return pd.read_csv(_ATLAS_CSV)

    cols = ["Latitude", "Longitude"]
    coords = observations[cols].dropna().drop_duplicates()

    # Region-batched fetch for any points not already cached (few API calls).
    uncached = coords[~coords.apply(
        lambda r: dem_is_cached(r["Longitude"], r["Latitude"]), axis=1)]
    if len(uncached):
        try:
            prepare_regions(uncached)
        except RuntimeError:
            pass  # fall back to per-point / cached tiles

    feature_keys = list(path_features(PathGeometry(lonlat=[(0.0, 0.0)], elev=[0.0])))
    rows: list[dict[str, object]] = []
    for lat, lon in coords.itertuples(index=False):
        row: dict[str, object] = {"Latitude": lat, "Longitude": lon, "trace_error": ""}
        try:
            dem = fetch_dem(lon, lat)
            geom = trace_path(dem, lon, lat)
            row.update(path_features(geom))
        except (RuntimeError, rasterio.RasterioIOError, ValueError) as exc:
            row.update({k: math.nan for k in feature_keys})
            row["trace_error"] = str(exc)
        rows.append(row)

    atlas = pd.DataFrame(rows)
    DEM_CACHE.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(_ATLAS_CSV, index=False)
    return atlas


# ── raster / geometry helpers ─────────────────────────────────────────────────

def _smooth(dem: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian-smooth the DEM, preserving NaN by nan-aware normalisation."""
    mask = np.isnan(dem)
    if not mask.any():
        return gaussian_filter(dem, sigma)
    filled = np.where(mask, 0.0, dem)
    sm = gaussian_filter(filled, sigma)
    weight = gaussian_filter((~mask).astype("float64"), sigma)
    out = sm / np.where(weight == 0, np.nan, weight)
    out[mask] = np.nan
    return out


def _rowcol(transform, lon: float, lat: float) -> tuple[int, int]:
    col, row = ~transform * (lon, lat)
    return int(round(row)), int(round(col))


def _xy(transform, r: int, c: int) -> tuple[float, float]:
    lon, lat = transform * (c + 0.5, r + 0.5)
    return lon, lat


def _elev(dem: np.ndarray, r: int, c: int) -> float:
    return float(dem[r, c])


def _snap_to_valid(dem: np.ndarray, r: int, c: int, k: int = 3) -> tuple[int, int]:
    """If the seed cell is NaN, snap to the nearest valid cell within ±k."""
    if not math.isnan(_elev(dem, r, c)):
        return r, c
    for rad in range(1, k + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1] \
                        and not math.isnan(_elev(dem, nr, nc)):
                    return nr, nc
    raise ValueError("Seed coordinate falls in a NaN region of the DEM.")


def _cell_distance_m(transform, r0, c0, r1, c1) -> float:
    lon0, lat0 = _xy(transform, r0, c0)
    lon1, lat1 = _xy(transform, r1, c1)
    return _haversine_m(lat0, lon0, lat1, lon1)


def _slope_aspect(dem, transform, r, c) -> tuple[float, float]:
    """Slope (deg) and aspect (deg from N, clockwise) from a 3×3 neighbourhood."""
    if not (1 <= r < dem.shape[0] - 1 and 1 <= c < dem.shape[1] - 1):
        return math.nan, math.nan
    win = dem[r - 1:r + 2, c - 1:c + 2]
    if np.isnan(win).any():
        return math.nan, math.nan

    # metre resolution of one cell in x and y at this latitude
    dx = _cell_distance_m(transform, r, c, r, c + 1)
    dy = _cell_distance_m(transform, r, c, r + 1, c)
    dzdx = ((win[0, 2] + 2 * win[1, 2] + win[2, 2]) -
            (win[0, 0] + 2 * win[1, 0] + win[2, 0])) / (8 * dx)
    dzdy = ((win[2, 0] + 2 * win[2, 1] + win[2, 2]) -
            (win[0, 0] + 2 * win[0, 1] + win[0, 2])) / (8 * dy)

    slope = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
    aspect = math.degrees(math.atan2(dzdy, -dzdx))
    aspect = (90.0 - aspect) % 360.0
    return slope, aspect


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
