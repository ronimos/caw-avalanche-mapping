"""
snowpack_io.py — SNOWPACK file I/O.

Parses both .pro and .smet simulation output files, and provides
coordinate-based station matching.

.pro parsing delegates to xsnow (https://gitlab.com/avacollabra/postprocessing/xsnow),
reshaped back to this project's long-format layer DataFrame. .smet parsing stays
hand-rolled: xsnow's unit registry rejects our files (dIntEnergySnow declared in
kJ/m2, xsnow accepts only W m-2), and even unit_validation="light" raises.

Public API:
  parse_snow_data          .pro  → long-format layer DataFrame
  parse_smet               .smet → time-series DataFrame
  extract_pro_coordinates  .pro header → (lat, lon)
  find_nearest_pro         (lat, lon) → nearest .pro Path
"""

import functools
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xsnow

# xsnow logs an INFO line per file read; keep pipeline output clean.
_QUIET_LOGGER = logging.getLogger('snowpack_io.xsnow')
_QUIET_LOGGER.setLevel(logging.WARNING)

# Aspect letter → numeric suffix appended to base station ID in .pro filenames.
# Flat terrain has no suffix (empty string).
ASPECT_SUFFIX: dict[str, str] = {
    'N': '1', 'E': '2', 'S': '3', 'W': '4',
}


# ── .pro parsing ──────────────────────────────────────────────────────────────

def parse_snow_data(file_path: str | Path) -> pd.DataFrame:
    """
    Parse a SNOWPACK .pro file into a long-format DataFrame with one row
    per layer per timestep.

    Columns:
        timestamp     — datetime index
        total_height  — snowpack surface height (cm), top of uppermost layer
        layer_z       — height of this layer from ground (cm)
        burial_depth  — total_height − layer_z (cm); depth below surface
        sn38          — natural stability index Sn38 for this layer

    Layers shallower than 20 cm burial depth are excluded.

    Parsing is delegated to xsnow, which returns layer heights and HS in
    metres; values are converted back to the cm convention used throughout
    this project.
    """
    MIN_BURIAL_DEPTH = 20.0  # cm

    # add_surface_sh_as_layer=False: don't synthesize a 1 mm surface-hoar layer
    # from record 0514 — total_height must stay the top of the uppermost real
    # layer, and the synthetic layer shifts burial depths at the 20 cm cutoff.
    xs = xsnow.read(str(file_path), logger=_QUIET_LOGGER,
                    add_surface_sh_as_layer=False)
    if xs is None:
        return pd.DataFrame(
            columns=['total_height', 'layer_z', 'burial_depth', 'sn38'],
            index=pd.DatetimeIndex([], name='timestamp'),
        )

    # Single-station file: collapse the singleton location/slope/realization dims,
    # leaving (time, layer) arrays padded with NaN beyond each timestep's layer count.
    ds = xs.data.squeeze(['location', 'slope', 'realization'])

    layer_z = ds['height'].values.astype(np.float64) * 100.0  # m → cm, layer top from ground
    sn38    = ds['sn38'].values.astype(np.float64)
    # total_height = top of uppermost layer, matching the previous parser
    # (not the .pro HS record, which can differ at the last decimals).
    with np.errstate(all='ignore'):
        total = np.nanmax(layer_z, axis=1)

    n_layer = layer_z.shape[1]
    times = np.repeat(ds['time'].values, n_layer)
    total_height = np.repeat(total, n_layer)
    layer_z = layer_z.ravel()
    sn38 = sn38.ravel()
    burial_depth = total_height - layer_z

    # 0.001 cm tolerance: xsnow stores heights as float32 metres, so a layer at
    # exactly 20.00 cm burial can land a rounding error below the threshold.
    # .pro heights have 0.01 cm resolution, so this cannot admit extra layers.
    keep = ~np.isnan(layer_z) & (burial_depth >= MIN_BURIAL_DEPTH - 1e-3)

    df = pd.DataFrame(
        {
            'timestamp':    times[keep],
            'total_height': total_height[keep],
            'layer_z':      layer_z[keep],
            'burial_depth': burial_depth[keep],
            'sn38':         sn38[keep],
        }
    )
    df.set_index('timestamp', inplace=True)
    return df


# ── .smet parsing ─────────────────────────────────────────────────────────────

def parse_smet(path: Path) -> pd.DataFrame:
    """
    Parse a SNOWPACK .smet file into a DataFrame indexed by timestamp.

    Applies units_multiplier to each field.  Temperature fields (TA, TSS_mod,
    T_bottom) are left in °C — the 273.15 offset that converts to Kelvin for
    SNOWPACK's internal physics is intentionally NOT applied here.
    """
    fields:      list[str]   = []
    multipliers: list[float] = []
    nodata   = -999.0
    rows:    list[list[str]] = []
    in_data  = False

    with open(path, errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('fields'):
                fields = line.split('=', 1)[1].strip().split()
            elif line.startswith('units_multiplier'):
                multipliers = [float(x) for x in line.split('=', 1)[1].strip().split()]
            elif line.startswith('nodata'):
                nodata = float(line.split('=', 1)[1].strip())
            elif line == '[DATA]':
                in_data = True
            elif in_data and line:
                rows.append(line.split())

    if not fields or not rows:
        return pd.DataFrame()

    data_fields = fields[1:]  # fields[0] == 'timestamp'
    records = []
    for parts in rows:
        if len(parts) < len(fields):
            continue
        record: dict = {'timestamp': pd.Timestamp(parts[0])}
        for i, fname in enumerate(data_fields, start=1):
            try:
                v = float(parts[i])
            except (ValueError, IndexError):
                record[fname] = np.nan
                continue
            if v == nodata:
                v = np.nan
            elif i < len(multipliers):
                v = v * multipliers[i]
            record[fname] = v
        records.append(record)

    return pd.DataFrame(records).set_index('timestamp')


# ── station matching ──────────────────────────────────────────────────────────

def extract_pro_coordinates(pro_file: Path) -> tuple[float, float] | None:
    """
    Extract (latitude, longitude) from a SNOWPACK .pro file header.
    Returns None if coordinates cannot be found.
    """
    lat, lon = None, None
    lat_re = re.compile(r'(?i)latitude\s*=\s*([-\d.]+)')
    lon_re = re.compile(r'(?i)longitude\s*=\s*([-\d.]+)')

    with open(pro_file) as f:
        for line in f:
            if lat is None:
                m = lat_re.search(line)
                if m:
                    lat = float(m.group(1))
            if lon is None:
                m = lon_re.search(line)
                if m:
                    lon = float(m.group(1))
            if lat is not None and lon is not None:
                break

    return (lat, lon) if lat is not None and lon is not None else None


def find_nearest_pro(
    lat: float,
    lon: float,
    sim_dir: Path,
    aspect: str = '',
    stations_csv: Path | None = None,
) -> Path | None:
    """
    Return the .pro file in sim_dir that best matches (lat, lon, aspect).

    When stations_csv is provided (preferred): find the nearest base station
    by haversine distance, then return the aspect-specific file
    ({base_id}{suffix}_res.pro).  Falls back to the flat file
    ({base_id}_res.pro) when the aspect variant is missing.

    When stations_csv is not provided: scan all .pro file headers for
    coordinates and return the nearest file (aspect-unaware legacy behaviour).
    """
    if stations_csv is not None and stations_csv.exists():
        return _find_nearest_pro_from_csv(lat, lon, aspect, sim_dir, stations_csv)

    best_file, best_dist = None, float('inf')
    for pro_file in sim_dir.glob('*.pro'):
        coords = extract_pro_coordinates(pro_file)
        if coords is None:
            continue
        dist = _haversine_km(lat, lon, coords[0], coords[1])
        if dist < best_dist:
            best_dist, best_file = dist, pro_file
    return best_file


@functools.lru_cache(maxsize=4)
def _load_stations(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _find_nearest_pro_from_csv(
    lat: float,
    lon: float,
    aspect: str,
    sim_dir: Path,
    stations_csv: Path,
) -> Path | None:
    """
    Find nearest base station from CSV that has simulation files in sim_dir,
    then return the aspect-specific .pro path.

    Walks stations in ascending distance order so that stations without any
    .pro files in sim_dir are skipped transparently.
    """
    stations = _load_stations(stations_csv)
    suffix   = ASPECT_SUFFIX.get(str(aspect).strip().upper(), '')

    dists = stations.apply(
        lambda r: _haversine_km(lat, lon, float(r['Latitude']), float(r['Longitude'])),
        axis=1,
    )

    for idx in dists.argsort():
        base_id = str(stations.loc[idx, 'Folder_Name'])
        # Prefer exact aspect match, then flat, then any variant for this station
        for stem in [f"{base_id}{suffix}_res", f"{base_id}_res"]:
            path = sim_dir / f"{stem}.pro"
            if path.exists():
                return path
        matches = sorted(sim_dir.glob(f"{base_id}*_res.pro"))
        if matches:
            return matches[0]

    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))
