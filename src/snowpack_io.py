"""
snowpack_io.py — SNOWPACK file I/O.

Parses both .pro and .smet simulation output files, and provides
coordinate-based station matching.

Public API:
  parse_snow_data          .pro  → long-format layer DataFrame
  parse_smet               .smet → time-series DataFrame
  extract_pro_coordinates  .pro header → (lat, lon)
  find_nearest_pro         (lat, lon) → nearest .pro Path
"""

import functools
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    Record codes used:
        0500  — timestep date
        0501  — per-layer heights bottom→top (cm from ground)
        0532  — per-layer Sn38 values (same order as 0501)
    """
    MIN_BURIAL_DEPTH = 20.0  # cm

    rows: list[dict[str, Any]] = []
    cur: dict[str, Any] = {'timestamp': None, 'layer_heights': None, 'sn38_values': None}

    with open(file_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if not parts or not parts[0].isdigit():
                continue

            code = parts[0]

            if code == '0500' and parts[1] != 'Date':
                if cur['timestamp'] and cur['layer_heights'] and cur['sn38_values']:
                    rows.extend(_expand_layers(cur, MIN_BURIAL_DEPTH))
                cur = {'timestamp': parts[1], 'layer_heights': None, 'sn38_values': None}

            elif code == '0501' and parts[1] != 'nElems':
                try:
                    n = int(parts[1])
                    cur['layer_heights'] = [float(x) for x in parts[2:2 + n]]
                except (ValueError, IndexError):
                    pass

            elif code == '0532' and parts[1] != 'nElems':
                try:
                    n = int(parts[1])
                    cur['sn38_values'] = [float(x) for x in parts[2:2 + n]]
                except (ValueError, IndexError):
                    pass

    # Flush last timestep
    if cur['timestamp'] and cur['layer_heights'] and cur['sn38_values']:
        rows.extend(_expand_layers(cur, MIN_BURIAL_DEPTH))

    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df.set_index('timestamp', inplace=True)
    return df


def _expand_layers(cur: dict[str, Any], min_burial: float) -> list[dict[str, Any]]:
    """Expand a single timestep dict into one row dict per qualifying layer."""
    heights = cur['layer_heights']
    sn38s   = cur['sn38_values']
    if len(heights) != len(sn38s):
        return []

    total_height = max(heights)
    return [
        {
            'timestamp':    cur['timestamp'],
            'total_height': total_height,
            'layer_z':      z,
            'burial_depth': total_height - z,
            'sn38':         sn,
        }
        for z, sn in zip(heights, sn38s)
        if (total_height - z) >= min_burial
    ]


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
