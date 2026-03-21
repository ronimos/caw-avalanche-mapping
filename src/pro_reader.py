"""
pro_reader.py — SNOWPACK .pro file I/O.

Provides:
  - extract_pro_coordinates  Read (lat, lon) from a .pro header.
  - find_nearest_pro         Match a coordinate to the closest .pro file.
  - parse_snow_data          Parse a .pro file into a long-format DataFrame.
"""

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ── header parsing ────────────────────────────────────────────────────────────

def extract_pro_coordinates(pro_file: Path) -> tuple[float, float] | None:
    """
    Extracts (latitude, longitude) from a SNOWPACK .pro file header.
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


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def find_nearest_pro(lat: float, lon: float, data_dir: Path) -> Path | None:
    """
    Returns the .pro file in data_dir whose header coordinates are closest
    to (lat, lon). Returns None if no .pro files with parseable coords exist.
    """
    best_file, best_dist = None, float('inf')
    for pro_file in data_dir.glob('*.pro'):
        coords = extract_pro_coordinates(pro_file)
        if coords is None:
            continue
        dist = _haversine_km(lat, lon, coords[0], coords[1])
        if dist < best_dist:
            best_dist, best_file = dist, pro_file
    return best_file


# ── data parsing ──────────────────────────────────────────────────────────────

def parse_snow_data(file_path: str | Path) -> pd.DataFrame:
    """
    Parses a SNOWPACK .pro file into a long-format DataFrame with one row
    per layer per timestep.

    Columns:
        timestamp     - datetime index
        total_height  - snowpack surface height (cm), top of uppermost layer
        layer_z       - height of this layer from ground (cm)
        burial_depth  - total_height - layer_z (cm); depth below surface
        sn38          - natural stability index Sn38 for this layer

    Layers shallower than 20 cm burial depth are excluded.
    Record codes used:
        0500  - timestep date
        0501  - per-layer heights bottom→top (cm from ground)
        0532  - per-layer Sn38 values (same order as 0501)
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
    """Expands a single timestep dict into one row dict per qualifying layer."""
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
