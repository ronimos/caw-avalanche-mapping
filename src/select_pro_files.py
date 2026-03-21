#!/usr/bin/env python3
"""
select_pro_files.py

Run on the remote server. For every avalanche observation in the CSV, finds
the closest .pro file in PRO_DIR whose SlopeAzi matches the observed aspect,
then copies the unique matched files to OUT_DIR.

Aspect matching: the .pro file's SlopeAzi must be within ±45° of the
cardinal/intercardinal direction given in the CSV. If no aspect-matched file
exists for an observation, falls back to nearest regardless of aspect and
prints a warning.

Usage (defaults shown):
    python select_pro_files.py \
        --csv   "Avalanche Information.csv" \
        --pro-dir /hdd1/snowpack/output.caw.2025 \
        --out-dir /home/ron/temp
"""

import argparse
import csv
import math
import re
import shutil
from pathlib import Path

# ── defaults ─────────────────────────────────────────────────────────────────
PRO_DIR_DEFAULT = "/hdd1/snowpack/output.caw.2025"
OUT_DIR_DEFAULT = "/home/ron/temp"
CSV_DEFAULT     = "Avalanche Information.csv"

ASPECT_TOLERANCE_DEG = 45.0   # half-width of matching window

ASPECT_TO_AZI = {
    "N":   0.0,   "NNE":  22.5, "NE":  45.0, "ENE":  67.5,
    "E":  90.0,   "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0,   "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0,   "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}


# ── geometry helpers ──────────────────────────────────────────────────────────

def _azi_diff(a: float, b: float) -> float:
    """Smallest angular difference between two azimuths (degrees, 0-360)."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _aspect_matches(slope_azi: float, aspect_str: str) -> bool:
    """True if slope_azi is within ASPECT_TOLERANCE_DEG of aspect_str."""
    key = aspect_str.strip().upper()
    if key not in ASPECT_TO_AZI:
        return True  # unknown string → don't filter out
    return _azi_diff(slope_azi, ASPECT_TO_AZI[key]) <= ASPECT_TOLERANCE_DEG


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── .pro header parsing ───────────────────────────────────────────────────────

_LAT_RE = re.compile(r"(?i)latitude\s*=\s*([-\d.]+)")
_LON_RE = re.compile(r"(?i)longitude\s*=\s*([-\d.]+)")
_AZI_RE = re.compile(r"(?i)slopeazi\s*=\s*([-\d.]+)")


def _parse_pro_header(path: Path):
    """Return (lat, lon, slope_azi) or None if any field is missing."""
    lat = lon = azi = None
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if line.startswith("[DATA]"):
                    break
                if lat is None:
                    m = _LAT_RE.search(line)
                    if m:
                        lat = float(m.group(1))
                if lon is None:
                    m = _LON_RE.search(line)
                    if m:
                        lon = float(m.group(1))
                if azi is None:
                    m = _AZI_RE.search(line)
                    if m:
                        azi = float(m.group(1))
                if lat is not None and lon is not None and azi is not None:
                    break
    except OSError as e:
        print(f"  WARN cannot read {path.name}: {e}")
        return None

    if lat is None or lon is None or azi is None:
        return None
    return lat, lon, azi


def _build_index(pro_dir: Path) -> list:
    """Scan pro_dir for *.pro files; return list of (path, lat, lon, azi)."""
    index = []
    skipped = 0
    for p in sorted(pro_dir.rglob("*.pro")):
        result = _parse_pro_header(p)
        if result is None:
            skipped += 1
        else:
            index.append((p, *result))
    if skipped:
        print(f"  ({skipped} files skipped — could not parse header)")
    return index


# ── matching ──────────────────────────────────────────────────────────────────

def _find_best(lat: float, lon: float, aspect: str, index: list):
    """
    Returns (path, dist_km, aspect_matched).
    Prefers closest aspect-matching file; falls back to closest overall.
    """
    scored = sorted(
        ((p, _haversine_km(lat, lon, plat, plon), pazi) for p, plat, plon, pazi in index),
        key=lambda x: x[1],
    )

    for path, dist, pazi in scored:
        if _aspect_matches(pazi, aspect):
            return path, dist, True

    # fallback
    if scored:
        path, dist, _ = scored[0]
        return path, dist, False

    return None, None, None


# ── CSV loading ───────────────────────────────────────────────────────────────

def _load_csv(csv_path: Path) -> list:
    obs = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            try:
                obs.append({
                    "name":   (row.get("Place") or row.get("Placemark Name") or f"row-{i}").strip(),
                    "lat":    float(row.get("Lat") or row.get("Latitude")),
                    "lon":    float(row.get("Long") or row.get("Longitude")),
                    "aspect": (row.get("Aspect") or "").strip(),
                    "date":   (row.get("Date") or "").strip(),
                })
            except (TypeError, ValueError):
                print(f"  WARN skipping row {i} — missing/invalid lat or lon")
    return obs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Select closest aspect-matched .pro files for each avalanche observation."
    )
    ap.add_argument("--csv",     default=CSV_DEFAULT,     help="Path to avalanche observations CSV")
    ap.add_argument("--pro-dir", default=PRO_DIR_DEFAULT, help="Directory containing .pro files")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help="Destination directory for copies")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    pro_dir  = Path(args.pro_dir)
    out_dir  = Path(args.out_dir)

    print(f"CSV:     {csv_path}")
    print(f"PRO dir: {pro_dir}")
    print(f"Out dir: {out_dir}\n")

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1
    if not pro_dir.is_dir():
        print(f"ERROR: PRO dir not found: {pro_dir}")
        return 1

    observations = _load_csv(csv_path)
    print(f"Loaded {len(observations)} observations\n")

    print(f"Indexing .pro files in {pro_dir} ...")
    index = _build_index(pro_dir)
    print(f"Indexed {len(index)} .pro files\n")

    if not index:
        print("No .pro files found — aborting.")
        return 1

    to_copy: dict[Path, list[str]] = {}

    print(f"{'Observation':<38} {'File':<30} {'km':>7}  {'Aspect'}")
    print("-" * 85)
    for obs in observations:
        path, dist, matched = _find_best(obs["lat"], obs["lon"], obs["aspect"], index)
        if path is None:
            print(f"  {'NO MATCH':<38} {obs['name']}")
            continue

        aspect_note = obs["aspect"] if matched else f"{obs['aspect']} [fallback]"
        print(f"  {obs['name']:<36} {path.name:<30} {dist:>6.1f}  {aspect_note}")

        to_copy.setdefault(path, []).append(obs["name"])

    print(f"\n{len(to_copy)} unique .pro file(s) matched.")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Copying to {out_dir} ...")
    for pro_path in sorted(to_copy):
        dest = out_dir / pro_path.name
        shutil.copy2(pro_path, dest)
        print(f"  {pro_path.name}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
