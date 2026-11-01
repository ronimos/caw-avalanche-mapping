#!/usr/bin/env python3
"""
Copy SNOWPACK simulation files for one season into data/simulations/<SEASON>/.

Two selection modes:

  observations (default) — one .pro/.smet pair per unique (nearest-station,
      aspect) combination found across all avalanches_*.csv files.
  --focus KML            — the nearest station to every Point placemark in the
      KML, with all five files (flat + N/E/S/W) per station.  Station
      coordinates are read from the .pro headers of the source run, so the
      source must be a local directory.  A manifest
      (focus_stations.csv: point → station, distance) is written next to the
      copied files.

  --selection JSON       — Add / Remove edits exported from the focus map
      editor (model_locations.json), applied on top of the --focus baseline
      (or on top of nothing without --focus).  --prune then deletes .pro/.smet
      files in the season folder that are no longer selected.

The source is either a local run directory (--source /path/to/output.caw.YYYY)
or, when --source is omitted, the remote server over SSH (needs .env).

Usage:
    uv run python scripts/sync_pro_files.py [--dry-run]
    uv run python scripts/sync_pro_files.py --season 2026-2027 \\
        --source /mnt/HC_Volume_106760790/model/snowpack/runs/output.caw.2026 \\
        --focus data/FOCUS_Ava.kml --selection data/model_locations.json --prune
"""

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snowpack_io import ASPECT_SUFFIX, _haversine_km

REMOTE_HOST  = "ron@home.mtnweather.info"
REMOTE_BASE  = "/hdd1/snowpack/output.caw.2024"
OBS_DIR      = ROOT / "data" / "observations"
STATIONS_CSV = ROOT / "data" / "snowpack_stations_locations.csv"
SSH_KEY      = Path(os.environ.get("SSH_KEY", str(Path.home() / ".ssh" / "my_key.pem"))).expanduser()
KML_NS       = {"k": "http://www.opengis.net/kml/2.2"}
ALL_SUFFIXES = ["", *ASPECT_SUFFIX.values()]


def load_env() -> dict[str, str]:
    env_file = ROOT / ".env"
    if not env_file.exists():
        sys.exit("ERROR: .env not found. Copy .env.example to .env and set SSH_PORT.")
    env: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, _, v = line.partition('=')
            env[k.strip()] = v.strip()
    return env


def find_needed_pairs() -> list[tuple[str, str]]:
    """
    Return sorted list of (base_station_id, aspect_suffix) pairs that cover
    all observations.  aspect_suffix '' means the flat file.
    """
    stations = pd.read_csv(STATIONS_CSV)

    obs_frames = [
        pd.read_csv(p) for p in sorted(OBS_DIR.glob("avalanches_*.csv"))
    ]
    if not obs_frames:
        sys.exit("ERROR: No avalanches_*.csv files found in data/observations/")
    obs = pd.concat(obs_frames, ignore_index=True)

    needed: set[tuple[str, str]] = set()
    for _, row in obs.iterrows():
        try:
            lat = float(row['Lat'])
            lon = float(row['Long'])
        except (ValueError, KeyError):
            continue

        dists = stations.apply(
            lambda r: _haversine_km(lat, lon, float(r['Latitude']), float(r['Longitude'])),
            axis=1,
        )
        base_id = str(stations.loc[dists.idxmin(), 'Folder_Name'])
        suffix  = ASPECT_SUFFIX.get(str(row.get('Aspect', '')).strip().upper(), '')
        needed.add((base_id, suffix))

    return sorted(needed)


def read_kml_points(kml: Path) -> pd.DataFrame:
    """Name/lat/lon of every Point placemark; other geometries are skipped."""
    rows = []
    for pm in ET.parse(kml).getroot().iterfind(".//k:Placemark", KML_NS):
        coords = pm.find(".//k:Point/k:coordinates", KML_NS)
        if coords is None or not coords.text:
            continue
        lon, lat, *_ = (float(v) for v in coords.text.strip().split(","))
        name = (pm.findtext("k:name", default="", namespaces=KML_NS) or "").strip()
        rows.append({"point": name or f"unnamed_{len(rows)}", "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def read_run_stations(source: Path) -> pd.DataFrame:
    """Station id/lat/lon/elevation from the flat .pro header of each station folder."""
    rows = []
    for d in sorted(source.iterdir()):
        pro = d / f"{d.name}_res.pro"
        if not pro.is_file():
            continue
        with pro.open() as fh:
            head = fh.read(1500)
        vals = {k: re.search(rf"^{k}\s*=\s*(\S+)", head, re.M) for k in ("Latitude", "Longitude", "Altitude")}
        if not all(vals.values()):
            print(f"  WARNING: no coordinates in {pro}")
            continue
        rows.append({
            "station": d.name,
            "lat": float(vals["Latitude"].group(1)),
            "lon": float(vals["Longitude"].group(1)),
            "elevation": float(vals["Altitude"].group(1)),
        })
    return pd.DataFrame(rows)


def match_focus_points(kml: Path, source: Path) -> pd.DataFrame:
    """Nearest run station for every KML point (one row per point)."""
    points   = read_kml_points(kml)
    stations = read_run_stations(source)
    rows = []
    for p in points.itertuples():
        dists = [_haversine_km(p.lat, p.lon, s.lat, s.lon) for s in stations.itertuples()]
        i = min(range(len(dists)), key=dists.__getitem__)
        s = stations.iloc[i]
        rows.append({
            "point": p.point, "point_lat": p.lat, "point_lon": p.lon,
            "station": s.station, "station_lat": s.lat, "station_lon": s.lon,
            "station_elevation": s.elevation, "distance_km": round(dists[i], 2),
        })
    return pd.DataFrame(rows)


ASPECT_NAME_SUFFIX = {"Flat": "", **ASPECT_SUFFIX}


def apply_selection(pairs: set[tuple[str, str]], path: Path) -> set[tuple[str, str]]:
    """
    Apply model_locations.json edits to a set of (station, suffix) pairs:
    'add' adds the listed aspects of a station, 'remove' drops them.
    """
    doc = json.loads(path.read_text())
    if doc.get("schema") != "caw-model-locations/1":
        sys.exit(f"ERROR: {path} is not a caw-model-locations/1 file")
    out = set(pairs)
    n_add = n_rm = 0
    for rec in doc.get("selections", []):
        sid, action = str(rec["station"]), rec["action"]
        unknown = set(rec["aspects"]) - ASPECT_NAME_SUFFIX.keys()
        if action not in ("add", "remove") or unknown:
            sys.exit(f"ERROR: bad selection for station {sid}: action={action!r} aspects={rec['aspects']}")
        for aspect in rec["aspects"]:
            pair = (sid, ASPECT_NAME_SUFFIX[aspect])
            if action == "add" and pair not in out:
                out.add(pair); n_add += 1
            elif action == "remove" and pair in out:
                out.discard(pair); n_rm += 1
    print(f"Selection {path.name}: {len(doc.get('selections', []))} edit(s) → "
          f"+{n_add} / -{n_rm} file pair(s)")
    return out


def prune(local_dir: Path, pairs: list[tuple[str, str]], dry_run: bool) -> None:
    """Delete .pro/.smet files in local_dir that are not in pairs."""
    keep = {f"{sid}{sfx}_res{ext}" for sid, sfx in pairs for ext in (".pro", ".smet")}
    stale = sorted(p for p in local_dir.glob("*_res.*")
                   if p.suffix in (".pro", ".smet") and p.name not in keep)
    print(f"Prune: {len(stale)} file(s) no longer selected"
          + (" (dry run — not deleted)" if dry_run else ""))
    for p in stale:
        print(f"  {'[dry-run] ' if dry_run else ''}rm {p.name}")
        if not dry_run:
            p.unlink()


def rsync_file(base_id: str, suffix: str, src_base: str, local_dir: Path,
               ssh_cmd: str | None, dry_run: bool) -> bool:
    """rsync a single (base_id, suffix) .pro/.smet pair. Returns True on success."""
    stem = f"{base_id}{suffix}_res"
    src  = f"{REMOTE_HOST}:{src_base}/{base_id}/" if ssh_cmd else f"{src_base}/{base_id}/"

    cmd = [
        "rsync", "-a" if ssh_cmd is None else "-az",
        "--include", f"{stem}.pro",
        "--include", f"{stem}.smet",
        "--exclude", "*",
        *(["-e", ssh_cmd] if ssh_cmd else []),
        src,
        str(local_dir) + "/",
    ]
    if dry_run:
        print("  [dry-run]", " ".join(cmd))
        return True

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: rsync failed for {stem}\n    {result.stderr.strip()}")
        return False
    if not (local_dir / f"{stem}.pro").exists():
        print(f"  WARNING: {stem}.pro not found in source")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", default="2024-2025",
                        help="Destination season folder under data/simulations/ (default: 2024-2025).")
    parser.add_argument("--source", type=Path,
                        help="Local SNOWPACK run directory. Omit to pull from the remote server over SSH.")
    parser.add_argument("--focus", type=Path,
                        help="KML of point locations: copy the nearest station (all aspects) to each point.")
    parser.add_argument("--selection", type=Path,
                        help="model_locations.json from the focus map editor: Add/Remove station aspects.")
    parser.add_argument("--prune", action="store_true",
                        help="Delete .pro/.smet files in the season folder that are not selected.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print rsync commands without executing them.")
    args = parser.parse_args()

    local_dir = ROOT / "data" / "simulations" / args.season

    if args.source is None:
        env      = load_env()
        ssh_port = env.get("SSH_PORT", "").strip()
        if not ssh_port:
            sys.exit("ERROR: SSH_PORT is not set in .env")
        if env.get("SSH_KEY"):
            global SSH_KEY
            SSH_KEY = Path(env["SSH_KEY"]).expanduser()
        if not args.dry_run and not SSH_KEY.exists():
            sys.exit(f"ERROR: SSH key not found at {SSH_KEY}")
        ssh_cmd: str | None = f"ssh -i {SSH_KEY} -p {ssh_port} -o BatchMode=yes"
        src_base = REMOTE_BASE
    else:
        if not args.source.is_dir():
            sys.exit(f"ERROR: source directory not found: {args.source}")
        ssh_cmd  = None
        src_base = str(args.source.resolve())

    if args.focus:
        if args.source is None:
            sys.exit("ERROR: --focus needs a local --source (station coordinates come from .pro headers)")
        matches = match_focus_points(args.focus, args.source)
        print(f"{len(matches)} point(s) in {args.focus} → {matches['station'].nunique()} nearest station(s); "
              f"distance median {matches['distance_km'].median():.1f} km, max {matches['distance_km'].max():.1f} km")
        pairs = sorted((sid, sfx) for sid in matches["station"].unique() for sfx in ALL_SUFFIXES)
    elif args.selection:
        matches = None
        pairs   = []
    else:
        matches = None
        pairs   = find_needed_pairs()
        print(f"Observations require {len(pairs)} (station, aspect) file pair(s):")
        for base_id, suffix in pairs:
            aspect_label = {v: k for k, v in ASPECT_SUFFIX.items()}.get(suffix, 'flat')
            print(f"  {base_id}{suffix}_res.pro  [{aspect_label}]")
    if args.selection:
        if args.source is None:
            sys.exit("ERROR: --selection needs a local --source")
        pairs = sorted(apply_selection(set(pairs), args.selection))
        missing = [sid for sid in {s for s, _ in pairs} if not (args.source / sid).is_dir()]
        if missing:
            sys.exit(f"ERROR: station(s) not in {args.source}: {', '.join(sorted(missing))}")
        print(f"Model set: {len({s for s, _ in pairs})} station(s), {len(pairs)} file pair(s)")
    print()

    if not args.dry_run:
        local_dir.mkdir(parents=True, exist_ok=True)
        if matches is not None:
            manifest = local_dir / "focus_stations.csv"
            matches.to_csv(manifest, index=False)
            print(f"Wrote {manifest}")

    ok = failed = 0
    for base_id, suffix in pairs:
        if rsync_file(base_id, suffix, src_base, local_dir, ssh_cmd, dry_run=args.dry_run):
            ok += 1
        else:
            failed += 1

    print(f"\nDone: {ok} succeeded, {failed} failed.")
    if args.prune:
        if failed:
            print("Prune skipped: some copies failed.")
        else:
            prune(local_dir, pairs, args.dry_run)
    if not args.dry_run:
        print(f"Files written to {local_dir}")


if __name__ == "__main__":
    main()
