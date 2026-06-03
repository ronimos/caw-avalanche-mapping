#!/usr/bin/env python3
"""
Download 2024-2025 SNOWPACK simulation files from the remote server.

Only downloads files relevant to the avalanche observations — one .pro/.smet
pair per unique (nearest-station, aspect) combination found across all
avalanches_*.csv files.

Usage:
    uv run python scripts/sync_pro_files.py [--dry-run]

Requires:
    .env           — SSH_PORT=<port>
    ~/.ssh/my_key.pem — SSH private key
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snowpack_io import ASPECT_SUFFIX, _haversine_km

REMOTE_HOST  = "ron@home.mtnweather.info"
REMOTE_BASE  = "/hdd1/snowpack/output.caw.2024"
LOCAL_DIR    = ROOT / "data" / "simulations" / "2024-2025"
OBS_DIR      = ROOT / "data" / "observations"
STATIONS_CSV = ROOT / "data" / "snowpack_stations_locations.csv"
SSH_KEY      = Path(os.environ.get("SSH_KEY", str(Path.home() / ".ssh" / "my_key.pem"))).expanduser()


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


def rsync_file(base_id: str, suffix: str, ssh_port: str, dry_run: bool) -> bool:
    """rsync a single (base_id, suffix) .pro/.smet pair. Returns True on success."""
    stem        = f"{base_id}{suffix}_res"
    remote_dir  = f"{REMOTE_HOST}:{REMOTE_BASE}/{base_id}/"
    ssh_cmd     = f"ssh -i {SSH_KEY} -p {ssh_port} -o BatchMode=yes"

    cmd = [
        "rsync", "-az",
        "--include", f"{stem}.pro",
        "--include", f"{stem}.smet",
        "--exclude", "*",
        "-e", ssh_cmd,
        remote_dir,
        str(LOCAL_DIR) + "/",
    ]
    if dry_run:
        print("  [dry-run]", " ".join(cmd))
        return True

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: rsync failed for {stem}\n    {result.stderr.strip()}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print rsync commands without executing them.")
    args = parser.parse_args()

    env      = load_env()
    ssh_port = env.get("SSH_PORT", "").strip()
    if not ssh_port:
        sys.exit("ERROR: SSH_PORT is not set in .env")
    if env.get("SSH_KEY"):
        global SSH_KEY
        SSH_KEY = Path(env["SSH_KEY"]).expanduser()

    pairs = find_needed_pairs()
    print(f"Observations require {len(pairs)} (station, aspect) file pair(s):")
    for base_id, suffix in pairs:
        aspect_label = {v: k for k, v in ASPECT_SUFFIX.items()}.get(suffix, 'flat')
        print(f"  {base_id}{suffix}_res.pro  [{aspect_label}]")
    print()

    if not args.dry_run:
        if not SSH_KEY.exists():
            sys.exit(f"ERROR: SSH key not found at {SSH_KEY}")
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    ok = failed = 0
    for base_id, suffix in pairs:
        if rsync_file(base_id, suffix, ssh_port, dry_run=args.dry_run):
            ok += 1
        else:
            failed += 1

    print(f"\nDone: {ok} succeeded, {failed} failed.")
    if not args.dry_run:
        print(f"Files written to {LOCAL_DIR}")
        print("Run: uv run python src/main.py --retrain")


if __name__ == "__main__":
    main()
