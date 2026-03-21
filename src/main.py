from pathlib import Path

import pandas as pd

from pro_reader import extract_pro_coordinates, find_nearest_pro, parse_snow_data
from visualization import create_avalanche_map, plot_interactive_stability

# ── season config ─────────────────────────────────────────────────────────────
# Change SEASON to switch between years. Expects:
#   data/simulations/<SEASON>/  *.pro files
#   data/observations/          observation CSVs
#   output/<SEASON>/            generated HTML (created automatically)

SEASON = "2025-2026"


def main():
    root_dir  = Path(__file__).resolve().parents[1]
    sim_dir   = root_dir / "data" / "simulations" / SEASON
    obs_dir   = root_dir / "data" / "observations"
    out_dir       = root_dir / "output" / SEASON
    stability_dir = out_dir / "stability"
    stability_dir.mkdir(parents=True, exist_ok=True)

    avalanche_csv = obs_dir / "Avalanche Information.csv"
    if not avalanche_csv.exists():
        print(f"CSV not found: {avalanche_csv}")
        return
    if not sim_dir.is_dir():
        print(f"Simulations directory not found: {sim_dir}")
        return

    # --- Load observations ---
    df = pd.read_csv(avalanche_csv)
    df = df.rename(columns={
        'Lat':   'Latitude',
        'Long':  'Longitude',
        'Place': 'Placemark Name',
    })
    df['date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%B %d, %Y').fillna('')

    print(f"Season:  {SEASON}")
    print(f"Loaded {len(df)} observations from {avalanche_csv.name}")

    # --- Match each observation to its nearest .pro file ---
    pro_cache: dict[Path, str] = {}

    def stability_html_for(pro_file: Path) -> str:
        """Returns the relative URL used by the map to link to this station's plot."""
        if pro_file not in pro_cache:
            pro_cache[pro_file] = f"stability/stability_{pro_file.stem}.html"
        return pro_cache[pro_file]

    target_urls:       list[str]        = []
    matched_pro_files: list[Path | None] = []
    pro_dates:         dict[Path, list]  = {}

    for _, row in df.iterrows():
        pro_file = find_nearest_pro(row['Latitude'], row['Longitude'], sim_dir)
        if pro_file is None:
            print(f"  WARNING: no .pro file found for {row['Placemark Name']} — skipping")
            target_urls.append("")
            matched_pro_files.append(None)
        else:
            coords = extract_pro_coordinates(pro_file)
            print(
                f"  {row['Placemark Name']:30s} → {pro_file.name}"
                + (f"  ({coords[0]:.4f}, {coords[1]:.4f})" if coords else "")
            )
            target_urls.append(stability_html_for(pro_file))
            matched_pro_files.append(pro_file)

            raw_date = pd.to_datetime(row.get('Date', ''), errors='coerce')
            if pd.notna(raw_date):
                pro_dates.setdefault(pro_file, [])
                if raw_date not in pro_dates[pro_file]:
                    pro_dates[pro_file].append(raw_date)

    df['target_url'] = target_urls

    # --- Parse and plot each unique matched .pro file ---
    unique_pro_files = {p for p in matched_pro_files if p is not None}
    if not unique_pro_files:
        print("No .pro files matched — stability plots skipped.")
    else:
        for pro_file in sorted(unique_pro_files):
            out_path   = stability_dir / f"stability_{pro_file.stem}.html"
            station_id = pro_file.stem
            dates      = sorted(pro_dates.get(pro_file, []))
            print(f"\nParsing {pro_file.name}...")
            plot_interactive_stability(
                parse_snow_data(pro_file), out_path, station_id, event_dates=dates
            )
            print(f"  → {out_path}")

    # --- Generate the map ---
    map_path = out_dir / "avalanche_map.html"
    print(f"\nGenerating map → {map_path}")
    create_avalanche_map(df, map_path)

    print("\nDone.")
    print(f"Open {map_path} and click any marker to view its nearest station's stability.")


if __name__ == "__main__":
    main()
