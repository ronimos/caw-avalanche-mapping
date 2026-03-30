import argparse
from pathlib import Path

import joblib
import pandas as pd

from classifier import predict_proba_series, train_station
from features import build_daily_features
from snowpack_io import find_nearest_pro, parse_snow_data
from visualization import create_avalanche_map, plot_interactive_stability

# ── season config ─────────────────────────────────────────────────────────────
# SEASON selects the simulation data used for prediction and stability plots.
# Models are trained on ALL seasons combined and saved without a season prefix.
#
# Data layout:
#   data/simulations/<SEASON>/*.pro  — SNOWPACK station outputs per season
#   data/observations/               — multi-year observation CSV
#   models/                          — per-station classifiers (all seasons)
#   output/assets/<SEASON>/          — stability HTML plots

SEASON = "2025-2026"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Avalanche stability analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--forecast-date", default="2026-01-23", metavar="YYYY-MM-DD",
        help="Forecast reference date. Map markers are coloured by max probability "
             "in the window [date−1d, date+2d]; stability plots highlight the same window.",
    )
    p.add_argument(
        "--retrain", action="store_true",
        help="Force retraining of all per-station classifiers even if saved models exist. "
             "Training uses data from ALL seasons combined.",
    )
    return p.parse_args()


def _collect_all_season_features(station_id: str, sims_root: Path) -> pd.DataFrame | None:
    """
    Scan every season subdirectory under sims_root for <station_id>.pro + .smet.
    Concatenate daily features across all found seasons and return a combined
    DataFrame sorted by date (duplicates dropped).  Returns None if no data found.
    """
    parts: list[pd.DataFrame] = []
    for season_dir in sorted(sims_root.iterdir()):
        if not season_dir.is_dir():
            continue
        pro  = season_dir / f"{station_id}.pro"
        smet = season_dir / f"{station_id}.smet"
        if pro.exists() and smet.exists():
            daily = build_daily_features(pro, smet)
            if not daily.empty:
                parts.append(daily)

    if not parts:
        return None

    combined = pd.concat(parts).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined


def main() -> None:
    args          = _parse_args()
    forecast_date = pd.Timestamp(args.forecast_date)

    root_dir      = Path(__file__).resolve().parents[1]
    sims_root     = root_dir / "data" / "simulations"
    sim_dir       = sims_root / SEASON          # current season for prediction & plots
    obs_dir       = root_dir / "data" / "observations"
    out_dir       = root_dir / "output"
    stability_dir = root_dir / "output" / "assets" / SEASON
    models_dir    = root_dir / "models"         # season-agnostic

    stability_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    avalanche_csv = obs_dir / "Avalanche Information.csv"
    if not avalanche_csv.exists():
        print(f"CSV not found: {avalanche_csv}")
        return
    if not sim_dir.is_dir():
        print(f"Simulations directory not found: {sim_dir}")
        return

    print(f"Season (prediction): {SEASON}")
    print(f"Forecast date:       {forecast_date.date()}  "
          f"(window: {(forecast_date - pd.Timedelta(days=1)).date()} – "
          f"{(forecast_date + pd.Timedelta(days=2)).date()})")
    print(f"Retrain:             {args.retrain}")

    # --- Load ALL observations (multi-year CSV) ---
    df_all = pd.read_csv(avalanche_csv)
    df_all = df_all.rename(columns={
        'Lat':   'Latitude',
        'Long':  'Longitude',
        'Place': 'Placemark Name',
    })
    df_all['date'] = (
        pd.to_datetime(df_all['Date'], errors='coerce')
        .dt.strftime('%B %d, %Y').fillna('')
    )
    print(f"Loaded {len(df_all)} total observations from {avalanche_csv.name}")

    # --- Match every observation to its nearest current-season .pro file ---
    # We use current-season coordinates to establish station locations, then
    # accumulate event dates from ALL seasons for training.
    pro_cache:    dict[Path, str]        = {}
    station_dates: dict[str, list]       = {}   # station_id -> all event dates

    def stability_html_for(pro_file: Path) -> str:
        if pro_file not in pro_cache:
            pro_cache[pro_file] = f"assets/{SEASON}/stability_{pro_file.stem}.html"
        return pro_cache[pro_file]

    target_urls:       list[str]         = []
    matched_pro_files: list[Path | None] = []

    for _, row in df_all.iterrows():
        pro_file = find_nearest_pro(row['Latitude'], row['Longitude'], sim_dir)
        if pro_file is None:
            target_urls.append("")
            matched_pro_files.append(None)
        else:
            station_id = pro_file.stem
            target_urls.append(stability_html_for(pro_file))
            matched_pro_files.append(pro_file)

            raw_date = pd.to_datetime(row.get('Date', ''), errors='coerce')
            if pd.notna(raw_date):
                station_dates.setdefault(station_id, [])
                if raw_date not in station_dates[station_id]:
                    station_dates[station_id].append(raw_date)

    # Print match summary
    matched = sum(1 for p in matched_pro_files if p is not None)
    print(f"Matched {matched}/{len(df_all)} observations to current-season stations")

    df_all['target_url'] = target_urls

    # --- Per-station: train/load classifier, generate stability plot ---
    unique_pro_files = {p for p in matched_pro_files if p is not None}

    win_start = forecast_date - pd.Timedelta(days=1)
    win_end   = forecast_date + pd.Timedelta(days=2)

    prob_series_dict: dict[Path, pd.Series] = {}

    if not unique_pro_files:
        print("No .pro files matched — stability plots skipped.")
    else:
        print(f"\nProcessing {len(unique_pro_files)} stations...")
        for pro_file in sorted(unique_pro_files):
            out_path   = stability_dir / f"stability_{pro_file.stem}.html"
            model_path = models_dir / f"{pro_file.stem}.joblib"
            station_id = pro_file.stem
            event_dates = sorted(station_dates.get(station_id, []))

            prob  = None
            daily = None  # current-season features for prediction & plot
            smet_file = pro_file.with_suffix('.smet')

            if not smet_file.exists():
                print(f"\n  {pro_file.name}  (no .smet — classifier skipped)")
            else:
                print(f"\n  {pro_file.name}")
                # Current-season features (always needed for prediction)
                daily = build_daily_features(pro_file, smet_file)

                fitted: tuple | None = None

                if not args.retrain and model_path.exists():
                    # ── Load saved model ──────────────────────────────────
                    fitted = joblib.load(model_path)
                    print(f"    → loaded model ({model_path.name})")
                else:
                    # ── Train on ALL seasons combined ─────────────────────
                    print(f"    collecting multi-season features...")
                    combined = _collect_all_season_features(station_id, sims_root)
                    if combined is None:
                        combined = daily   # fallback: current season only

                    n_seasons = sum(
                        1 for sd in sims_root.iterdir()
                        if sd.is_dir()
                        and (sd / f"{station_id}.pro").exists()
                        and (sd / f"{station_id}.smet").exists()
                    )
                    print(f"    training on {n_seasons} season(s) of data  "
                          f"({len(event_dates)} event dates)")

                    fitted = train_station(combined, event_dates)
                    if fitted is not None:
                        joblib.dump(fitted, model_path)
                        print(f"    → saved model to {model_path.name}")

                if fitted is not None:
                    model, scaler = fitted
                    prob = predict_proba_series(model, scaler, daily)
                    prob_series_dict[pro_file] = prob

            print(f"  Plotting {pro_file.name}...")
            plot_interactive_stability(
                parse_snow_data(pro_file), out_path, station_id,
                event_dates=event_dates, prob_series=prob, daily_df=daily,
                forecast_date=forecast_date,
            )
            print(f"  → {out_path}")

    # --- Attach max forecast-window probability to each observation row ---
    forecast_probs: list[float] = []
    for pf in matched_pro_files:
        if pf is None or pf not in prob_series_dict:
            forecast_probs.append(float('nan'))
        else:
            s      = prob_series_dict[pf]
            window = s[(s.index >= win_start) & (s.index <= win_end)]
            forecast_probs.append(
                float(window.max()) if not window.dropna().empty else float('nan')
            )
    df_all['forecast_prob'] = forecast_probs

    # --- Generate the map ---
    map_path = out_dir / "avalanche_map.html"
    print(f"\nGenerating map → {map_path}")
    create_avalanche_map(df_all, map_path, forecast_date=forecast_date)

    print("\nDone.")
    print(f"Open {map_path} and click any marker to view its nearest station's stability.")


if __name__ == "__main__":
    main()
