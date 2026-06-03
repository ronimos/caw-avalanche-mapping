import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from classifier import (aggregate_predictions, blend_probabilities,
                        evaluate_event_level, evaluate_regional, evaluate_station,
                        predict_proba_series, train_regional, train_station)
from features import build_daily_features
from snowpack_io import find_nearest_pro, parse_snow_data
from visualization import create_avalanche_map, plot_interactive_stability

# ── season config ─────────────────────────────────────────────────────────────
# SEASON:       current / test season — used for prediction, plots, and the map.
# TRAIN_SEASON: historical season used exclusively for training classifiers.
#               Features come from TRAIN_SEASON .pro/.smet files; labels come
#               from TRAIN_SEASON observation rows.
#
# Data layout:
#   data/simulations/<SEASON>/*.pro          — SNOWPACK outputs per season
#   data/observations/avalanches_<SEASON>.csv — per-season observation CSV
#   models/                                  — per-station classifiers (saved)
#   output/assets/<SEASON>/                  — stability HTML plots
#   output/evaluation.csv                    — held-out test-set metrics

SEASON       = "2025-2026"
TRAIN_SEASON = "2024-2025"


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
        help="Force retraining of all per-station classifiers even if saved models exist.",
    )
    return p.parse_args()


def _load_observations(obs_dir: Path) -> pd.DataFrame:
    """
    Load all avalanches_<SEASON>.csv files from obs_dir, tag each row with
    source_season, and return a combined DataFrame.
    """
    parts: list[pd.DataFrame] = []
    for csv_path in sorted(obs_dir.glob("avalanches_*.csv")):
        season = csv_path.stem[len("avalanches_"):]
        df = pd.read_csv(csv_path)
        df['source_season'] = season
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.rename(columns={
        'Lat':   'Latitude',
        'Long':  'Longitude',
        'Place': 'Placemark Name',
    })
    combined['date'] = (
        pd.to_datetime(combined['Date'], errors='coerce')
        .dt.strftime('%B %d, %Y').fillna('')
    )
    return combined


def _collect_all_season_features(
    station_id: str,
    sims_root: Path,
    exclude_seasons: set[str] | None = None,
) -> pd.DataFrame | None:
    """
    Concatenate daily features across season subdirectories under sims_root,
    skipping any season listed in exclude_seasons.
    Returns None if no data is found.
    """
    parts: list[pd.DataFrame] = []
    for season_dir in sorted(sims_root.iterdir()):
        if not season_dir.is_dir():
            continue
        if exclude_seasons and season_dir.name in exclude_seasons:
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
    sim_dir       = sims_root / SEASON
    obs_dir       = root_dir / "data" / "observations"
    out_dir       = root_dir / "output"
    stability_dir = out_dir / "assets" / SEASON
    models_dir    = root_dir / "models"

    stability_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    stations_csv  = root_dir / "data" / "snowpack_stations_locations.csv"

    if not sim_dir.is_dir():
        print(f"Simulations directory not found: {sim_dir}")
        return

    print(f"Season (prediction): {SEASON}")
    print(f"Train season:        {TRAIN_SEASON}")
    print(f"Forecast date:       {forecast_date.date()}  "
          f"(window: {(forecast_date - pd.Timedelta(days=1)).date()} – "
          f"{(forecast_date + pd.Timedelta(days=2)).date()})")
    print(f"Retrain:             {args.retrain}")

    # --- Load all observations across seasons ---
    df_all = _load_observations(obs_dir)
    if df_all.empty:
        print(f"No observation CSVs found in {obs_dir}")
        return

    n_train_obs = (df_all['source_season'] == TRAIN_SEASON).sum()
    n_test_obs  = (df_all['source_season'] == SEASON).sum()
    print(f"Loaded {len(df_all)} observations  "
          f"({n_train_obs} from {TRAIN_SEASON}, {n_test_obs} from {SEASON})")

    # --- Match every observation to the nearest current-season .pro station ---
    matched_pro_files: list[Path | None] = []
    target_urls:       list[str]         = []

    for _, row in df_all.iterrows():
        pro_file = find_nearest_pro(
            row['Latitude'], row['Longitude'], sim_dir,
            aspect=str(row.get('Aspect', '')),
            stations_csv=stations_csv,
        )
        matched_pro_files.append(pro_file)
        target_urls.append(
            f"assets/{SEASON}/stability_{pro_file.stem}.html" if pro_file else ""
        )

    df_all['target_url'] = target_urls
    matched = sum(1 for p in matched_pro_files if p is not None)
    print(f"Matched {matched}/{len(df_all)} observations to {SEASON} stations")

    # --- Assign train / test split ---
    # Test: any SEASON observation whose station_id was active in TRAIN_SEASON
    #       (aspect relaxed — same physical location is enough for cross-season
    #       validation even if the reported aspect differs slightly).
    train_station_ids: set[str] = set()
    for idx, (_, row) in enumerate(df_all.iterrows()):
        pf = matched_pro_files[idx]
        if pf is not None and row['source_season'] == TRAIN_SEASON:
            train_station_ids.add(pf.stem)

    splits: list[str] = []
    for idx, (_, row) in enumerate(df_all.iterrows()):
        pf = matched_pro_files[idx]
        if pf is not None and row['source_season'] == SEASON:
            splits.append('test' if pf.stem in train_station_ids else 'train')
        else:
            splits.append('train')
    df_all['split'] = splits

    n_test_flagged = splits.count('test')
    print(f"Test set: {n_test_flagged} observation(s) "
          f"(station matched to a {TRAIN_SEASON} event, any aspect)")

    # --- Build per-station train / test event date lists ---
    # Training labels come ONLY from TRAIN_SEASON observations so that
    # 2025-26 dates never contaminate the 2024-25 feature date range.
    station_train_dates: dict[str, list[pd.Timestamp]] = {}
    station_test_dates:  dict[str, list[pd.Timestamp]] = {}

    for idx, (_, row) in enumerate(df_all.iterrows()):
        pf = matched_pro_files[idx]
        if pf is None:
            continue
        raw_date = pd.to_datetime(row.get('Date', ''), errors='coerce')
        if pd.isna(raw_date):
            continue
        station_id = pf.stem
        if row['split'] == 'test':
            bucket = station_test_dates
        elif row['source_season'] == TRAIN_SEASON:
            bucket = station_train_dates
        else:
            continue  # 2025-26 non-test obs: not used as training labels
        bucket.setdefault(station_id, [])
        if raw_date not in bucket[station_id]:
            bucket[station_id].append(raw_date)

    # --- Per-station: train / load classifier, predict, plot ---
    unique_pro_files = {p for p in matched_pro_files if p is not None}

    win_start = forecast_date - pd.Timedelta(days=1)
    win_end   = forecast_date + pd.Timedelta(days=2)

    prob_series_dict:  dict[Path, pd.Series]    = {}
    fitted_dict:       dict[Path, tuple]        = {}
    daily_dict:        dict[Path, pd.DataFrame] = {}
    train_features_dict: dict[str, pd.DataFrame] = {}  # station_id → 2024-25 features

    if not unique_pro_files:
        print("No .pro files matched — stability plots skipped.")
    else:
        print(f"\nProcessing {len(unique_pro_files)} stations...")
        for pro_file in sorted(unique_pro_files):
            station_id        = pro_file.stem
            out_path          = stability_dir / f"stability_{station_id}.html"
            model_path        = models_dir / f"{station_id}.joblib"
            train_event_dates = sorted(station_train_dates.get(station_id, []))
            test_event_dates  = sorted(station_test_dates.get(station_id, []))

            print(f"\n  {pro_file.name}  "
                  f"(train events: {len(train_event_dates)}, "
                  f"test events: {len(test_event_dates)})")

            smet_file = pro_file.with_suffix('.smet')
            daily:  pd.DataFrame | None = None
            fitted: tuple | None        = None

            if not smet_file.exists():
                print(f"    (no .smet — classifier skipped)")
            else:
                # Current-season features used for prediction and plots
                daily = build_daily_features(pro_file, smet_file)
                daily_dict[pro_file] = daily

                if not args.retrain and model_path.exists():
                    fitted = joblib.load(model_path)
                    print(f"    → loaded model ({model_path.name})")
                else:
                    # Training features: TRAIN_SEASON only (never leak test season)
                    train_pro  = sims_root / TRAIN_SEASON / f"{station_id}.pro"
                    train_smet = sims_root / TRAIN_SEASON / f"{station_id}.smet"

                    if train_pro.exists() and train_smet.exists():
                        print(f"    collecting {TRAIN_SEASON} features...")
                        train_daily: pd.DataFrame | None = build_daily_features(
                            train_pro, train_smet
                        )
                        if train_daily is not None and not train_daily.empty:
                            train_features_dict[station_id] = train_daily
                    else:
                        # Fallback: any season except the test season
                        print(f"    collecting multi-season features (excl. {SEASON})...")
                        train_daily = _collect_all_season_features(
                            station_id, sims_root, exclude_seasons={SEASON}
                        )

                    if train_daily is not None and not train_daily.empty:
                        n_seasons = sum(
                            1 for sd in sims_root.iterdir()
                            if sd.is_dir() and sd.name != SEASON
                            and (sd / f"{station_id}.pro").exists()
                            and (sd / f"{station_id}.smet").exists()
                        )
                        print(f"    training on {n_seasons} season(s)  "
                              f"({len(train_event_dates)} train event dates)")
                        fitted = train_station(train_daily, train_event_dates)
                        if fitted is not None:
                            joblib.dump(fitted, model_path)
                            print(f"    → saved {model_path.name}")
                    else:
                        print(f"    → no training data found for {TRAIN_SEASON}")

                if fitted is not None:
                    fitted_dict[pro_file] = fitted
                    model, scaler = fitted
                    prob = predict_proba_series(model, scaler, daily)
                    prob_series_dict[pro_file] = prob

            print(f"  Plotting {pro_file.name}...")
            plot_interactive_stability(
                parse_snow_data(pro_file), out_path, station_id,
                event_dates=train_event_dates,
                test_dates=test_event_dates,
                prob_series=prob_series_dict.get(pro_file),
                daily_df=daily,
                forecast_date=forecast_date,
            )
            print(f"  → {out_path}")

    # --- Evaluate on held-out test set ---
    stations_with_test = [
        sid for sid, dates in station_test_dates.items() if dates
    ]
    if stations_with_test:
        print(f"\nEvaluating on test set ({len(stations_with_test)} station(s))...")
        eval_rows = []

        for station_id in sorted(stations_with_test):
            pf = next((p for p in unique_pro_files if p.stem == station_id), None)
            if pf is None or pf not in fitted_dict or pf not in daily_dict:
                print(f"  {station_id}: no model or features — skipped")
                continue

            model, scaler   = fitted_dict[pf]
            test_event_dates = sorted(station_test_dates[station_id])
            metrics = evaluate_station(model, scaler, daily_dict[pf], test_event_dates)

            if not metrics:
                print(f"  {station_id}: insufficient test data — skipped")
                continue

            test_aspects = sorted({
                str(row.get('Aspect', '')).strip()
                for idx, (_, row) in enumerate(df_all.iterrows())
                if (pf := matched_pro_files[idx]) is not None
                and pf.stem == station_id
                and row['split'] == 'test'
            })

            eval_rows.append({
                'station_id':     station_id,
                'aspects':        ';'.join(test_aspects),
                'n_train_events': len(station_train_dates.get(station_id, [])),
                'n_test_events':  len(test_event_dates),
                **metrics,
            })
            print(f"  {station_id}  aspects={';'.join(test_aspects)}  "
                  f"AUC={metrics.get('auc', float('nan')):.3f}  "
                  f"F1={metrics.get('f1', float('nan')):.3f}  "
                  f"precision={metrics.get('precision', float('nan')):.3f}  "
                  f"recall={metrics.get('recall', float('nan')):.3f}")

        if eval_rows:
            eval_df   = pd.DataFrame(eval_rows)
            eval_path = out_dir / "evaluation.csv"
            eval_df.to_csv(eval_path, index=False)
            print(f"\n→ Evaluation saved: {eval_path}")
        else:
            print("  No stations had sufficient test data for evaluation.")

    # --- Regional pooled model ---
    # Train on all 2024-25 stations combined; evaluate on all 2025-26 observations.
    if train_features_dict:
        print(f"\nTraining regional model "
              f"({len(train_features_dict)} stations, {TRAIN_SEASON} features)...")
        regional_path = models_dir / "regional.joblib"
        regional_fitted: tuple | None = None

        if not args.retrain and regional_path.exists():
            regional_fitted = joblib.load(regional_path)
            print(f"  → loaded regional model")
        else:
            regional_fitted = train_regional(train_features_dict, station_train_dates)
            if regional_fitted is not None:
                joblib.dump(regional_fitted, regional_path)
                print(f"  → saved {regional_path.name}")

        if regional_fitted is not None:
            reg_model, reg_scaler = regional_fitted

            # All 2025-26 event dates per station (used by regional + blended)
            station_2526_dates: dict[str, list[pd.Timestamp]] = {}
            for idx, (_, row) in enumerate(df_all.iterrows()):
                pf = matched_pro_files[idx]
                if pf is None or row['source_season'] != SEASON:
                    continue
                raw_date = pd.to_datetime(row.get('Date', ''), errors='coerce')
                if pd.isna(raw_date):
                    continue
                station_2526_dates.setdefault(pf.stem, [])
                if raw_date not in station_2526_dates[pf.stem]:
                    station_2526_dates[pf.stem].append(raw_date)

            test_features = {pf.stem: df for pf, df in daily_dict.items()}

            # Per-station regional probabilities (needed for blending)
            regional_prob_dict: dict[str, pd.Series] = {
                sid: predict_proba_series(reg_model, reg_scaler, df)
                for sid, df in test_features.items()
            }

            reg_metrics = evaluate_regional(
                reg_model, reg_scaler, test_features, station_2526_dates
            )

            if reg_metrics:
                print(f"\n  Regional evaluation ({SEASON}):")
                print(f"    stations={int(reg_metrics['n_stations'])}  "
                      f"events={int(reg_metrics['n_events'])}  "
                      f"AUC={reg_metrics.get('auc', float('nan')):.3f}  "
                      f"F1={reg_metrics.get('f1', float('nan')):.3f}  "
                      f"precision={reg_metrics.get('precision', float('nan')):.3f}  "
                      f"recall={reg_metrics.get('recall', float('nan')):.3f}")
                pd.DataFrame([reg_metrics]).to_csv(
                    out_dir / "evaluation_regional.csv", index=False
                )
                print(f"  → saved evaluation_regional.csv")
            else:
                print("  Regional: insufficient test data for evaluation.")

            # --- Blended model: confidence-weighted per-station + regional ---
            print(f"\nBuilding blended model (per-station + regional, max_events=5)...")

            # For each station build three probability series aligned to regional
            blended_prob_dict:  dict[str, pd.Series] = {}
            fallback_prob_dict: dict[str, pd.Series] = {}  # station if exists, else regional

            for pro_file in sorted(unique_pro_files):
                sid     = pro_file.stem
                p_reg   = regional_prob_dict.get(sid)
                if p_reg is None:
                    continue
                p_sta   = prob_series_dict.get(pro_file)   # None if no per-station model
                n_train = len(station_train_dates.get(sid, []))

                blended_prob_dict[sid]  = blend_probabilities(p_sta, p_reg, n_train)
                fallback_prob_dict[sid] = (
                    p_sta.reindex(p_reg.index).fillna(p_reg)
                    if p_sta is not None else p_reg.copy()
                )

            # Evaluate blended model directly from aggregated predictions
            yt_blend, yp_blend = aggregate_predictions(
                blended_prob_dict, station_2526_dates, test_features
            )
            if len(yt_blend) > 0 and yt_blend.sum() > 0:
                yp_blend_bin = (yp_blend >= 0.5).astype(int)
                blend_metrics: dict[str, float] = {
                    'n_stations': float(len(blended_prob_dict)),
                    'n_events':   float(sum(len(v) for v in
                                            station_2526_dates.values())),
                    'precision':  float(precision_score(yt_blend, yp_blend_bin,
                                                        zero_division=0)),
                    'recall':     float(recall_score(yt_blend, yp_blend_bin,
                                                     zero_division=0)),
                    'f1':         float(f1_score(yt_blend, yp_blend_bin,
                                                 zero_division=0)),
                }
                if len(set(yt_blend.tolist())) > 1:
                    blend_metrics['auc'] = float(roc_auc_score(yt_blend, yp_blend))
                print(f"  Blended:  "
                      f"AUC={blend_metrics.get('auc', float('nan')):.3f}  "
                      f"F1={blend_metrics.get('f1', float('nan')):.3f}  "
                      f"precision={blend_metrics.get('precision', float('nan')):.3f}  "
                      f"recall={blend_metrics.get('recall', float('nan')):.3f}")
                pd.DataFrame([blend_metrics]).to_csv(
                    out_dir / "evaluation_blended.csv", index=False
                )
                print("  → saved evaluation_blended.csv")

            # --- Event-level Precision-Recall curves (3-day pre-event window) ---
            PRE_EVENT_WINDOW = 3
            print(f"\nGenerating event-level PR curves "
                  f"(pre-event window = {PRE_EVENT_WINDOW} days)...")

            pr_curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []
            n_events_total = n_pos_windows = n_neg_windows = 0

            for label, prob_dict in [
                ("Regional",               regional_prob_dict),
                ("Per-station + fallback", fallback_prob_dict),
                ("Blended (weighted)",     blended_prob_dict),
            ]:
                yt, ys = evaluate_event_level(
                    prob_dict, station_2526_dates, PRE_EVENT_WINDOW
                )
                if len(yt) == 0 or yt.sum() == 0:
                    print(f"  {label}: no positive event windows — skipped")
                    continue

                # Store counts from the regional model (reference)
                if label == "Regional":
                    n_pos_windows = int(yt.sum())
                    n_neg_windows = int((yt == 0).sum())
                    n_events_total = n_pos_windows

                prec, rec, _  = precision_recall_curve(yt, ys)
                ap             = float(average_precision_score(yt, ys))
                auc_roc        = float(roc_auc_score(yt, ys)) if len(set(yt.tolist())) > 1 else float('nan')
                pr_curves.append((label, prec, rec, ap))
                print(f"  {label:30s}  AP={ap:.3f}  AUC-ROC={auc_roc:.3f}  "
                      f"(+windows={int(yt.sum())}  -windows={int((yt==0).sum())})")

            if pr_curves:
                prevalence = n_pos_windows / (n_pos_windows + n_neg_windows) \
                    if (n_pos_windows + n_neg_windows) > 0 else 0.0

                fig, ax = plt.subplots(figsize=(7, 5))
                colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']
                for (label, prec, rec, ap), color in zip(pr_curves, colors):
                    ax.plot(rec, prec, color=color, linewidth=2,
                            label=f"{label}  (AP = {ap:.3f})")
                ax.axhline(prevalence, color='grey', linestyle='--', linewidth=1,
                           label=f"No skill  (prevalence = {prevalence:.2f})")
                ax.set_xlabel("Recall  (fraction of events detected)", fontsize=11)
                ax.set_ylabel("Precision  (fraction of alarms correct)", fontsize=11)
                ax.set_title(
                    f"Event-Level Precision–Recall  "
                    f"({PRE_EVENT_WINDOW}-day pre-event window)\n"
                    f"Train: {TRAIN_SEASON}  →  Test: {SEASON}  "
                    f"(n events = {n_events_total}  |  "
                    f"non-event windows = {n_neg_windows})",
                    fontsize=10,
                )
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                plt.tight_layout()
                pr_path = out_dir / "pr_curves.png"
                fig.savefig(str(pr_path), dpi=150)
                plt.close(fig)
                print(f"  → saved {pr_path}")

    # --- Attach max forecast-window probability to each observation ---
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
