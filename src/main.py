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
from snowpack_io import extract_pro_coordinates, find_nearest_pro, parse_snow_data
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

# Confidence tiers from §4.5 operational threshold analysis (FA rate at
# recall-maximising threshold).  Used both in the map overlay and plot titles.
READY_STATIONS    = frozenset({'160942_res', '153203_res', '180343_res', '164801_res'})
MARGINAL_STATIONS = frozenset({'272401_res', '176522_res', '250224_res'})

_ASPECT_SUFFIX = {'1': 'N', '2': 'E', '3': 'S', '4': 'W'}


def _station_aspect(stem: str) -> str:
    numeric = stem.replace('_res', '')
    return _ASPECT_SUFFIX.get(numeric[-1:], 'Flat') if numeric else 'Flat'


def _station_tier(stem: str) -> str:
    if stem in READY_STATIONS:
        return 'ready'
    if stem in MARGINAL_STATIONS:
        return 'marginal'
    return 'not_ready'


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
    p.add_argument(
        "--no-hierarchical", action="store_true",
        help="Skip the hierarchical Bayesian model (avoids the PyMC dependency / MCMC sampling).",
    )
    return p.parse_args()


PRE_EVENT_WINDOW = 3  # days before event counted as a "warning window"


def _run_loo_folds(
    station_id: str,
    sims_root: Path,
    all_events: list[tuple[pd.Timestamp, str]],
) -> list[dict]:
    """
    Temporal leave-one-out CV for one station.

    Events are sorted chronologically.  Fold i trains on events[0..i-1] and
    tests on events[i].  Features are loaded from whichever season each event
    belongs to, so the model always trains on data it could have seen in
    real-time.

    Returns a list of per-fold result dicts.
    """
    results: list[dict] = []
    window_td = pd.Timedelta(days=PRE_EVENT_WINDOW)

    # Grace zone: days within ±(window+1) of any known event (excluded from
    # non-event probability estimation so nearby events don't contaminate it)
    grace: set[pd.Timestamp] = set()
    for ev_date, _ in all_events:
        ed = ev_date.normalize()
        for d in range(-(PRE_EVENT_WINDOW + 1), PRE_EVENT_WINDOW + 2):
            grace.add(ed + pd.Timedelta(days=d))

    for i in range(1, len(all_events)):
        test_date, test_season = all_events[i]
        train_evs  = all_events[:i]
        train_dates = [ev[0] for ev in train_evs]

        # Load test-season features
        test_pro  = sims_root / test_season / f"{station_id}.pro"
        test_smet = sims_root / test_season / f"{station_id}.smet"
        if not (test_pro.exists() and test_smet.exists()):
            continue
        test_daily = build_daily_features(test_pro, test_smet)

        # Concatenate training features across all seasons with training events
        train_parts: list[pd.DataFrame] = []
        for season in {ev[1] for ev in train_evs}:
            pro  = sims_root / season / f"{station_id}.pro"
            smet = sims_root / season / f"{station_id}.smet"
            if pro.exists() and smet.exists():
                df = build_daily_features(pro, smet)
                if not df.empty:
                    train_parts.append(df)
        if not train_parts:
            continue

        train_daily = pd.concat(train_parts).sort_index()
        train_daily = train_daily[~train_daily.index.duplicated(keep='first')]

        fold: dict = {
            'station_id':       station_id,
            'n_train_events':   len(train_dates),
            'test_date':        test_date.date(),
            'test_season':      test_season,
            'trained':          False,
            'event_prob':       float('nan'),
            'non_event_median': float('nan'),
        }

        fitted = train_station(train_daily, train_dates, verbose=False)
        if fitted is not None:
            model, scaler = fitted
            prob = predict_proba_series(model, scaler, test_daily)

            # Max probability in the pre-event window
            w_start = (test_date - window_td).normalize()
            w_end   = test_date.normalize()
            idx_norm = prob.index.floor('D')
            win = prob[(idx_norm >= w_start) & (idx_norm <= w_end)]
            fold['event_prob']       = float(win.max()) if not win.dropna().empty else float('nan')
            fold['trained']          = True

            # Median non-event probability (background rate at this station)
            non_grace = prob[~prob.index.floor('D').isin(grace)].dropna()
            fold['non_event_median'] = float(non_grace.median()) if not non_grace.empty else float('nan')

        results.append(fold)

    return results


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
        pd.to_datetime(combined['Date'], errors='coerce', format='mixed')
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
    stability_dir = root_dir / "assets" / SEASON
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
                aspect=_station_aspect(station_id),
                confidence_tier=_station_tier(station_id),
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

    regional_prob_dict: dict[str, pd.Series] = {}

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
            regional_prob_dict = {
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

            # --- Hierarchical Bayesian model (partial pooling) ---
            # One multilevel model replaces per-station + regional + blend; each
            # station's coefficients shrink toward the regional mean by an amount
            # learned from its data. Posterior spread gives per-station uncertainty.
            hier_prob_dict: dict[str, pd.Series] = {}
            if not args.no_hierarchical:
                try:
                    from hierarchical import train_hierarchical
                    print(f"\nTraining hierarchical Bayesian model "
                          f"({len(train_features_dict)} stations)...")
                    hmodel = train_hierarchical(train_features_dict, station_train_dates)
                except ImportError:
                    hmodel = None
                    print("\n  PyMC not available — hierarchical model skipped.")

                if hmodel is not None:
                    hier_std_rows: list[dict] = []
                    for sid, df in test_features.items():
                        mean_p, std_p = hmodel.predict(sid, df)
                        hier_prob_dict[sid] = mean_p
                        # Forecast-window uncertainty summary (for map confidence)
                        win = std_p[(std_p.index >= win_start) & (std_p.index <= win_end)]
                        hier_std_rows.append({
                            'station_id':       sid,
                            'in_training':      sid in hmodel.station_list,
                            'mean_prob_window': float(
                                mean_p[(mean_p.index >= win_start) &
                                       (mean_p.index <= win_end)].max()
                            ) if not mean_p.dropna().empty else float('nan'),
                            'std_window':       float(win.mean()) if not win.dropna().empty else float('nan'),
                        })

                    yt_h, ys_h = evaluate_event_level(
                        hier_prob_dict, station_2526_dates, 3
                    )
                    if len(yt_h) > 0 and yt_h.sum() > 0:
                        h_metrics = {
                            'auc': float(roc_auc_score(yt_h, ys_h))
                                   if len(set(yt_h.tolist())) > 1 else float('nan'),
                            'ap':  float(average_precision_score(yt_h, ys_h)),
                            'n_stations': float(len(hier_prob_dict)),
                        }
                        print(f"  Hierarchical:  AUC-ROC={h_metrics['auc']:.3f}  "
                              f"AP={h_metrics['ap']:.3f}")
                        pd.DataFrame([h_metrics]).to_csv(
                            out_dir / "evaluation_hierarchical.csv", index=False)
                        pd.DataFrame(hier_std_rows).to_csv(
                            out_dir / "hierarchical_uncertainty.csv", index=False)
                        print("  → saved evaluation_hierarchical.csv, "
                              "hierarchical_uncertainty.csv")

            # --- Event-level Precision-Recall curves (3-day pre-event window) ---
            print(f"\nGenerating event-level PR curves "
                  f"(pre-event window = {PRE_EVENT_WINDOW} days)...")

            pr_curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []
            n_events_total = n_pos_windows = n_neg_windows = 0

            pr_model_list = [
                ("Regional",               regional_prob_dict),
                ("Per-station + fallback", fallback_prob_dict),
                ("Blended (weighted)",     blended_prob_dict),
            ]
            if hier_prob_dict:
                pr_model_list.append(("Hierarchical (Bayesian)", hier_prob_dict))

            for label, prob_dict in pr_model_list:
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
                colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
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

    # --- Temporal LOO cross-validation + performance-by-events plot ---
    print("\nRunning temporal LOO cross-validation...")

    # Build per-station event list (all seasons, sorted chronologically, deduplicated)
    station_all_events: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for idx, (_, row) in enumerate(df_all.iterrows()):
        pf = matched_pro_files[idx]
        if pf is None:
            continue
        raw_date = pd.to_datetime(row.get('Date', ''), errors='coerce')
        if pd.isna(raw_date):
            continue
        evs = station_all_events.setdefault(pf.stem, [])
        entry = (raw_date.normalize(), row['source_season'])
        if entry not in evs:
            evs.append(entry)

    for sid in station_all_events:
        station_all_events[sid].sort(key=lambda x: x[0])

    loo_rows: list[dict] = []
    for pro_file in sorted(unique_pro_files):
        sid = pro_file.stem
        evs = station_all_events.get(sid, [])
        if len(evs) < 2:
            continue
        folds = _run_loo_folds(sid, sims_root, evs)
        loo_rows.extend(folds)
        n_trained = sum(1 for f in folds if f['trained'])
        print(f"  {sid}: {len(evs)} events → {len(folds)} folds  ({n_trained} trained)")

    if loo_rows:
        loo_df = pd.DataFrame(loo_rows)
        loo_df.to_csv(out_dir / "evaluation_loo.csv", index=False)
        print(f"  → saved evaluation_loo.csv  ({len(loo_df)} folds total)")

        # ── Performance-by-events plot ─────────────────────────────────────
        plot_df = loo_df[loo_df['trained'] & loo_df['event_prob'].notna()].copy()

        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))

            # Jitter x slightly for visibility
            rng = np.random.default_rng(42)
            jitter   = rng.uniform(-0.12, 0.12, size=len(plot_df))
            x        = plot_df['n_train_events'].to_numpy(dtype=float) + jitter
            ep       = plot_df['event_prob'].to_numpy(dtype=float)
            detected = ep >= 0.5

            ax.scatter(x[detected],  ep[detected],
                       color='#2ca02c', s=70, alpha=0.8, zorder=3,
                       label='Detected (prob ≥ 0.5)')
            ax.scatter(x[~detected], ep[~detected],
                       color='#d62728', s=70, alpha=0.8, zorder=3,
                       label='Missed (prob < 0.5)')

            # Mean per n_train
            mean_by_n = plot_df.groupby('n_train_events')['event_prob'].mean()
            ax.plot(mean_by_n.index.to_numpy(dtype=float), mean_by_n.to_numpy(dtype=float),
                    'k--o', linewidth=1.8, markersize=7, zorder=4, label='Mean')

            # Non-event background reference
            non_event_ref = plot_df['non_event_median'].median()
            if not pd.isna(non_event_ref):
                ax.axhline(non_event_ref, color='steelblue', linestyle=':',
                           linewidth=1.2,
                           label=f'Median non-event prob ({non_event_ref:.3f})')

            ax.axhline(0.5, color='grey', linestyle='--', linewidth=1,
                       alpha=0.6, label='0.5 threshold')

            # Fold counts per n_train
            for n, cnt in plot_df.groupby('n_train_events').size().items():
                ax.text(float(n), -0.06, f'n={cnt}', ha='center',
                        fontsize=9, color='#555', transform=ax.get_xaxis_transform())

            ax.set_xlabel('Number of training events', fontsize=12)
            ax.set_ylabel(f'Event window probability\n'
                          f'(max prob in {PRE_EVENT_WINDOW} days before event)',
                          fontsize=11)
            ax.set_title(
                'Per-Station Classifier Performance vs. Training Data Size\n'
                'Temporal LOO cross-validation — all stations, both seasons',
                fontsize=11,
            )
            n_max = int(plot_df['n_train_events'].max())
            ax.set_xlim(0.5, n_max + 0.5)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(range(1, n_max + 1))
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            perf_path = out_dir / "loo_performance_by_events.png"
            fig.savefig(str(perf_path), dpi=150)
            plt.close(fig)
            print(f"  → saved {perf_path}")

            # Print summary by n_train
            print("\n  LOO summary by number of training events:")
            print(f"  {'n_train':>8}  {'folds':>6}  {'mean_prob':>10}  "
                  f"{'detected':>9}  {'detection_rate':>14}")
            for n, grp in plot_df.groupby('n_train_events'):
                det = (grp['event_prob'] >= 0.5).sum()
                print(f"  {n:>8}  {len(grp):>6}  "
                      f"{grp['event_prob'].mean():>10.3f}  "
                      f"{det:>9}  {det/len(grp):>14.1%}")

    # --- Operational threshold analysis ---
    # For each station: set threshold = min(LOO event prob) = recall-maximizing.
    # Count non-event DAYS (not windows) above that threshold to measure false alarm rate.
    if loo_rows:
        loo_df_op = pd.DataFrame(loo_rows)
        trained   = loo_df_op[loo_df_op['trained'] & loo_df_op['event_prob'].notna()]
        if not trained.empty:
            print("\nRunning operational threshold analysis...")
            op_rows: list[dict] = []

            for pro_file in sorted(unique_pro_files):
                sid = pro_file.stem
                station_folds = trained[trained['station_id'] == sid]
                if station_folds.empty:
                    continue
                model_path = models_dir / f"{sid}.joblib"
                smet_file  = pro_file.with_suffix('.smet')
                if not (model_path.exists() and smet_file.exists()):
                    continue

                T     = float(station_folds['event_prob'].min())
                model, scaler = joblib.load(model_path)
                daily = build_daily_features(pro_file, smet_file)
                prob  = predict_proba_series(model, scaler, daily)

                # Grace zone: ±(PRE_EVENT_WINDOW+1) around every known event
                ev_dates: set[pd.Timestamp] = set()
                for d_str in station_folds['test_date']:
                    ev_dates.add(pd.Timestamp(d_str).normalize())
                grace: set[pd.Timestamp] = set()
                for ed in ev_dates:
                    for delta in range(-(PRE_EVENT_WINDOW + 1), PRE_EVENT_WINDOW + 2):
                        grace.add(ed + pd.Timedelta(days=delta))

                non_event = prob[~prob.index.floor('D').isin(grace)].dropna()
                fa_days   = int((non_event >= T).sum())
                ne_days   = len(non_event)
                fa_rate   = fa_days / ne_days if ne_days else float('nan')

                op_rows.append({
                    'station_id':        sid,
                    'n_folds':           len(station_folds),
                    'threshold':         round(T, 3),
                    'false_alarm_days':  fa_days,
                    'non_event_days':    ne_days,
                    'false_alarm_rate':  round(fa_rate, 3) if not np.isnan(fa_rate) else float('nan'),
                    'pct_ne_days':       round(fa_rate * 100, 1) if not np.isnan(fa_rate) else float('nan'),
                })
                print(f"  {sid}: T={T:.3f}  FA days={fa_days}/{ne_days}  "
                      f"({fa_rate*100:.1f}%)")

            if op_rows:
                op_df = pd.DataFrame(op_rows).sort_values('false_alarm_rate').reset_index(drop=True)
                op_df.to_csv(out_dir / "evaluation_operational.csv", index=False)
                print(f"  → saved evaluation_operational.csv")

                # ── Operational threshold figure ──────────────────────────────
                def _tier_color(fa: float) -> str:
                    if np.isnan(fa): return '#aaaaaa'
                    if fa <= 0.10:   return '#2ca02c'
                    if fa <= 0.25:   return '#ff7f0e'
                    return '#d62728'

                colors = [_tier_color(r) for r in op_df['false_alarm_rate']]
                labels = [r.replace('_res', '') for r in op_df['station_id']]

                fig, axes = plt.subplots(1, 2, figsize=(13, 6))

                # Panel 1: false alarm days per station
                ax1 = axes[0]
                ax1.barh(range(len(op_df)), op_df['pct_ne_days'],
                         color=colors, edgecolor='white', height=0.7)
                ax1.axvline(10, color='black', linestyle='--', linewidth=1.2,
                            label='10 % operational threshold')
                ax1.set_yticks(range(len(op_df)))
                ax1.set_yticklabels(labels, fontsize=9)
                ax1.set_xlabel('False alarm rate  (% of non-event days)', fontsize=10)
                ax1.set_title('False Alarm Rate at\nRecall-Maximizing Threshold', fontsize=11)
                ax1.set_xlim(0, 108)
                for i, (_, row) in enumerate(op_df.iterrows()):
                    fa_pct = row['pct_ne_days'] if not np.isnan(row['pct_ne_days']) else 0
                    ax1.text(min(fa_pct + 1.5, 105), i,
                             f"T={row['threshold']:.3f}  {int(row['false_alarm_days'])} / "
                             f"{int(row['non_event_days'])} days",
                             va='center', fontsize=7.5, color='#333')
                ax1.legend(fontsize=9, loc='lower right')
                ax1.grid(axis='x', alpha=0.3)

                # Panel 2: threshold vs false alarm rate
                ax2 = axes[1]
                for _, row in op_df.iterrows():
                    if np.isnan(row['false_alarm_rate']): continue
                    ax2.scatter(row['threshold'], row['pct_ne_days'],
                                color=_tier_color(row['false_alarm_rate']),
                                s=80 + row['n_folds'] * 30,
                                zorder=3, edgecolors='white', linewidth=0.8)
                    ax2.annotate(row['station_id'].replace('_res', ''),
                                 (row['threshold'], row['pct_ne_days']),
                                 textcoords='offset points', xytext=(5, 3),
                                 fontsize=7.5, color='#333')
                ax2.axhline(10, color='black', linestyle='--', linewidth=1.2,
                            label='10% operational threshold')
                ax2.set_xlabel('Recall-maximizing threshold\n'
                               '(min event prob across LOO folds)', fontsize=10)
                ax2.set_ylabel('False alarm rate  (% of non-event days)', fontsize=10)
                ax2.set_title('Threshold vs False Alarm Rate\n'
                              '(bubble size ∝ training folds)', fontsize=11)
                ax2.set_xlim(-0.05, 1.05)
                ax2.set_ylim(-3, 105)
                from matplotlib.patches import Patch
                ax2.legend(handles=[
                    Patch(color='#2ca02c', label='Ready  (FA ≤ 10%)'),
                    Patch(color='#ff7f0e', label='Marginal  (10–25%)'),
                    Patch(color='#d62728', label='Not ready  (> 25%)'),
                ], fontsize=8.5, loc='upper right')
                ax2.grid(alpha=0.3)

                plt.suptitle(
                    'Operational Threshold Analysis — Per-Station Recall-Maximizing Thresholds\n'
                    f'{SEASON} season  |  Threshold = min(event prob across LOO folds)',
                    fontsize=11, y=1.01,
                )
                plt.tight_layout()
                fig.savefig(str(out_dir / "operational_thresholds.png"),
                            dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  → saved operational_thresholds.png")

    # --- Operational fit: blended model trained on ALL available data -----------
    # Uses observations and features from every season so nothing is held out.
    # Training: combined multi-season features + all event dates (both seasons).
    # Prediction: 2025-26 features only — the map always shows the current season.

    op_event_dates: dict[str, list] = {}
    for _idx, (_, _row) in enumerate(df_all.iterrows()):
        _pf = matched_pro_files[_idx]
        if _pf is None:
            continue
        _rd = pd.to_datetime(_row.get('Date', ''), errors='coerce')
        if pd.isna(_rd):
            continue
        op_event_dates.setdefault(_pf.stem, [])
        if _rd not in op_event_dates[_pf.stem]:
            op_event_dates[_pf.stem].append(_rd)

    n_op_events = sum(len(v) for v in op_event_dates.values())
    n_op_seasons = len({r['source_season'] for _, r in df_all.iterrows()})
    print(f"\nFitting operational blended model on full dataset "
          f"({n_op_events} events across {len(op_event_dates)} stations, "
          f"{n_op_seasons} season(s))...")

    # Combined features per station (both seasons) used only for training.
    # daily_dict (2025-26 only) is used for prediction so the map time series
    # covers the current season.
    op_train_daily: dict[str, pd.DataFrame] = {}
    for _pf in sorted(unique_pro_files):
        _sid = _pf.stem
        _combined = _collect_all_season_features(_sid, sims_root)
        if _combined is not None and not _combined.empty:
            op_train_daily[_sid] = _combined

    # Per-station operational models: train and predict on combined features so the
    # probability time series spans ALL seasons (enables full-range time slider).
    op_ml_stations: set[str] = set()
    op_prob_dict:   dict[str, pd.Series] = {}
    for _pf in sorted(unique_pro_files):
        _sid        = _pf.stem
        _all_daily  = op_train_daily.get(_sid)   # combined both seasons
        if _all_daily is None:
            continue
        _ev = op_event_dates.get(_sid, [])
        if _ev:
            _fitted_op = train_station(_all_daily, _ev, verbose=False)
            if _fitted_op is not None:
                _m, _sc = _fitted_op
                op_prob_dict[_sid] = predict_proba_series(_m, _sc, _all_daily)
                op_ml_stations.add(_sid)

    # Operational regional model: train and predict on combined features.
    print(f"  Training operational regional model (all seasons)...")
    _op_reg_fitted = train_regional(op_train_daily, op_event_dates)
    op_reg_prob_dict: dict[str, pd.Series] = {}
    if _op_reg_fitted is not None:
        _op_rm, _op_rsc = _op_reg_fitted
        op_reg_prob_dict = {
            _sid: predict_proba_series(_op_rm, _op_rsc, _df)
            for _sid, _df in op_train_daily.items()
        }

    # Blended: per-station weighted by event count, backed by regional, then Sn38.
    # All series span combined seasons — the time slider covers historical dates.
    op_blended_dict: dict[str, pd.Series] = {}
    for _pf in sorted(unique_pro_files):
        _sid      = _pf.stem
        _all_daily = op_train_daily.get(_sid)
        _daily_26  = daily_dict.get(_pf)          # 2025-26 only (fallback Sn38)
        _ref       = _all_daily if _all_daily is not None else _daily_26
        if _ref is None:
            continue
        _p_reg = op_reg_prob_dict.get(_sid)
        _p_sta = op_prob_dict.get(_sid) if _sid in op_ml_stations else None
        _n_ev  = len(op_event_dates.get(_sid, []))
        if _p_reg is not None:
            op_blended_dict[_sid] = blend_probabilities(_p_sta, _p_reg, _n_ev)
        elif _p_sta is not None:
            op_blended_dict[_sid] = _p_sta
        elif 'sn38_min' in _ref.columns:
            _sn38 = _ref['sn38_min'].fillna(3.0)
            op_blended_dict[_sid] = 1.0 / (1.0 + np.exp(2.0 * (_sn38 - 1.0)))

    # Update regional overlay with operational probs; is_ready = has per-station ML model
    regional_prob_dict = op_reg_prob_dict
    _ml_pro_files: set[Path] = {_pf for _pf in unique_pro_files if _pf.stem in op_ml_stations}

    # Map-facing probability series and forecast-window values
    prob_by_station: dict[str, pd.Series] = op_blended_dict
    forecast_probs: list[float] = []
    for pf in matched_pro_files:
        _sid = pf.stem if pf is not None else None
        if _sid is None or _sid not in prob_by_station:
            forecast_probs.append(float('nan'))
        else:
            s      = prob_by_station[_sid]
            window = s[(s.index >= win_start) & (s.index <= win_end)]
            forecast_probs.append(
                float(window.max()) if not window.dropna().empty else float('nan')
            )
    df_all['forecast_prob'] = forecast_probs
    df_all['station_id'] = [pf.stem if pf is not None else None
                            for pf in matched_pro_files]

    # --- Build per-station simulation metadata for the map overlay ---
    sim_stations: list[dict] = []
    for pro_file in sorted(sim_dir.glob('*.pro')):
        coords = extract_pro_coordinates(pro_file)
        if coords is None:
            continue
        lat, lon = coords
        stem    = pro_file.stem
        prob    = None
        if stem in prob_by_station:
            s = prob_by_station[stem]
            win = s[(s.index >= win_start) & (s.index <= win_end)]
            v   = float(win.max()) if not win.dropna().empty else None
            prob = round(v, 4) if v is not None else None
        reg_prob = None
        if stem in regional_prob_dict:
            s_reg = regional_prob_dict[stem]
            win_reg = s_reg[(s_reg.index >= win_start) & (s_reg.index <= win_end)]
            v_reg   = float(win_reg.max()) if not win_reg.dropna().empty else None
            reg_prob = round(v_reg, 4) if v_reg is not None else None
        sim_stations.append({
            'id':               stem,
            'lat':              lat,
            'lon':              lon,
            'aspect':           _station_aspect(stem),
            'forecast_prob':    prob,
            'confidence_tier':  _station_tier(stem),
            'regional_prob':    reg_prob,
            'url':              f"assets/{SEASON}/stability_{stem}.html",
        })

    # --- Generate the map ---
    map_path = root_dir / "index.html"
    print(f"\nGenerating map → {map_path}")
    create_avalanche_map(df_all, map_path, forecast_date=forecast_date,
                         prob_by_station=prob_by_station,
                         sim_stations=sim_stations)

    print("\nDone.")
    print(f"Open {map_path} and click any marker to view its nearest station's stability.")


if __name__ == "__main__":
    main()
