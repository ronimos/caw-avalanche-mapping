"""
Forecast-only mode (main.py --forecast-only).

Operational counterpart of main(): no held-out test season and no evaluation.

  Training:   every season under data/simulations/ except FORECAST_SEASON.
              Observations from those seasons are matched to their own season's
              stations (as main() does) and supply the event labels.
  Prediction: every .pro/.smet station in data/simulations/FORECAST_SEASON/
              (the FOCUS stations copied by scripts/sync_pro_files.py --focus).

Models are the same three as the operational fit in main(): per-station
(stations with >= 2 positive days of history), pooled regional, and the
hierarchical Bayesian model, combined through the same model switch.

Confidence tiers and the hierarchical reliability flag cannot be computed
without held-out events, so they are read from the last evaluation run
(output/evaluation_operational.csv, output/hierarchical_uncertainty.csv) when
present.  Stations missing from those files are 'not_ready' / unchecked.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from classifier import (blend_probabilities, predict_proba_series,
                        train_regional, train_station)
from features import build_daily_features
from main import (FORECAST_SEASON, HIER_RELIABILITY_THRESHOLD, TIER_FA_MARGINAL,
                  TIER_FA_READY, _build_sim_stations, _choose_models,
                  _collect_all_season_features, _hier_known_event_ref,
                  _load_observations, _render_stability_plots,
                  _station_event_dates)
from snowpack_io import find_nearest_pro
from visualization import create_avalanche_map

# Any id not in HierModel.station_list selects the population-level prior.
_UNSEEN_STATION = "__population__"


def _load_tiers(out_dir: Path) -> dict[str, str]:
    """station_id → ready/marginal/not_ready from the last evaluation run."""
    path = out_dir / "evaluation_operational.csv"
    if not path.exists():
        return {}
    op_df = pd.read_csv(path).dropna(subset=['false_alarm_rate'])
    return {
        r.station_id: ('ready' if r.false_alarm_rate <= TIER_FA_READY
                       else 'marginal' if r.false_alarm_rate <= TIER_FA_MARGINAL
                       else 'not_ready')
        for r in op_df.itertuples()
    }


def _load_hier_reliability(out_dir: Path) -> dict[str, bool]:
    """station_id → hierarchical reliability flag from the last evaluation run."""
    path = out_dir / "hierarchical_uncertainty.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path).dropna(subset=['reliable'])
    return {r.station_id: bool(r.reliable) for r in df.itertuples()}


def _population_event_ref(
    hmodel, train_feats: dict[str, pd.DataFrame],
    station_events: dict[str, list[pd.Timestamp]],
) -> float | None:
    """
    Regional counterpart of main._hier_known_event_ref, for stations with no
    history: the mean probability the hierarchical model's population-level
    prior (what an unseen station gets) assigns to every known training
    avalanche day across all stations.  1.0x on these stations' charts then
    means "as strong a signal as a typical real event in the region".
    """
    preds: list[float] = []
    for sid, events in station_events.items():
        feats = train_feats.get(sid)
        if feats is None or not events:
            continue
        mean_p, _ = hmodel.predict(_UNSEEN_STATION, feats)
        for d in {pd.Timestamp(e).normalize() for e in events}:
            if d in mean_p.index and pd.notna(mean_p[d]):
                preds.append(float(mean_p[d]))
    return sum(preds) / len(preds) if preds else None


def run_forecast(args: argparse.Namespace) -> None:
    root_dir   = Path(__file__).resolve().parents[1]
    sims_root  = root_dir / "data" / "simulations"
    fc_dir     = sims_root / FORECAST_SEASON
    obs_dir    = root_dir / "data" / "observations"
    out_dir    = root_dir / "output"
    models_dir = root_dir / "models" / "forecast"
    site_dir   = (args.site_dir or out_dir / "forecast" / FORECAST_SEASON).resolve()
    assets_dir = site_dir / "assets" / FORECAST_SEASON
    stations_csv = root_dir / "data" / "snowpack_stations_locations.csv"

    if not fc_dir.is_dir():
        print(f"Forecast simulations not found: {fc_dir}\n"
              f"Run scripts/sync_pro_files.py --season {FORECAST_SEASON} --focus ... first.")
        return
    hist_seasons = sorted(d.name for d in sims_root.iterdir()
                          if d.is_dir() and d.name != FORECAST_SEASON)
    for d in (models_dir, assets_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Forecast-only mode:  {FORECAST_SEASON}")
    print(f"Training seasons:    {', '.join(hist_seasons)}")
    print(f"Site dir:            {site_dir}")

    # --- Observations → historical event labels ---------------------------
    # Each observation is matched within its own season's simulation folder.
    # Forecast-season observations are not used for training: that folder only
    # holds the FOCUS stations, so nearest-station matching would snap a
    # distant observation onto an unrelated FOCUS station.
    df_obs = _load_observations(obs_dir)
    df_hist = df_obs[df_obs['source_season'].isin(hist_seasons)].copy() if not df_obs.empty else df_obs
    n_fc_obs = int((df_obs['source_season'] == FORECAST_SEASON).sum()) if not df_obs.empty else 0
    if df_hist.empty:
        print("No historical observations — nothing to train on.")
        return

    df_hist['station_id'] = [
        (pf.stem if (pf := find_nearest_pro(r['Latitude'], r['Longitude'],
                                            sims_root / r['source_season'],
                                            aspect=str(r.get('Aspect', '')),
                                            stations_csv=stations_csv)) else None)
        for _, r in df_hist.iterrows()
    ]
    df_hist['event_date'] = pd.to_datetime(df_hist['Date'], errors='coerce', format='mixed')
    station_events = _station_event_dates(df_hist)
    print(f"Training labels:     {sum(map(len, station_events.values()))} event-days at "
          f"{len(station_events)} stations ({len(df_hist)} observations"
          + (f"; {n_fc_obs} {FORECAST_SEASON} observation(s) not used)" if n_fc_obs else ")"))

    # --- Training features: all historical seasons ------------------------
    train_feats: dict[str, pd.DataFrame] = {}
    for sid in sorted(station_events):
        feats = _collect_all_season_features(sid, sims_root, exclude_seasons={FORECAST_SEASON})
        if feats is not None and not feats.empty:
            train_feats[sid] = feats
    print(f"Training features:   {len(train_feats)} station(s)")

    # --- Forecast-season features -----------------------------------------
    fc_pros = sorted(p for p in fc_dir.glob("*.pro") if p.with_suffix('.smet').exists())
    print(f"\nBuilding {FORECAST_SEASON} features for {len(fc_pros)} station files...")
    fc_daily: dict[str, pd.DataFrame] = {}
    for pf in fc_pros:
        daily = build_daily_features(pf, pf.with_suffix('.smet'))
        if not daily.empty:
            fc_daily[pf.stem] = daily
    if not fc_daily:
        print("No forecast-season features — nothing to predict.")
        return
    last_day = max(df.index.max() for df in fc_daily.values())
    forecast_date = pd.Timestamp(args.forecast_date) if args.forecast_date else last_day.normalize()
    win_start = forecast_date - pd.Timedelta(days=1)
    win_end   = forecast_date + pd.Timedelta(days=2)
    print(f"Forecast date:       {forecast_date.date()}  (data through {last_day.date()})")

    # --- Per-station models (forecast stations with their own history) -----
    sta_prob: dict[str, pd.Series] = {}
    for sid in sorted(set(fc_daily) & set(train_feats)):
        fitted = train_station(train_feats[sid], station_events.get(sid, []), verbose=False)
        if fitted is not None:
            sta_prob[sid] = predict_proba_series(*fitted, fc_daily[sid])
    print(f"\nPer-station models:  {len(sta_prob)} forecast station(s) have enough history")

    # --- Regional pooled model --------------------------------------------
    reg_prob: dict[str, pd.Series] = {}
    reg_fitted = train_regional(train_feats, station_events)
    if reg_fitted is None:
        print("Regional model:      not enough positives — skipped")
    else:
        reg_prob = {sid: predict_proba_series(*reg_fitted, df) for sid, df in fc_daily.items()}
        print(f"Regional model:      trained on {len(train_feats)} station(s)")

    blended: dict[str, pd.Series] = {}
    for sid in fc_daily:
        if sid in reg_prob:
            blended[sid] = blend_probabilities(sta_prob.get(sid), reg_prob[sid],
                                               len(station_events.get(sid, [])))
        elif sid in sta_prob:
            blended[sid] = sta_prob[sid]

    # --- Hierarchical Bayesian model (cached; MCMC only on --retrain) ------
    hier_prob: dict[str, pd.Series] = {}
    hier_ci:   dict[str, tuple[pd.Series, pd.Series]] = {}
    hier_ref:  dict[str, float] = {}
    regional_ref_ids: set[str] = set()
    if not args.no_hierarchical:
        hier_path = models_dir / "hierarchical.joblib"
        hmodel = None
        if not args.retrain and hier_path.exists():
            hmodel = joblib.load(hier_path)
            print(f"Hierarchical model:  loaded {hier_path.relative_to(root_dir)}")
        else:
            try:
                from hierarchical import train_hierarchical
                print(f"\nTraining hierarchical Bayesian model ({len(train_feats)} stations)...")
                hmodel = train_hierarchical(train_feats, station_events)
                if hmodel is not None:
                    joblib.dump(hmodel, hier_path)
                    print(f"  → saved {hier_path.relative_to(root_dir)}")
            except ImportError:
                print("PyMC not available — hierarchical model skipped.")
        if hmodel is not None:
            for sid, df in fc_daily.items():
                hier_prob[sid], _ = hmodel.predict(sid, df)
                _, lo, hi = hmodel.predict_interval(sid, df, ci=0.95)
                hier_ci[sid] = (lo, hi)
                if sid in train_feats:
                    ref = _hier_known_event_ref(hmodel, sid, train_feats[sid],
                                                station_events.get(sid, []))
                    if ref is not None:
                        hier_ref[sid] = ref
            n_seen = sum(sid in hmodel.station_list for sid in fc_daily)
            print(f"Hierarchical model:  {n_seen}/{len(fc_daily)} forecast stations were in "
                  f"training; the rest use the population-level prior")
            pop_ref = _population_event_ref(hmodel, train_feats, station_events)
            if pop_ref is not None:
                regional_ref_ids = {sid for sid in fc_daily if sid not in hier_ref}
                for sid in regional_ref_ids:
                    hier_ref[sid] = pop_ref
                print(f"Regional event ref:  p={pop_ref:.4f} on known training event days "
                      f"(used for {len(regional_ref_ids)} stations without their own)")

    # --- Model switch (tiers / reliability from the last evaluation run) ---
    tiers    = _load_tiers(out_dir)
    reliable = _load_hier_reliability(out_dir)
    if not tiers:
        print("\nNo output/evaluation_operational.csv — every station is tier 'not_ready'. "
              "Run main.py without --forecast-only to compute tiers.")
    if not reliable:
        print(f"No output/hierarchical_uncertainty.csv — hierarchical reliability "
              f"(≥ {HIER_RELIABILITY_THRESHOLD}x known event) is unchecked.")
    prob_by_station, model_used = _choose_models(blended, hier_prob, tiers, reliable)
    counts = pd.Series(model_used).value_counts().to_dict()
    print(f"Model used:          {counts}")

    # --- Stability charts + map --------------------------------------------
    plot_queue = [{
        'pro_file':      fc_dir / f"{sid}.pro",
        'out_path':      assets_dir / f"stability_{sid}.html",
        'station_id':    sid,
        'event_dates':   [],
        'test_dates':    [],
        'daily_df':      fc_daily[sid],
        'forecast_date': forecast_date,
    } for sid in sorted(fc_daily)]
    _render_stability_plots(plot_queue, prob_by_station, model_used,
                            hier_ref, hier_ci, tiers, regional_ref_ids=regional_ref_ids,
                            viewer_year=FORECAST_SEASON[:4])

    sim_stations = _build_sim_stations(
        fc_dir, FORECAST_SEASON, prob_by_station, reg_prob, model_used,
        hier_ref, tiers, win_start, win_end, regional_ref_ids=regional_ref_ids)

    # Forecast-season observations are shown on the map (unmatched, so no
    # chart link); with none yet the observation layer is simply empty.
    df_map = (df_obs[df_obs['source_season'] == FORECAST_SEASON].copy()
              if not df_obs.empty else pd.DataFrame())
    for col in ('Latitude', 'Longitude', 'Placemark Name'):
        if col not in df_map.columns:
            df_map[col] = pd.Series(dtype=object)
    df_map['target_url'] = ''

    map_path = site_dir / "index.html"
    create_avalanche_map(df_map, map_path, forecast_date=forecast_date,
                         prob_by_station=prob_by_station, sim_stations=sim_stations,
                         highway_geojson=root_dir / "data" / "geo" / "pamir_highway.geojson")
    print(f"\n→ Map: {map_path}")
    print(f"→ Charts: {assets_dir} ({len(plot_queue)} files)")
