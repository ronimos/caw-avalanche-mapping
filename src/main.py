import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from classifier import (aggregate_predictions, blend_probabilities,
                        evaluate_event_level, evaluate_regional, evaluate_station,
                        plot_feature_importance, predict_proba_series,
                        train_regional, train_station)
from features import build_daily_features
from snowpack_io import extract_pro_coordinates, find_nearest_pro, parse_snow_data
from visualization import (create_avalanche_map, plot_interactive_stability,
                           plot_static_map)

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
EVAL_FORECAST_DATE = "2026-01-23"

# Forecast-only mode (--forecast-only, see src/forecast.py): no held-out
# evaluation.  Models are trained on every season under data/simulations/
# except FORECAST_SEASON, then predict every station in FORECAST_SEASON
# (populated by scripts/sync_pro_files.py --focus).
FORECAST_SEASON = "2026-2027"

# Best-performing station; also gets a static PNG export of its stability
# chart for the case-study figure in the paper (docs/paper/figures/).
CASE_STUDY_STATION = "160942_res"

# Confidence-tier false-alarm-rate cutoffs (§4.5 operational threshold
# analysis). Tiers themselves are computed live each run from
# evaluation_operational.csv (see station_tier_map in main()) rather than
# hand-maintained here, so they track each station's most recent FA rate as
# it accumulates events across seasons.
TIER_FA_READY    = 0.10  # FA <= this -> 'ready'
TIER_FA_MARGINAL = 0.25  # FA <= this -> 'marginal'; above -> 'not_ready'

# A station's hierarchical prediction on a real, held-out future event must
# reach at least this fraction of the model's own average prediction on that
# station's KNOWN training event(s) to count as "reliable" (see
# _hier_known_event_ref / hier_reliable in main()) — otherwise the model
# isn't generalizing there even by its own calibration, whatever its raw
# number says, and the station falls back to blended regardless of tier.
HIER_RELIABILITY_THRESHOLD = 0.5

_ASPECT_SUFFIX = {'1': 'N', '2': 'E', '3': 'S', '4': 'W'}

# Station ids are 5 digits; an aspect file appends one digit (1-4). Going by
# length, not by the last digit: a flat file whose id ends in 1-4 (13874_res)
# is not west-facing.
STATION_ID_DIGITS = 5

# niViz viewer of a station's simulation, all five aspects (from the
# snowpack-mapping repo, served next to the CAW conditions map). It only has
# .pro files for the current run, so charts link to it in forecast mode only.
PROFILE_VIEWER_URL = "/nwp/caw/conditions/profiles.html?year={year}&id={station}"


def _split_stem(stem: str) -> tuple[str, str]:
    """'231581_res' → ('23158', 'N'); '13874_res' → ('13874', 'Flat')."""
    numeric = stem.replace('_res', '')
    if len(numeric) == STATION_ID_DIGITS + 1 and numeric[-1] in _ASPECT_SUFFIX:
        return numeric[:-1], _ASPECT_SUFFIX[numeric[-1]]
    return numeric, 'Flat'


def _station_aspect(stem: str) -> str:
    return _split_stem(stem)[1]


def _station_tier(stem: str, tier_map: dict[str, str]) -> str:
    """Ready/marginal/not_ready from the live per-run false-alarm-rate
    analysis. Stations with no operational threshold yet (too few events to
    run the LOO sweep) default to 'not_ready'."""
    return tier_map.get(stem, 'not_ready')


def _hier_known_event_ref(
    hmodel, station_id: str, train_daily: pd.DataFrame, train_event_dates: list,
) -> float | None:
    """
    Average probability the hierarchical model itself assigns to a station's
    own known TRAIN_SEASON avalanche day(s) — "what does a real event look
    like here, according to this model." This is both:

      - the normalization reference for that station's chart (a day is
        reported as Nx this reference, not a raw percentage that isn't
        comparable across stations or against the blended model's scale —
        hierarchical numbers are shrunk toward the ~1-2% regional base rate
        and so aren't meaningfully read as absolute percentages at all), and
      - the reliability check below (hier_reliable in main()): a model whose
        held-out prediction on an actual future event never approaches this
        reference isn't generalizing for that station, whatever its raw
        number says.

    Returns None if the station has no known training events.
    """
    if not train_event_dates:
        return None
    mean_p, _ = hmodel.predict(station_id, train_daily)
    preds = [
        float(mean_p[d]) for d in (pd.Timestamp(x).normalize() for x in train_event_dates)
        if d in mean_p.index and pd.notna(mean_p[d])
    ]
    return sum(preds) / len(preds) if preds else None


def _hier_known_event_axis(
    mean_series: pd.Series, lower_series: pd.Series, upper_series: pd.Series,
    known_event_ref: float,
    regional: bool = False,
) -> dict:
    """
    Normalizes a hierarchical model's probability (and its 95% CI) by
    known_event_ref (see _hier_known_event_ref) so the chart reads "Nx a
    known avalanche day" — 1.0 meaning as strong a signal as this station's
    own confirmed history — instead of a raw percentage that means something
    different at every station and isn't comparable to the blended model's
    scale. Log-scaled: the ratio genuinely spans two-plus orders of
    magnitude both across stations and within one station's season (observed
    range so far: ~0.07x to ~33x).

    regional=True labels the reference as a regional avalanche day: used for
    stations with no history of their own, whose reference is the model's
    population-level prediction on all known training events (see
    forecast._population_event_ref).
    """
    what = 'a regional avalanche day' if regional else 'a known avalanche day'
    return {
        'prob_series': mean_series / known_event_ref,
        'prob_lower':  lower_series / known_event_ref,
        'prob_upper':  upper_series / known_event_ref,
        'y_axis_type': 'log',
        'y_range': [-2, 2],  # 0.01x-100x: comfortable margin over the observed 0.07x-33x
        'y_tickformat': '',
        'y_ticksuffix': 'x',
        'y_axis_title': f'Relative to {what}',
        'curve_hovertemplate': f"%{{y:.2f}}x {'regional' if regional else 'known'} event<extra></extra>",
        'curve_fill': None,
        'prob_bands': [],
        'prob_thresholds': [(1.0, '#555555', 'Regional event' if regional else 'Known event')],
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Avalanche stability analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--forecast-date", default=None, metavar="YYYY-MM-DD",
        help="Forecast reference date. Map markers are coloured by max probability "
             "in the window [date−1d, date+2d]; stability plots highlight the same window. "
             f"Default: {EVAL_FORECAST_DATE}, or with --forecast-only the last simulated day.",
    )
    p.add_argument(
        "--forecast-only", action="store_true",
        help=f"Operational forecast for {FORECAST_SEASON}: train on all other seasons, "
             "skip evaluation, predict every station in the forecast season.",
    )
    p.add_argument(
        "--site-dir", type=Path, default=None,
        help="Where --forecast-only writes index.html and assets/<SEASON>/. "
             f"Default: output/forecast/{FORECAST_SEASON}/ (the repo-root index.html is left alone).",
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

LOO_THRESHOLD    = 0.5  # probability above which a day counts as a warning

# Learning-curve figure (Figure 3 in the ISSW paper).  Sizes are points per
# inch of canvas, not points on the page: the figure is authored 8 in wide and
# printed at ~3.4 in column width, so on-page type is ~0.41x these values.
# Raising fontsize and figsize together is a no-op — only the ratio matters.
LOO_LABEL_SIZE  = 20
LOO_TICK_SIZE   = 18
LOO_LEGEND_SIZE = 15
LOO_ANNOT_SIZE  = 15


def _run_loo_folds(
    station_id: str,
    sims_root: Path,
    all_events: list[tuple[pd.Timestamp, str]],
    daily_sink: list[dict] | None = None,
) -> list[dict]:
    """
    Temporal leave-one-out CV for one station.

    Events are sorted chronologically.  Fold i trains on events[0..i-1] and
    tests on events[i].  Features are loaded from whichever season each event
    belongs to, so the model always trains on data it could have seen in
    real-time.

    Returns a list of per-fold result dicts.  If `daily_sink` is given, one
    record per fold-day is appended to it (labelled event_window / grace /
    non_event) so threshold sweeps can be replayed without refitting.
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
            'n_non_event_days': 0,
            'false_alarm_days': 0,
            'false_alarms':     0,
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

            # False alarms on non-event days.  Two counts: raw days above the
            # threshold, and alarm episodes (runs of consecutive warning days
            # collapsed to one) — a forecaster reacts to the episode, not to
            # each day of a multi-day alarm.
            daily_ne = non_grace.groupby(non_grace.index.floor('D')).max().sort_index()
            fold['n_non_event_days'] = int(len(daily_ne))
            if not daily_ne.empty:
                warned = daily_ne >= LOO_THRESHOLD
                fold['false_alarm_days'] = int(warned.sum())
                # New episode where a warning day follows a day that was not a
                # consecutive warning day.
                prev_day_warned = warned.shift(1, fill_value=False) & (
                    daily_ne.index.to_series().diff().eq(pd.Timedelta(days=1)).fillna(False).to_numpy()
                )
                fold['false_alarms'] = int((warned & ~prev_day_warned).sum())

            if daily_sink is not None:
                daily = prob.dropna()
                daily = daily.groupby(daily.index.floor('D')).max().sort_index()
                for day, p in daily.items():
                    kind = ('event_window' if w_start <= day <= w_end
                            else 'grace' if day in grace else 'non_event')
                    daily_sink.append({
                        'station_id':     station_id,
                        'n_train_events': len(train_dates),
                        'test_date':      test_date.date(),
                        'day':            day.date(),
                        'prob':           float(p),
                        'kind':           kind,
                    })

        results.append(fold)

    return results


def _count_alarm_episodes(warned: pd.Series, days: pd.Series) -> int:
    """Runs of consecutive warned days, counted once each (calendar gaps split runs)."""
    m = warned.to_numpy()
    consecutive = np.r_[False, (days.diff() == pd.Timedelta(days=1)).to_numpy()[1:]]
    prev_warned = np.r_[False, m[:-1]] & consecutive
    return int((m & ~prev_warned).sum())


def _print_threshold_sweep(
    daily: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95),
) -> None:
    """
    Detection rate / precision / F1 by number of training events at several
    absolute probability thresholds, plus a per-fold background percentile.

    A fold is detected when any day in its event window clears the threshold;
    a false positive is one run of consecutive non-event days above it.
    """
    daily = daily.copy()
    daily['day'] = pd.to_datetime(daily['day'])

    def report(label: str, thr_of_fold) -> None:
        print(f"\n  Threshold: {label}")
        print(f"  {'n_train':>8}  {'folds':>6}  {'detection_rate':>14}  "
              f"{'false_alarms':>13}  {'fa_day_rate':>12}  "
              f"{'precision':>10}  {'f1':>6}")
        for n, grp_n in daily.groupby('n_train_events'):
            tp = n_folds = fp = fa_days = ne_days = 0
            for _, fold in grp_n.groupby(['station_id', 'test_date']):
                fold = fold.sort_values('day')
                window = fold[fold['kind'] == 'event_window']
                if window.empty:
                    continue
                thr = thr_of_fold(fold)
                n_folds += 1
                tp += int(window['prob'].max() >= thr)
                non_event = fold[fold['kind'] == 'non_event']
                warned = non_event['prob'] >= thr
                fa_days += int(warned.sum())
                ne_days += len(non_event)
                fp += _count_alarm_episodes(warned, non_event['day'])
            if not n_folds:
                continue
            recall    = tp / n_folds
            precision = tp / (tp + fp) if (tp + fp) else float('nan')
            f1 = (2 * precision * recall / (precision + recall)
                  if precision and recall and not pd.isna(precision) else 0.0)
            print(f"  {n:>8}  {n_folds:>6}  {recall:>14.1%}  {fp:>13}  "
                  f"{fa_days/ne_days if ne_days else float('nan'):>12.1%}  "
                  f"{precision:>10.3f}  {f1:>6.3f}")

    print("\n  LOO threshold sweep:")
    for t in thresholds:
        report(f"absolute {t}", lambda _fold, t=t: t)
    report("p95 of each fold's non-event background",
           lambda fold: fold.loc[fold['kind'] == 'non_event', 'prob'].quantile(0.95))


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


def _station_event_dates(
    df: pd.DataFrame,
    mask: pd.Series | None = None,
) -> dict[str, list[pd.Timestamp]]:
    """
    Build station_id → unique event dates (insertion order) from observation
    rows.  Requires the 'station_id' and 'event_date' columns added in main();
    rows without a matched station or a parseable date are skipped.
    """
    sub = df if mask is None else df[mask]
    sub = sub[sub['station_id'].notna() & sub['event_date'].notna()]

    out: dict[str, list[pd.Timestamp]] = {}
    for sid, date in zip(sub['station_id'], sub['event_date']):
        dates = out.setdefault(sid, [])
        if date not in dates:
            dates.append(date)
    return out


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


def _choose_models(
    op_blended_dict: dict[str, pd.Series],
    hier_prob_dict: dict[str, pd.Series],
    station_tier_map: dict[str, str],
    hier_reliable: dict[str, bool],
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """
    Per-station model switch (see the --- Model switch --- comment in main()).
    Returns (prob_by_station, model_used).
    """
    model_used: dict[str, str] = {}
    prob_by_station: dict[str, pd.Series] = {}
    for _sid in set(op_blended_dict) | set(hier_prob_dict):
        _tier = station_tier_map.get(_sid, 'not_ready')
        _hier_ok = hier_reliable.get(_sid, True)  # benefit of the doubt if unchecked (no test event yet)
        if _tier == 'ready' and _sid in op_blended_dict:
            prob_by_station[_sid] = op_blended_dict[_sid]
            model_used[_sid] = 'blended'
        elif _sid in hier_prob_dict and _hier_ok:
            prob_by_station[_sid] = hier_prob_dict[_sid]
            model_used[_sid] = 'hierarchical'
        elif _sid in op_blended_dict:
            prob_by_station[_sid] = op_blended_dict[_sid]
            model_used[_sid] = 'blended' if _hier_ok else 'blended_fallback'
        else:
            prob_by_station[_sid] = hier_prob_dict[_sid]
            model_used[_sid] = 'hierarchical'
    return prob_by_station, model_used


def _render_stability_plots(
    plot_queue: list[dict],
    prob_by_station: dict[str, pd.Series],
    model_used: dict[str, str],
    hier_known_event_ref: dict[str, float],
    hier_ci_dict: dict[str, tuple[pd.Series, pd.Series]],
    station_tier_map: dict[str, str],
    png_dir: Path | None = None,
    regional_ref_ids: set[str] = frozenset(),
    viewer_year: str | None = None,
) -> None:
    """Write one interactive stability chart per plot_queue item. When png_dir
    is given, CASE_STUDY_STATION also gets a static PNG of its probability row.
    viewer_year (season start year) adds an "Open SNOWPACK viewer" button."""
    if not plot_queue:
        return
    print(f"\nGenerating {len(plot_queue)} stability plots...")
    for item in plot_queue:
        sid = item['station_id']
        print(f"  Plotting {item['pro_file'].name}...")
        sid_model = model_used.get(sid, 'blended')
        prob_series = prob_by_station.get(sid)
        prob_lower = prob_upper = None
        prob_bands = prob_thresholds = y_tickvals = y_ticktext = None
        y_axis_type, y_range = 'linear', [0, 1]
        y_tickformat, y_ticksuffix = '.0%', ''
        curve_hovertemplate, curve_fill = 'Prob: %{y:.1%}<extra></extra>', 'tozeroy'
        y_axis_title = 'Probability'
        right_margin = 140
        known_ref = hier_known_event_ref.get(sid)
        if sid_model == 'hierarchical' and prob_series is not None and known_ref is not None:
            lower_s, upper_s = hier_ci_dict.get(sid, (None, None))
            axis = _hier_known_event_axis(prob_series, lower_s, upper_s, known_ref,
                                          regional=sid in regional_ref_ids)
            prob_series          = axis['prob_series']
            prob_lower           = axis['prob_lower']
            prob_upper           = axis['prob_upper']
            prob_bands           = axis['prob_bands']
            prob_thresholds      = axis['prob_thresholds']
            y_axis_type          = axis['y_axis_type']
            y_range              = axis['y_range']
            y_tickformat         = axis['y_tickformat']
            y_ticksuffix         = axis['y_ticksuffix']
            y_axis_title         = axis['y_axis_title']
            curve_hovertemplate  = axis['curve_hovertemplate']
            curve_fill           = axis['curve_fill']
            right_margin         = 280
        is_case_study = png_dir is not None and sid == CASE_STUDY_STATION
        plot_interactive_stability(
            parse_snow_data(item['pro_file']),
            item['out_path'],
            _split_stem(sid)[0],  # title shows the station; the aspect has its own field
            event_dates=item['event_dates'],
            test_dates=item['test_dates'],
            prob_series=prob_series,
            prob_lower=prob_lower, prob_upper=prob_upper,
            prob_bands=prob_bands, prob_thresholds=prob_thresholds,
            y_tickvals=y_tickvals, y_ticktext=y_ticktext, y_axis_title=y_axis_title,
            y_axis_type=y_axis_type, y_range=y_range,
            y_tickformat=y_tickformat, y_ticksuffix=y_ticksuffix,
            curve_hovertemplate=curve_hovertemplate, curve_fill=curve_fill,
            right_margin=right_margin,
            daily_df=item['daily_df'],
            forecast_date=item['forecast_date'],
            aspect=_station_aspect(sid),
            confidence_tier=_station_tier(sid, station_tier_map),
            model_used=sid_model,
            png_path=(png_dir / f"fig4_stability_{CASE_STUDY_STATION.replace('_res', '')}.png"
                      if is_case_study else None),
            png_rows=['prob'] if is_case_study else None,
            viewer_url=(PROFILE_VIEWER_URL.format(year=viewer_year, station=_split_stem(sid)[0])
                        if viewer_year else None),
        )
        print(f"  → {item['out_path']}  (model: {sid_model})")


def _build_sim_stations(
    sim_dir: Path,
    season: str,
    prob_by_station: dict[str, pd.Series],
    regional_prob_dict: dict[str, pd.Series],
    model_used: dict[str, str],
    hier_known_event_ref: dict[str, float],
    station_tier_map: dict[str, str],
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
    regional_ref_ids: set[str] = frozenset(),
) -> list[dict]:
    """Per-station metadata (location, window-max probability, tier, chart
    URL) for the map's simulation-station overlay."""
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
        # For a hierarchical-driven station, forecast_prob alone reads as a
        # tiny, unalarming number (shrunk toward the ~1-2% regional base
        # rate) — the same reason the chart itself is normalized (see
        # _hier_known_event_axis). Carry the same ratio here so the map
        # tooltip doesn't contradict the chart it links to.
        known_ratio = None
        if model_used.get(stem) == 'hierarchical' and prob is not None:
            ref = hier_known_event_ref.get(stem)
            if ref:
                known_ratio = round(prob / ref, 3)
        sim_stations.append({
            'id':                stem,
            'lat':               lat,
            'lon':               lon,
            'aspect':            _station_aspect(stem),
            'forecast_prob':     prob,
            'confidence_tier':   _station_tier(stem, station_tier_map),
            'model_used':        model_used.get(stem, 'blended'),
            'known_event_ratio': known_ratio,
            'known_event_basis': 'regional' if stem in regional_ref_ids else 'station',
            'regional_prob':     reg_prob,
            'url':               f"assets/{season}/stability_{stem}.html",
        })
    return sim_stations


def main() -> None:
    args = _parse_args()
    if args.forecast_only:
        from forecast import run_forecast
        run_forecast(args)
        return
    forecast_date = pd.Timestamp(args.forecast_date or EVAL_FORECAST_DATE)

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
    # station_id and event_date become columns so every later stage can use
    # ordinary pandas filtering instead of positional row/list alignment.
    matched_pro_files: list[Path | None] = []
    for _, row in df_all.iterrows():
        matched_pro_files.append(find_nearest_pro(
            row['Latitude'], row['Longitude'], sim_dir,
            aspect=str(row.get('Aspect', '')),
            stations_csv=stations_csv,
        ))

    df_all['station_id'] = [pf.stem if pf is not None else None
                            for pf in matched_pro_files]
    df_all['event_date'] = pd.to_datetime(df_all['Date'], errors='coerce',
                                          format='mixed')
    df_all['target_url'] = df_all['station_id'].map(
        lambda sid: f"assets/{SEASON}/stability_{sid}.html" if sid else ""
    )
    matched = int(df_all['station_id'].notna().sum())
    print(f"Matched {matched}/{len(df_all)} observations to {SEASON} stations")

    # --- Assign train / test split ---
    # Test: any SEASON observation whose station_id was active in TRAIN_SEASON
    #       (aspect relaxed — same physical location is enough for cross-season
    #       validation even if the reported aspect differs slightly).
    train_station_ids = set(
        df_all.loc[(df_all['source_season'] == TRAIN_SEASON)
                   & df_all['station_id'].notna(), 'station_id']
    )
    is_test = (
        (df_all['source_season'] == SEASON)
        & df_all['station_id'].isin(train_station_ids)
    )
    df_all['split'] = np.where(is_test, 'test', 'train')

    print(f"Test set: {int(is_test.sum())} observation(s) "
          f"(station matched to a {TRAIN_SEASON} event, any aspect)")

    # --- Build per-station train / test event date lists ---
    # Training labels come ONLY from TRAIN_SEASON observations so that
    # 2025-26 dates never contaminate the 2024-25 feature date range.
    station_train_dates = _station_event_dates(
        df_all, (df_all['split'] != 'test')
        & (df_all['source_season'] == TRAIN_SEASON))
    station_test_dates = _station_event_dates(df_all, df_all['split'] == 'test')

    # --- Per-station: train / load classifier, predict, plot ---
    unique_pro_files = {p for p in matched_pro_files if p is not None}

    win_start = forecast_date - pd.Timedelta(days=1)
    win_end   = forecast_date + pd.Timedelta(days=2)

    prob_series_dict:  dict[Path, pd.Series]    = {}
    fitted_dict:       dict[Path, tuple]        = {}
    daily_dict:        dict[Path, pd.DataFrame] = {}
    train_features_dict: dict[str, pd.DataFrame] = {}  # station_id → 2024-25 features
    plot_queue: list[dict] = []  # deferred until prob_by_station is available

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

                # Training features: TRAIN_SEASON only (never leak test season).
                # Collected unconditionally — the regional/blended/hierarchical
                # stage needs them even when the per-station model loads from
                # cache (previously they were only built on the retrain path,
                # which silently skipped that whole stage on cached runs).
                train_pro  = sims_root / TRAIN_SEASON / f"{station_id}.pro"
                train_smet = sims_root / TRAIN_SEASON / f"{station_id}.smet"
                train_daily: pd.DataFrame | None = None
                if train_pro.exists() and train_smet.exists():
                    print(f"    collecting {TRAIN_SEASON} features...")
                    train_daily = build_daily_features(train_pro, train_smet)
                    if train_daily is not None and not train_daily.empty:
                        train_features_dict[station_id] = train_daily

                if not args.retrain and model_path.exists():
                    fitted = joblib.load(model_path)
                    print(f"    → loaded model ({model_path.name})")
                else:
                    if train_daily is None or train_daily.empty:
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

            plot_queue.append({
                'pro_file':        pro_file,
                'out_path':        out_path,
                'station_id':      station_id,
                'event_dates':     train_event_dates,
                'test_dates':      test_event_dates,
                'daily_df':        daily,
                'forecast_date':   forecast_date,
            })

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

            test_rows    = df_all[(df_all['station_id'] == station_id)
                                  & (df_all['split'] == 'test')]
            test_aspects = sorted({str(a).strip()
                                   for a in test_rows.get('Aspect', pd.Series())})

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

            fi_path = out_dir / "feature_importance.png"
            plot_feature_importance(reg_model, reg_scaler, fi_path)
            print(f"  → saved {fi_path.name}")

            # All 2025-26 event dates per station (used by regional + blended)
            station_2526_dates = _station_event_dates(
                df_all, df_all['source_season'] == SEASON)

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
            # Per-station (lower, upper) 95% credible interval, keyed like
            # hier_prob_dict — used to shade the chart when a station is
            # switched over to the hierarchical model (see model_used below).
            hier_ci_dict: dict[str, tuple[pd.Series, pd.Series]] = {}
            # Per-station known-event reference and reliability flag (see
            # _hier_known_event_ref / HIER_RELIABILITY_THRESHOLD) — hoisted out
            # here, like the two dicts above, so they're defined even when
            # --no-hierarchical or pymc is missing.
            hier_known_event_ref: dict[str, float] = {}
            hier_reliable: dict[str, bool] = {}
            hmodel = None  # defined even when --no-hierarchical, so later code can check it safely
            if not args.no_hierarchical:
                # Cached like the per-station/regional models: MCMC only runs on
                # --retrain (or a cold cache), but predictions/evaluation refresh
                # on every run.
                hier_path = models_dir / "hierarchical.joblib"
                if not args.retrain and hier_path.exists():
                    hmodel = joblib.load(hier_path)
                    print(f"\n  → loaded hierarchical model ({hier_path.name})")
                else:
                    try:
                        from hierarchical import train_hierarchical
                        print(f"\nTraining hierarchical Bayesian model "
                              f"({len(train_features_dict)} stations)...")
                        hmodel = train_hierarchical(train_features_dict,
                                                    station_train_dates)
                        if hmodel is not None:
                            joblib.dump(hmodel, hier_path)
                            print(f"  → saved {hier_path.name}")
                    except ImportError:
                        print("\n  PyMC not available — hierarchical model skipped.")

                if hmodel is not None:
                    hier_std_rows: list[dict] = []
                    # hier_known_event_ref / hier_reliable populated below — see
                    # _hier_known_event_ref and HIER_RELIABILITY_THRESHOLD.
                    for sid, df in test_features.items():
                        mean_p, std_p = hmodel.predict(sid, df)
                        hier_prob_dict[sid] = mean_p
                        _, lower_p, upper_p = hmodel.predict_interval(sid, df, ci=0.95)
                        hier_ci_dict[sid] = (lower_p, upper_p)

                        train_daily = train_features_dict.get(sid)
                        ref = (
                            _hier_known_event_ref(hmodel, sid, train_daily, station_train_dates.get(sid, []))
                            if train_daily is not None else None
                        )
                        if ref is not None:
                            hier_known_event_ref[sid] = ref
                            signals = []
                            for ed in station_test_dates.get(sid, []):
                                ew = mean_p[(mean_p.index >= ed - pd.Timedelta(days=PRE_EVENT_WINDOW))
                                            & (mean_p.index <= ed)].dropna()
                                if not ew.empty:
                                    signals.append(float(ew.max()))
                            if signals:
                                hier_reliable[sid] = (max(signals) / ref) >= HIER_RELIABILITY_THRESHOLD

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
                            'known_event_ref':  ref,
                            'reliable':         hier_reliable.get(sid),
                        })

                    if hier_reliable:
                        n_bad = sum(not ok for ok in hier_reliable.values())
                        print(f"  → hierarchical reliability: {n_bad}/{len(hier_reliable)} station(s) "
                              f"scored below {HIER_RELIABILITY_THRESHOLD}x their own known-event "
                              f"reference on a real 2025-26 event")

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
            # One row per event-level window, tagged with its station, so AUC/AP
            # confidence intervals can be bootstrapped without refitting.
            event_rows: list[dict] = []

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

                # evaluate_event_level treats stations independently, so calling
                # it one station at a time reproduces the pooled arrays exactly
                # while recording which station each window came from.
                for sid, p in prob_dict.items():
                    yt_s, ys_s = evaluate_event_level(
                        {sid: p}, station_2526_dates, PRE_EVENT_WINDOW
                    )
                    event_rows.extend(
                        {'model': label, 'station_id': sid,
                         'y_true': int(t), 'y_score': float(s)}
                        for t, s in zip(yt_s, ys_s)
                    )

            if event_rows:
                ev_df = pd.DataFrame(event_rows)
                ev_df.to_csv(out_dir / "evaluation_event_level.csv", index=False)
                print(f"  → saved evaluation_event_level.csv "
                      f"({len(ev_df)} model-windows)")

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
    ev_rows = df_all[df_all['station_id'].notna() & df_all['event_date'].notna()]
    for sid, date, season in zip(ev_rows['station_id'], ev_rows['event_date'],
                                 ev_rows['source_season']):
        evs = station_all_events.setdefault(sid, [])
        entry = (date.normalize(), season)
        if entry not in evs:
            evs.append(entry)

    for sid in station_all_events:
        station_all_events[sid].sort(key=lambda x: x[0])

    loo_rows: list[dict] = []
    loo_daily: list[dict] = []
    for pro_file in sorted(unique_pro_files):
        sid = pro_file.stem
        evs = station_all_events.get(sid, [])
        if len(evs) < 2:
            continue
        folds = _run_loo_folds(sid, sims_root, evs, daily_sink=loo_daily)
        loo_rows.extend(folds)
        n_trained = sum(1 for f in folds if f['trained'])
        print(f"  {sid}: {len(evs)} events → {len(folds)} folds  ({n_trained} trained)")

    if loo_rows:
        loo_df = pd.DataFrame(loo_rows)
        loo_df.to_csv(out_dir / "evaluation_loo.csv", index=False)
        print(f"  → saved evaluation_loo.csv  ({len(loo_df)} folds total)")

        if loo_daily:
            daily_df = pd.DataFrame(loo_daily)
            daily_df.to_csv(out_dir / "evaluation_loo_daily.csv", index=False)
            print(f"  → saved evaluation_loo_daily.csv  ({len(daily_df)} fold-days)")

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

            # Marker shape ('o' vs 'x') is redundant with color so the
            # detected/missed distinction survives grayscale printing and
            # red-green color blindness, not just the color channel.
            # '#1b7837' (not the lighter '#2ca02c') keeps ≥3:1 contrast
            # against white at this alpha (WCAG 1.4.11 non-text contrast).
            ax.scatter(x[detected],  ep[detected],
                       color='#1b7837', marker='o', s=90, alpha=0.8, zorder=3,
                       label='Detected (prob ≥ 0.5)')
            ax.scatter(x[~detected], ep[~detected],
                       color='#d62728', marker='X', s=90, alpha=0.8, zorder=3,
                       label='Missed (prob < 0.5)')

            # Mean per n_train
            mean_by_n = plot_df.groupby('n_train_events')['event_prob'].mean()
            ax.plot(mean_by_n.index.to_numpy(dtype=float), mean_by_n.to_numpy(dtype=float),
                    'k--o', linewidth=2.0, markersize=9, zorder=4, label='Mean')

            # Non-event background reference
            non_event_ref = plot_df['non_event_median'].median()
            if not pd.isna(non_event_ref):
                ax.axhline(non_event_ref, color='steelblue', linestyle=':',
                           linewidth=1.5,
                           label=f'Median non-event prob ({non_event_ref:.3f})')

            # '#595959' at full opacity clears 3:1 against white; the old
            # 'grey' at alpha=0.6 landed near 2:1 (WCAG 1.4.11 non-text contrast).
            ax.axhline(0.5, color='#595959', linestyle='--', linewidth=1.2,
                       label='0.5 threshold')

            # Fold counts per n_train
            for n, cnt in plot_df.groupby('n_train_events').size().items():
                ax.text(float(n), -0.11, f'n={cnt}', ha='center',
                        fontsize=LOO_ANNOT_SIZE, color='#555',
                        transform=ax.get_xaxis_transform())

            ax.set_xlabel('Number of training events', fontsize=LOO_LABEL_SIZE,
                          labelpad=44)
            # Shorter than the x-label and a size down: rotated text is bounded
            # by the axes height, which is the tighter constraint here.
            ax.set_ylabel(f'Event window probability\n'
                          f'({PRE_EVENT_WINDOW}-day pre-event max)',
                          fontsize=LOO_TICK_SIZE)
            ax.tick_params(axis='both', labelsize=LOO_TICK_SIZE)
            n_max = int(plot_df['n_train_events'].max())
            ax.set_xlim(0.5, n_max + 0.5)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(range(1, n_max + 1))
            # Legend sits below the x-axis label so it cannot cover the n=1
            # cluster, which is the densest part of the plot.
            ax.legend(fontsize=LOO_LEGEND_SIZE, loc='upper center',
                      bbox_to_anchor=(0.5, -0.38), ncol=2,
                      frameon=True, borderaxespad=0.0)
            ax.grid(True, alpha=0.3)
            perf_path = out_dir / "loo_performance_by_events.png"
            fig.savefig(str(perf_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  → saved {perf_path}")

            # Print summary by n_train.  Precision / F1 treat each detected
            # event window as one true positive and each non-event alarm
            # episode as one false positive.
            print("\n  LOO summary by number of training events:")
            print(f"  {'n_train':>8}  {'folds':>6}  {'mean_prob':>10}  "
                  f"{'detected':>9}  {'detection_rate':>14}  "
                  f"{'false_alarms':>13}  {'fa_days':>8}  {'fa_day_rate':>12}  "
                  f"{'precision':>10}  {'f1':>6}")
            for n, grp in plot_df.groupby('n_train_events'):
                tp = int((grp['event_prob'] >= LOO_THRESHOLD).sum())
                fp = int(grp['false_alarms'].sum())
                fa_days = int(grp['false_alarm_days'].sum())
                ne_days = int(grp['n_non_event_days'].sum())
                recall    = tp / len(grp)
                precision = tp / (tp + fp) if (tp + fp) else float('nan')
                f1 = (2 * precision * recall / (precision + recall)
                      if precision and recall and not pd.isna(precision) else 0.0)
                print(f"  {n:>8}  {len(grp):>6}  "
                      f"{grp['event_prob'].mean():>10.3f}  "
                      f"{tp:>9}  {recall:>14.1%}  "
                      f"{fp:>13}  {fa_days:>8}  "
                      f"{fa_days/ne_days if ne_days else float('nan'):>12.1%}  "
                      f"{precision:>10.3f}  {f1:>6.3f}")

            # Threshold sweep — shows that LOO_THRESHOLD is the F1-maximizing
            # absolute cutoff, and that higher cutoffs cost more detection than
            # they save in false alarms (§4.1 of docs/methods_and_results.md).
            if loo_daily:
                _print_threshold_sweep(pd.DataFrame(loo_daily))

    # --- Operational threshold analysis ---
    # For each station: set threshold = min(LOO event prob) = recall-maximizing.
    # Count non-event DAYS (not windows) above that threshold to measure false alarm rate.
    # station_tier_map (ready/marginal/not_ready) is filled in below from this
    # same analysis — defined here so it's always available even when the
    # analysis is skipped (e.g. no stations with >=2 events yet).
    station_tier_map: dict[str, str] = {}
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

                for _, _r in op_df.iterrows():
                    if pd.isna(_r['false_alarm_rate']):
                        continue
                    station_tier_map[_r['station_id']] = (
                        'ready' if _r['false_alarm_rate'] <= TIER_FA_READY
                        else 'marginal' if _r['false_alarm_rate'] <= TIER_FA_MARGINAL
                        else 'not_ready'
                    )
                print(f"  → confidence tiers: "
                      f"{sum(t == 'ready' for t in station_tier_map.values())} ready, "
                      f"{sum(t == 'marginal' for t in station_tier_map.values())} marginal, "
                      f"{sum(t == 'not_ready' for t in station_tier_map.values())} not ready")

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

    op_event_dates: dict[str, list] = _station_event_dates(df_all)

    n_op_events = sum(len(v) for v in op_event_dates.values())
    n_op_seasons = int(df_all['source_season'].nunique())
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

    # Update regional overlay with operational probs
    regional_prob_dict = op_reg_prob_dict

    # --- Model switch ---
    # Stations whose blended point estimate fails the operational false-alarm
    # bar (station_tier_map != 'ready') hand off to the hierarchical model's
    # mean instead: with only 1-2 logged events the per-station piece of the
    # blend is fit on almost no positive examples and can spike on lookalike
    # non-event days nearly as high as on the real one (see
    # docs/presentation/), whereas the hierarchical model's regional shrinkage
    # gave much better separation from those lookalikes for exactly these
    # sparse stations. Ready stations, already validated at that bar, keep
    # the sharper blended estimate.
    #
    # But hierarchical isn't automatically trustworthy just because blended
    # failed: hier_reliable (see HIER_RELIABILITY_THRESHOLD) is a per-station
    # check of whether the hierarchical model's own held-out prediction on a
    # real 2025-26 event even reached a reasonable fraction of what it
    # predicts on that station's KNOWN training event(s) — two stations
    # (224101_res, 224102_res) fail this outright (0.07x-0.28x). For those we
    # fall back to blended too, as the only model showing any skill there,
    # but tag it 'blended_fallback' (not plain 'blended') so the map/chart can
    # still flag the station as low-confidence overall rather than implying
    # blended was validated the normal way.
    prob_by_station, model_used = _choose_models(
        op_blended_dict, hier_prob_dict, station_tier_map, hier_reliable)

    # --- Generate stability plots (deferred so prob_by_station is available) ---
    _render_stability_plots(plot_queue, prob_by_station, model_used,
                            hier_known_event_ref, hier_ci_dict, station_tier_map,
                            png_dir=out_dir)

    def _window_max(sid: str | None) -> float:
        if sid is None or sid not in prob_by_station:
            return float('nan')
        s      = prob_by_station[sid]
        window = s[(s.index >= win_start) & (s.index <= win_end)]
        return float(window.max()) if not window.dropna().empty else float('nan')

    df_all['forecast_prob'] = [_window_max(sid) for sid in df_all['station_id']]

    # --- Build per-station simulation metadata for the map overlay ---
    sim_stations = _build_sim_stations(
        sim_dir, SEASON, prob_by_station, regional_prob_dict, model_used,
        hier_known_event_ref, station_tier_map, win_start, win_end)

    # --- Generate the map ---
    map_path = root_dir / "index.html"
    print(f"\nGenerating map → {map_path}")
    create_avalanche_map(df_all, map_path, forecast_date=forecast_date,
                         prob_by_station=prob_by_station,
                         sim_stations=sim_stations)

    fig1_path = out_dir / "fig1_map.png"
    plot_static_map(df_all, sim_stations, forecast_date, fig1_path)
    print(f"  → saved {fig1_path.name}")

    print("\nDone.")
    print(f"Open {map_path} and click any marker to view its nearest station's stability.")


if __name__ == "__main__":
    main()
