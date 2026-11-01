#!/usr/bin/env python3
"""
Generate the blended-vs-hierarchical model comparison figures used in the
presentation deck (docs/presentation/figures/): for one station, three
probability-only charts sharing the blended model's own background (the
fixed 33/50/65/80% threshold bands, percentage y-axis) —

    1. the current operational (blended) model,
    2. the hierarchical model trained on only that station's first
       N_REDUCED_EVENTS logged event(s),
    3. the hierarchical model with its full, current training.

Panels 2 and 3 exist to show how the forecast for one station changes as it
accumulates logged events; panel 1 is the operational baseline for
comparison. All three plot the probability line only — no credible-interval
band, no severity bands — for a consistent, easy side-by-side read.

Usage:
    uv run python scripts/generate_hierarchical_comparison.py [STATION_ID]

STATION_ID defaults to 164801_res. Requires the main pipeline to have been
run at least once (`cd src && uv run python main.py`) so
models/{STATION_ID}.joblib, models/regional.joblib, and
models/hierarchical.joblib already exist — this script does not retrain
those. It does train (and cache) the reduced-event hierarchical variant
itself if not already cached, which takes a few seconds.

Outputs (docs/presentation/figures/, named without the "_res" suffix to match
the existing files there):
    {station}_1_blended_model.png
    {station}_2_hierarchical_reduced.png
    {station}_3_hierarchical_full.png
"""
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import main as pipeline  # noqa: E402 — needs sys.path set up first
from classifier import blend_probabilities, predict_proba_series  # noqa: E402
from features import build_daily_features  # noqa: E402
from hierarchical import train_hierarchical  # noqa: E402
from snowpack_io import parse_snow_data  # noqa: E402
from visualization import plot_interactive_stability  # noqa: E402

FORECAST_DATE = pd.Timestamp("2026-01-23")
N_REDUCED_EVENTS = 1  # "before" state: only this station's first N events
OUT_DIR = ROOT / "docs" / "presentation" / "figures"
SIM_DIR = ROOT / "data" / "simulations" / pipeline.SEASON


def _station_train_dates() -> dict[str, list[pd.Timestamp]]:
    """
    Reconstructs each station's TRAIN_SEASON event dates the same way
    main.py's pipeline does, without re-running the whole pipeline (training
    the hierarchical model needs the full pooled corpus, not just one
    station's).
    """
    df_all = pipeline._load_observations(ROOT / "data" / "observations")
    matched = [
        pipeline.find_nearest_pro(
            row["Latitude"], row["Longitude"], SIM_DIR,
            aspect=str(row.get("Aspect", "")),
            stations_csv=ROOT / "data" / "snowpack_stations_locations.csv",
        )
        for _, row in df_all.iterrows()
    ]
    df_all["station_id"] = [pf.stem if pf is not None else None for pf in matched]
    df_all["event_date"] = pd.to_datetime(df_all["Date"], errors="coerce", format="mixed")
    train_ids = set(
        df_all.loc[
            (df_all["source_season"] == pipeline.TRAIN_SEASON) & df_all["station_id"].notna(),
            "station_id",
        ]
    )
    is_test = (df_all["source_season"] == pipeline.SEASON) & df_all["station_id"].isin(train_ids)
    df_all["split"] = ["test" if t else "train" for t in is_test]
    return pipeline._station_event_dates(
        df_all, (df_all["split"] != "test") & (df_all["source_season"] == pipeline.TRAIN_SEASON)
    )


def _get_reduced_hierarchical_model(station_id: str, n_reduced_events: int):
    """Loads (or trains and caches) a hierarchical model identical to the
    full one except that `station_id`'s own training events are truncated to
    its first `n_reduced_events` — the rest of the 13-station pooled corpus
    is unchanged, so this isolates the effect of that one station's history."""
    cache_path = ROOT / "models" / f"hierarchical_{station_id}_{n_reduced_events}event.joblib"
    if cache_path.exists():
        print(f"  -> loaded cached {cache_path.name}")
        return joblib.load(cache_path)

    full_model = joblib.load(ROOT / "models" / "hierarchical.joblib")
    station_list = full_model.station_list
    if station_id not in station_list:
        raise ValueError(
            f"{station_id} is not in the hierarchical model's training set "
            f"({station_list}); pick a station that is."
        )

    all_train_dates = _station_train_dates()
    reduced_dates = dict(all_train_dates)
    reduced_dates[station_id] = sorted(all_train_dates[station_id])[:n_reduced_events]
    reduced_dates = {sid: reduced_dates[sid] for sid in station_list}

    train_features = {}
    for sid in station_list:
        pro = ROOT / "data" / "simulations" / pipeline.TRAIN_SEASON / f"{sid}.pro"
        smet = pro.with_suffix(".smet")
        train_features[sid] = build_daily_features(pro, smet)

    print(f"  Training reduced hierarchical model "
          f"({station_id} truncated to {n_reduced_events} event(s))...")
    model = train_hierarchical(train_features, reduced_dates)
    joblib.dump(model, cache_path)
    print(f"  -> saved {cache_path.name}")
    return model


def _render(prob_series: pd.Series, layer_df, station_id: str, out_path: Path) -> None:
    """One probability-only chart, using plot_interactive_stability's
    defaults for everything background-related (the same fixed 33/50/65/80%
    bands and percentage axis the blended model's chart uses) — no CI band,
    no custom severity bands, so all three figures share one visual frame.
    plot_interactive_stability always also writes an interactive HTML file;
    that goes to a scratch temp dir, not docs/ — only the PNG is a tracked
    deliverable here."""
    scratch_html = Path(tempfile.gettempdir()) / f"{out_path.stem}.html"
    plot_interactive_stability(
        layer_df, scratch_html, station_id,
        event_dates=[], test_dates=[FORECAST_DATE],
        prob_series=prob_series,
        daily_df=None, forecast_date=FORECAST_DATE,
        aspect=pipeline._station_aspect(station_id), confidence_tier="ready",
        png_path=out_path, png_rows=["prob"],
    )
    print(f"  -> saved {out_path}")


def main() -> None:
    station_id = sys.argv[1] if len(sys.argv) > 1 else "164801_res"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    daily = build_daily_features(SIM_DIR / f"{station_id}.pro", SIM_DIR / f"{station_id}.smet")
    layer_df = parse_snow_data(SIM_DIR / f"{station_id}.pro")
    n_train_events = len(_station_train_dates().get(station_id, []))
    label = station_id.replace("_res", "")  # matches the existing figure filenames

    print(f"Station {station_id}  (n_train_events={n_train_events})")

    # 1. Blended (operational) model — current default.
    print("Blended model...")
    station_model, station_scaler = joblib.load(ROOT / "models" / f"{station_id}.joblib")
    regional_model, regional_scaler = joblib.load(ROOT / "models" / "regional.joblib")
    p_station = predict_proba_series(station_model, station_scaler, daily)
    p_regional = predict_proba_series(regional_model, regional_scaler, daily)
    p_blended = blend_probabilities(p_station, p_regional, n_train_events)
    _render(p_blended, layer_df, station_id, OUT_DIR / f"{label}_1_blended_model.png")

    # 2. Hierarchical, reduced training ("before" state).
    print("Hierarchical model (reduced training)...")
    hm_reduced = _get_reduced_hierarchical_model(station_id, N_REDUCED_EVENTS)
    assert hm_reduced is not None, "hierarchical training failed (see train_hierarchical logs above)"
    mean_p_reduced, _ = hm_reduced.predict(station_id, daily)
    _render(mean_p_reduced, layer_df, station_id, OUT_DIR / f"{label}_2_hierarchical_reduced.png")

    # 3. Hierarchical, full training ("current" state).
    print("Hierarchical model (full training)...")
    hm_full = joblib.load(ROOT / "models" / "hierarchical.joblib")
    mean_p_full, _ = hm_full.predict(station_id, daily)
    _render(mean_p_full, layer_df, station_id, OUT_DIR / f"{label}_3_hierarchical_full.png")


if __name__ == "__main__":
    main()
