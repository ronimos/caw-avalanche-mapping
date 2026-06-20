"""
classifier.py — Avalanche probability classifier.

Trains a per-station logistic regression on daily feature tables and returns
a per-day probability of avalanche occurrence.

Features are expected to come from features.build_daily_features().
Training labels are built internally: a station-day is positive (label=1)
if an avalanche was observed on that day OR the following day (one-day
reporting lag).  Class imbalance is handled with balanced weights.

Public API:
  train_station        fit a model for one station → (model, scaler) | None
  predict_proba_series predict daily probabilities → Series
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# Full 9-feature set — retained for comparison / ablation, but no longer the
# default.  With ~32 positive training days this set badly over-parameterizes
# the model (≈3.5 events per predictor vs. the 10–20 rule of thumb).
FULL_FEATURE_COLS = [
    'HS',             # total snow height (m)  — size proxy; varies with start-zone distance
    'HN24',           # new snow last 24 h (m) — loading signal
    'HN72',           # new snow last 3 days (m) — sustained loading
    'rain_sum',       # daily sum of rain rate — rain-trigger signal
    'TA_max',         # daily max air temp (°C) — phase proxy
    'wet_flag',       # 1 if TA_max > 0 — explicit wet vs dry mechanism indicator
    'sn38_upper_min', # min Sn38 in upper zone — near-surface weakness
    'sn38_lower_min', # min Sn38 in lower zone — deep persistent weakness
    'depth_lower_wl', # burial depth of weakest lower-zone layer (cm)
]

# Reduced 5-feature set (default).  Each feature is physically distinct, chosen
# to minimize collinearity given the small event count:
#   - dropped HN72 (collinear with HN24)
#   - dropped wet_flag (a threshold of TA_max — collinear)
#   - dropped rain_sum (sparse; its signal overlaps TA_max)
#   - collapsed sn38_upper/lower into a single whole-profile sn38_min
REDUCED_FEATURE_COLS = [
    'HS',             # total snow height (m)  — size / valley-reach proxy
    'HN24',           # new snow last 24 h (m) — primary loading trigger
    'TA_max',         # daily max air temp (°C) — wet vs dry mechanism
    'sn38_min',       # min Sn38 across whole profile — weakest layer strength
    'depth_lower_wl', # burial depth of weakest lower-zone layer (cm)
]

# Active feature set used for training and prediction.
FEATURE_COLS = REDUCED_FEATURE_COLS

# Fallback when zone features are unavailable on positive days (e.g. early
# season with a thin snowpack that lacks reliable Sn38 values).
SMET_ONLY_COLS = ['HS', 'HN24', 'TA_max']


def train_station(
    daily_df: pd.DataFrame,
    event_dates: list[pd.Timestamp],
    min_positives: int = 2,
    verbose: bool = True,
) -> tuple[LogisticRegression, StandardScaler] | None:
    """
    Train a logistic regression for a single station.

    Tries the full FEATURE_COLS set first; falls back to SMET_ONLY_COLS when
    zone features are NaN on all positive days.  Returns None when neither
    feature set produces enough positive labels to train.

    Args:
        daily_df:      Output of features.build_daily_features().
        event_dates:   All observed avalanche timestamps at this station.
        min_positives: Minimum labeled positive days required to attempt fit.
    """
    y     = _make_labels(daily_df, event_dates)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if verbose:
        print(f"    samples={n_pos + n_neg}  pos={n_pos}  neg={n_neg}")

    if n_pos < min_positives:
        if verbose:
            print(f"    → skipped (< {min_positives} positive days)")
        return None

    available = [c for c in FEATURE_COLS if c in daily_df.columns]

    for col_set, label in [
        (available,                                     "full"),
        ([c for c in SMET_ONLY_COLS if c in daily_df], "smet-only fallback"),
    ]:
        X      = daily_df[col_set].copy()
        mask   = X.notna().all(axis=1) & y.notna()
        if y[mask].sum() >= min_positives:
            if verbose:
                print(f"    → training ({label})")
            return _fit(X[mask], y[mask], verbose=verbose)

    if verbose:
        print(f"    → skipped (< {min_positives} positives in both feature sets)")
    return None


def predict_proba_series(
    model: LogisticRegression,
    scaler: StandardScaler,
    daily_df: pd.DataFrame,
) -> pd.Series:
    """
    Return a daily probability Series for a single station.

    Core weather features (HS, HN*, rain_sum, TA_max, wet_flag) must be non-NaN
    for a day to be predicted.  Zone features (sn38_*, depth_lower_wl) that are
    NaN — e.g. on thin-snowpack days where no qualifying layers exist — are
    imputed with the training mean (zero after StandardScaler), representing
    neutral / average stability rather than silently dropping the day.
    """
    fitted_cols  = list(scaler.feature_names_in_)
    zone_cols    = {'sn38_upper_min', 'sn38_lower_min', 'sn38_min', 'depth_lower_wl'}
    core_cols    = [c for c in fitted_cols if c not in zone_cols]

    X = daily_df.reindex(columns=fitted_cols).copy()

    # Impute zone NaNs with training mean (neutral — does not bias toward high/low risk)
    for i, col in enumerate(fitted_cols):
        if col in zone_cols:
            X[col] = X[col].fillna(float(scaler.mean_[i]))

    # Only predict on days where core weather features are fully available
    mask = X[core_cols].notna().all(axis=1)

    proba = pd.Series(np.nan, index=daily_df.index, name='avalanche_prob')
    if mask.any():
        proba[mask] = model.predict_proba(scaler.transform(X[mask]))[:, 1]

    return proba


def blend_probabilities(
    p_station: pd.Series | None,
    p_regional: pd.Series,
    n_train_events: int,
    max_events: int = 5,
) -> pd.Series:
    """
    Confidence-weighted blend of per-station and regional probabilities.

    Weight w = min(n_train_events / max_events, 1.0):
      - 0 events  → 100 % regional
      - max_events+ → 100 % per-station
    Where p_station is None or NaN, falls back to p_regional.
    """
    if p_station is None or n_train_events == 0:
        return p_regional.copy()
    w       = min(n_train_events / max_events, 1.0)
    blended = p_station.reindex(p_regional.index) * w + p_regional * (1.0 - w)
    return blended.fillna(p_regional)


def evaluate_event_level(
    station_probs:  dict[str, pd.Series],
    station_events: dict[str, list[pd.Timestamp]],
    pre_event_window: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Event-level precision-recall evaluation.

    Positive sample  (label=1): max(prob) in [event_day - window, event_day].
    Negative sample  (label=0): max(prob) in a same-length non-overlapping window
                                 that lies outside a grace zone around all events.

    This frames the task as "did the model issue a timely warning?" rather than
    "did it correctly classify every individual day?"

    Args:
    
        station_probs:    station_id → daily probability Series.
        station_events:   station_id → avalanche event dates.
        pre_event_window: days before (and including) the event to consider.

    Returns:
        (y_true, y_scores) as numpy arrays for sklearn PR / ROC functions.
    """
    window_td  = pd.Timedelta(days=pre_event_window)
    y_true_list:  list[int]   = []
    y_score_list: list[float] = []

    for station_id, prob in station_probs.items():
        events = sorted(station_events.get(station_id, []))
        if not events or prob.dropna().empty:
            continue

        # ── Positive samples: max prob in pre-event window ────────────────────
        for ev in events:
            ed = ev.normalize()
            win = prob[
                (prob.index.normalize() >= (ed - window_td).normalize()) &
                (prob.index.normalize() <= ed)
            ]
            if not win.dropna().empty:
                y_true_list.append(1)
                y_score_list.append(float(win.max()))

        # ── Negative samples: non-overlapping windows away from events ────────
        # Grace zone: exclude days within ±(window+1) of any event
        grace: set[pd.Timestamp] = set()
        for ev in events:
            ed = ev.normalize()
            for delta in range(-(pre_event_window + 1), pre_event_window + 2):
                grace.add(ed + pd.Timedelta(days=delta))

        non_grace = prob[~prob.index.normalize().isin(grace)].dropna()
        dates      = list(non_grace.index)
        i = 0
        while i + pre_event_window < len(dates):
            chunk = non_grace.iloc[i: i + pre_event_window + 1]
            if not chunk.empty:
                y_true_list.append(0)
                y_score_list.append(float(chunk.max()))
            i += pre_event_window + 1  # non-overlapping step

    return np.array(y_true_list, dtype=int), np.array(y_score_list)


def aggregate_predictions(
    station_probs:    dict[str, pd.Series],
    station_events:   dict[str, list[pd.Timestamp]],
    station_features: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate (y_true, y_prob) pairs across all stations that have test events.

    Args:
        station_probs:    station_id → pre-computed probability Series.
        station_events:   station_id → avalanche dates (used to build labels).
        station_features: station_id → daily feature DataFrame (for label alignment).

    Returns:
        (y_true, y_prob) as numpy arrays, suitable for sklearn PR / ROC functions.
    """
    all_p: list[float] = []
    all_y: list[int]   = []

    for station_id, prob in station_probs.items():
        event_dates = station_events.get(station_id, [])
        daily_df    = station_features.get(station_id)
        if not event_dates or daily_df is None:
            continue
        y    = _make_labels(daily_df, event_dates)
        mask = prob.notna()
        if mask.sum() == 0:
            continue
        all_p.extend(prob[mask].tolist())
        all_y.extend(y[mask].tolist())

    return np.array(all_y, dtype=int), np.array(all_p)


def train_regional(
    station_features: dict[str, pd.DataFrame],
    station_events: dict[str, list[pd.Timestamp]],
    min_positives: int = 3,
) -> tuple[LogisticRegression, StandardScaler] | None:
    """
    Train a single logistic regression pooling data from all stations.

    Features are standardized globally (one StandardScaler fit on all training
    rows), so HS and other magnitude features become relative to the pooled
    distribution rather than any single station's absolute values.

    Args:
        station_features: station_id → daily feature DataFrame (training season).
        station_events:   station_id → observed avalanche dates (training season).
        min_positives:    minimum positive-label rows required across all stations.
    """
    X_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series]    = []

    for station_id, daily_df in station_features.items():
        event_dates = station_events.get(station_id, [])
        y           = _make_labels(daily_df, event_dates)
        available   = [c for c in FEATURE_COLS if c in daily_df.columns]
        X           = daily_df[available].copy()
        mask        = X.notna().all(axis=1) & y.notna()
        if mask.sum() > 0:
            X_parts.append(X[mask])
            y_parts.append(y[mask])

    if not X_parts:
        return None

    X_all = pd.concat(X_parts, ignore_index=True)
    y_all = pd.concat(y_parts, ignore_index=True)

    n_pos = int(y_all.sum())
    n_neg = int((y_all == 0).sum())
    print(f"    pooled: {len(station_features)} stations  "
          f"samples={n_pos + n_neg}  pos={n_pos}  neg={n_neg}")

    if n_pos < min_positives:
        print(f"    → skipped (< {min_positives} positive days)")
        return None

    best_C = _tune_C(X_all, y_all, n_pos)
    return _fit(X_all, y_all, C=best_C)


def _tune_C(X: pd.DataFrame, y: pd.Series, n_pos: int) -> float:
    """
    Select the L2 regularization strength C by stratified k-fold CV, scoring on
    average precision (appropriate for the heavy class imbalance).  Falls back to
    a strong fixed C when there are too few positives to cross-validate.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline

    n_splits = min(5, n_pos)
    if n_splits < 3:
        return 0.1  # too few positives to CV — use strong fixed regularization

    Cs  = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegressionCV(
        Cs=Cs, cv=cv, scoring='average_precision',
        class_weight='balanced', max_iter=1000, random_state=42,
    )
    # Scale inside the CV to avoid leakage across folds
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X, y)
    best_C = float(pipe.named_steps['logisticregressioncv'].C_[0])
    print(f"    tuned C = {best_C:g}  (stratified {n_splits}-fold, scoring=AP)")
    return best_C


def evaluate_regional(
    model: LogisticRegression,
    scaler: StandardScaler,
    station_features: dict[str, pd.DataFrame],
    station_events: dict[str, list[pd.Timestamp]],
) -> dict[str, float]:
    """
    Evaluate the regional model by aggregating predictions across all stations
    that have at least one test event.  Returns an empty dict on failure.
    """
    all_probs:  list[float] = []
    all_labels: list[int]   = []
    n_events = 0

    for station_id, daily_df in station_features.items():
        event_dates = station_events.get(station_id, [])
        if not event_dates:
            continue
        n_events += len(event_dates)
        prob = predict_proba_series(model, scaler, daily_df)
        y    = _make_labels(daily_df, event_dates)
        mask = prob.notna()
        if mask.sum() == 0:
            continue
        all_probs.extend(prob[mask].tolist())
        all_labels.extend(y[mask].tolist())

    if not all_labels or sum(all_labels) == 0:
        return {}

    p      = np.array(all_probs)
    y      = np.array(all_labels, dtype=int)
    y_pred = (p >= 0.5).astype(int)

    metrics: dict[str, float] = {
        'n_stations': float(len(station_features)),
        'n_events':   float(n_events),
        'precision':  float(precision_score(y, y_pred, zero_division=0)),
        'recall':     float(recall_score(y, y_pred, zero_division=0)),
        'f1':         float(f1_score(y, y_pred, zero_division=0)),
    }
    if len(set(y.tolist())) > 1:
        metrics['auc'] = float(roc_auc_score(y, p))
    return metrics


def evaluate_station(
    model: LogisticRegression,
    scaler: StandardScaler,
    daily_df: pd.DataFrame,
    test_dates: list[pd.Timestamp],
) -> dict[str, float]:
    """
    Evaluate a trained model on held-out test dates.

    Returns a dict with precision, recall, f1 (at 0.5 threshold) and auc
    (when both classes are present).  Returns an empty dict when there is
    insufficient data (no positive labels, or no non-NaN predictions).
    """
    if not test_dates:
        return {}

    prob   = predict_proba_series(model, scaler, daily_df)
    y_true = _make_labels(daily_df, test_dates)
    mask   = prob.notna()

    if mask.sum() == 0 or int(y_true[mask].sum()) == 0:
        return {}

    p      = prob[mask].values
    y      = y_true[mask].values
    y_pred = (p >= 0.5).astype(int)

    metrics: dict[str, float] = {
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall':    float(recall_score(y, y_pred, zero_division=0)),
        'f1':        float(f1_score(y, y_pred, zero_division=0)),
    }
    if len(set(y.tolist())) > 1:
        metrics['auc'] = float(roc_auc_score(y, p))

    return metrics


# ── internal helpers ──────────────────────────────────────────────────────────

def _make_labels(
    daily_df: pd.DataFrame,
    event_dates: list[pd.Timestamp],
) -> pd.Series:
    """
    Build a binary label Series.  Label=1 if the day OR the previous day
    had an observed avalanche (accounts for one-day reporting lag).
    """
    positive_days: set = set()
    for d in event_dates:
        day = d.normalize()
        positive_days.add(day)
        positive_days.add(day - pd.Timedelta(days=1))

    return pd.Series(
        [1 if pd.Timestamp(d) in positive_days else 0 for d in daily_df.index],
        index=daily_df.index,
        dtype=int,
    )


def _fit(
    X_feat: pd.DataFrame,
    y: pd.Series,
    verbose: bool = True,
    C: float = 0.1,
) -> tuple[LogisticRegression, StandardScaler]:
    """
    Scale and fit an L2-regularized logistic regression; optionally print weights.

    The default C=0.1 applies strong regularization — appropriate for per-station
    models where only 1–2 positive samples are available and the data cannot
    support a tuned penalty.  The regional model overrides C with a CV-tuned value.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    model = LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42, C=C,
    )
    model.fit(X_scaled, y)

    if verbose:
        coef_df = pd.Series(model.coef_[0], index=X_feat.columns).sort_values(
            key=abs, ascending=False
        )
        print(f"    weights (C={C:g}):",
              "  ".join(f"{f}:{w:+.2f}" for f, w in coef_df.items()))
    return model, scaler
