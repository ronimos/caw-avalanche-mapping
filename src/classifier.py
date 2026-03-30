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
from sklearn.preprocessing import StandardScaler


# Full feature set — requires both .smet and .pro zone data.
FEATURE_COLS = [
    'HS',             # total snow height (m)  — size proxy
    'HN24',           # new snow last 24 h (m) — loading signal
    'HN72',           # new snow last 3 days (m) — sustained loading
    'rain_sum',       # daily sum of rain rate — rain-trigger signal
    'TA_max',         # daily max air temp (°C) — phase proxy
    'sn38_upper_min', # min Sn38 in upper zone — near-surface weakness
    'sn38_lower_min', # min Sn38 in lower zone — deep persistent weakness
    'depth_lower_wl', # burial depth of weakest lower-zone layer (cm)
]

# Fallback when zone features are unavailable on positive days (e.g. early
# season with a thin snowpack that lacks reliable Sn38 values).
SMET_ONLY_COLS = ['HS', 'HN24', 'HN72', 'rain_sum', 'TA_max']


def train_station(
    daily_df: pd.DataFrame,
    event_dates: list[pd.Timestamp],
    min_positives: int = 2,
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
    print(f"    samples={n_pos + n_neg}  pos={n_pos}  neg={n_neg}")

    if n_pos < min_positives:
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
            print(f"    → training ({label})")
            return _fit(X[mask], y[mask])

    print(f"    → skipped (< {min_positives} positives in both feature sets)")
    return None


def predict_proba_series(
    model: LogisticRegression,
    scaler: StandardScaler,
    daily_df: pd.DataFrame,
) -> pd.Series:
    """
    Return a daily probability Series for a single station.
    Days with any missing feature are returned as NaN.
    Uses the exact feature columns the scaler was fitted on.
    """
    fitted_cols = list(scaler.feature_names_in_)
    X    = daily_df[[c for c in fitted_cols if c in daily_df.columns]].copy()
    mask = X.notna().all(axis=1)

    proba = pd.Series(np.nan, index=daily_df.index, name='avalanche_prob')
    if mask.any():
        proba[mask] = model.predict_proba(scaler.transform(X[mask]))[:, 1]

    return proba


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
) -> tuple[LogisticRegression, StandardScaler]:
    """Scale and fit a logistic regression; print feature weights."""
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    coef_df = pd.Series(model.coef_[0], index=X_feat.columns).sort_values(
        key=abs, ascending=False
    )
    print("    weights:", "  ".join(f"{f}:{w:+.2f}" for f, w in coef_df.items()))
    return model, scaler
