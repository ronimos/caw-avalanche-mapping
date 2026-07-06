"""
hierarchical.py — Bayesian hierarchical (partial-pooling) avalanche classifier.

A single multilevel logistic regression replaces the separate per-station,
regional, and hand-blended models.  Each station gets its own coefficients,
drawn from a shared regional prior; the amount of shrinkage toward the regional
mean is learned from the data rather than set by a fixed weight.  Stations with
many events keep their own signal; stations with one event shrink strongly
toward the population.

The posterior also yields a per-day probability *and* its uncertainty (spread
across posterior draws), which feeds the map confidence encoding directly.

Public API:
  train_hierarchical    fit the model → HierModel
  HierModel.predict     per-station daily (mean_prob, std_prob) Series

Requires pymc + arviz.  Imported lazily so the rest of the pipeline runs
without these heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from classifier import FEATURE_COLS, _make_labels

# Default static terrain predictors on the intercept. Deliberately minimal — the
# dataset is event-sparse, so extra group-level params overfit (see project
# thesis). `alpha` is the governing-path runout angle (the consequence threshold).
TERRAIN_FEATURE_COLS = ["alpha"]


@dataclass
class HierModel:
    """Fitted hierarchical model and everything needed to predict from it."""
    idata:        object           # arviz.InferenceData (posterior)
    scaler:       StandardScaler
    feature_cols: list[str]
    station_list: list[str]        # training stations, in coefficient-index order
    # Optional static terrain predictors on the station intercept (group-level).
    terrain_cols:   list[str] | None = None
    terrain_scaler: StandardScaler | None = None

    # ── posterior coefficient summaries (drawn lazily) ──────────────────────────
    def _posterior_arrays(self):
        """Return stacked posterior draws: alpha (S,D), beta (S,F,D), and the
        population-level mu_alpha (D,) and mu_beta (F,D) for unseen stations."""
        post = self.idata.posterior  # type: ignore[attr-defined]
        # dims: (chain, draw, station[, feature]) → stack chain+draw → samples
        alpha = post['alpha'].stack(sample=('chain', 'draw')).values      # (S, N)
        beta  = post['beta'].stack(sample=('chain', 'draw')).values       # (S, F, N)
        mu_a  = post['mu_alpha'].stack(sample=('chain', 'draw')).values   # (N,)
        mu_b  = post['mu_beta'].stack(sample=('chain', 'draw')).values    # (F, N)
        return alpha, beta, mu_a, mu_b

    def _terrain_intercept(self, terrain: dict | None, mu_a: np.ndarray) -> np.ndarray:
        """
        Population intercept draws for an unseen station, shifted by its terrain:
        mu_alpha + gamma . terrain_scaled. Falls back to mu_alpha when the model
        has no terrain predictors or none is supplied.
        """
        if not self.terrain_cols or self.terrain_scaler is None or terrain is None:
            return mu_a
        post  = self.idata.posterior  # type: ignore[attr-defined]
        gamma = post['gamma'].stack(sample=('chain', 'draw')).values      # (Ft, N)
        x     = np.array([[terrain.get(c, np.nan) for c in self.terrain_cols]], dtype=float)
        if np.isnan(x).any():
            return mu_a
        xs = self.terrain_scaler.transform(x)[0]                          # (Ft,)
        return mu_a + xs @ gamma                                          # (N,)

    def predict(self, station_id: str, daily_df: pd.DataFrame,
                terrain: dict | None = None) -> tuple[pd.Series, pd.Series]:
        """
        Predict daily avalanche probability for one station.

        Returns (mean_prob, std_prob) Series indexed like daily_df.  std_prob is
        the posterior standard deviation of the probability — a direct measure of
        model confidence at that station/day.

        Stations absent from training use the population-level (regional) prior;
        if a `terrain` dict is supplied, the intercept is shifted by the learned
        terrain effect, so path geometry transfers to unseen/unproven stations.
        """
        alpha, beta, mu_a, mu_b = self._posterior_arrays()

        if station_id in self.station_list:
            s   = self.station_list.index(station_id)
            a_s = alpha[s]          # (N,)
            b_s = beta[s]           # (F, N)
        else:
            a_s = self._terrain_intercept(terrain, mu_a)   # (N,)
            b_s = mu_b              # (F, N)

        # Impute zone NaNs with training mean (0 after scaling); require core feats
        zone_cols = {'sn38_upper_min', 'sn38_lower_min', 'sn38_min', 'depth_lower_wl'}
        core_cols = [c for c in self.feature_cols if c not in zone_cols]

        X_raw    = daily_df.reindex(columns=self.feature_cols).astype(float)
        fill_vals = pd.Series(self.scaler.mean_, index=self.feature_cols)  # type: ignore[arg-type]
        Xs       = pd.DataFrame(
            self.scaler.transform(X_raw.fillna(fill_vals)),
            index=daily_df.index, columns=self.feature_cols,
        )
        valid = X_raw[core_cols].notna().all(axis=1)

        # logit_p[t, n] = a_s[n] + sum_f Xs[t,f] * b_s[f,n]   → (T, N)
        logit = a_s[None, :] + Xs.values @ b_s            # (T, N)
        p     = 1.0 / (1.0 + np.exp(-logit))              # (T, N)

        mean_p = pd.Series(p.mean(axis=1), index=daily_df.index, name='avalanche_prob')
        std_p  = pd.Series(p.std(axis=1),  index=daily_df.index, name='avalanche_prob_std')
        mean_p[~valid] = np.nan
        std_p[~valid]  = np.nan
        return mean_p, std_p


def train_hierarchical(
    station_features: dict[str, pd.DataFrame],
    station_events: dict[str, list[pd.Timestamp]],
    station_terrain: dict[str, dict] | None = None,
    terrain_cols: list[str] | None = None,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
    seed: int = 42,
) -> HierModel | None:
    """
    Fit a hierarchical Bayesian logistic regression pooling all stations.

    Model (non-centered parameterization for stable sampling):
        logit(p_{s,t}) = alpha_s + beta_s . x_{s,t}
        alpha_s = (mu_alpha + gamma . terrain_s) + sigma_alpha * z^alpha_s
        beta_s  = mu_beta  + sigma_beta  * z^beta_s
    with weakly-informative hyperpriors.  The intercept prior is centered on the
    empirical log-odds of an event day, so probabilities stay calibrated to the
    low (~1%) base rate rather than being rebalanced.

    When `station_terrain` (station_id → {terrain_col: value}) is supplied, the
    static path geometry enters as **group-level predictors on the intercept**
    (gamma): terrain modulates each station's base rate and, being a fixed
    effect, transfers to stations with little/no event history — the data-sparse
    fix. `terrain_cols` selects which keys to use (default: all shared keys).
    """
    import pymc as pm

    feature_cols = list(FEATURE_COLS)

    # ── Assemble pooled design matrix with station index ────────────────────────
    X_parts, y_parts, sidx_parts = [], [], []
    station_list: list[str] = []

    for station_id, daily_df in station_features.items():
        y         = _make_labels(daily_df, station_events.get(station_id, []))
        available = [c for c in feature_cols if c in daily_df.columns]
        if len(available) != len(feature_cols):
            continue
        X    = daily_df[feature_cols].copy()
        mask = X.notna().all(axis=1) & y.notna()
        # Only include stations with at least one event day: event-free stations
        # add unconstrained per-station offsets (pure noise) to the hierarchy
        # without contributing positive signal. Such stations are still served
        # at prediction time by the population-level (regional) coefficients.
        if int(y[mask].sum()) == 0:
            continue
        s = len(station_list)
        station_list.append(station_id)
        X_parts.append(X[mask].to_numpy(dtype=float))
        y_parts.append(y[mask].to_numpy(dtype=int))
        sidx_parts.append(np.full(int(mask.sum()), s, dtype=int))

    if not X_parts:
        return None

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)
    sidx  = np.concatenate(sidx_parts)

    n_pos = int(y_all.sum())
    n_obs, n_feat = X_all.shape
    n_stations    = len(station_list)
    print(f"    hierarchical: {n_stations} stations  obs={n_obs}  pos={n_pos}")
    if n_pos < 3:
        print("    → skipped (< 3 positive days)")
        return None

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X_all)

    base_rate = max(n_pos / n_obs, 1e-4)
    logit_prev = float(np.log(base_rate / (1 - base_rate)))

    # ── Optional group-level terrain design matrix (aligned to station_list) ────
    T = None
    t_cols: list[str] | None = None
    terrain_scaler: StandardScaler | None = None
    if station_terrain:
        t_cols = terrain_cols or [
            c for c in TERRAIN_FEATURE_COLS
            if any(np.isfinite(station_terrain.get(sid, {}).get(c, np.nan))
                   for sid in station_list)
        ]
        if t_cols:
            T_raw = np.array(
                [[station_terrain.get(sid, {}).get(c, np.nan) for c in t_cols]
                 for sid in station_list], dtype=float)
            # Stations missing a terrain value get the column mean → no intercept shift.
            col_mean = np.nanmean(T_raw, axis=0)
            nan_r, nan_c = np.where(~np.isfinite(T_raw))
            T_raw[nan_r, nan_c] = np.take(col_mean, nan_c)
            terrain_scaler = StandardScaler()
            T = terrain_scaler.fit_transform(T_raw)
            print(f"    terrain predictors on intercept: {t_cols}")
        else:
            t_cols = None

    coords = {"station": station_list, "feature": feature_cols}
    if t_cols:
        coords["terrain"] = t_cols
    with pm.Model(coords=coords) as model:
        station_idx = pm.Data("station_idx", sidx)
        X_data      = pm.Data("X_data", Xs)

        # Population-level (regional) coefficients
        mu_alpha    = pm.Normal("mu_alpha", mu=logit_prev, sigma=1.5)
        mu_beta     = pm.Normal("mu_beta", mu=0.0, sigma=1.0, dims="feature")
        # Between-station spread (pooling strength, learned)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)
        sigma_beta  = pm.HalfNormal("sigma_beta", sigma=1.0, dims="feature")

        # Static terrain shifts the station intercept mean (group-level predictor)
        if T is not None:
            T_data     = pm.Data("T_data", T)
            gamma      = pm.Normal("gamma", 0.0, 1.0, dims="terrain")
            alpha_mean = mu_alpha + pm.math.dot(T_data, gamma)
        else:
            alpha_mean = mu_alpha

        # Non-centered station offsets
        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="station")
        z_beta  = pm.Normal("z_beta", 0.0, 1.0, dims=("station", "feature"))
        alpha   = pm.Deterministic("alpha", alpha_mean + sigma_alpha * z_alpha, dims="station")
        beta    = pm.Deterministic("beta", mu_beta + sigma_beta * z_beta, dims=("station", "feature"))

        logit_p = alpha[station_idx] + (X_data * beta[station_idx]).sum(axis=-1)
        pm.Bernoulli("y_obs", logit_p=logit_p, observed=y_all)

        idata = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept,
            chains=2, cores=2, random_seed=seed, progressbar=False,
        )

    # Report divergences as a sampling-health check
    n_div = int(idata.sample_stats["diverging"].sum())  # type: ignore[index]
    print(f"    sampled: {draws}×2 draws  divergences={n_div}")

    return HierModel(idata=idata, scaler=scaler,
                     feature_cols=feature_cols, station_list=station_list,
                     terrain_cols=t_cols, terrain_scaler=terrain_scaler)


# ── standalone smoke test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import build_daily_features

    root = Path(__file__).resolve().parents[1]
    sims = root / "data" / "simulations"
    TRAIN, TEST = "2024-2025", "2025-2026"

    # Minimal event lists from observation CSVs via nearest-station matching
    from main import _load_observations
    from snowpack_io import find_nearest_pro

    obs          = _load_observations(root / "data" / "observations")
    stations_csv = root / "data" / "snowpack_stations_locations.csv"
    sim_dir      = sims / TEST

    train_feats: dict[str, pd.DataFrame] = {}
    train_events: dict[str, list] = {}
    for _, r in obs[obs['source_season'] == TRAIN].iterrows():
        pf = find_nearest_pro(r['Latitude'], r['Longitude'], sim_dir,
                              aspect=str(r.get('Aspect', '')), stations_csv=stations_csv)
        if pf is None:
            continue
        sid = pf.stem
        tp, ts = sims / TRAIN / f"{sid}.pro", sims / TRAIN / f"{sid}.smet"
        if tp.exists() and ts.exists() and sid not in train_feats:
            train_feats[sid] = build_daily_features(tp, ts)
        d = pd.to_datetime(r['Date'], errors='coerce')
        if pd.notna(d):
            train_events.setdefault(sid, []).append(d)

    print(f"Training hierarchical on {len(train_feats)} stations...")
    hm = train_hierarchical(train_feats, train_events, draws=500, tune=500)
    if hm is not None:
        sid = hm.station_list[0]
        mp, sp = hm.predict(sid, train_feats[sid])
        print(f"\nStation {sid}: mean_prob range [{mp.min():.4f}, {mp.max():.4f}]  "
              f"max std {sp.max():.4f}")
