"""
hierarchical.py — Bayesian hierarchical (partial-pooling) avalanche classifier.

A single multilevel logistic regression replaces the separate per-station,
regional, and hand-blended models.  Each station gets its own intercept (base
rate), drawn from a shared regional prior, while all stations share one slope
vector; the amount of intercept shrinkage toward the regional mean is learned
from the data rather than set by a fixed weight.  Stations with many events
keep their own signal; stations with one event shrink strongly toward the
population.  (Per-station slopes are available via varying_slopes=True but are
not identifiable at the current event count — see train_hierarchical.)

The posterior also yields a per-day probability *and* its uncertainty (spread
across posterior draws), which feeds the map confidence encoding directly.

Public API:
  train_hierarchical    fit the model → HierModel
  HierModel.predict     per-station daily (mean_prob, std_prob) Series

Requires pymc + arviz.  Imported lazily so the rest of the pipeline runs
without these heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

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
    # Optional terrain × feature interaction (e.g. terrain moderates the HS slope).
    interact_feature: str | None = None
    # Whether slopes vary by station (True) or are pooled to a single shared
    # vector (False — random intercepts only, the default; see train_hierarchical).
    varying_slopes: bool = False
    _var_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def _stacked_var(self, name: str) -> np.ndarray:
        """Stack (and cache) one posterior variable's draws."""
        if name not in self._var_cache:
            post = self.idata.posterior  # type: ignore[attr-defined]
            self._var_cache[name] = post[name].stack(sample=('chain', 'draw')).values
        return self._var_cache[name]

    # ── posterior coefficient summaries (stacked once, then cached) ─────────────
    @cached_property
    def _stacked(self):
        """Stacked posterior draws: alpha (S,N), beta (S,F,N), mu_alpha (N,),
        mu_beta (F,N). With pooled slopes, beta is mu_beta broadcast per station.
        Cached — stacking the full posterior is expensive and predict() is called
        once per station when building the map."""
        post = self.idata.posterior  # type: ignore[attr-defined]
        # dims: (chain, draw, station[, feature]) → stack chain+draw → samples
        alpha = post['alpha'].stack(sample=('chain', 'draw')).values      # (S, N)
        mu_a  = post['mu_alpha'].stack(sample=('chain', 'draw')).values   # (N,)
        mu_b  = post['mu_beta'].stack(sample=('chain', 'draw')).values    # (F, N)
        if self.varying_slopes:
            beta = post['beta'].stack(sample=('chain', 'draw')).values    # (S, F, N)
        else:
            beta = np.broadcast_to(mu_b, (alpha.shape[0], *mu_b.shape))
        return alpha, beta, mu_a, mu_b

    def _terrain_intercept(self, terrain: dict | None, mu_a: np.ndarray) -> np.ndarray:
        """
        Population intercept draws for an unseen station, shifted by its terrain:
        mu_alpha + gamma . terrain_scaled. Falls back to mu_alpha when the model
        has no terrain predictors or none is supplied.
        """
        if not self.terrain_cols or self.terrain_scaler is None or terrain is None:
            return mu_a
        gamma = self._stacked_var('gamma')                                # (Ft, N)
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
        alpha, beta, mu_a, mu_b = self._stacked

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

        # terrain × feature interaction: theta . (scaled_feature_t * scaled_terrain_s)
        if (self.interact_feature and self.terrain_cols and self.terrain_scaler is not None
                and terrain is not None and self.interact_feature in self.feature_cols):
            x = np.array([[terrain.get(c, np.nan) for c in self.terrain_cols]], dtype=float)
            if not np.isnan(x).any():
                theta = self._stacked_var('theta')                              # (Ft, N)
                ts    = self.terrain_scaler.transform(x)[0]                     # (Ft,)
                fcol  = Xs[self.interact_feature].to_numpy(dtype=float)         # (T,) scaled
                Xi    = fcol[:, None] * ts[None, :]                             # (T, Ft)
                logit = logit + Xi @ theta                                     # (T, N)

        p = 1.0 / (1.0 + np.exp(-logit))                  # (T, N)

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
    interact_with: str | None = None,
    varying_slopes: bool = False,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
    seed: int = 42,
) -> HierModel | None:
    """
    Fit a hierarchical Bayesian logistic regression pooling all stations.

    Default model — random intercepts, shared slopes (non-centered):
        logit(p_{s,t}) = alpha_s + mu_beta . x_{s,t}
        alpha_s = (mu_alpha + gamma . terrain_s) + sigma_alpha * z^alpha_s
    with weakly-informative hyperpriors.  The intercept prior is centered on the
    empirical log-odds of an event day, so probabilities stay calibrated to the
    low (~1%) base rate rather than being rebalanced.

    `varying_slopes=True` additionally gives each station its own slope vector
        beta_s = mu_beta + sigma_beta * z^beta_s
    Evaluated head-to-head on the 2024-25 corpus, the varying-slopes variant
    spends ~9 extra effective parameters (PSIS-LOO p_eff 21.3 vs 12.5) for zero
    predictive gain (elpd tie; cross-station LOO AUC 0.702 vs 0.711 in favour of
    shared slopes) — with events this scarce, per-station slopes are not
    identifiable, so shared slopes are the default by parsimony.

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
        # Only include stations with at least one event day. This is an
        # under-reporting (positive-unlabeled) choice, not a statistical one:
        # AKAH observations are opportunistic, so a station with zero recorded
        # events more likely means "nobody was watching" than "no avalanches" —
        # its all-negative labels would teach the model the wrong lesson. Such
        # stations are still served at prediction time by the population-level
        # (regional) coefficients.
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

    # Resolve the terrain × feature interaction (only if terrain is present).
    interact_feat: str | None = None
    feat_idx: int | None = None
    if T is not None and interact_with and interact_with in feature_cols:
        interact_feat = interact_with
        feat_idx = feature_cols.index(interact_with)
        print(f"    terrain × {interact_feat} interaction enabled")

    coords = {"station": station_list, "feature": feature_cols}
    if t_cols:
        coords["terrain"] = t_cols
    with pm.Model(coords=coords) as model:
        station_idx = pm.Data("station_idx", sidx)
        X_data      = pm.Data("X_data", Xs)

        # Population-level (regional) coefficients
        mu_alpha    = pm.Normal("mu_alpha", mu=logit_prev, sigma=1.5)
        mu_beta     = pm.Normal("mu_beta", mu=0.0, sigma=1.0, dims="feature")
        # Between-station intercept spread (pooling strength, learned)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)

        # Static terrain shifts the station intercept mean (group-level predictor)
        if T is not None:
            T_data     = pm.Data("T_data", T)
            gamma      = pm.Normal("gamma", 0.0, 1.0, dims="terrain")
            alpha_mean = mu_alpha + pm.math.dot(T_data, gamma)
        else:
            alpha_mean = mu_alpha

        # Non-centered station intercept offsets
        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="station")
        alpha   = pm.Deterministic("alpha", alpha_mean + sigma_alpha * z_alpha, dims="station")

        if varying_slopes:
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0, dims="feature")
            z_beta     = pm.Normal("z_beta", 0.0, 1.0, dims=("station", "feature"))
            beta       = pm.Deterministic("beta", mu_beta + sigma_beta * z_beta,
                                          dims=("station", "feature"))
            logit_p    = alpha[station_idx] + (X_data * beta[station_idx]).sum(axis=-1)
        else:
            # Random intercepts only: one shared slope vector for all stations.
            logit_p = alpha[station_idx] + pm.math.dot(X_data, mu_beta)

        # terrain × feature interaction: terrain moderates the slope of one feature
        # (e.g. HS) — the "how much snow is needed depends on the path" mechanism.
        if T is not None and interact_feat is not None:
            # (obs, terrain) = scaled feature at each obs × its station's terrain row
            Xi_np  = Xs[:, feat_idx][:, None] * T[sidx, :]
            Xi     = pm.Data("Xi_data", Xi_np)
            theta  = pm.Normal("theta", 0.0, 1.0, dims="terrain")
            logit_p = logit_p + (Xi * theta).sum(axis=-1)

        pm.Bernoulli("y_obs", logit_p=logit_p, observed=y_all)

        idata = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept,
            chains=4, cores=4, random_seed=seed, progressbar=False,
            # Pointwise log-likelihood → arviz PSIS-LOO / az.compare work directly.
            idata_kwargs={"log_likelihood": True},
        )

    # Sampling-health checks: divergences and split-R̂ on the population params.
    import arviz as az

    n_div = int(idata.sample_stats["diverging"].sum())  # type: ignore[index]
    check_vars = ["mu_alpha", "mu_beta", "sigma_alpha"]
    rhat = az.rhat(idata, var_names=check_vars)
    rhat_ds = rhat.to_dataset() if hasattr(rhat, "to_dataset") else rhat  # arviz ≥1 → DataTree
    rhat_max = float(max(float(rhat_ds[v].max()) for v in rhat_ds.data_vars))
    print(f"    sampled: {draws}×4 draws  divergences={n_div}  max R̂={rhat_max:.3f}")
    if n_div > 0 or rhat_max > 1.01:
        print("    ⚠ sampling-health warning: inspect trace before trusting results")

    return HierModel(idata=idata, scaler=scaler,
                     feature_cols=feature_cols, station_list=station_list,
                     terrain_cols=t_cols, terrain_scaler=terrain_scaler,
                     interact_feature=interact_feat, varying_slopes=varying_slopes)


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
