"""
features.py — Feature engineering for the avalanche classifier.

Transforms raw .pro/.smet data into a daily feature table ready for
model training and prediction.

Public API:
  build_daily_features   (pro_path, smet_path) → daily DataFrame
  UPPER_ZONE_FRAC        fraction constant (must match _UPPER_ZONE_FRAC in visualization.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from snowpack_io import parse_smet, parse_snow_data


# Fraction of snowpack depth (from surface down) that defines the upper zone.
# E.g. 0.40 → top 40% of total snow height.
# NOTE: visualization.py has a matching _UPPER_ZONE_FRAC = 0.40 constant.
UPPER_ZONE_FRAC = 0.40


def build_daily_features(pro_path: Path, smet_path: Path) -> pd.DataFrame:
    """
    Combine .pro and .smet data into a daily feature DataFrame for one station.

    .smet  → HS, HN24, HN72, rain_sum, TA_max  (aggregated from sub-daily to daily)
    .pro   → sn38_upper_min, sn38_lower_min, depth_lower_wl  (daily extremes)

    Returns a DataFrame indexed by date (midnight).
    """
    snow_df  = parse_snow_data(pro_path)
    zone_df  = _layer_zone_features(snow_df)
    smet_df  = parse_smet(smet_path)

    # Select and rename .smet columns we care about
    smet_rename = {
        'HS_mod':   'HS',
        'HN24':     'HN24',
        'HN72_24':  'HN72',
        'MS_Rain':  'rain_rate',
        'TA':       'TA',
    }
    present  = {k: v for k, v in smet_rename.items() if k in smet_df.columns}
    smet_sub = smet_df[list(present.keys())].rename(columns=present)

    agg_map: dict[str, str] = {}
    if 'HS'        in smet_sub: agg_map['HS']        = 'max'
    if 'HN24'      in smet_sub: agg_map['HN24']      = 'max'
    if 'HN72'      in smet_sub: agg_map['HN72']      = 'max'
    if 'rain_rate' in smet_sub: agg_map['rain_rate']  = 'sum'
    if 'TA'        in smet_sub: agg_map['TA']        = 'max'

    smet_daily = smet_sub.resample('D').agg(agg_map)  # type: ignore[arg-type]
    if 'rain_rate' in smet_daily:
        smet_daily = smet_daily.rename(columns={'rain_rate': 'rain_sum'})
    if 'TA' in smet_daily.columns:
        smet_daily = smet_daily.rename(columns={'TA': 'TA_max'})
    if 'TA_max' in smet_daily.columns:
        # 1 = above-freezing day (wet-avalanche regime), 0 = cold/dry
        smet_daily['wet_flag'] = (smet_daily['TA_max'] > 0).astype(float)

    zone_daily = zone_df.resample('D').agg({  # type: ignore[arg-type]
        'sn38_upper_min': 'min',
        'sn38_lower_min': 'min',
        'depth_lower_wl': 'max',
    })

    # Combined whole-profile minimum stability index — the single cleanest
    # "how weak is the weakest layer anywhere" signal, used by the reduced
    # feature set to avoid the collinearity between the upper/lower zone mins.
    zone_daily['sn38_min'] = zone_daily[['sn38_upper_min', 'sn38_lower_min']].min(axis=1)

    # Forward-fill zone features up to 7 days: when the snowpack is too thin
    # to resolve an upper/lower zone, carry the last known stability state
    # forward rather than dropping the day from classifier predictions.
    zone_daily = zone_daily.ffill(limit=7)

    return smet_daily.join(zone_daily, how='outer')


# ── internal helpers ──────────────────────────────────────────────────────────

def _layer_zone_features(snow_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each timestep, split layers into upper and lower zones based on
    burial depth relative to total HS, then compute zone stability metrics.

    Upper zone: burial_depth <  UPPER_ZONE_FRAC * total_height
    Lower zone: burial_depth >= UPPER_ZONE_FRAC * total_height

    Returns a DataFrame indexed by timestamp with columns:
        sn38_upper_min  — min Sn38 in upper zone (near-surface weakness)
        sn38_lower_min  — min Sn38 in lower zone (deep persistent weakness)
        depth_lower_wl  — burial depth of weakest lower-zone layer (cm)
    """
    results = []
    for ts, group in snow_df.groupby(snow_df.index):
        hs        = group['total_height'].iloc[0]
        threshold = UPPER_ZONE_FRAC * hs

        upper = group[group['burial_depth'] <  threshold]
        lower = group[group['burial_depth'] >= threshold]

        sn38_upper = upper['sn38'].min() if len(upper) else np.nan
        sn38_lower = lower['sn38'].min() if len(lower) else np.nan

        if len(lower) and not lower['sn38'].isna().all():
            idx            = int(lower['sn38'].values.argmin())
            depth_lower_wl = float(lower['burial_depth'].iloc[idx])
        else:
            depth_lower_wl = np.nan

        results.append({
            'timestamp':      ts,
            'sn38_upper_min': sn38_upper,
            'sn38_lower_min': sn38_lower,
            'depth_lower_wl': depth_lower_wl,
        })

    return pd.DataFrame(results).set_index('timestamp')
