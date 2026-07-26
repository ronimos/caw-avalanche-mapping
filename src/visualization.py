"""
visualization.py — Snowpack stability analysis and interactive output.

Provides:
  - plot_interactive_stability  Multi-panel Plotly HTML stability chart.
  - create_avalanche_map        Folium map of avalanche observations.
  - create_trace_validation_map Folium overlay of debris→start-zone terrain traces.
"""

import json
import math
from pathlib import Path

import contextily as ctx
import folium
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from folium.plugins import Geocoder, TimestampedGeoJson
from matplotlib.lines import Line2D
from plotly.subplots import make_subplots


_LAYER_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#17becf',
]

# Standardized font sizes for every print figure in the paper (study-area
# map, learning curve, feature importance, stability chart), so panels read
# as one system regardless of whether they're matplotlib or Plotly. Figures
# carry no title text of their own — captions in the paper cover that.
FIG_LABEL_SIZE        = 13
FIG_TICK_SIZE         = 12
FIG_LEGEND_SIZE       = 11
FIG_LEGEND_TITLE_SIZE = 12
FIG_ANNOT_SIZE        = 11

# Plotly font sizes are CSS px, not points, and print smaller than the same
# number in matplotlib at this figure's canvas proportions — scaled up to
# roughly match the FIG_* sizes above at print size without crowding the panel.
PLOTLY_TITLE_SIZE  = 22
PLOTLY_LABEL_SIZE  = 38
PLOTLY_TICK_SIZE   = 36
PLOTLY_LEGEND_SIZE = 32
PLOTLY_ANNOT_SIZE  = 32

# Top fraction of snowpack (by burial depth) treated as the "upper zone".
# Must match UPPER_ZONE_FRAC in classifier.py.
_UPPER_ZONE_FRAC = 0.40

# Colors are darkened from the "obvious" red/orange/gold so the threshold
# labels meet WCAG AA text contrast (>=3:1 at this large font size) against
# the white margin they're drawn on; plain 'orange'/'gold' fail that check.
_PROB_THRESHOLDS = [
    (0.80, 'black'),
    (0.65, '#CC0000'),
    (0.50, '#B35900'),
    (0.33, '#8B6508'),
]


# ── stability analysis ────────────────────────────────────────────────────────

def _get_dominant_layers(df: pd.DataFrame, z_tolerance: float = 3.0) -> pd.DataFrame:
    """
    Returns the subset of df representing layers that were at some point the
    minimum-Sn38 layer, starting from the timestep each layer first claimed
    that minimum.

    Algorithm:
      1. Walk timesteps in order; at each step find the min-Sn38 layer.
      2. If that layer's z is not within z_tolerance of any known cluster,
         register it as a new dominant layer with its first-seen timestamp.
      3. For each cluster, include only rows from first_seen onward within
         the ±z_tolerance band around the cluster center.
    """
    df_reset   = df.reset_index()
    timesteps  = sorted(df_reset['timestamp'].unique())
    clusters: list[tuple] = []  # (z_center, label, first_seen_timestamp)

    def find_cluster(z: float) -> int | None:
        for idx, (z_center, _, _) in enumerate(clusters):
            if abs(z - z_center) <= z_tolerance:
                return idx
        return None

    for ts in timesteps:
        group = df_reset[df_reset['timestamp'] == ts]
        z = group.loc[group['sn38'].idxmin(), 'layer_z']
        if find_cluster(z) is None:
            clusters.append((z, f"Layer {len(clusters) + 1}", ts))

    parts: list[pd.DataFrame] = []
    for z_center, label, first_seen in clusters:
        in_band = (
            (df['layer_z'] >= z_center - z_tolerance) &
            (df['layer_z'] <= z_center + z_tolerance)
        )
        sub = df[in_band & (df.index >= first_seen)].copy()
        sub['layer_label'] = label
        parts.append(sub)

    return pd.concat(parts).sort_index() if parts else df.iloc[0:0]


# ── plotting ──────────────────────────────────────────────────────────────────

def _sanitize_timestamps(obj):
    """
    Recursively replace pandas Timestamps with plain datetimes.

    kaleido's orjson-based serializer doesn't recognize pd.Timestamp (a C
    extension type) even though it subclasses datetime.datetime, so any
    Timestamp reachable from fig.to_dict() (annotations, shapes, axis ranges)
    must be converted before a static image export.
    """
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    if isinstance(obj, dict):
        return {k: _sanitize_timestamps(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_timestamps(v) for v in obj]
    return obj


def plot_interactive_stability(
    df: pd.DataFrame,
    output_path: Path,
    station_id: str = "",
    event_dates: list[pd.Timestamp] | None = None,
    test_dates: list[pd.Timestamp] | None = None,
    prob_series: pd.Series | None = None,
    daily_df: pd.DataFrame | None = None,
    forecast_date: pd.Timestamp | None = None,
    aspect: str = "",
    confidence_tier: str = "",
    png_path: Path | None = None,
    png_rows: list[str] | None = None,
) -> None:
    """
    Generates an interactive Plotly HTML stability chart.

    Panel 1 — Probability (top, requires prob_series): blended model daily
               avalanche probability for the current season, with threshold lines.
    Panel 2 — Snow stratigraphy: HS surface, dominant weak layers coloured by
               Sn38 (circle = upper zone, diamond = lower zone).
    Panel 3 — Loading (optional, requires daily_df): daily new snow (HN24) and
               air temperature (TA).

    Args:
        png_path: When given, also renders a static PNG snapshot (e.g. for a
                  print figure) with the rangeslider widget hidden — the
                  rangeslider is an HTML-only interactive control and has no
                  place in a static image.
        png_rows: Restricts the PNG export to a subset of {'prob', 'strat',
                  'load'} (e.g. ['prob'] for a probability-only print figure).
                  The interactive HTML always keeps every available panel;
                  this only trims the static image.
    """
    dominant = _get_dominant_layers(df)
    hs_ts    = df.groupby(df.index)['total_height'].first()

    # Zone classification: upper = burial depth < UPPER_ZONE_FRAC * HS
    dominant = dominant.copy()
    dominant['is_upper'] = (
        dominant['burial_depth'] < _UPPER_ZONE_FRAC * dominant['total_height']
    )
    zone_boundary_z = hs_ts * (1.0 - _UPPER_ZONE_FRAC)  # height from ground

    has_prob    = prob_series is not None and not prob_series.dropna().empty
    has_loading = daily_df is not None and not daily_df.empty

    # Pin the x-axis to the current-season .pro file date range so that no
    # historical-season data from the combined prob_series or daily_df bleeds in.
    x_min_raw = df.index.min()
    x_max_raw = df.index.max()
    if has_prob and prob_series is not None:
        prob_series = prob_series[
            (prob_series.index >= x_min_raw) & (prob_series.index <= x_max_raw)
        ]
        has_prob = not prob_series.dropna().empty
    if has_loading and daily_df is not None:
        daily_df = daily_df[
            (daily_df.index >= x_min_raw) & (daily_df.index <= x_max_raw)
        ]
        has_loading = not daily_df.empty

    x_min = x_min_raw
    x_max = x_max_raw

    def _build(allowed: set[str] | None, include_title: bool = True) -> tuple[go.Figure, str, int]:
        """
        Builds the multi-panel figure, optionally restricted to `allowed` row keys.

        include_title: Panel titles and the overall station/aspect/confidence
                       title are HTML-only — the print figure relies on its
                       LaTeX caption instead, so it's built with no titles at all.
        """
        # ── Build subplot layout: prob on top, then strat, then loading ───────
        row_keys: list[str] = []
        if has_prob and (allowed is None or 'prob' in allowed):
            row_keys.append('prob')
        if allowed is None or 'strat' in allowed:
            row_keys.append('strat')
        if has_loading and (allowed is None or 'load' in allowed):
            row_keys.append('load')

        n_rows = len(row_keys)
        specs  = [[{"secondary_y": True} if k == 'load' else {}] for k in row_keys]

        if n_rows == 1:
            row_heights = [1.0]
        elif n_rows == 2:
            row_heights = [0.40, 0.60] if row_keys[0] == 'prob' else [0.60, 0.40]
        else:
            row_heights = [0.35, 0.40, 0.25]

        panel_titles = {
            'prob':  "Avalanche Probability — 2025–26 Season",
            'strat': "Snow Stratigraphy — Weak Layers by Height",
            'load':  "New Snow (HN24) & Air Temperature",
        }

        fig = make_subplots(
            rows=n_rows, cols=1,
            specs=specs,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=(tuple(panel_titles[k] for k in row_keys)
                             if include_title else None),
            row_heights=row_heights,
        )
        if include_title:
            # subplot_titles are the only annotations on the figure at this
            # point, so this sizes just the panel titles (not the
            # event/forecast labels added further below).
            for ann in fig.layout.annotations:
                ann.update(font=dict(size=PLOTLY_TITLE_SIZE))

        strat_row = row_keys.index('strat') + 1 if 'strat' in row_keys else None
        load_row  = row_keys.index('load') + 1 if 'load' in row_keys else None
        prob_row  = row_keys.index('prob') + 1 if 'prob' in row_keys else None

        # Row that carries the event/forecast-date text annotations (shapes
        # still span every row); prefers stratigraphy, else whatever exists.
        label_row = strat_row or prob_row or load_row

        # ── Panel: Stratigraphy ────────────────────────────────────────────────
        if strat_row is not None:
            # Faint blue bands on rain days
            if has_loading and 'rain_sum' in daily_df.columns:  # type: ignore[union-attr]
                for rday in daily_df[daily_df['rain_sum'] > 0].index:  # type: ignore[index]
                    fig.add_shape(
                        type='rect',
                        x0=rday, x1=rday + pd.Timedelta(days=1),
                        y0=0, y1=hs_ts.max() * 1.15,
                        fillcolor='rgba(30,100,220,0.07)', line_width=0,
                        row=strat_row, col=1,
                    )

            # Snow surface
            fig.add_trace(go.Scatter(
                x=hs_ts.index, y=hs_ts.values,
                name="Snow Surface (HS)",
                line=dict(color='rgba(140,140,140,0.55)', width=2),
                hovertemplate="HS: %{y:.1f} cm<extra></extra>",
            ), row=strat_row, col=1)

            # Zone boundary (60 % of HS height from ground)
            fig.add_trace(go.Scatter(
                x=zone_boundary_z.index, y=zone_boundary_z.values,
                name="Zone boundary (40 % depth)",
                line=dict(color='rgba(80,80,80,0.30)', width=1, dash='dot'),
                hovertemplate="Zone split: %{y:.1f} cm<extra></extra>",
            ), row=strat_row, col=1)

            # Dominant layers: circle = upper zone, diamond = lower zone
            colorbar_shown = False
            for i, label in enumerate(sorted(dominant['layer_label'].unique())):
                sub    = dominant[dominant['layer_label'] == label]
                colour = _LAYER_PALETTE[i % len(_LAYER_PALETTE)]

                for is_upper, symbol in [(True, 'circle'), (False, 'diamond')]:
                    sub_z = sub[sub['is_upper'] == is_upper]
                    if sub_z.empty:
                        continue
                    zone_name = 'upper' if is_upper else 'lower'
                    show_cb   = not colorbar_shown
                    if show_cb:
                        colorbar_shown = True

                    fig.add_trace(go.Scatter(
                        x=sub_z.index,
                        y=sub_z['layer_z'],
                        mode='markers',
                        name=f"{label} ({zone_name})",
                        marker=dict(
                            size=9, symbol=symbol,
                            color=sub_z['sn38'],
                            colorscale='RdYlGn', cmin=1.0, cmax=6.0,
                            showscale=show_cb,
                            colorbar=dict(title="Sn38", thickness=14, len=0.45, y=0.75),
                            line=dict(width=1.2, color=colour),
                        ),
                        customdata=sub_z[['burial_depth', 'sn38']].values,
                        hovertemplate=(
                            f"<b>{label} ({zone_name})</b><br>"
                            "Height: %{y:.1f} cm<br>"
                            "Burial: %{customdata[0]:.1f} cm<br>"
                            "Sn38: %{customdata[1]:.2f}<extra></extra>"
                        ),
                    ), row=strat_row, col=1)

        # ── Panel: Loading / Temperature ──────────────────────────────────────
        if has_loading and load_row is not None and daily_df is not None:
            if 'HN24' in daily_df.columns:
                fig.add_trace(go.Bar(
                    x=daily_df.index, y=daily_df['HN24'],
                    name="HN24 (m)",
                    marker_color='rgba(70,130,200,0.72)',
                    hovertemplate="HN24: %{y:.3f} m<extra></extra>",
                ), row=load_row, col=1, secondary_y=False)

            if 'TA_max' in daily_df.columns:
                ta = daily_df['TA_max']
                # 0 °C reference
                fig.add_trace(go.Scatter(
                    x=[x_min, x_max], y=[0, 0],
                    mode='lines', showlegend=False, hoverinfo='skip',
                    line=dict(color='rgba(220,80,60,0.40)', width=1, dash='dash'),
                ), row=load_row, col=1, secondary_y=True)
                # Temperature line
                fig.add_trace(go.Scatter(
                    x=ta.index, y=ta.values,
                    name="TA max (°C)",
                    line=dict(color='rgba(220,80,60,0.85)', width=1.5),
                    hovertemplate="TA: %{y:.1f} °C<extra></extra>",
                ), row=load_row, col=1, secondary_y=True)

            fig.update_yaxes(
                title_text="HN24 (m)", title_font=dict(size=PLOTLY_LABEL_SIZE),
                rangemode='tozero',
                row=load_row, col=1, secondary_y=False,
            )
            fig.update_yaxes(
                title_text="TA (°C)", title_font=dict(size=PLOTLY_LABEL_SIZE),
                row=load_row, col=1, secondary_y=True,
            )

        # ── Panel: Probability ─────────────────────────────────────────────────
        if has_prob and prob_series is not None and prob_row is not None:
            prob_clean = prob_series.dropna()

            # Coloured background zones as filled polygons (avoids axis-ref complexity)
            bands = [
                (0.80, 1.00, 'rgba(0,0,0,0.08)'),
                (0.65, 0.80, 'rgba(200,50,50,0.10)'),
                (0.50, 0.65, 'rgba(255,140,0,0.12)'),
                (0.33, 0.50, 'rgba(255,215,0,0.13)'),
            ]
            for y_lo, y_hi, fill in bands:
                fig.add_trace(go.Scatter(
                    x=[x_min, x_max, x_max, x_min, x_min],
                    y=[y_lo,  y_lo,  y_hi,  y_hi,  y_lo],
                    fill='toself', fillcolor=fill,
                    line=dict(width=0), mode='lines',
                    showlegend=False, hoverinfo='skip',
                ), row=prob_row, col=1)

            # Probability curve
            fig.add_trace(go.Scatter(
                x=prob_clean.index, y=prob_clean.values,
                name="Avalanche Probability",
                mode='lines',
                line=dict(color='#333333', width=2),
                fill='tozeroy', fillcolor='rgba(100,100,200,0.10)',
                hovertemplate="Prob: %{y:.1%}<extra></extra>",
            ), row=prob_row, col=1)

            # Threshold lines + labels as traces (avoids axis-ref issues)
            for threshold, colour in _PROB_THRESHOLDS:
                fig.add_trace(go.Scatter(
                    x=[x_min, x_max], y=[threshold, threshold],
                    mode='lines', showlegend=False, hoverinfo='skip',
                    line=dict(color=colour, width=1.2, dash='dot'),
                ), row=prob_row, col=1)
                fig.add_annotation(
                    x=x_max, y=threshold,
                    text=f"  {int(threshold * 100)} %",
                    showarrow=False, xanchor='left',
                    font=dict(color=colour, size=PLOTLY_ANNOT_SIZE),
                    row=prob_row, col=1,
                )

            fig.update_yaxes(
                title_text="Probability", title_font=dict(size=PLOTLY_LABEL_SIZE),
                tickformat='.0%',
                range=[0, 1], row=prob_row, col=1,
            )

        # ── Train event markers (red) ──────────────────────────────────────────
        for edate in (event_dates or []):
            for r in range(1, n_rows + 1):
                fig.add_shape(
                    type='line',
                    x0=edate, x1=edate, y0=0, y1=1,
                    yref='y domain',
                    line=dict(color='red', width=1.5, dash='dash'),
                    row=r, col=1,
                )
            fig.add_annotation(
                x=edate, y=1, yref='y domain',
                text=edate.strftime('%b %d'),
                showarrow=False, xanchor='left',
                font=dict(color='red', size=PLOTLY_ANNOT_SIZE),
                row=label_row, col=1,
            )

        # ── Test event markers (blue, held-out) ────────────────────────────────
        for edate in (test_dates or []):
            for r in range(1, n_rows + 1):
                fig.add_shape(
                    type='line',
                    x0=edate, x1=edate, y0=0, y1=1,
                    yref='y domain',
                    line=dict(color='royalblue', width=1.5, dash='dot'),
                    row=r, col=1,
                )
            fig.add_annotation(
                x=edate, y=0.88, yref='y domain',
                text=edate.strftime('%b %d') + ' (test)',
                showarrow=False, xanchor='left',
                font=dict(color='royalblue', size=PLOTLY_ANNOT_SIZE),
                row=label_row, col=1,
            )

        # ── Forecast window highlight ──────────────────────────────────────────
        if forecast_date is not None:
            fw_start = forecast_date - pd.Timedelta(days=1)
            fw_end   = forecast_date + pd.Timedelta(days=2)
            for r in range(1, n_rows + 1):
                # Semi-transparent green band covering the window
                fig.add_shape(
                    type='rect',
                    x0=fw_start, x1=fw_end,
                    y0=0, y1=1, yref='y domain',
                    fillcolor='rgba(60,180,100,0.10)',
                    line=dict(width=0),
                    row=r, col=1,
                )
                # Dashed "now" line at the forecast reference date
                fig.add_shape(
                    type='line',
                    x0=forecast_date, x1=forecast_date,
                    y0=0, y1=1, yref='y domain',
                    line=dict(color='rgba(40,150,80,0.80)', width=1.5, dash='dash'),
                    row=r, col=1,
                )
            fig.add_annotation(
                x=forecast_date, y=1, yref='y domain',
                text=f"  {forecast_date.strftime('%b %d')} (now)",
                showarrow=False, xanchor='left',
                font=dict(color='rgb(40,150,80)', size=PLOTLY_ANNOT_SIZE),
                row=label_row, col=1,
            )

        # ── Layout ──────────────────────────────────────────────────────────────
        if strat_row is not None:
            fig.update_yaxes(title_text="Height from ground (cm)",
                              title_font=dict(size=PLOTLY_LABEL_SIZE),
                              row=strat_row, col=1)

        height = {1: 800, 2: 750, 3: 1000}.get(n_rows, 1000)

        # Rangeslider on the bottom-most available panel.
        slider_row = load_row or strat_row or prob_row
        slider_key = (f"xaxis{slider_row}_rangeslider_visible" if slider_row > 1
                      else "xaxis_rangeslider_visible")

        # Build title: station, aspect, confidence (HTML only — see include_title)
        title_text = ""
        if include_title:
            sid_clean  = station_id.replace('_res', '')
            tier_label = {'ready': 'Ready', 'marginal': 'Marginal', 'not_ready': 'Not ready'}.get(
                confidence_tier, ''
            )
            title_parts = [f"Station {sid_clean}"]
            if aspect:
                title_parts.append(f"Aspect: {aspect}")
            if tier_label:
                title_parts.append(f"Confidence: {tier_label}")
            title_text = "  |  ".join(title_parts)

        fig.update_layout(
            height=height,
            title_text=title_text,
            title_font=dict(size=PLOTLY_TITLE_SIZE),
            template="plotly_white",
            hovermode="x unified",
            xaxis_range=[x_min, x_max],
            xaxis_autorange=False,
            **{slider_key: True, slider_key.replace("visible", "thickness"): 0.04},
            legend=dict(orientation='h', y=-0.08, font=dict(size=PLOTLY_LEGEND_SIZE)),
            bargap=0.15,
            margin=dict(r=140),
        )
        # Enforce range on every x-axis in the figure (shared axes get their own key)
        fig.update_xaxes(range=[x_min, x_max], autorange=False,
                          tickfont=dict(size=PLOTLY_TICK_SIZE))
        fig.update_yaxes(tickfont=dict(size=PLOTLY_TICK_SIZE))

        return fig, slider_key, height

    html_fig, html_slider_key, html_height = _build(None)
    html_fig.write_html(str(output_path))

    if png_path is not None:
        if png_rows is not None:
            png_fig, png_slider_key, png_height = _build(set(png_rows), include_title=False)
        else:
            png_fig, png_slider_key, png_height = html_fig, html_slider_key, html_height
        png_fig.update_layout(**{png_slider_key: False})
        clean_fig = go.Figure(_sanitize_timestamps(png_fig.to_dict()))
        clean_fig.write_image(str(png_path), width=1600, height=png_height, scale=2)


# ── map ───────────────────────────────────────────────────────────────────────

def _date_nav_html(features_json: str) -> str:
    """Returns the HTML+JS for the date navigator overlay.

    features_json: JSON-serialised list of GeoJSON feature dicts.  The nav
    reads dates and station URLs from these features to drive the sim-layer
    date-coloring without rendering any markers itself.
    """
    return f"""
    <div id="date-nav" style="
        position: fixed; bottom: 12px; left: 50%; transform: translateX(-50%);
        z-index: 1000; background: rgba(255,255,255,0.95);
        padding: 6px 14px; border-radius: 8px; border: 1px solid #ccc;
        font-family: sans-serif; font-size: 13px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.18);
        display: flex; align-items: center; gap: 8px; white-space: nowrap;
    ">
        <button onclick="navDate(-1)" title="Previous date"
                style="background:none;border:1px solid #aaa;border-radius:4px;padding:2px 9px;cursor:pointer;font-size:15px">&#8592;</button>
        <span id="date-label" style="min-width:150px;text-align:center;font-weight:bold">All dates</span>
        <button onclick="navDate(1)" title="Next date"
                style="background:none;border:1px solid #aaa;border-radius:4px;padding:2px 9px;cursor:pointer;font-size:15px">&#8594;</button>
        <span style="color:#aaa">|</span>
        <button onclick="showAllDates()" title="Show all dates"
                style="background:none;border:1px solid #aaa;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:11px;color:#555">All</button>
        <span id="date-counter" style="color:#888;font-size:11px"></span>
    </div>
    <script>
    window.addEventListener('load', function() {{
        var _features = {features_json};

        var _dates = (function() {{
            var seen = {{}};
            _features.forEach(function(f) {{ var d = f.properties.date; if (d) seen[d] = true; }});
            return Object.keys(seen).sort(function(a, b) {{ return new Date(a) - new Date(b); }});
        }})();

        var _idx = -1;

        function _refresh() {{
            var show = _idx === -1
                ? _features
                : _features.filter(function(f) {{ return f.properties.date === _dates[_idx]; }});
            var lbl = document.getElementById('date-label');
            var ctr = document.getElementById('date-counter');
            if (lbl) lbl.textContent = _idx === -1 ? 'All dates' : _dates[_idx];
            if (ctr) ctr.textContent = _idx === -1
                ? '(' + _dates.length + ' dates)'
                : '(' + (_idx + 1) + ' / ' + _dates.length + ')';
            var obs = document.getElementById('obs-count');
            if (obs) obs.textContent = show.length + ' observation' + (show.length !== 1 ? 's' : '');
            window._dateNavState = {{idx: _idx, dates: _dates, features: _features}};
            if (typeof window._obsRefresh  === 'function') window._obsRefresh();
            if (typeof window._simRefresh  === 'function') window._simRefresh();
        }}

        window._dateNavState = {{idx: -1, dates: _dates, features: _features}};

        window.navDate = function(dir) {{
            if (_dates.length === 0) return;
            if (_idx === -1) {{
                _idx = dir > 0 ? 0 : _dates.length - 1;
            }} else {{
                _idx = Math.max(0, Math.min(_dates.length - 1, _idx + dir));
            }}
            _refresh();
        }};

        window.showAllDates = function() {{ _idx = -1; _refresh(); }};

        var ctr = document.getElementById('date-counter');
        if (ctr) ctr.textContent = '(' + _dates.length + ' dates)';
        var obs = document.getElementById('obs-count');
        if (obs) obs.textContent = _features.length + ' observation' + (_features.length !== 1 ? 's' : '');
    }});
    </script>
    """


def _sim_layer_html(
    map_name: str,
    layer_ctrl_name: str,
    stations: list[dict],
    prob_series_json: str = '{}',
) -> str:
    """
    Returns HTML+JS that adds a simulation-stations overlay and aspect-filter UI.

    Each entry in `stations` must have keys:
        id, lat, lon, aspect, forecast_prob (float|None),
        confidence_tier ('ready' | 'marginal' | 'not_ready')

    prob_series_json: JSON-serialised dict  {station_id: {"YYYY-MM-DD": prob, ...}}
        used to recolour _simLayer as the time player scrubs through dates.
        The TimestampedGeoJson layer drives the time slider but its own circles are
        hidden (weight 0, fillOpacity 0) — _simLayer is the sole visual layer.
    """
    stations_js = json.dumps(stations)
    return f"""
    <div id="aspect-filter" style="
        position: fixed; bottom: 40px; right: 12px; z-index: 1000;
        background: rgba(255,255,255,0.95);
        padding: 8px 10px; border-radius: 8px; border: 1px solid #ccc;
        font-family: sans-serif; font-size: 11px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.18);
    ">
        <div style="font-weight:bold;color:#444;margin-bottom:5px">Aspect filter</div>
        <div style="display:flex;gap:3px;flex-wrap:wrap;max-width:160px">
            <button id="asp-All"  onclick="filterAspect('All')"  style="background:#444;color:#fff;border:1px solid #444;border-radius:4px;padding:2px 7px;cursor:pointer;font-size:11px">All</button>
            <button id="asp-N"    onclick="filterAspect('N')"    style="background:none;color:#555;border:1px solid #aaa;border-radius:4px;padding:2px 7px;cursor:pointer;font-size:11px">N</button>
            <button id="asp-E"    onclick="filterAspect('E')"    style="background:none;color:#555;border:1px solid #aaa;border-radius:4px;padding:2px 7px;cursor:pointer;font-size:11px">E</button>
            <button id="asp-S"    onclick="filterAspect('S')"    style="background:none;color:#555;border:1px solid #aaa;border-radius:4px;padding:2px 7px;cursor:pointer;font-size:11px">S</button>
            <button id="asp-W"    onclick="filterAspect('W')"    style="background:none;color:#555;border:1px solid #aaa;border-radius:4px;padding:2px 7px;cursor:pointer;font-size:11px">W</button>
            <button id="asp-Flat" onclick="filterAspect('Flat')" style="background:none;color:#555;border:1px solid #aaa;border-radius:4px;padding:2px 7px;cursor:pointer;font-size:11px">Flat</button>
        </div>
        <hr style="margin:6px 0;border:none;border-top:1px solid #ddd">
        <div style="font-weight:bold;color:#444;margin-bottom:4px">Layers</div>
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer;margin-bottom:3px">
            <input type="checkbox" id="sim-toggle" checked onchange="window.toggleSim(this.checked)">
            <span>Simulation stations</span>
        </label>
        <label style="display:flex;align-items:center;gap:5px;cursor:pointer">
            <input type="checkbox" id="reg-toggle" onchange="window.toggleRegional(this.checked)">
            <span>Regional model</span>
        </label>
    </div>
    <script>
    window.addEventListener('load', function() {{
        var _SIM        = {stations_js};
        var _PROB_SERIES = {prob_series_json};  // {{station_id: {{"YYYY-MM-DD": prob}}}}

        function _simColors(p) {{
            if (p === null || p === undefined) return {{fill:'#808080', stroke:'#505050'}};
            if (p >= 0.80) return {{fill:'#7b0000', stroke:'#3d0000'}};
            if (p >= 0.65) return {{fill:'#d62728', stroke:'#8b0000'}};
            if (p >= 0.50) return {{fill:'#ff7f0e', stroke:'#b35a00'}};
            if (p >= 0.33) return {{fill:'#e6c200', stroke:'#9e8600'}};
            return {{fill:'#2ca02c', stroke:'#1a6b1a'}};
        }}

        var _simAspect = 'All';
        var _simLayer = L.geoJson(null, {{
            pointToLayer: function(feature, latlng) {{
                var p    = feature.properties.forecast_prob;
                var c    = _simColors(p);
                var tier = feature.properties.confidence_tier;
                // Ring colour   = probability (vivid — primary risk signal)
                // Fill opacity  = confidence tier (§4.5):
                //   0.82 (solid)  → Ready     (FA ≤ 10%)
                //   0.42 (faded)  → Marginal  (FA 10–25%)
                //   0.12 (ghost)  → Not ready (FA > 25%)
                // Ring weight also varies by tier for a redundant confidence cue.
                var fillOp = tier === 'ready' ? 0.82 : tier === 'marginal' ? 0.42 : 0.12;
                var weight = tier === 'ready' ? 3.5  : tier === 'marginal' ? 2.5  : 1.5;
                return L.circleMarker(latlng, {{
                    radius:      7,
                    fillColor:   c.fill,
                    fillOpacity: fillOp,
                    color:       c.fill,
                    weight:      weight,
                    opacity:     1,
                }});
            }},
            onEachFeature: function(feature, layer) {{
                var pr = feature.properties;
                var probStr = (pr.forecast_prob !== null && pr.forecast_prob !== undefined)
                    ? Math.round(pr.forecast_prob * 100) + '%' : 'n/a';
                var tier = pr.confidence_tier;
                var tierLabel = tier === 'ready'
                    ? '<b>Ready</b> (FA &le;10%)'
                    : tier === 'marginal'
                        ? '<b>Marginal</b> (FA 10–25%)'
                        : (pr.forecast_prob !== null && pr.forecast_prob !== undefined)
                            ? 'Not ready (FA &gt;25%)'
                            : 'No data';
                layer.bindTooltip(
                    '<b>' + pr.id.replace('_res','') + '</b><br>'
                    + 'Aspect: ' + pr.aspect + '<br>'
                    + 'Prob: <b>' + probStr + '</b><br>'
                    + 'Confidence: ' + tierLabel,
                    {{sticky: true}}
                );
                layer.on('click', function() {{
                    if (pr.url) window.open(pr.url, '_blank');
                }});
            }}
        }});

        // Parse "Month DD, YYYY" (date nav format) → "YYYY-MM-DD" for PROB_SERIES lookup.
        function _navDateToISO(dateStr) {{
            var d = new Date(dateStr);
            if (isNaN(d)) return null;
            return d.getUTCFullYear() + '-'
                + String(d.getUTCMonth() + 1).padStart(2, '0') + '-'
                + String(d.getUTCDate()).padStart(2, '0');
        }}

        // Return the max value in series ({{YYYY-MM-DD: prob}}) within ±halfDays of isoDate.
        function _maxProbInWindow(series, isoDate, halfDays) {{
            var centre = new Date(isoDate + 'T00:00:00Z');
            if (isNaN(centre)) return undefined;
            var best;
            for (var d = -halfDays; d <= halfDays; d++) {{
                var dt  = new Date(centre.getTime() + d * 86400000);
                var key = dt.getUTCFullYear() + '-'
                    + String(dt.getUTCMonth() + 1).padStart(2, '0') + '-'
                    + String(dt.getUTCDate()).padStart(2, '0');
                var p = series[key];
                if (p !== undefined && (best === undefined || p > best)) best = p;
            }}
            return best;
        }}
        window._maxProbInWindow = _maxProbInWindow;  // shared with obs script

        // tdDate: "YYYY-MM-DD" from time player, or null/undefined for forecast window.
        function _doSimRefresh(tdDate) {{
            var state = window._dateNavState || {{idx: -1, features: [], dates: []}};

            // Build a per-station probability override for the current context.
            // Stations not in the override keep their all-window forecast_prob (never gray).
            var probOverride = null;

            if (state.idx !== -1) {{
                // Date nav active: recolour all stations by max prob in ±3-day window.
                var isoDate = _navDateToISO(state.dates[state.idx]);
                if (isoDate) {{
                    probOverride = {{}};
                    _SIM.forEach(function(s) {{
                        var p = _maxProbInWindow(_PROB_SERIES[s.id] || {{}}, isoDate, 3);
                        if (p !== undefined) probOverride[s.id] = p;
                    }});
                }}
            }} else if (tdDate) {{
                // Time player: exact date (preserves day-by-day resolution).
                probOverride = {{}};
                _SIM.forEach(function(s) {{
                    var p = (_PROB_SERIES[s.id] || {{}})[tdDate];
                    if (p !== undefined) probOverride[s.id] = p;
                }});
            }}

            _simLayer.clearLayers();

            // Apply prob override to get each station's current probability.
            var candidates = _SIM.map(function(s) {{
                var prob = s.forecast_prob;
                if (probOverride !== null && s.id in probOverride) prob = probOverride[s.id];
                return {{id:s.id, lat:s.lat, lon:s.lon, aspect:s.aspect,
                         forecast_prob:prob, confidence_tier:s.confidence_tier, url:s.url}};
            }});

            var filtered;
            if (_simAspect === 'All') {{
                // One marker per unique location, coloured by max probability across all aspects.
                var byLoc = {{}};
                candidates.forEach(function(s) {{
                    var key = s.lat.toFixed(4) + ',' + s.lon.toFixed(4);
                    var cur = byLoc[key];
                    var sProb = s.forecast_prob !== null && s.forecast_prob !== undefined ? s.forecast_prob : -1;
                    var cProb = cur && cur.forecast_prob !== null && cur.forecast_prob !== undefined ? cur.forecast_prob : -1;
                    if (!cur || sProb > cProb) byLoc[key] = s;
                }});
                filtered = Object.values(byLoc);
            }} else {{
                filtered = candidates.filter(function(s) {{ return s.aspect === _simAspect; }});
            }}

            var features = filtered.map(function(s) {{
                return {{type:'Feature',
                         geometry:{{type:'Point', coordinates:[s.lon, s.lat]}},
                         properties:{{id:s.id, aspect:s.aspect, forecast_prob:s.forecast_prob,
                                      confidence_tier:s.confidence_tier, url:s.url}}}};
            }});
            _simLayer.addData({{type:'FeatureCollection', features:features}});
        }}

        window._simRefresh = _doSimRefresh;

        // Time player: poll timeDimension and recolour sim markers when it advances.
        // Skipped when date nav has a specific date selected (date nav takes priority).
        // _tdSimSeen: skip first tick so initial TD time doesn't override forecast-window colours.
        var _lastTdTimeForSim = null;
        var _tdSimSeen        = false;
        setInterval(function() {{
            var _map = {map_name};
            if (!_map.timeDimension) return;
            var state = window._dateNavState || {{idx: -1}};
            if (state.idx !== -1) return;
            var t = _map.timeDimension.getCurrentTime();
            if (t === null) return;
            if (!_tdSimSeen) {{ _tdSimSeen = true; _lastTdTimeForSim = t; return; }}
            if (t === _lastTdTimeForSim) return;
            _lastTdTimeForSim = t;
            var d = new Date(t);
            var dateStr = d.getUTCFullYear() + '-'
                + String(d.getUTCMonth() + 1).padStart(2, '0') + '-'
                + String(d.getUTCDate()).padStart(2, '0');
            _doSimRefresh(dateStr);
        }}, 150);

        // ── Regional model overlay layer ──────────────────────────────────
        var _regionalLayer = L.geoJson(null, {{
            pointToLayer: function(feature, latlng) {{
                var p = feature.properties.regional_prob;
                var c = _simColors(p);
                return L.circleMarker(latlng, {{
                    radius: 7,
                    fillColor: c.fill,
                    color: c.stroke,
                    weight: 3.5,
                    opacity: 1,
                    fillOpacity: p !== null && p !== undefined ? 0.78 : 0.15
                }});
            }},
            onEachFeature: function(feature, layer) {{
                var pr = feature.properties;
                var probStr = (pr.regional_prob !== null && pr.regional_prob !== undefined)
                    ? Math.round(pr.regional_prob * 100) + '%' : 'n/a';
                layer.bindTooltip(
                    '<b>' + pr.id.replace('_res','') + '</b><br>'
                    + 'Aspect: ' + pr.aspect + '<br>'
                    + 'Regional prob: <b>' + probStr + '</b><br>'
                    + '<span style="color:#888">&#x25CB; Regional model</span>',
                    {{sticky: true}}
                );
                layer.on('click', function() {{
                    if (pr.url) window.open(pr.url, '_blank');
                }});
            }}
        }});

        function _doRegionalRefresh() {{
            _regionalLayer.clearLayers();
            var show = _SIM.filter(function(s) {{ return _simAspect === 'All' || s.aspect === _simAspect; }});
            var features = show.map(function(s) {{
                return {{type:'Feature',
                         geometry:{{type:'Point', coordinates:[s.lon, s.lat]}},
                         properties:{{id:s.id, aspect:s.aspect, regional_prob:s.regional_prob, url:s.url}}}};
            }});
            _regionalLayer.addData({{type:'FeatureCollection', features:features}});
        }}

        window.filterAspect = function(asp) {{
            _simAspect = asp;
            ['All','N','E','S','W','Flat'].forEach(function(a) {{
                var btn = document.getElementById('asp-' + a);
                if (!btn) return;
                btn.style.background  = a === asp ? '#444' : 'none';
                btn.style.color       = a === asp ? '#fff' : '#555';
                btn.style.borderColor = a === asp ? '#444' : '#aaa';
            }});
            _doSimRefresh();
            _doRegionalRefresh();
        }};

        window.toggleSim = function(show) {{
            if (show) {{ _simLayer.addTo({map_name}); _doSimRefresh(); }}
            else       {map_name}.removeLayer(_simLayer);
        }};

        window.toggleRegional = function(show) {{
            if (show) {{ _regionalLayer.addTo({map_name}); _doRegionalRefresh(); }}
            else       {map_name}.removeLayer(_regionalLayer);
        }};

        window.filterAspect('All');
        _simLayer.addTo({map_name});
        // Regional overlay starts hidden (checkbox is unchecked by default)
    }});
    </script>
    """


def create_avalanche_map(
    df: pd.DataFrame,
    output_path: Path,
    target_url: str = "stability_analysis.html",
    forecast_date: pd.Timestamp | None = None,
    prob_by_station: dict[str, pd.Series] | None = None,
    sim_stations: list[dict] | None = None,
) -> None:
    """
    Build and write a Folium forecast map.

    Observation triangles are coloured by the blended model probability within
    the forecast window (forecast_date − 1 day to forecast_date + 2 days).
    Six-level colour scale (gray / green / gold / orange / red / dark red)
    matches the JS thresholds: < 33 %, 33–50 %, 50–65 %, 65–80 %, ≥ 80 %.

    Args:
        df:              Must contain Latitude, Longitude, Placemark Name
                         columns, and optionally Aspect, date, target_url,
                         forecast_prob, station_id.
        output_path:     Destination HTML file.
        target_url:      Fallback URL when df has no target_url column.
        forecast_date:   Reference date shown in the map legend.
        prob_by_station: station_id → daily probability Series.  When supplied,
                         enables the time slider and date navigator recolouring.
        sim_stations:    List of station metadata dicts (id, lat, lon, aspect,
                         forecast_prob, confidence_tier, regional_prob, url).
    """
    if 'target_url' not in df.columns:
        df = df.copy()
        df['target_url'] = target_url

    m = folium.Map(
        location=[df['Latitude'].mean(), df['Longitude'].mean()],
        zoom_start=9,
    )
    Geocoder().add_to(m)

    folium.TileLayer(
        tiles='https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap',
        name='Topographic',
    ).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite',
    ).add_to(m)
    folium.TileLayer('OpenStreetMap', name='Standard').add_to(m)

    # Serialise prob_by_station early — needed by both the obs script and sim layer.
    prob_series_dict: dict[str, dict[str, float]] = {}
    if prob_by_station:
        for _sid, _series in prob_by_station.items():
            _clean = _series.dropna()
            prob_series_dict[_sid] = {
                pd.Timestamp(_ts).strftime('%Y-%m-%d'): round(float(_p), 4)
                for _ts, _p in _clean.items()
            }
    prob_series_json = json.dumps(prob_series_dict)

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['Longitude'], row['Latitude']],
            },
            "properties": {
                "name":         row['Placemark Name'],
                "aspect":       str(row.get('Aspect', '') or ''),
                "date":         str(row.get('date', '') or ''),
                "url":          row['target_url'],
                "station_id":   str(row.get('station_id', '') or ''),
                "forecast_prob": (
                    None if (
                        'forecast_prob' not in df.columns
                        or math.isnan(row.get('forecast_prob', float('nan')))
                    )
                    else round(float(row['forecast_prob']), 4)
                ),
            },
        }
        for _, row in df.iterrows()
    ]

    m.get_root().html.add_child(folium.Element(_date_nav_html(json.dumps(features))))

    # ── Observation markers (triangles, date-filtered) ────────────────────────
    point_to_layer = folium.JsCode("""
        function(feature, latlng) {
            var p = feature.properties.forecast_prob;
            var fill, stroke;
            if (p === null || p === undefined) {
                fill = '#808080'; stroke = '#505050';
            } else if (p >= 0.80) {
                fill = '#7b0000'; stroke = '#3d0000';
            } else if (p >= 0.65) {
                fill = '#d62728'; stroke = '#8b0000';
            } else if (p >= 0.50) {
                fill = '#ff7f0e'; stroke = '#b35a00';
            } else if (p >= 0.33) {
                fill = '#e6c200'; stroke = '#9e8600';
            } else {
                fill = '#2ca02c'; stroke = '#1a6b1a';
            }
            var svg = '<svg width="22" height="20" viewBox="0 0 22 20" xmlns="http://www.w3.org/2000/svg">'
                + '<polygon points="11,1 21,19 1,19" fill="' + fill + '" stroke="' + stroke
                + '" stroke-width="2.5" stroke-linejoin="round"/></svg>';
            return L.marker(latlng, {
                icon: L.divIcon({
                    html: svg, className: '',
                    iconSize: [22, 20], iconAnchor: [11, 19], tooltipAnchor: [0, -20]
                })
            });
        }
    """)

    on_each_feature = folium.JsCode("""
        function(feature, layer) {
            var p = feature.properties;
            var tip = '<b>' + p.name + '</b>';
            if (p.date)         tip += '<br>Date: '   + p.date;
            if (p.aspect)       tip += '<br>Aspect: ' + p.aspect;
            if (p.forecast_prob !== null && p.forecast_prob !== undefined)
                tip += '<br><b>Prob: ' + Math.round(p.forecast_prob * 100) + '%</b>';
            layer.bindTooltip(tip, {sticky: true});
            layer.on('click', function() { if (p.url) window.open(p.url, '_blank'); });
        }
    """)

    obs_layer = folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name='Avalanche Observations',
        point_to_layer=point_to_layer,
        on_each_feature=on_each_feature,
    )
    obs_layer.add_to(m)

    # Wire date nav + time-series player → observation layer filtering
    obs_layer_name = obs_layer.get_name()
    map_name       = m.get_name()
    m.get_root().html.add_child(folium.Element(f"""
    <script>
    window.addEventListener('load', function() {{
        var _obsLayer    = {obs_layer_name};
        var _map         = {map_name};
        var _OBS_PROB_SERIES = {prob_series_json};

        function _navDateToISOObs(dateStr) {{
            var d = new Date(dateStr);
            if (isNaN(d)) return null;
            return d.getUTCFullYear() + '-'
                + String(d.getUTCMonth() + 1).padStart(2, '0') + '-'
                + String(d.getUTCDate()).padStart(2, '0');
        }}

        function _filterObs(features) {{
            _obsLayer.clearLayers();
            _obsLayer.addData({{type: 'FeatureCollection', features: features}});
        }}

        var _maxProbInWindowObs = window._maxProbInWindow || function(s, d, h) {{ return s[d]; }};

        // Date nav: filter to selected date and recolour triangles by max prob in ±3-day window.
        window._obsRefresh = function() {{
            var state = window._dateNavState || {{idx: -1, dates: [], features: []}};
            if (state.idx === -1) {{
                _filterObs(state.features);
                return;
            }}
            var sel     = state.dates[state.idx];
            var isoDate = _navDateToISOObs(sel);
            var show = state.features
                .filter(function(f) {{ return f.properties.date === sel; }})
                .map(function(f) {{
                    if (!isoDate) return f;
                    var sid = f.properties.station_id;
                    var p   = sid ? _maxProbInWindowObs(_OBS_PROB_SERIES[sid] || {{}}, isoDate, 3) : undefined;
                    if (p === undefined) return f;
                    return {{
                        type: f.type,
                        geometry: f.geometry,
                        properties: Object.assign({{}}, f.properties, {{forecast_prob: p}})
                    }};
                }});
            _filterObs(show);
        }};

        // Time-series player: poll timeDimension and filter + recolour triangles.
        // _tdObsSeen: skip first tick so initial TD time doesn't hide all triangles.
        var _lastTdTime = null;
        var _tdObsSeen  = false;
        var _months = ['January','February','March','April','May','June',
                       'July','August','September','October','November','December'];
        setInterval(function() {{
            if (!_map.timeDimension) return;
            var state = window._dateNavState || {{idx: -1}};
            if (state.idx !== -1) return;
            var t = _map.timeDimension.getCurrentTime();
            if (t === null) return;
            if (!_tdObsSeen) {{ _tdObsSeen = true; _lastTdTime = t; return; }}
            if (t === _lastTdTime) return;
            _lastTdTime = t;
            var d       = new Date(t);
            var isoDate = d.getUTCFullYear() + '-'
                + String(d.getUTCMonth() + 1).padStart(2, '0') + '-'
                + String(d.getUTCDate()).padStart(2, '0');
            var str = _months[d.getUTCMonth()] + ' '
                    + String(d.getUTCDate()).padStart(2, '0') + ', '
                    + d.getUTCFullYear();
            var all = (window._dateNavState || {{}}).features || [];
            var show = all
                .filter(function(f) {{ return f.properties.date === str; }})
                .map(function(f) {{
                    var sid = f.properties.station_id;
                    var p   = sid ? (_OBS_PROB_SERIES[sid] || {{}})[isoDate] : undefined;
                    if (p === undefined) return f;
                    return {{
                        type: f.type,
                        geometry: f.geometry,
                        properties: Object.assign({{}}, f.properties, {{forecast_prob: p}})
                    }};
                }});
            _filterObs(show);
        }}, 150);
    }});
    </script>
    """))

    # ── Forecast window info box + probability legend ─────────────────────────
    if forecast_date is not None:
        win_start_str = (forecast_date - pd.Timedelta(days=1)).strftime('%b %d')
        win_end_str   = (forecast_date + pd.Timedelta(days=2)).strftime('%b %d')
        now_str       = forecast_date.strftime('%b %d, %Y')
    else:
        win_start_str = win_end_str = now_str = "—"

    legend_html = f"""
    <div style="
        position: fixed; bottom: 40px; left: 12px; z-index: 1000;
        background: rgba(255,255,255,0.94); padding: 10px 14px;
        border-radius: 8px; border: 1px solid #ccc;
        font-family: sans-serif; font-size: 12px; line-height: 1.7;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    ">
        <b>Forecast window</b><br>
        <span style="color:#555">Now: {now_str}</span><br>
        <span style="color:#555">Window: {win_start_str} – {win_end_str}</span><br>
        <hr style="margin:6px 0">
        <b>Max probability in window</b><br>
        <span style="color:#808080">&#9679;</span> No classifier<br>
        <span style="color:#2ca02c">&#9679;</span> &lt; 33 %<br>
        <span style="color:#e6c200">&#9679;</span> 33 – 50 %<br>
        <span style="color:#ff7f0e">&#9679;</span> 50 – 65 %<br>
        <span style="color:#d62728">&#9679;</span> 65 – 80 %<br>
        <span style="color:#7b0000">&#9679;</span> &ge; 80 %
        <hr style="margin:6px 0">
        <span id="obs-count" style="color:#555"></span>
        <hr style="margin:6px 0">
        <b>Simulation stations</b><br>
        <span style="color:#555;font-size:11px">Ring colour &#8594; probability<br>
        Solid fill &#8594; <b>Ready</b> (FA &le;10%)<br>
        Faded fill &#8594; Marginal (FA 10–25%)<br>
        Ghost fill &#8594; Not ready (FA &gt;25%)</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Time-series player: daily probability at each sim station ────────────
    if prob_by_station and sim_stations:
        # Build a coord lookup: station_id → (lon, lat)
        station_coords = {s['id']: (s['lon'], s['lat']) for s in sim_stations}

        # Circles are invisible — they exist only to populate timeDimension so
        # the time slider works.  _simLayer is the sole visual layer and
        # recolours itself via setInterval polling timeDimension.getCurrentTime().
        _invisible = {
            "fillColor": "transparent", "color": "transparent",
            "fillOpacity": 0, "weight": 0, "radius": 1,
        }
        ts_features: list[dict] = []
        for sid, series in prob_by_station.items():
            coords = station_coords.get(sid)
            if coords is None:
                continue
            for ts, p in series.dropna().items():
                ts_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": list(coords)},
                    "properties": {
                        "times":     [pd.Timestamp(ts).strftime('%Y-%m-%d')],
                        "icon":      "circle",
                        "iconstyle": _invisible,
                        "popup": (f"<b>{sid.replace('_res','')}</b><br>"
                                  f"{pd.Timestamp(ts).strftime('%b %d, %Y')}<br>"
                                  f"Prob: {float(p)*100:.0f}%"),
                    },
                })

        td_layer_js_name: str | None = None
        if ts_features:
            TimestampedGeoJson(
                {"type": "FeatureCollection", "features": ts_features},
                period="P1D", duration="P1D", transition_time=150,
                auto_play=False, loop=False, max_speed=15,
                date_options="YYYY-MM-DD", time_slider_drag_update=True,
                add_last_point=False,
            ).add_to(m)

    layer_ctrl = folium.LayerControl()
    layer_ctrl.add_to(m)

    # ── Simulation stations overlay (aspect-filterable) ───────────────────────
    if sim_stations:
        m.get_root().html.add_child(
            folium.Element(_sim_layer_html(
                m.get_name(), layer_ctrl.get_name(), sim_stations,
                json.dumps(prob_series_dict),
            ))
        )

    m.save(str(output_path))


# Six-level probability color scale, matching the JS point_to_layer /
# _sim_layer_html scales used by the interactive map above.
_MAP_PROB_COLORS = [
    (0.80, '#7b0000', '≥ 80 %'),
    (0.65, '#d62728', '65 – 80 %'),
    (0.50, '#ff7f0e', '50 – 65 %'),
    (0.33, '#e6c200', '33 – 50 %'),
    (0.00, '#2ca02c', '< 33 %'),
]
_MAP_PROB_NO_CLASSIFIER = '#808080'

# Admin-0 country polygons for the study region (Natural Earth 1:50m,
# trimmed to Central/South Asia), bundled locally so the study-area map
# doesn't depend on an external vector source at render time. Drawn as
# plain polylines rather than a basemap raster overlay so line width and
# label placement are under our control.
_COUNTRY_BORDERS_PATH = Path(__file__).resolve().parent.parent / 'data' / 'geo' / 'central_asia_countries.geojson'
_country_borders_cache: list[dict] | None = None


def _map_prob_color(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return _MAP_PROB_NO_CLASSIFIER
    for threshold, color, _ in _MAP_PROB_COLORS:
        if p >= threshold:
            return color
    return _MAP_PROB_COLORS[-1][1]


def _load_country_borders() -> list[dict]:
    global _country_borders_cache
    if _country_borders_cache is None:
        with open(_COUNTRY_BORDERS_PATH) as fh:
            _country_borders_cache = json.load(fh)['features']
    return _country_borders_cache


def _point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    """Standard PNPOLY even-odd ray-casting test."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _point_in_country(x: float, y: float, polys: list) -> bool:
    for poly in polys:
        if _point_in_ring(x, y, poly[0]) and not any(
            _point_in_ring(x, y, hole) for hole in poly[1:]
        ):
            return True
    return False


def _label_position(
    in_view: list[tuple[float, float]],
    polys: list,
    avoid_points: list[tuple[float, float]],
) -> tuple[float, float]:
    """
    Pick a label anchor near the centroid of `in_view` vertices that stays
    clear of `avoid_points` (station/observation markers). Tries the
    centroid plus eight offsets around it, discards any candidate that
    falls outside the country's own polygon (so a nudge never lands the
    label in a neighboring country), and keeps whichever of the rest
    maximizes distance to the nearest avoid_point.
    """
    xs = [p[0] for p in in_view]
    ys = [p[1] for p in in_view]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    if not avoid_points:
        return cx, cy

    dx = (max(xs) - min(xs)) * 0.22 or 0.3
    dy = (max(ys) - min(ys)) * 0.22 or 0.3
    candidates = [(cx, cy)] + [
        (cx + fx * dx, cy + fy * dy)
        for fx in (-1, 0, 1) for fy in (-1, 0, 1) if not (fx == 0 and fy == 0)
    ]
    valid = [pt for pt in candidates if _point_in_country(pt[0], pt[1], polys)]
    if not valid:
        valid = [(cx, cy)]

    def min_dist(pt: tuple[float, float]) -> float:
        return min(math.hypot(pt[0] - ax_, pt[1] - ay_) for ax_, ay_ in avoid_points)

    return max(valid, key=min_dist)


def _draw_country_borders(
    ax,
    extent: tuple[float, float, float, float],
    avoid_points: list[tuple[float, float]] | None = None,
    linewidth: float = 2.2,
    color: str = 'black',
) -> None:
    """
    Draw country border polylines and name labels within `extent`
    (lon_min, lon_max, lat_min, lat_max), from the bundled Natural Earth
    admin-0 polygons. A country's label is anchored near the centroid of
    its own vertices that fall inside `extent` (so labels stay on-map even
    though the source polygons extend far beyond it), nudged away from
    `avoid_points` so it doesn't land under a station/observation marker.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    avoid_points = avoid_points or []
    for feat in _load_country_borders():
        geom = feat['geometry']
        polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
        in_view: list[tuple[float, float]] = []
        for poly in polys:
            for ring in poly:
                xs = [pt[0] for pt in ring]
                ys = [pt[1] for pt in ring]
                ax.plot(xs, ys, color=color, linewidth=linewidth,
                        solid_capstyle='round', zorder=2)
                in_view.extend(
                    (x, y) for x, y in zip(xs, ys)
                    if lon_min <= x <= lon_max and lat_min <= y <= lat_max
                )
        if in_view:
            label_x, label_y = _label_position(in_view, polys, avoid_points)
            ax.text(
                label_x, label_y, feat['properties']['name'].upper(),
                color='black', fontsize=FIG_LABEL_SIZE, fontweight='bold',
                ha='center', va='center', zorder=2,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')],
            )


def plot_static_map(
    df: pd.DataFrame,
    sim_stations: list[dict],
    forecast_date: pd.Timestamp,
    output_path: Path,
) -> None:
    """
    Render a static, print-quality study-area map: SNOWPACK virtual stations
    as circles ringed by their forecast-window max probability, and that
    day's avalanche observations as triangles colored the same way.

    Stations that have never been matched to an observation (see
    Observation-to-Station Matching) are drawn at reduced opacity so the
    reader's eye goes to stations with a real event history.

    Args:
        df:            Must contain Latitude, Longitude, date, forecast_prob,
                       station_id columns (as built in main._load_observations
                       / main's forecast_prob assignment).
        sim_stations:  List of station metadata dicts (id, lat, lon,
                       forecast_prob), as built in main.py for create_avalanche_map.
        forecast_date: Reference date; the window shown is [date-1, date+2].
        output_path:   Destination PNG file.
    """
    matched_stations = set(df['station_id'].dropna())

    fig, ax = plt.subplots(figsize=(8, 6.2))

    for s in sim_stations:
        color   = _map_prob_color(s.get('forecast_prob'))
        matched = s['id'] in matched_stations
        ax.scatter(
            s['lon'], s['lat'],
            s=150 if matched else 90, marker='o',
            facecolor=color, edgecolor='white',
            linewidth=1.4 if matched else 0.9,
            alpha=1.0 if matched else 0.55, zorder=3,
        )

    day_str = forecast_date.strftime('%B %d, %Y')
    obs_day = df[df['date'] == day_str]
    for _, row in obs_day.iterrows():
        color = _map_prob_color(row.get('forecast_prob'))
        ax.scatter(
            row['Longitude'], row['Latitude'], s=190, marker='^',
            facecolor=color, edgecolor='black', linewidth=1.3, zorder=4,
        )

    ax.set_xlabel('Longitude (°E)', fontsize=FIG_LABEL_SIZE)
    ax.set_ylabel('Latitude (°N)', fontsize=FIG_LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=FIG_TICK_SIZE)

    # Extent must cover stations AND that day's observations — an outlier
    # report can fall well outside the station network's bounding box.
    lons = [s['lon'] for s in sim_stations] + obs_day['Longitude'].tolist()
    lats = [s['lat'] for s in sim_stations] + obs_day['Latitude'].tolist()
    mean_lat = float(np.mean(lats)) if lats else 0.0
    ax.set_aspect(1 / max(math.cos(math.radians(mean_lat)), 1e-6))

    # Pad the data extent slightly, then fix the limits before fetching tiles
    # so contextily's basemap request matches exactly what we plot.
    pad_lon = (max(lons) - min(lons)) * 0.08 or 0.1
    pad_lat = (max(lats) - min(lats)) * 0.08 or 0.1
    xlim = (min(lons) - pad_lon, max(lons) + pad_lon)
    ylim = (min(lats) - pad_lat, max(lats) + pad_lat)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    try:
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.Esri.WorldImagery,
                         attribution_size=6, zorder=0)
    except Exception as exc:  # noqa: BLE001 — tile fetch can fail offline; keep the scatter
        print(f"  (basemap fetch failed, falling back to plain background: {exc})")
        ax.grid(True, alpha=0.25, linestyle='--')

    ax.autoscale(False)  # country polylines must not expand the fixed map extent
    _draw_country_borders(ax, extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                           avoid_points=list(zip(lons, lats)))

    prob_handles = [
        Line2D([0], [0], marker='o', linestyle='', markerfacecolor=color,
               markeredgecolor='#333333', markersize=9, label=label)
        for _, color, label in _MAP_PROB_COLORS
    ] + [
        Line2D([0], [0], marker='o', linestyle='', markerfacecolor=_MAP_PROB_NO_CLASSIFIER,
               markeredgecolor='#333333', markersize=9, label='No classifier'),
    ]
    shape_handles = [
        Line2D([0], [0], marker='o', linestyle='', markerfacecolor='white',
               markeredgecolor='#333333', markersize=9, label='SNOWPACK station'),
        Line2D([0], [0], marker='^', linestyle='', markerfacecolor='white',
               markeredgecolor='black', markersize=9, label='Avalanche observation'),
        Line2D([0], [0], marker='o', linestyle='', markerfacecolor=_MAP_PROB_NO_CLASSIFIER,
               markeredgecolor='#333333', markersize=7, alpha=0.55,
               label='Station, no matched observation'),
    ]
    # Legends sit outside the axes (never over the data) since real stations
    # can fall anywhere, including the corners a legend would normally occupy.
    leg1 = ax.legend(handles=prob_handles, title='Max probability in window',
                      loc='upper left', bbox_to_anchor=(1.02, 1.0),
                      fontsize=FIG_LEGEND_SIZE, title_fontsize=FIG_LEGEND_TITLE_SIZE,
                      framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, loc='upper left', bbox_to_anchor=(1.02, 0.45),
              fontsize=FIG_LEGEND_SIZE, framealpha=0.9)

    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ── terrain-trace validation map ──────────────────────────────────────────────

def _trace_quality(feats: dict, trace_error: str) -> tuple[str, str, str]:
    """
    Classify a traced path for visual QA.

    Returns (level, colour, reason) where level is 'good' | 'check' | 'fail'.
    The heuristics flag traces that likely did not reach a real start zone at
    30 m resolution — the reviewer confirms or rejects them by eye.
    """
    if trace_error or feats.get('vertical_drop') is None \
            or (isinstance(feats.get('vertical_drop'), float) and math.isnan(feats['vertical_drop'])):
        return 'fail', '#d62728', trace_error or 'no steep start zone found'

    # Governing start zone is steep by construction, so QA flags degenerate paths:
    # too short, or too little vertical drop.
    L = feats.get('travel_distance', float('nan'))
    H = feats.get('vertical_drop', float('nan'))

    reasons = []
    if not (L >= 100):
        reasons.append(f'short path L={L:.0f} m')
    if not (H >= 50):
        reasons.append(f'small drop H={H:.0f} m')

    if reasons:
        return 'check', '#ff7f0e', '; '.join(reasons)
    return 'good', '#2ca02c', 'plausible'


def create_trace_validation_map(
    observations: pd.DataFrame,
    output_path: Path,
    radius_km: float = 4.0,
) -> pd.DataFrame:
    """
    Draw each governing avalanche-path terrain trace on a Folium map for visual QA.

    For every unique observation coordinate, traces the reverse-watershed
    governing path (governing start zone → valley floor) and draws:
      - a debris seed marker (red-filled when the trace failed, to stand out),
      - the traced polyline (governing start zone → valley floor),
      - a filled start-zone marker coloured by trace quality (green / orange /
        red), with H, L, alpha, start-zone slope/aspect/elevation and catchment,
      - a hollow valley-floor marker at the bottom of the path,
      - a faint dashed step-up reference line (the single-thread up-trace), and
      - a toggleable catchment-polygon layer.

    Satellite imagery is the default base layer so traces can be checked against
    real gullies. Returns the per-observation feature/quality table it built.

    Args:
        observations: must contain 'Latitude' and 'Longitude'; 'Placemark Name',
                      'date', 'Size', 'Remarks' are shown in tooltips if present.
        output_path:  destination HTML file.
        radius_km:    DEM window half-size passed to terrain.fetch_dem.
    """
    import terrain  # local import: pulls in rasterio only when this map is built

    coords = observations[['Latitude', 'Longitude']].dropna().drop_duplicates()

    # Region-batched fetch: download one regional DEM per cluster of *uncached*
    # points, so each is sliced locally instead of a per-point API call.
    uncached = coords[~coords.apply(
        lambda r: terrain.dem_is_cached(r['Longitude'], r['Latitude']), axis=1)]
    if len(uncached):
        try:
            regions = terrain.prepare_regions(uncached, radius_km=radius_km)
            print(f"prepared {len(regions)} regional DEM(s) for {len(uncached)} uncached points")
        except Exception as exc:  # noqa: BLE001 — fall back to per-point / cached
            print(f"region prep failed ({exc}); using cached tiles only")

    m = folium.Map(
        location=[coords['Latitude'].mean(), coords['Longitude'].mean()],
        zoom_start=11,
    )
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery', name='Satellite',
    ).add_to(m)
    folium.TileLayer(
        tiles='https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap', name='Topographic',
    ).add_to(m)
    folium.TileLayer('OpenStreetMap', name='Standard').add_to(m)

    groups = {
        'good':  folium.FeatureGroup(name='Traces — plausible'),
        'check': folium.FeatureGroup(name='Traces — needs review'),
        'fail':  folium.FeatureGroup(name='Traces — failed'),
    }
    catchment_grp = folium.FeatureGroup(name='Catchments (reverse-watershed)', show=False)

    # metadata keyed by rounded coordinate, for tooltips
    meta = {
        (round(r['Latitude'], 5), round(r['Longitude'], 5)): r
        for _, r in observations.iterrows()
    }

    records = []
    for lat, lon in coords.itertuples(index=False):
        trace_error = ''
        feats: dict = {}
        polyline: list[tuple[float, float]] = []
        geom = None
        try:
            dem = terrain.fetch_dem(lon, lat, radius_km=radius_km)
            geom = terrain.trace_path(dem, lon, lat)
            feats = terrain.path_features(geom)
            polyline = [(ll[1], ll[0]) for ll in geom.lonlat]  # (lat, lon) for folium
        except Exception as exc:  # noqa: BLE001 — QA map should never abort on one bad point
            trace_error = f'{type(exc).__name__}: {exc}'

        level, colour, reason = _trace_quality(feats, trace_error)
        grp = groups[level]

        info = meta.get((round(lat, 5), round(lon, 5)))
        name = info['Placemark Name'] if info is not None and 'Placemark Name' in info else ''

        # reverse-watershed catchment polygon (toggleable QA layer)
        if geom is not None and geom.catchment_rings:
            for ring in geom.catchment_rings:
                folium.Polygon(
                    [(y, x) for x, y in ring], color='#3186cc', weight=1,
                    fill=True, fill_color='#3186cc', fill_opacity=0.10,
                    tooltip=folium.Tooltip(
                        f"<b>{name or 'catchment'}</b><br>"
                        f"area = {geom.catchment_area_km2:.2f} km²<br>"
                        f"{geom.num_start_zones} start zone(s)"),
                ).add_to(catchment_grp)

        # step-up reference line (single-thread up-trace) — faint, for comparison
        if geom is not None and len(geom.uptrace_lonlat) > 1:
            folium.PolyLine(
                [(y, x) for x, y in geom.uptrace_lonlat],
                color='#666666', weight=1.5, opacity=0.55, dash_array='4,5',
                tooltip=folium.Tooltip('step-up trace (reference)'),
            ).add_to(grp)

        if polyline:
            folium.PolyLine(
                polyline, color=colour, weight=3, opacity=0.85,
            ).add_to(grp)
            # start zone = top of the path (start→valley); valley floor = bottom.
            start_lat, start_lon = polyline[0]
            folium.CircleMarker(
                [start_lat, start_lon], radius=6, color=colour, weight=2,
                fill=True, fill_color=colour, fill_opacity=0.9,
                tooltip=folium.Tooltip(
                    f"<b>{name or 'path'}</b> — governing start zone<br>"
                    f"H = {feats.get('vertical_drop', float('nan')):.0f} m &nbsp; "
                    f"L = {feats.get('travel_distance', float('nan')):.0f} m<br>"
                    f"&alpha; = {feats.get('alpha', float('nan')):.1f}° &nbsp; "
                    f"start slope = {feats.get('startzone_slope', float('nan')):.0f}°<br>"
                    f"start elev = {feats.get('startzone_elev', float('nan')):.0f} m &nbsp; "
                    f"aspect = {feats.get('startzone_aspect', float('nan')):.0f}°<br>"
                    f"catchment {feats.get('catchment_area_km2', float('nan')):.2f} km² · "
                    f"{int(feats.get('num_start_zones', 0))} start zone(s)<br>"
                    f"<i>{reason}</i>"
                ),
            ).add_to(grp)
            # valley-floor marker (bottom of the traced path)
            vf_lat, vf_lon = polyline[-1]
            folium.CircleMarker(
                [vf_lat, vf_lon], radius=4, color=colour, weight=1.5,
                fill=True, fill_color='#ffffff', fill_opacity=0.9,
                tooltip=folium.Tooltip(f"<b>{name or 'path'}</b> — valley floor"),
            ).add_to(grp)

        # debris seed marker (always drawn — red-filled when the trace failed,
        # so failed locations are easy to spot).
        seed_fill = colour if level == 'fail' else '#ffffff'
        seed_edge = colour if level == 'fail' else '#222222'
        folium.CircleMarker(
            [lat, lon], radius=6 if level == 'fail' else 4,
            color=seed_edge, weight=1.5,
            fill=True, fill_color=seed_fill, fill_opacity=0.95,
            tooltip=folium.Tooltip(
                f"<b>{name or 'debris'}</b><br>"
                f"{info['date'] if info is not None and 'date' in info else ''}<br>"
                f"<i>{trace_error or reason}</i>"
            ),
        ).add_to(grp)

        rec = {'Latitude': lat, 'Longitude': lon, 'quality': level,
               'reason': reason, 'trace_error': trace_error}
        rec.update(feats)
        records.append(rec)

    for grp in groups.values():
        grp.add_to(m)
    catchment_grp.add_to(m)

    legend_html = """
    <div style="
        position: fixed; bottom: 40px; left: 12px; z-index: 1000;
        background: rgba(255,255,255,0.94); padding: 10px 14px;
        border-radius: 8px; border: 1px solid #ccc;
        font-family: sans-serif; font-size: 12px; line-height: 1.7;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    ">
        <b>Terrain trace QA</b><br>
        <span style="color:#2ca02c">&#9679;</span> Plausible<br>
        <span style="color:#ff7f0e">&#9679;</span> Needs review<br>
        <span style="color:#d62728">&#9679;</span> Failed<br>
        <hr style="margin:6px 0">
        <span style="color:#222">&#9679;</span> Debris seed &nbsp;
        &#9679; Start zone (governing) &nbsp;
        &#9711; Valley floor<br>
        <span style="color:#666">- -</span> Step-up trace (reference) &nbsp;
        <span style="color:#3186cc">&#9632;</span> Catchment (toggle layer)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(output_path))
    return pd.DataFrame(records)
