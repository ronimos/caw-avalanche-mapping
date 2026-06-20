"""
visualization.py — Snowpack stability analysis and interactive output.

Provides:
  - plot_interactive_stability  Multi-panel Plotly HTML stability chart.
  - create_avalanche_map        Folium map of avalanche observations.
"""

import math
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_LAYER_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#17becf',
]

# Top fraction of snowpack (by burial depth) treated as the "upper zone".
# Must match UPPER_ZONE_FRAC in classifier.py.
_UPPER_ZONE_FRAC = 0.40

_PROB_THRESHOLDS = [
    (0.95, 'black'),
    (0.80, 'red'),
    (0.65, 'orange'),
    (0.50, 'gold'),
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

def plot_interactive_stability(
    df: pd.DataFrame,
    output_path: Path,
    station_id: str = "",
    event_dates: list[pd.Timestamp] | None = None,
    test_dates: list[pd.Timestamp] | None = None,
    prob_series: pd.Series | None = None,
    daily_df: pd.DataFrame | None = None,
    forecast_date: pd.Timestamp | None = None,
) -> None:
    """
    Generates an interactive Plotly HTML stability chart.

    Panel 1 — Snow stratigraphy: HS surface, dominant weak layers coloured by
               Sn38 (circle = upper zone, diamond = lower zone), dynamic zone
               boundary, and faint blue bands on rain days.
    Panel 2 — Loading (optional, requires daily_df): daily new snow bars (HN24)
               with air temperature (TA) on a secondary axis and a 0 °C line.
    Panel 3 — Probability (optional, requires prob_series): daily avalanche
               probability with threshold lines at 50 % / 65 % / 80 % / 95 %.

    Vertical dashed red lines mark training event dates on all panels.
    Vertical dashed blue lines mark held-out test event dates on all panels.
    A green highlight band marks the forecast window (±1/+2 days around
    forecast_date) with a dashed "now" line on all panels.

    Args:
        df:            Output of parse_snow_data().
        output_path:   Destination HTML file.
        station_id:    Label shown in the chart title.
        event_dates:   Training avalanche dates (red lines).
        test_dates:    Held-out test avalanche dates (blue lines).
        prob_series:   Daily probability Series (index=date, values 0–1).
        daily_df:      Output of build_daily_features() — used for loading panel.
        forecast_date: Reference date for the forecast window highlight.
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

    # ── Build subplot layout ──────────────────────────────────────────────────
    row_keys: list[str] = ['strat']
    if has_loading:
        row_keys.append('load')
    if has_prob:
        row_keys.append('prob')

    n_rows = len(row_keys)
    specs  = [[{"secondary_y": True} if k == 'load' else {}] for k in row_keys]

    if n_rows == 1:
        row_heights = [1.0]
    elif n_rows == 2:
        row_heights = [0.60, 0.40]
    else:
        row_heights = [0.50, 0.25, 0.25]

    panel_titles = {
        'strat': "Snow Stratigraphy — Weak Layers by Height",
        'load':  "New Snow (HN24) & Air Temperature",
        'prob':  "Avalanche Probability",
    }

    fig = make_subplots(
        rows=n_rows, cols=1,
        specs=specs,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=tuple(panel_titles[k] for k in row_keys),
        row_heights=row_heights,
    )

    strat_row = row_keys.index('strat') + 1
    load_row  = row_keys.index('load') + 1 if 'load' in row_keys else None
    prob_row  = row_keys.index('prob') + 1 if 'prob' in row_keys else None

    x_min = df.index.min()
    x_max = df.index.max()

    # ── Panel 1: Stratigraphy ─────────────────────────────────────────────────

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

    # ── Panel 2: Loading / Temperature ───────────────────────────────────────
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
            title_text="HN24 (m)", rangemode='tozero',
            row=load_row, col=1, secondary_y=False,
        )
        fig.update_yaxes(
            title_text="TA (°C)",
            row=load_row, col=1, secondary_y=True,
        )

    # ── Panel 3: Probability ──────────────────────────────────────────────────
    if has_prob and prob_series is not None and prob_row is not None:
        prob_clean = prob_series.dropna()

        # Coloured background zones as filled polygons (avoids axis-ref complexity)
        bands = [
            (0.95, 1.00, 'rgba(0,0,0,0.08)'),
            (0.80, 0.95, 'rgba(200,50,50,0.10)'),
            (0.65, 0.80, 'rgba(255,140,0,0.12)'),
            (0.50, 0.65, 'rgba(255,215,0,0.13)'),
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
                font=dict(color=colour, size=10),
                row=prob_row, col=1,
            )

        fig.update_yaxes(
            title_text="Probability", tickformat='.0%',
            range=[0, 1], row=prob_row, col=1,
        )

    # ── Train event markers (red) ─────────────────────────────────────────────
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
            font=dict(color='red', size=11),
            row=strat_row, col=1,
        )

    # ── Test event markers (blue, held-out) ───────────────────────────────────
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
            font=dict(color='royalblue', size=11),
            row=strat_row, col=1,
        )

    # ── Forecast window highlight ─────────────────────────────────────────────
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
            font=dict(color='rgb(40,150,80)', size=11),
            row=strat_row, col=1,
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_yaxes(title_text="Height from ground (cm)", row=strat_row, col=1)

    height = {1: 750, 2: 950, 3: 1100}.get(n_rows, 1100)

    # Rangeslider on loading panel when present, otherwise on bottom panel
    slider_row = load_row if load_row else (prob_row if prob_row else strat_row)
    slider_key = f"xaxis{slider_row}_rangeslider_visible" if slider_row > 1 else "xaxis_rangeslider_visible"

    fig.update_layout(
        height=height,
        title_text=f"Stability Analysis — Station {station_id}",
        template="plotly_white",
        hovermode="x unified",
        xaxis_range=[x_min, x_max],
        **{slider_key: True, slider_key.replace("visible", "thickness"): 0.04},
        legend=dict(orientation='h', y=-0.08),
        bargap=0.15,
    )
    fig.write_html(str(output_path))


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
    import json as _json
    stations_js = _json.dumps(stations)
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
            if (p >= 0.95) return {{fill:'#7b0000', stroke:'#3d0000'}};
            if (p >= 0.80) return {{fill:'#d62728', stroke:'#8b0000'}};
            if (p >= 0.65) return {{fill:'#ff7f0e', stroke:'#b35a00'}};
            if (p >= 0.50) return {{fill:'#e6c200', stroke:'#9e8600'}};
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

        // tdDate: "YYYY-MM-DD" from time player, or null/undefined for forecast window.
        function _doSimRefresh(tdDate) {{
            var state = window._dateNavState || {{idx: -1, features: [], dates: []}};

            // Build a per-station probability override for the current context.
            // Stations not in the override keep their all-window forecast_prob (never gray).
            var probOverride = null;

            if (state.idx !== -1) {{
                // Date nav active: recolour all stations from PROB_SERIES for this date.
                var isoDate = _navDateToISO(state.dates[state.idx]);
                if (isoDate) {{
                    probOverride = {{}};
                    _SIM.forEach(function(s) {{
                        var p = (_PROB_SERIES[s.id] || {{}})[isoDate];
                        if (p !== undefined) probOverride[s.id] = p;
                    }});
                }}
            }} else if (tdDate) {{
                // Time player: look up each station's prob for that specific date.
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


def _prob_to_colors(p: float | None) -> tuple[str, str]:
    """Map a probability to (fill, stroke) colours — mirrors the map's JS scale."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return '#808080', '#505050'
    if p >= 0.95: return '#7b0000', '#3d0000'
    if p >= 0.80: return '#d62728', '#8b0000'
    if p >= 0.65: return '#ff7f0e', '#b35a00'
    if p >= 0.50: return '#e6c200', '#9e8600'
    return '#2ca02c', '#1a6b1a'


def create_avalanche_map(
    df: pd.DataFrame,
    output_path: Path,
    target_url: str = "stability_analysis.html",
    forecast_date: pd.Timestamp | None = None,
    prob_by_station: dict[str, pd.Series] | None = None,
    sim_stations: list[dict] | None = None,
) -> None:
    """
    Creates a Folium map with circle markers for each avalanche observation.
    Marker colour reflects the maximum avalanche probability within the forecast
    window (forecast_date − 1 day to forecast_date + 2 days):
      gray        — no classifier available
      green       — < 50 %
      gold        — 50–65 %
      orange      — 65–80 %
      red         — 80–95 %
      dark red    — ≥ 95 %

    Hovering shows name, date, aspect, elevation, size, and remarks.
    Clicking opens the linked stability HTML in a new tab.

    When `prob_by_station` is supplied, the map also gains a date slider
    (TimestampedGeoJson): scrubbing through the season recolours each marker by
    its station's probability on that date, so the user can watch risk evolve.

    Args:
        df:            Must contain Latitude, Longitude, Placemark Name columns,
                       and optionally Aspect, date, Elevation (M), Size, Remarks,
                       target_url, forecast_prob, station_id.
        output_path:   Destination HTML file.
        target_url:    Fallback URL when df has no target_url column.
        forecast_date: Reference date shown in the map legend.
        prob_by_station: station_id → daily probability Series. Enables the date
                       slider; rows are matched to a series via the station_id column.
    """
    if 'target_url' not in df.columns:
        df = df.copy()
        df['target_url'] = target_url

    from folium.plugins import Geocoder

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

    import json as _json

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

    m.get_root().html.add_child(folium.Element(_date_nav_html(_json.dumps(features))))

    # ── Observation markers (triangles, date-filtered) ────────────────────────
    point_to_layer = folium.JsCode("""
        function(feature, latlng) {
            var p = feature.properties.forecast_prob;
            var fill, stroke;
            if (p === null || p === undefined) {
                fill = '#808080'; stroke = '#505050';
            } else if (p >= 0.95) {
                fill = '#7b0000'; stroke = '#3d0000';
            } else if (p >= 0.80) {
                fill = '#d62728'; stroke = '#8b0000';
            } else if (p >= 0.65) {
                fill = '#ff7f0e'; stroke = '#b35a00';
            } else if (p >= 0.50) {
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
        var _obsLayer = {obs_layer_name};
        var _map      = {map_name};

        function _filterObs(features) {{
            _obsLayer.clearLayers();
            _obsLayer.addData({{type: 'FeatureCollection', features: features}});
        }}

        // Date nav: filter by selected observation date (or show all)
        window._obsRefresh = function() {{
            var state = window._dateNavState || {{idx: -1, dates: [], features: []}};
            var show = state.idx === -1
                ? state.features
                : state.features.filter(function(f) {{ return f.properties.date === state.dates[state.idx]; }});
            _filterObs(show);
        }};

        // Time-series player: poll timeDimension and filter triangles when it advances.
        // Only applies when the date nav is at "All dates" (idx === -1); date nav takes
        // priority when a specific date is selected.
        // _tdObsSeen: skip the first tick so the initial TD time doesn't hide all triangles.
        var _lastTdTime = null;
        var _tdObsSeen  = false;
        var _months = ['January','February','March','April','May','June',
                       'July','August','September','October','November','December'];
        setInterval(function() {{
            if (!_map.timeDimension) return;
            var state = window._dateNavState || {{idx: -1}};
            if (state.idx !== -1) return;          // date nav has priority
            var t = _map.timeDimension.getCurrentTime();
            if (t === null) return;
            if (!_tdObsSeen) {{ _tdObsSeen = true; _lastTdTime = t; return; }}
            if (t === _lastTdTime) return;
            _lastTdTime = t;
            var d   = new Date(t);
            var str = _months[d.getUTCMonth()] + ' '
                    + String(d.getUTCDate()).padStart(2, '0') + ', '
                    + d.getUTCFullYear();
            var all = (window._dateNavState || {{}}).features || [];
            _filterObs(all.filter(function(f) {{ return f.properties.date === str; }}));
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
        <span style="color:#2ca02c">&#9679;</span> &lt; 50 %<br>
        <span style="color:#e6c200">&#9679;</span> 50 – 65 %<br>
        <span style="color:#ff7f0e">&#9679;</span> 65 – 80 %<br>
        <span style="color:#d62728">&#9679;</span> 80 – 95 %<br>
        <span style="color:#7b0000">&#9679;</span> &ge; 95 %
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
        from folium.plugins import TimestampedGeoJson

        # Build a coord lookup: station_id → (lon, lat)
        station_coords = {s['id']: (s['lon'], s['lat']) for s in sim_stations}

        ts_features: list[dict] = []
        for sid, series in prob_by_station.items():
            coords = station_coords.get(sid)
            if coords is None:
                continue
            clean = series.dropna()
            for ts, p in clean.items():
                fill, stroke = _prob_to_colors(float(p))
                ts_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": list(coords)},
                    "properties": {
                        "times":     [pd.Timestamp(ts).strftime('%Y-%m-%d')],
                        "icon":      "circle",
                        "iconstyle": {
                            "fillColor": fill, "color": stroke,
                            "fillOpacity": 0.85, "weight": 1.5, "radius": 8,
                        },
                        "popup": (f"<b>{sid.replace('_res','')}</b><br>"
                                  f"{pd.Timestamp(ts).strftime('%b %d, %Y')}<br>"
                                  f"Prob: {float(p)*100:.0f}%"),
                    },
                })

        td_layer_js_name: str | None = None
        if ts_features:
            # Circles are invisible (weight=0, fillOpacity=0) — they exist only to
            # populate the time dimension so the time slider works.  _simLayer
            # (in _sim_layer_html) is the sole visual layer and recolours itself via
            # a setInterval that polls timeDimension.getCurrentTime().
            for f in ts_features:
                f["properties"]["iconstyle"] = {
                    "fillColor": "transparent", "color": "transparent",
                    "fillOpacity": 0, "weight": 0, "radius": 1,
                }
            TimestampedGeoJson(
                {"type": "FeatureCollection", "features": ts_features},
                period="P1D", duration="P1D", transition_time=150,
                auto_play=False, loop=False, max_speed=15,
                date_options="YYYY-MM-DD", time_slider_drag_update=True,
                add_last_point=False,
            ).add_to(m)

    # Serialise prob_by_station time series for _sim_layer_html's time-player polling.
    import json as _json
    prob_series_dict: dict[str, dict[str, float]] = {}
    if prob_by_station:
        for sid, series in prob_by_station.items():
            clean = series.dropna()
            prob_series_dict[sid] = {
                pd.Timestamp(ts).strftime('%Y-%m-%d'): round(float(p), 4)
                for ts, p in clean.items()
            }

    layer_ctrl = folium.LayerControl()
    layer_ctrl.add_to(m)

    # ── Simulation stations overlay (aspect-filterable) ───────────────────────
    if sim_stations:
        m.get_root().html.add_child(
            folium.Element(_sim_layer_html(
                m.get_name(), layer_ctrl.get_name(), sim_stations,
                _json.dumps(prob_series_dict),
            ))
        )

    m.save(str(output_path))
