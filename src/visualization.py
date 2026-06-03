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

def create_avalanche_map(
    df: pd.DataFrame,
    output_path: Path,
    target_url: str = "stability_analysis.html",
    forecast_date: pd.Timestamp | None = None,
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

    Args:
        df:            Must contain Latitude, Longitude, Placemark Name columns,
                       and optionally Aspect, date, Elevation (M), Size, Remarks,
                       target_url, forecast_prob.
        output_path:   Destination HTML file.
        target_url:    Fallback URL when df has no target_url column.
        forecast_date: Reference date shown in the map legend.
    """
    if 'target_url' not in df.columns:
        df = df.copy()
        df['target_url'] = target_url

    m = folium.Map(
        location=[df['Latitude'].mean(), df['Longitude'].mean()],
        zoom_start=9,
    )

    folium.TileLayer('OpenStreetMap', name='Standard').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite',
    ).add_to(m)
    folium.TileLayer(
        tiles='https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap',
        name='Topographic',
    ).add_to(m)

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['Longitude'], row['Latitude']],
            },
            "properties": {
                "name":         row['Placemark Name'],
                "aspect":       row.get('Aspect', ''),
                "date":         row.get('date', ''),
                "elevation":    row.get('Elevation (M)', ''),
                "size":         row.get('Size', ''),
                "remarks":      row.get('Remarks', ''),
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
            return L.circleMarker(latlng, {
                radius: 10,
                fillColor: fill,
                color: stroke,
                weight: 1.5,
                opacity: 1,
                fillOpacity: 0.88
            });
        }
    """)

    on_each_feature = folium.JsCode("""
        function(feature, layer) {
            function wrapText(text, maxLen) {
                var words = text.split(' ');
                var lines = [], current = '';
                for (var i = 0; i < words.length; i++) {
                    var word = words[i];
                    var candidate = current ? current + ' ' + word : word;
                    if (candidate.length <= maxLen) {
                        current = candidate;
                    } else {
                        if (current) lines.push(current);
                        current = word;
                    }
                }
                if (current) lines.push(current);
                return lines.join('<br>');
            }

            var p = feature.properties;
            var tip = '<b>' + p.name + '</b>';
            if (p.date)      tip += '<br>Date: '      + p.date;
            if (p.aspect)    tip += '<br>Aspect: '    + p.aspect;
            if (p.elevation) tip += '<br>Elevation: ' + p.elevation + ' m';
            if (p.size)      tip += '<br>Size: '      + p.size;
            if (p.forecast_prob !== null && p.forecast_prob !== undefined) {
                tip += '<br><b>Forecast prob: ' + Math.round(p.forecast_prob * 100) + '%</b>';
            } else {
                tip += '<br>Forecast prob: <i>n/a</i>';
            }
            if (p.remarks)   tip += '<br><i>'         + wrapText(p.remarks, 50) + '</i>';
            layer.bindTooltip(tip, {sticky: true});
            layer.on('click', function() {
                if (p.url) { window.open(p.url, '_blank'); }
            });
        }
    """)

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name='Avalanche Observations',
        point_to_layer=point_to_layer,
        on_each_feature=on_each_feature,
    ).add_to(m)

    # ── Map title ─────────────────────────────────────────────────────────────
    title_date = forecast_date.strftime('%B %d, %Y') if forecast_date is not None else ""
    title_html = f"""
    <div style="
        position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
        z-index: 1000; background: rgba(255,255,255,0.93);
        padding: 7px 22px; border-radius: 7px; border: 1px solid #bbb;
        font-family: sans-serif; font-size: 15px; font-weight: bold;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.18); white-space: nowrap;
    ">
        Avalanche Observations &nbsp;|&nbsp; Forecast: {title_date}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

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
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    m.save(str(output_path))
