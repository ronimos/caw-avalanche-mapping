"""
visualization.py — Snowpack stability analysis and interactive output.

Provides:
  - plot_interactive_stability  Two-panel Plotly HTML stability chart.
  - create_avalanche_map        Folium map of avalanche observations.
"""

from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_LAYER_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#17becf',
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
) -> None:
    """
    Generates a two-panel interactive Plotly HTML stability chart.

    Upper panel: snow surface height + dominant weak layers coloured by Sn38.
    Lower panel: minimum Sn38 across all layers at each timestep.
    Vertical dashed red lines are drawn at each date in event_dates.

    Args:
        df:           Output of parse_snow_data().
        output_path:  Destination HTML file.
        station_id:   Label shown in the chart title.
        event_dates:  Avalanche occurrence dates to mark on both panels.
    """
    dominant = _get_dominant_layers(df)
    hs_ts    = df.groupby(df.index)['total_height'].first()
    min_sn38 = df.groupby(df.index)['sn38'].min()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Dominant Weak Layers by Height (layers that were ever minimum Sn38)",
            "Minimum Sn38 Across All Layers",
        ),
        row_heights=[0.65, 0.35],
    )

    # Snow surface
    fig.add_trace(go.Scatter(
        x=hs_ts.index,
        y=hs_ts.values,
        name="Snow Surface (HS)",
        line=dict(color='rgba(150,150,150,0.5)', width=2),
        hovertemplate="HS: %{y:.2f} cm<extra></extra>",
    ), row=1, col=1)

    # One scatter trace per dominant layer
    for i, label in enumerate(sorted(dominant['layer_label'].unique())):
        sub    = dominant[dominant['layer_label'] == label]
        colour = _LAYER_PALETTE[i % len(_LAYER_PALETTE)]
        fig.add_trace(go.Scatter(
            x=sub.index,
            y=sub['layer_z'],
            mode='markers',
            name=label,
            marker=dict(
                size=8,
                color=sub['sn38'],
                colorscale='RdYlGn',
                cmin=1.0, cmax=6.0,
                showscale=(i == 0),
                colorbar=dict(title="Sn38", thickness=14, len=0.55, y=0.72),
                line=dict(width=1, color=colour),
            ),
            customdata=sub[['burial_depth', 'sn38']].values,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Height: %{y:.2f} cm<br>"
                "Burial: %{customdata[0]:.2f} cm<br>"
                "Sn38: %{customdata[1]:.2f}<extra></extra>"
            ),
        ), row=1, col=1)

    # Min Sn38 timeseries
    fig.add_trace(go.Scatter(
        x=min_sn38.index,
        y=min_sn38.values,
        name="Min Sn38",
        mode='lines+markers',
        marker=dict(size=5, color=min_sn38.values, colorscale='RdYlGn', cmin=1.0, cmax=6.0),
        line=dict(width=1.5, color='rgba(100,100,100,0.6)'),
        hovertemplate="Min Sn38: %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    # Avalanche date markers
    for edate in (event_dates or []):
        for panel_row in (1, 2):
            fig.add_shape(
                type='line',
                x0=edate, x1=edate,
                y0=0, y1=1,
                yref='y domain',
                line=dict(color='red', width=1.5, dash='dash'),
                row=panel_row, col=1,
            )
        fig.add_annotation(
            x=edate,
            y=1,
            yref='y domain',
            text=edate.strftime('%b %d'),
            showarrow=False,
            xanchor='left',
            font=dict(color='red', size=11),
            row=1, col=1,
        )

    fig.update_yaxes(title_text="Height from ground (cm)", row=1, col=1)
    fig.update_yaxes(title_text="Sn38",                    row=2, col=1)

    fig.update_layout(
        height=900,
        title_text=f"Stability Analysis — Station {station_id}",
        template="plotly_white",
        hovermode="x unified",
        xaxis_range=[df.index.min(), df.index.max()],
        xaxis2_range=[df.index.min(), df.index.max()],
        xaxis2_rangeslider_visible=True,
        xaxis2_rangeslider_thickness=0.05,
        legend=dict(orientation='h', y=-0.12),
    )
    fig.write_html(str(output_path))


# ── map ───────────────────────────────────────────────────────────────────────

def create_avalanche_map(
    df: pd.DataFrame,
    output_path: Path,
    target_url: str = "stability_analysis.html",
) -> None:
    """
    Creates a Folium map with red circle markers for each avalanche observation.
    Hovering shows name, date, aspect, elevation, size, and remarks.
    Clicking opens the linked stability HTML in a new tab.

    Args:
        df:           Must contain Latitude, Longitude, Placemark Name columns,
                      and optionally Aspect, date, Elevation (M), Size, Remarks,
                      target_url. Rows without a target_url column use the
                      target_url argument as a fallback.
        output_path:  Destination HTML file.
        target_url:   Fallback URL for all markers when df has no target_url column.
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
                "name":      row['Placemark Name'],
                "aspect":    row.get('Aspect', ''),
                "date":      row.get('date', ''),
                "elevation": row.get('Elevation (M)', ''),
                "size":      row.get('Size', ''),
                "remarks":   row.get('Remarks', ''),
                "url":       row['target_url'],
            },
        }
        for _, row in df.iterrows()
    ]

    point_to_layer = folium.JsCode("""
        function(feature, latlng) {
            return L.circleMarker(latlng, {
                radius: 10,
                fillColor: 'red',
                color: 'darkred',
                weight: 1.5,
                opacity: 1,
                fillOpacity: 0.85
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

    folium.LayerControl().add_to(m)
    m.save(str(output_path))
