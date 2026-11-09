#!/usr/bin/env python3
"""
Folium map of the WMP (FOCUS) points, SNOWPACK stations and the Pamir Highway,
with an editor for choosing which stations and aspects the model runs on.

Reads the manifest written by sync_pro_files.py --focus
(data/simulations/<SEASON>/focus_stations.csv) and station coordinates from
the .pro headers of the SNOWPACK run.  Layers:

  WMP points        circles coloured by their matched station, dashed line to it
  WMP stations      squares, one colour per station (the baseline model set)
  Pamir Highway     M41 line from data/geo/pamir_highway.geojson, when present
  Highway stations  red triangles: nearest station along the road, within one
                    grid spacing (8 km); also written to
                    data/geo/pamir_highway_stations.csv
  All stations      every station in the run (off by default)
  Edits             the Add / Remove marks from the editor

Click any marker or anywhere on the map to mark the nearest station Add or
Remove for chosen aspects (scripts/focus_map_editor.js).  The Save button posts
to save_model_locations.php (copied next to the map), which writes
<out>/model_locations/model_locations.json and backs up the previous version
to model_locations/backups/model_locations_<UTC timestamp>.json.  That file is
the input to sync_pro_files.py --selection.  One-time setup of the folder
(PHP runs as apache):

    mkdir <out>/model_locations
    setfacl -m u:apache:rwx,d:u:apache:rwx,d:u:ops:rwx,d:m::rwx <out>/model_locations
  Place search (OpenStreetMap Nominatim), a
cursor lat/lon readout and a "Go to lat, lon" box help find locations.

Base tiles: OpenStreetMap and Esri World Imagery (satellite).

Usage:
    uv run python scripts/focus_map.py [--season 2026-2027] [--out DIR] [--source RUN_DIR]
"""

import argparse
import colorsys
import json
import shutil
from pathlib import Path

import folium
from folium.plugins import Geocoder, MousePosition
import pandas as pd

from sync_pro_files import HWY_MAX_KM, PAMIR_HWY, highway_stations, read_run_stations

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path("/mnt/HC_Volume_106760790/model/snowpack/runs/output.caw.2026")
ESRI_SAT = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
EDITOR_JS = Path(__file__).with_name("focus_map_editor.js")
SAVE_PHP  = Path(__file__).with_name("save_model_locations.php")


def station_colours(stations: list[str]) -> dict[str, str]:
    """Distinct hex colour per station: golden-angle hue steps, alternating lightness."""
    colours = {}
    for i, sid in enumerate(stations):
        hue   = (i * 0.618033988749895) % 1.0
        light = (0.45, 0.60, 0.35)[i % 3]
        r, g, b = colorsys.hls_to_rgb(hue, light, 0.85)
        colours[sid] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    return colours


def build_map(df: pd.DataFrame, run_stations: pd.DataFrame, season: str, run_dir: Path,
              hwy_stations: pd.DataFrame | None = None) -> folium.Map:
    """
    Map shell in folium (tiles, highway line, search, coordinates, empty
    layer groups); markers and the model-location editor are added in the
    browser by focus_map_editor.js from the data embedded here.
    """
    stations = sorted(df["station"].astype(str).unique(), key=int)
    colours  = station_colours(stations)
    df       = df.assign(station=df["station"].astype(str))

    m = folium.Map(location=[df["point_lat"].mean(), df["point_lon"].mean()],
                   zoom_start=8, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Base map").add_to(m)
    folium.TileLayer(ESRI_SAT, attr="Esri, Maxar, Earthstar Geographics",
                     name="Satellite").add_to(m)
    Geocoder(position="topright", collapsed=True, add_marker=True).add_to(m)
    MousePosition(position="bottomright", separator=", ", prefix="Lat, Lon:",
                  num_digits=5, empty_string="").add_to(m)

    if PAMIR_HWY.exists():
        hwy_layer = folium.FeatureGroup(name="Pamir Highway (M41)", show=True)
        folium.GeoJson(
            json.loads(PAMIR_HWY.read_text()),
            style_function=lambda _: {"color": "#d7191c", "weight": 4, "opacity": 0.85},
            tooltip="Pamir Highway (M41)",
        ).add_to(hwy_layer)
        hwy_layer.add_to(m)

    groups = {
        "points": folium.FeatureGroup(name="WMP points", show=True),
        "focus":  folium.FeatureGroup(name="WMP stations", show=True),
        "hwy":    folium.FeatureGroup(name="Pamir Highway stations", show=True),
        "all":    folium.FeatureGroup(name="All SNOWPACK stations", show=False),
        "edits":  folium.FeatureGroup(name="Model-location edits", show=True),
    }
    for g in groups.values():
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    data = {
        "map":      m.get_name(),
        "layers":   {k: g.get_name() for k, g in groups.items()},
        "season":   season,
        "run_dir":  str(run_dir),
        "save_url": SAVE_PHP.name,
        # niViz view of a station's simulation (snowpack-mapping's profiles.html)
        "viewer_url": f"/nwp/caw/conditions/profiles.html?year={season[:4]}&id=",
        "stations": [[str(r.station), round(r.lat, 5), round(r.lon, 5), round(r.elevation)]
                     for r in run_stations.itertuples()],
        # Baseline = what sync_pro_files.py --focus ... --highway copies.
        "baseline": {**({str(r.station): ["Pamir Highway"] for r in hwy_stations.itertuples()}
                        if hwy_stations is not None else {}),
                     **{sid: list(g["point"]) for sid, g in df.groupby("station")}},
        "colours":  colours,
        "points":   [[r.point, r.point_lat, r.point_lon, r.station, float(r.distance_km), colours[r.station]]
                     for r in df.itertuples()],
        "hwy":      ([] if hwy_stations is None else
                     [[str(r.station), float(r.distance_km)] for r in hwy_stations.itertuples()]),
    }
    # </ in JSON would close the <script> early; KML names are free text.
    payload = json.dumps(data).replace("</", "<\\/")
    m.get_root().script.add_child(folium.Element(
        f"window.FOCUS_EDITOR_DATA = {payload};\n" + EDITOR_JS.read_text()))

    m.fit_bounds([[df[["point_lat", "station_lat"]].min().min(), df[["point_lon", "station_lon"]].min().min()],
                  [df[["point_lat", "station_lat"]].max().max(), df[["point_lon", "station_lon"]].max().max()]])
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", default="2026-2027")
    parser.add_argument("--out", type=Path, default=Path("/home/ops/www/nwp/caw/test"))
    parser.add_argument("--source", type=Path, default=RUN_DIR,
                        help="SNOWPACK run directory (station coordinates for all station layers).")
    args = parser.parse_args()

    manifest = ROOT / "data" / "simulations" / args.season / "focus_stations.csv"
    df = pd.read_csv(manifest)
    if not args.source.is_dir():
        raise SystemExit(f"SNOWPACK run directory not found: {args.source}")
    run_stations = read_run_stations(args.source)
    hwy = None
    if PAMIR_HWY.exists():
        hwy = highway_stations(run_stations)
        hwy.to_csv(ROOT / "data" / "geo" / "pamir_highway_stations.csv", index=False)
        print(f"{len(hwy)} SNOWPACK stations within {HWY_MAX_KM:.0f} km of the Pamir Highway")
    m  = build_map(df, run_stations, args.season, args.source, hwy)
    m.get_root().html.add_child(folium.Element(
        f"<title>FOCUS points & SNOWPACK stations {args.season}</title>"))

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "focus_stations_map.html"
    m.save(str(out))
    shutil.copy2(SAVE_PHP, args.out / SAVE_PHP.name)
    if not (args.out / "model_locations").is_dir():
        print(f"NOTE: create {args.out / 'model_locations'} writable by the web server "
              f"(see the docstring) or the map's Save button will fail")
    print(f"{len(df)} points, {df['station'].nunique()} stations → {out}")


if __name__ == "__main__":
    main()
