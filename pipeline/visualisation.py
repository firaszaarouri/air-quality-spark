"""
pipeline/visualisation.py
─────────────────────────────────────────────────────────────────────────────
Interactive map generation using Folium.

Maps produced:
  1. Choropleth — average AQI by country (today)
  2. Circle markers — top 50 clusters by AQI improvement
  3. Cluster zones — shows sensor distribution coloured by cluster
  4. Streak map — popups with per-cluster streak histograms (Vega-Lite)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import pandas as pd
import altair as alt
import folium
from folium import Choropleth, CircleMarker, Marker
from folium.plugins import MarkerCluster
from branca.colormap import LinearColormap

import config


# ── Colour helpers ────────────────────────────────────────────────────────────

def aqi_to_colour(aqi: float) -> str:
    """Map AQI value (1–10) to a hex colour (green → red)."""
    colours = ['#00e400', '#92d050', '#ffff00', '#ff7e00',
               '#ff0000', '#8f3f97', '#7e0023', '#7e0023', '#7e0023', '#7e0023']
    idx = max(0, min(9, int(aqi) - 1))
    return colours[idx]


def improvement_to_colour(val: float, vmin: float, vmax: float) -> str:
    """Map an improvement value to a blue (bad) → green (good) colour."""
    if vmax == vmin:
        return '#808080'
    norm = (val - vmin) / (vmax - vmin)
    norm = max(0, min(1, norm))
    r = int(255 * (1 - norm))
    g = int(200 * norm + 55)
    return f'#{r:02x}{g:02x}55'


# ══════════════════════════════════════════════════════════════════════════════
#  MAP 1 — Choropleth: worst AQI by country today
# ══════════════════════════════════════════════════════════════════════════════

def map_country_aqi(worst_aqi_pdf: pd.DataFrame,
                     output_dir: str = 'output/') -> str:
    """
    Generate a choropleth map showing average AQI by country for today.

    Uses the world-countries GeoJSON from the Folium examples repository.

    Args:
        worst_aqi_pdf: pandas DataFrame with columns [country (alpha3), avg_AQI]
        output_dir:    directory to save the HTML file

    Returns:
        path to the saved HTML file
    """
    geojson_url = (
        "https://raw.githubusercontent.com/python-visualization/folium"
        "/main/examples/data/world-countries.json"
    )

    m = folium.Map(location=[20, 0], zoom_start=2.25, tiles='CartoDB positron')

    Choropleth(
        geo_data=geojson_url,
        name="Average AQI",
        data=worst_aqi_pdf,
        columns=['country', 'avg_AQI'],
        key_on='feature.id',
        fill_color='RdYlGn_r',    # reversed: green=low AQI (good), red=high (bad)
        fill_opacity=0.75,
        line_opacity=0.1,
        legend_name='Average Air Quality Index (1=best, 10=worst)',
        nan_fill_color='#ededed',
        nan_fill_opacity=0.4
    ).add_to(m)

    folium.LayerControl().add_to(m)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'map_country_aqi.html')
    m.save(path)
    print(f"[Visualisation] Map 1 saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  MAP 2 — Top 50 cluster improvement circles
# ══════════════════════════════════════════════════════════════════════════════

def map_cluster_improvement(top50_pdf: pd.DataFrame,
                              output_dir: str = 'output/') -> str:
    """
    Map showing the top 50 clusters with circle markers coloured by improvement.

    Larger circles = greater improvement. Green = improved, red = worsened.

    Args:
        top50_pdf: pandas DataFrame from compute_aqi_improvement_by_cluster()
                   must have columns: cluster, AQI_improvement, centroid_lat, centroid_lon
    """
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')

    vmin = top50_pdf['AQI_improvement'].min()
    vmax = top50_pdf['AQI_improvement'].max()

    colormap = LinearColormap(
        colors=['#e74c3c', '#f39c12', '#2ecc71'],
        vmin=vmin, vmax=vmax
    )
    colormap.add_to(m)
    colormap.caption = 'AQI Improvement (positive = better air)'

    for _, row in top50_pdf.iterrows():
        if pd.isna(row['centroid_lat']) or pd.isna(row['centroid_lon']):
            continue

        val       = row['AQI_improvement']
        radius    = max(5, min(20, abs(val) * 8))
        colour    = colormap(val)
        tooltip   = (f"Cluster {int(row['cluster'])}<br>"
                     f"AQI improvement: {val:+.3f}")

        CircleMarker(
            location=[row['centroid_lat'], row['centroid_lon']],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.8,
            tooltip=tooltip
        ).add_to(m)

    path = os.path.join(output_dir, 'map_cluster_improvement.html')
    m.save(path)
    print(f"[Visualisation] Map 2 saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  MAP 3 — Cluster zone distribution (sensors coloured by cluster ID)
# ══════════════════════════════════════════════════════════════════════════════

def map_cluster_zones(sensor_pdf: pd.DataFrame, model_centers: list,
                       output_dir: str = 'output/') -> str:
    """
    Visualise the geographic distribution of clusters.

    Each sensor dot is coloured by its assigned cluster.
    Cluster centroids are shown as blue markers with tooltips.

    Args:
        sensor_pdf:    pandas DataFrame with lat, lon, cluster columns
        model_centers: list of (lat, lon) arrays from KMeansModel.clusterCenters()
    """
    m = folium.Map(location=[30, 10], zoom_start=2, tiles='CartoDB positron')

    # Colour palette cycling through many colours
    np.random.seed(42)
    palette = [f'#{np.random.randint(0,0xFFFFFF):06x}' for _ in range(config.K_CLUSTERS)]

    # Sensor dots (sample for performance)
    sample = sensor_pdf.dropna(subset=['latitude', 'longitude', 'cluster'])
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)

    for _, row in sample.iterrows():
        color = palette[int(row['cluster']) % len(palette)]
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=3000,
            color=color,
            fill=True,
            fill_opacity=0.6,
            weight=0
        ).add_to(m)

    # Centroid markers
    for i, center in enumerate(model_centers):
        Marker(
            location=[center[0], center[1]],
            tooltip=f'Cluster {i} centroid',
            icon=folium.Icon(color='blue', icon='info-sign', prefix='glyphicon')
        ).add_to(m)

    path = os.path.join(output_dir, 'map_cluster_zones.html')
    m.save(path)
    print(f"[Visualisation] Map 3 saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  MAP 4 — Streak map with Vega-Lite popup histograms
# ══════════════════════════════════════════════════════════════════════════════

def map_streak_popups(streak_count_pdf: pd.DataFrame,
                       model_centers: list,
                       output_dir: str = 'output/') -> str:
    """
    Interactive map where clicking a cluster marker shows a histogram
    of its good air quality streak distribution.

    Uses Altair (Vega-Lite) charts embedded in Folium popups.

    Args:
        streak_count_pdf: pandas DataFrame with columns:
                          cluster, streak, count(streak)
        model_centers:    list of (lat, lon) from KMeansModel.clusterCenters()
    """
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')

    max_streak_overall = int(streak_count_pdf['streak'].max()) if not streak_count_pdf.empty else 22

    for cluster_id, center in enumerate(model_centers):
        cluster_data = streak_count_pdf[streak_count_pdf['cluster'] == cluster_id].copy()

        if cluster_data.empty:
            # No good air days — simple grey marker
            Marker(
                location=[center[0], center[1]],
                tooltip=f'Cluster {cluster_id}: no good air days recorded',
                icon=folium.Icon(color='gray', icon='remove', prefix='glyphicon')
            ).add_to(m)
            continue

        max_streak = int(cluster_data['streak'].max())

        # Build Altair chart
        chart_data = cluster_data[['streak', 'count(streak)']].rename(
            columns={'count(streak)': 'count'}
        )
        chart = (alt.Chart(chart_data)
                 .mark_bar(color='#3498db')
                 .encode(
                     x=alt.X('streak:Q', title='Streak length (days)',
                              scale=alt.Scale(domain=[0, max_streak_overall])),
                     y=alt.Y('count:Q', title='Frequency')
                 )
                 .properties(
                     width=300, height=200,
                     title=f'Cluster {cluster_id} — Good Air Streaks'
                 ))

        popup = folium.Popup(max_width=400).add_child(
            folium.VegaLite(chart.to_json(), width=350, height=230)
        )

        # Colour by max streak: longer = greener
        norm = min(1.0, max_streak / 14)
        r = int(255 * (1 - norm))
        g = int(180 * norm + 75)
        icon_color = 'green' if max_streak >= 7 else ('orange' if max_streak >= 3 else 'red')

        Marker(
            location=[center[0], center[1]],
            tooltip=f'Cluster {cluster_id} — max streak: {max_streak} days',
            popup=popup,
            icon=folium.Icon(color=icon_color, icon='leaf', prefix='glyphicon')
        ).add_to(m)

    path = os.path.join(output_dir, 'map_streak_popups.html')
    m.save(path)
    print(f"[Visualisation] Map 4 saved: {path}")
    return path
