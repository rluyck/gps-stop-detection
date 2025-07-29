import os
import folium
from matplotlib import cm, colors
import geopandas as gpd


def generate_map(gdf: gpd.GeoDataFrame, static_dir: str) -> None:
    # Assign unique color per (device_id, trace_number)
    pairs = list(gdf[['device_id', 'trace_number']].drop_duplicates().itertuples(index=False, name=None))
    cmap = cm.get_cmap('tab20', len(pairs))
    color_dict = {pair: colors.to_hex(cmap(i)) for i, pair in enumerate(pairs)}

    gdf['pair'] = list(zip(gdf['device_id'], gdf['trace_number']))
    gdf['base_color'] = gdf['pair'].map(color_dict)

    # Override with red for stopped points
    gdf['color'] = gdf.apply(
        lambda row: '#d62828' if row.get('stopped', False) else row['base_color'], axis=1
    )

    # Ensure stopped points (red) are plotted last
    gdf = gdf.sort_values(by='stopped')

    # Generate map
    center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
    m = folium.Map(location=center, zoom_start=12)

    # First: non-stopped points
    for _, row in gdf[~gdf['stopped']].iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color=row['base_color'],
            fill=True,
            fill_color=row['base_color'],
            fill_opacity=0.9,
            opacity=0.9,
            tooltip=f"Device: {row.device_id}, Trace: {row.trace_number}, TS: {row.timestamp}"
        ).add_to(m)

    # Then: stopped points in red
    for _, row in gdf[gdf['stopped']].iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=4,
            color="#d62828",
            fill=True,
            fill_color="#d62828",
            fill_opacity=1.0,
            opacity=1.0,
            tooltip=f"STOP | Device: {row.device_id}, Trace: {row.trace_number}, TS: {row.timestamp}"
        ).add_to(m)

    # Save map
    map_path = os.path.join(static_dir, "map.html")
    m.save(map_path)
