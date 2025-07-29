import numpy as np
import pandas as pd
from geopy.distance import geodesic

def add_features(preprocessed_gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches a preprocessed GPS DataFrame with engineered features:
    - Latitude and longitude
    - Distance between consecutive points (in meters)
    - Time difference between consecutive points (in seconds)
    - Speed in km/h

    Parameters
    ----------
    preprocessed_gdf : pd.DataFrame

    Returns
    -------
    enriched_gdf : pd.DataFrame
    """
    enriched_gdf = preprocessed_gdf.copy()

    def compute_features_for_trace(trace_df):
        device_id, trace_number = trace_df.name
        trace_df = trace_df.copy()
        trace_df['lat'] = trace_df.geometry.y
        trace_df['lon'] = trace_df.geometry.x

        distances = [0]
        time_diffs = [0]

        for i in range(1, len(trace_df)):
            coord_prev = (trace_df.iloc[i - 1]['lat'], trace_df.iloc[i - 1]['lon'])
            coord_curr = (trace_df.iloc[i]['lat'], trace_df.iloc[i]['lon'])
            distances.append(geodesic(coord_prev, coord_curr).meters)
            time_diffs.append((trace_df.iloc[i]['timestamp'] - trace_df.iloc[i - 1]['timestamp']).total_seconds())

        trace_df['distance_m'] = distances
        trace_df['time_diff_s'] = time_diffs
        trace_df['speed_kmh'] = np.where(
            trace_df['time_diff_s'] > 0,
            trace_df['distance_m'] / trace_df['time_diff_s'] * 3.6,
            0
        )
        return trace_df

    enriched_gdf = (
        enriched_gdf
        .groupby(['device_id', 'trace_number'], group_keys=False)
        .apply(compute_features_for_trace)
        .reset_index(drop=True)
    )

    return enriched_gdf
