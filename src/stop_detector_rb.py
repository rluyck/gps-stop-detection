import pandas as pd

def rule_based_detection(preprocessed_gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a simplified rule-based stop detection method on enriched GPS trace data.

    Rules applied:
    - A point is considered 'stopped' if speed_kmh < 1
    - Only stops with a duration >= 5 seconds are retained
    - All moving points are kept regardless of duration

    Parameters
    ----------
    preprocessed_gdf : pd.DataFrame
        A DataFrame enriched by `add_features`, with 'speed_kmh' and 'time_diff_s'.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with short stops (<5s) removed, keeping all moving points.
    """
    df = preprocessed_gdf.copy()

    # Mark points where the vehicle is stopped
    df['stopped'] = df['speed_kmh'] < 1

    # Keep all moving points and stopped points with sufficient duration
    filtered_df = df[
        (~df['stopped']) |                # moving points
        ((df['stopped']) & (df['time_diff_s'] >= 5))  # valid stop points
    ]

    return filtered_df
