import pandas as pd
import geopandas as gpd
from shapely import wkt

REQUIRED_COLUMNS = ['geom_wkt', 'trace_number', 'device_id', 'ts']

def read_and_preprocess(csv_bytes: bytes) -> gpd.GeoDataFrame:
    """
    Reads and preprocesses a CSV file containing GPS trace data, and returns a cleaned and structured GeoDataFrame.

    Parameters:
    ----------
    csv_bytes : bytes
        The content of a CSV file in bytes format. The CSV must contain the following columns:
        - 'geom_wkt': GPS points in WKT (Well-Known Text) format
        - 'trace_number': identifier for the route or trace
        - 'device_id': identifier for the device that recorded the data
        - 'ts': timestamp of the recorded point

    Returns:
    -------
    preprocessed_gdf : gpd.GeoDataFrame
        A GeoDataFrame with:
        - geometries converted from WKT to Shapely Point objects
        - timestamps parsed as datetime objects
        - sorted by 'device_id', 'trace_number', and 'timestamp'
        - coordinate reference system set to WGS84 (EPSG:4326)
    """
    raw_df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))

    if not all(col in raw_df.columns for col in REQUIRED_COLUMNS):
        raise ValueError(f"CSV must contain columns: {REQUIRED_COLUMNS}")

    preprocessed_gdf = (
        gpd.GeoDataFrame(
            raw_df.assign(
                geometry=lambda df: df["geom_wkt"].apply(wkt.loads),
                timestamp=lambda df: pd.to_datetime(df["ts"])
            ),
            geometry="geometry",
            crs="EPSG:4326"
        )
        .sort_values(by=["device_id", "trace_number", "timestamp"])
        .drop(columns=["geom_wkt", "ts"], axis=1)
        .reset_index(drop=True)
    )

    return preprocessed_gdf
