python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib seaborn ipykernel


EDA
dataset shape (5908, 4)
no whitespaces in column names
dtypes
- convert ts to timestamp dtype
- geom_wkt is an object (plain string)
    - convert to geometry object for plotting (to do)
        - need packages geopandas and shapely.wkt
            - pip install geopandas shapely
convert WKT strings to Shapely geometries
read up on what Shapely geometries are
    - python object that represents a geometric shape
        - Point, LineString, Polygon, MultiPoint, ec..
    - you can do nice stuff with these objects
        - calculate distances, check touch, area of a polygon, ..
    - building blocks for GeoPandas, Folium
create a geodataframe
    - like a regular df, but it contains the geometry column with dtype geometry
set crs to 4326
gdf.plot
    - lat and lon visible, but no background
    - downloaded ne_110m_admin_0_countries.shp
    - load background world map (shp file)
    - plot again
        - seems to be in China
        - narrow down by changing axis limits
            - no streets visible
                - pip install folium                
plotting with folium
    - a mess with just one color
        - generate color map based on device_id
apparently geopandas has an explore method
    - pip install mapclassify    
enough mapping, let's see what kind of speeds we are dealing with (car, bike, foot, ..)
    - pip install geopy
        - to calculate distances between coordinates









pip freeze > requirements.txt
to do: download other vehicle dataset to test model