# Notes while working on data wizards case

## Context
- Client TravelTrack Inc. provided GPS traces
    - Each trace contains timestamped GPS coordinates: lat and lon (e.g., 39.898573, 116.391305)
        - Coordinates are in CRS:4326 (Coordinate Reference System 4326)
            - EPSG:4326 is the official code from the European Petroleum Survey Group for this CRS
                - CRS 4326 corresponds to WGS 84 (World Geodetic System 1984), the standard for GPS coordinates

## Objective
- Gain insights into vehicle movement patterns, particulary identify stops
    - What kind of speeds are we dealing with?
        - From this we can derive what mode of transportation it is and create a definition for "stop"

## Dataset
- the GPS location is represented as text using the Well-Known Text (WKT) standard.
    - This format is widely used to encode geometric shapes (points, lines, polygons) in geospatial data.

- Where on the planet are we? 
    - China

Sample data:
'''
geom_wkt	trace_number	device_id	ts
POINT (116.391305 39.898573)	1	19	2008-12-11 04:42:14+00
POINT (116.391317 39.898617)	1	19	2008-12-11 04:42:16+00
POINT (116.390928 39.898613)	1	19	2008-12-11 04:43:26+00
POINT (116.390833 39.898635)	1	19	2008-12-11 04:43:32+00
POINT (116.38941  39.898723)	1	19	2008-12-11 04:43:47+00
'''

## My Task
Your mission is to develop algorithms to extract meaningful insights from the GPS traces. Specifically, we'd like you to:
- note: algorithms (plural)

1. Identify vehicle stops (locations where vehicles remain stationary for a certain period)
    - rule based: stop when speed = 0 m/s or speed < 1m/ for more than 5 sec
    - 
2. Create visualizations to communicate your findings
3. Nice to have: detect patterns or behaviors in the data
    - frequent routes
    - frequent short-cuts
    - traffick avoidance
    - correlation with hour of the day
    - correlation with day of the week
    - speed
    - speedlimit

Feel free to take some inspiration from existing solutions, but make sure to not just use a library for this task. This is the opportunity to show us all your creativity building advanced algorithms, your feature engineering skills and the mathematical knowledge you hold!

### Deliverables
We expect the following for this take-home assignment:

- A Python solution that processes the GPS data and creates a machine learning model to identify stops
    - labeling
- An API endpoint (using Flask or FastAPI) that allows processing of a single (or multiple) raw GPS trace
- A Dockerfile to containerize your solution
- A README file explaining your approach, assumptions, and instructions to run the solution


When you've completed the assignment, please share your private GitHub repository with us or send us a zipfile with your solution. We'll review it before the second interview, where you'll have the opportunity to present your approach, findings, and insights.

Should you have any questions, please don't hesitate to reach out.

Good luck!