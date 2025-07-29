# GPS Stop Detection App

This project provides a FastAPI web application for detecting stops in GPS trajectory data using machine learning. 

<img src="api/static/stops_example.png" alt="Example of GPS stops detection" width="600" style="display: block; margin: 0 left;"/>

## Features

- **File Upload**: Upload single or multiple GPS trajectory files (`.csv`) through a web interface
   - If you want to test the app and you don't have any traces, use the file in data/raw/gps_traces.csv.
- **Stop Detection**: Detect stops using a trained machine learning model (RandomForestClassifier)
- **Interactive Maps**: Visualize detected stops and paths on interactive Folium maps
- **Model Explainability**: Understand model predictions with SHAP feature importance plots
- **Batch Processing**: Process multiple traces simultaneously

## Data Format

The uploaded `.csv` traces must contain the following columns:

| Column         | Type              | Description                                                                 |
|----------------|-------------------|-----------------------------------------------------------------------------|
| `device_id`    | string            | Identifier for the GPS device (e.g., vehicle or user ID).                  |
| `trace_number` | integer           | Trace sequence number (e.g., session or trip ID).                          |
| `ts       `    | string (datetime) | Timestamp in ISO 8601 format, e.g., `2024-05-17T14:35:10`.                 |
| `geometry`     | string (WKT)      | GPS coordinates in Well-Known Text (WKT), e.g., `POINT(4.395 51.209)`.     |

### Example Data Format
```csv
device_id,trace_number,ts,geometry
device_001,1,2024-05-17T14:35:10,POINT(4.395 51.209)
device_001,1,2024-05-17T14:35:15,POINT(4.396 51.210)
device_001,1,2024-05-17T14:35:20,POINT(4.395 51.209)
```
## Approach

1. A dataset of GPS traces was provided by the client, TravelTrack Inc.
2. Data Exploration, see data_exploration.ipynb.
3. Label (target = stopped) the provided dataset using a **rule-based approach**:
   - A point is considered "stopped" if `speed_kmh < 1` and at least for 5 sec on the (exact) same location (GPS data might be jittery, so probably the exact same location might not be the best approach, but also keep points very close)
4. Feature Engineering
   - **Distance metrics**: Distance between consecutive GPS points.
   - **Spatial coordinates**: Latitude and longitude values.
   - To prevent data leakage, we exclude time (distance/time = one-to-one relationship with stop labels).
5. Model development
   - Removed unrealistic outliers.
   - Used unique device-trace combinations for train/test/val split.
      - We want to keep all GPS points from the same trace together (not part in train, part in test).
      - This avoids "cheating" by seeing parts of the same trace in both training and evaluation.
   - Split dataset in train 60% / validation 20% / test 20%
   - Train model on training set
      - Baseline - logistic regression
      - RandomForestClassifier and XGBOOST
         - Hyperparameter Search: Manual randomized search using validation set for evaluation

## Installation & Setup

### Option 1: Docker (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/rluyck/gps-stop-detection.git
cd gps-stop-detection
```

2. **Build the Docker image**
```bash
docker build --no-cache -t gps-stop-detector -f docker/Dockerfile .
```

3. **Start the container**
```bash
docker run -p 8000:8000 gps-stop-detector
```

4. **Access the web interface**
   - Open your browser and visit: http://localhost:8000

### Option 2: Local Development

1. **Clone and navigate to repository**
```bash
git clone https://github.com/rluyck/gps-stop-detection.git
cd gps-stop-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Usage

1. **Upload Files**: Select one or more CSV files with GPS traces
2. **Process Data**: Click "Run" to start stop detection
3. **View Results**: 
   - Interactive map showing detected stops (red markers) and paths (blue lines)
   - SHAP feature importance plots explaining model decisions
   - Summary statistics of detected stops

### Expected Output
- **Map Visualization**: Interactive Folium map with stop locations and trajectory paths
- **Stop Summary**: Number of stops detected, total stop duration, and stop locations
- **Feature Importance**: SHAP plots showing which features influenced stop predictions

---

## TO DO

- unit tests
- clean api script
- remove unused files from repo
- look for more data (open-source gps traces) to train on
- do testing on more data (open-source gps traces)
- ...

---
