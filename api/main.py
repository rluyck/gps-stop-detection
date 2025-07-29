"""
FastAPI application for GPS stop detection service + HTML map rendering

This application provides a web interface for uploading GPS trajectory data,
detecting vehicle stops using rule-based algorithms, and visualizing the results
on interactive maps.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add the src directory to Python path to import custom modules
# This allows importing from the parent directory's src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessor import read_and_preprocess
from stop_detector_ml import apply_stop_classifier, generate_simple_shap_plot
from map_generator import generate_map
from model_utils import get_feature_columns
from feature_engineering import add_features

# Configure logging to track application events and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB per file
ALLOWED_EXTENSIONS = {'.gpx', '.kml', '.csv', '.json', '.txt'}
MAX_FILES = 20

# Initialize FastAPI application with metadata
app = FastAPI(
    title="GPS Stop Detection API",
    description="API for detecting vehicle stops in GPS trajectory data",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI documentation endpoint
    redoc_url="/redoc"     # ReDoc documentation endpoint
)

# Enable Cross-Origin Resource Sharing (CORS) for web browser access
# This allows the frontend to make requests from different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow requests from any origin (use specific domains in production)
    allow_credentials=True,     # Allow cookies and authentication headers
    allow_methods=["*"],        # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],        # Allow all headers
)

# Set up directories for static files (CSS, JS, generated maps) and HTML templates
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(STATIC_DIR, exist_ok=True)  # Create static directory if it doesn't exist

# Mount static files directory to serve CSS, JS, and generated map files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Initialize Jinja2 templating engine for rendering HTML pages
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global variable to store processed GPS data across requests
# Note: In production, consider using a proper database or session storage
gdf_global = pd.DataFrame()


# Custom exception classes for better error handling
class FileValidationError(Exception):
    """Custom exception for file validation errors"""
    pass


class GPSAnalysisError(Exception):
    """Custom exception for GPS analysis errors"""
    pass


def validate_uploaded_files(files: List[UploadFile]) -> None:
    """
    Validate uploaded files before processing.
    
    Args:
        files: List of uploaded files to validate
        
    Raises:
        FileValidationError: If validation fails
    """
    if not files:
        raise FileValidationError("No files provided")
    
    if len(files) > MAX_FILES:
        raise FileValidationError(f"Too many files. Maximum {MAX_FILES} allowed")
    
    for file in files:
        if not file.filename:
            raise FileValidationError("All files must have a filename")
        
        # Check file extension (be more permissive than original)
        file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_ext and file_ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"File {file.filename} has extension {file_ext}, proceeding anyway")


async def process_uploaded_files(files: List[UploadFile]) -> pd.DataFrame:
    """
    Process and combine uploaded GPS files into a single DataFrame.
    
    Args:
        files: List of uploaded files
        
    Returns:
        Combined and preprocessed DataFrame
        
    Raises:
        GPSAnalysisError: If file processing fails
    """
    processed_dataframes = []
    processing_errors = []
    
    for i, file in enumerate(files):
        try:
            logger.info(f"Processing file {i+1}/{len(files)}: {file.filename}")
            
            # Read file contents
            contents = await file.read()
            if not contents:
                logger.warning(f"Skipping empty file: {file.filename}")
                continue
            
            # Preprocess the GPS data (keep original function signature)
            preprocessed_df = read_and_preprocess(contents)
            
            if preprocessed_df.empty:
                logger.warning(f"No valid data found in file: {file.filename}")
                continue
                
            # Add source file information for tracking
            if 'source_file' not in preprocessed_df.columns:
                preprocessed_df['source_file'] = file.filename
                
            processed_dataframes.append(preprocessed_df)
            logger.info(f"Successfully processed {len(preprocessed_df)} records from {file.filename}")
            
        except Exception as e:
            error_msg = f"Failed to process {file.filename}: {str(e)}"
            logger.error(error_msg)
            processing_errors.append(error_msg)
            # Continue processing other files instead of failing completely
    
    # Check if we have any successful processing
    if not processed_dataframes:
        if processing_errors:
            raise GPSAnalysisError(f"Failed to process any files. Errors: {'; '.join(processing_errors)}")
        else:
            raise GPSAnalysisError("No valid GPS data found in any uploaded files")
    
    # Log any partial failures
    if processing_errors:
        logger.warning(f"Some files failed to process: {'; '.join(processing_errors)}")
    
    # Combine all processed DataFrames
    try:
        combined_df = pd.concat(processed_dataframes, ignore_index=True)
        logger.info(f"Successfully combined {len(processed_dataframes)} files into {len(combined_df)} total records")
        return combined_df
    except Exception as e:
        raise GPSAnalysisError(f"Failed to combine processed files: {str(e)}")


def predict_stops_with_model(gdf: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering and predict the stops using the trained model.

    """
    try:
        logger.info("Start predicting stops...")
        
        # Feature engineering
        enriched_gdf = add_features(gdf)
        
        # Log df after feature engineering
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Feature engineering complete. Sample data:\n{enriched_gdf.head()}")
            
        if logger.isEnabledFor(logging.DEBUG):
            feature_cols = get_feature_columns(gdf)
            logger.debug(f"Generated features: {feature_cols}")
            logger.debug(f"Feature data sample:\n{enriched_gdf[feature_cols].head()}")
        
        # Apply the classifier to predict stops
        logger.info("Running ML model stop prediction...")
        predictions_gdf = apply_stop_classifier(enriched_gdf)
        
        # Log model prediction results
        if 'stopped' in predictions_gdf.columns:
            stops_predicted = predictions_gdf['stopped'].sum()
            logger.info(f"ML model prediction complete. Predicted {stops_predicted} stop points out of {len(predictions_gdf)} total points")
        else:
            logger.warning("Model prediction completed but 'stopped' column not found in results")
            
        logger.info(f"Final prediction dataset shape: {predictions_gdf.shape}")
        
        return predictions_gdf
        
    except Exception as e:
        logger.error(f"ML model stop prediction failed: {str(e)}")
        raise GPSAnalysisError(f"Stop prediction failed: {str(e)}")


@app.on_event("startup")
async def startup_event() -> None:
    """
    Application startup event handler.
    Logs that the application is ready and using rule-based detection.
    """
    logger.info("GPS Stop Detection API starting up...")
    logger.info("ML model will be used for stop prediction")
    logger.info(f"Static files directory: {STATIC_DIR}")
    logger.info(f"Templates directory: {TEMPLATES_DIR}")
    logger.info("Startup complete")


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """
    Root endpoint that serves the main upload page.
    
    Args:
        request: FastAPI Request object needed for template rendering
        
    Returns:
        HTMLResponse: Rendered index.html template with file upload form
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze/html")
async def analyze_and_list_traces(request: Request, files: List[UploadFile] = File(...)):
    """
    Main analysis endpoint that processes uploaded GPS files.
    
    This endpoint:
    1. Validates and accepts multiple uploaded files
    2. Preprocesses each file to standardize the data format
    3. Combines all files into a single DataFrame
    4. Applies feature engineering and ML model stop prediction
    5. Stores results globally for subsequent visualization
    6. Redirects to trace selector page
    
    Args:
        request: FastAPI Request object for template rendering
        files: List of uploaded files containing GPS trajectory data
        
    Returns:
        HTMLResponse: Rendered trace selector page showing analysis results
        
    Raises:
        HTTPException: If file processing fails or no valid data is found
    """
    global gdf_global
    
    try:
        logger.info(f"{len(files)} files uploaded")
        
        # Validate uploaded files
        validate_uploaded_files(files)
        logger.info("File validation passed")
        
        # Process uploaded files
        combined_gdf = await process_uploaded_files(files)
        logger.info(f"File processing complete. Combined dataset has {len(combined_gdf)} records")
        
        # Perform ML model stop prediction (feature engineering + model prediction)
        predictions_gdf = predict_stops_with_model(combined_gdf)
        
        # Store predicted data globally for use in other endpoints
        gdf_global = predictions_gdf.copy()
        logger.info("ML model prediction complete. Data stored globally for visualization")

        # Redirect to trace selector page to show analysis results
        return await render_trace_selector(request)

    except FileValidationError as e:
        logger.error(f"File validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File validation error: {str(e)}")
    
    except GPSAnalysisError as e:
        logger.error(f"ML model prediction failed: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Prediction error: {str(e)}")
    
    except Exception as e:
        logger.exception("Unexpected error during ML model prediction")
        raise HTTPException(status_code=500, detail=f"Failed to process traces: {str(e)}")


@app.get("/selector", response_class=HTMLResponse)
async def render_trace_selector(request: Request):
    """
    Renders the trace selector page showing statistics for all analyzed traces.
    
    This endpoint:
    1. Groups GPS data by device_id and trace_number
    2. Calculates statistics for each trace (total points, stops, etc.)
    3. Generates a simple SHAP plot for model explainability
    4. Renders a table allowing users to select traces for visualization
    
    Args:
        request: FastAPI Request object for template rendering
        
    Returns:
        HTMLResponse: Rendered trace selector page with statistics table and SHAP plot
        
    Raises:
        HTTPException: If no data is loaded or processing fails
    """
    global gdf_global
    
    try:
        # Check if data has been loaded and processed
        if gdf_global.empty:
            logger.warning("Trace selector accessed but no data is loaded")
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")

        logger.info("Generating trace selector statistics and SHAP plot")
        
        # Generate simple SHAP plot
        shap_plot_b64 = None
        try:
            shap_result = generate_simple_shap_plot(gdf_global)
            shap_plot_b64 = shap_result.get('plot_b64')
            logger.info("SHAP plot generated successfully")
        except Exception as e:
            logger.warning(f"Failed to generate SHAP plot: {str(e)}")
            # Continue without SHAP plot
        
        # Group data by device ID and trace number to analyze each trace separately
        trace_groups = gdf_global.groupby(["device_id", "trace_number"])
        rows = []
        
        # Calculate statistics for each trace
        for (device_id, trace_number), group in trace_groups:
            total_points = len(group)
            
            # Handle cases where 'stopped' column might not exist
            stopped_points = group.get("stopped", pd.Series(dtype=int)).sum()
            
            stop_ratio = round(100 * stopped_points / total_points, 2) if total_points > 0 else 0
            
            rows.append({
                "device_id": device_id,
                "trace_number": trace_number,
                "total_points": total_points,
                "stopped_points": int(stopped_points),
                "stop_ratio": stop_ratio
            })

        logger.info(f"Generated statistics for {len(rows)} traces")
        
        # Render the trace selector page with statistics
        return templates.TemplateResponse("trace_selector.html", {
            "request": request,
            "trace_stats": rows,
            "shap_plot": shap_plot_b64
        })

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.exception("Trace selector generation failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate trace selector: {str(e)}")


@app.post("/visualize/trace")
async def visualize_selected_trace(request: Request, device_id: int = Form(...), trace_number: int = Form(...)):
    """
    Generates and displays an interactive map for a selected trace.
    
    This endpoint:
    1. Filters the global dataset for the specified device and trace
    2. Generates an interactive HTML map showing the GPS trajectory and stops
    3. Renders a results page with the embedded map and statistics
    
    Args:
        request: FastAPI Request object for template rendering
        device_id: ID of the device whose trace to visualize
        trace_number: Number of the specific trace to visualize
        
    Returns:
        HTMLResponse: Rendered results page with embedded map and statistics
        
    Raises:
        HTTPException: If the specified trace is not found or map generation fails
    """
    global gdf_global
    
    try:
        logger.info(f"Visualizing trace for device {device_id}, trace {trace_number}")
        
        # Check if data is loaded
        if gdf_global.empty:
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")
        
        # Filter data for the specific device and trace requested
        subset = gdf_global[
            (gdf_global["device_id"] == device_id) &
            (gdf_global["trace_number"] == trace_number)
        ]

        # Check if the requested trace exists
        if subset.empty:
            logger.warning(f"No matching trace found for device {device_id}, trace {trace_number}")
            raise HTTPException(status_code=404, detail=f"No trace found for device {device_id}, trace {trace_number}")

        logger.info(f"Found trace with {len(subset)} points. Generating map...")
        
        # Generate interactive HTML map and save to static directory
        generate_map(subset, STATIC_DIR)
        logger.info("Map generation complete")

        # Calculate statistics for display
        total_stops = int(subset.get('stopped', pd.Series(dtype=int)).sum())
        
        # Render results page with embedded map and trace statistics
        return templates.TemplateResponse("result.html", {
            "request": request,
            "map_url": "/static/map.html",
            "stats": {
                "Device ID": device_id,
                "Trace number": trace_number,
                "Total points": len(subset),
                "Stops": total_stops
            }
        })

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.exception(f"Map generation failed for device {device_id}, trace {trace_number}")
        raise HTTPException(status_code=500, detail=f"Failed to generate map: {str(e)}")


@app.post("/visualize/all")
async def visualize_all_traces(request: Request):
    """
    Generates and displays an interactive map showing all traces overlaid on one map.
    
    This endpoint:
    1. Uses the entire global dataset (all devices and traces)
    2. Generates an interactive HTML map showing all GPS trajectories and stops
    3. Renders a results page with the embedded map and overall statistics
    
    Args:
        request: FastAPI Request object for template rendering
        
    Returns:
        HTMLResponse: Rendered results page with embedded map showing all traces
        
    Raises:
        HTTPException: If no data is loaded or map generation fails
    """
    global gdf_global
    
    try:
        logger.info("Visualizing all traces on combined map")
        
        # Check if data has been loaded and processed
        if gdf_global.empty:
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")

        logger.info(f"Generating combined map for {len(gdf_global)} total points")
        
        # Generate map using the entire dataset
        generate_map(gdf_global, STATIC_DIR)
        logger.info("Combined map generation complete")

        # Calculate overall statistics across all traces
        trace_groups = gdf_global.groupby(["device_id", "trace_number"])
        total_traces = len(trace_groups)
        unique_devices = gdf_global["device_id"].nunique()
        total_points = len(gdf_global)
        total_stops = int(gdf_global.get('stopped', pd.Series(dtype=int)).sum())
        
        # Calculate average stop ratio across all data
        stop_ratio = round(100 * total_stops / total_points, 2) if total_points > 0 else 0

        logger.info(f"Combined visualization complete: {unique_devices} devices, {total_traces} traces, {total_stops} stops")

        # Render results page with embedded map and overall statistics
        return templates.TemplateResponse("result.html", {
            "request": request,
            "map_url": "/static/map.html",
            "stats": {
                "View Type": "All Traces Combined",
                "Total Devices": unique_devices,
                "Total Traces": total_traces,
                "Total Points": total_points,
                "Total Stops": total_stops,
                "Overall Stop Ratio (%)": stop_ratio
            }
        })

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.exception("Combined map generation failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate combined map: {str(e)}")
    

@app.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint for monitoring application status.
    
    Returns basic information about the application state, useful for
    load balancers, monitoring systems, and debugging.
    
    Returns:
        Dict: Status information including health status, detection mode, and timestamp
    """
    global gdf_global
    
    return {
        "status": "healthy",
        "mode": "rule_based",
        "data_loaded": not gdf_global.empty,
        "total_records": len(gdf_global) if not gdf_global.empty else 0,
        "timestamp": datetime.now().isoformat()
    }


# Additional endpoint for clearing global data (useful for testing/debugging)
@app.post("/admin/clear")
async def clear_global_data():
    """
    Administrative endpoint to clear the global dataset.
    Useful for testing or when you need to start fresh.
    
    Returns:
        Dict: Confirmation message
    """
    global gdf_global
    
    records_cleared = len(gdf_global)
    gdf_global = pd.DataFrame()
    
    logger.info(f"Global dataset cleared. {records_cleared} records removed.")
    
    return {
        "status": "success",
        "message": f"Cleared {records_cleared} records from global dataset",
        "timestamp": datetime.now().isoformat()
    }