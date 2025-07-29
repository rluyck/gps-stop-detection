import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from model_utils import get_feature_columns
import logging

logger = logging.getLogger(__name__)


def apply_stop_classifier(enriched_gdf: pd.DataFrame, model_path: str = "models/stop_model_rfc.pkl") -> pd.DataFrame:
    """
    Loads the trained ML model and predicts stops for new data.

    Parameters
    ----------
    enriched_gdf : pd.DataFrame
        DataFrame with GPS features but without stop labels.

    model_path : str
        Path to the trained model.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with added prediction columns.
    """
    try:
        # Load the trained model
        clf = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Get the relevant feature columns
        feature_cols = get_feature_columns(enriched_gdf)
        logger.info(f"Using {len(feature_cols)} features for prediction")

        # Extract feature matrix (keep index intact)
        X = enriched_gdf[feature_cols]

        # Predict stops
        y_pred = clf.predict(X)
        y_pred_proba = clf.predict_proba(X)

        # Add predictions to a copy of the input DataFrame
        enriched_with_predictions = enriched_gdf.copy()
        enriched_with_predictions['stopped_predicted'] = y_pred
        enriched_with_predictions['stopped'] = y_pred  # For compatibility with existing code
        enriched_with_predictions['stop_probability'] = y_pred_proba[:, 1]  # Probability of being a stop

        logger.info(f"Predictions complete. {y_pred.sum()} stops predicted out of {len(y_pred)} points")
        
        return enriched_with_predictions
        
    except Exception as e:
        logger.error(f"Error in apply_stop_classifier: {str(e)}")
        raise


def generate_simple_shap_plot(enriched_gdf: pd.DataFrame, model_path: str = "models/stop_model_rfc.pkl", 
                             max_samples: int = 10000) -> dict:
    """
    Generate a simple SHAP feature importance plot for display on the trace selector page.
    
    Parameters
    ----------
    enriched_gdf : pd.DataFrame
        DataFrame with GPS features and predictions.
    model_path : str
        Path to the trained model.
    max_samples : int
        Maximum number of samples to use for SHAP explanation (for performance).
        
    Returns
    -------
    dict
        Dictionary containing base64-encoded plot image.
    """
    try:
        # Load the trained model
        clf = joblib.load(model_path)
        logger.info(f"Loaded model for SHAP analysis from {model_path}")
        
        # Get feature columns
        feature_cols = get_feature_columns(enriched_gdf)
        X = enriched_gdf[feature_cols]
        
        # Sample data if too large (SHAP can be slow on large datasets)
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} points from {len(X)} for SHAP analysis")
        else:
            X_sample = X
            logger.info(f"Using all {len(X)} points for SHAP analysis")
        
        # Create SHAP explainer
        logger.info("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(clf)
        
        # Calculate SHAP values
        logger.info("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP versions and output formats
        if isinstance(shap_values, list):
            # Older SHAP versions return list of arrays for each class
            shap_values_class = shap_values[1]  # Class 1 (stopped=True)
        else:
            # Newer SHAP versions might return 3D array
            if len(shap_values.shape) == 3:
                shap_values_class = shap_values[:, :, 1]  # Class 1
            else:
                shap_values_class = shap_values  # Binary classification, single output
        
        # Generate simple summary plot (bar chart)
        logger.info("Generating SHAP summary plot...")
        shap.summary_plot(shap_values_class, X_sample, show=False, max_display=10, plot_type="dot", alpha=0.5)
        fig = plt.gcf()
        fig.set_size_inches(12, 4)
        plt.xlabel("SHAP Value (impact on model output)", fontsize=12)
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_b64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        logger.info("Simple SHAP plot generation complete")
        
        return {
            'plot_b64': plot_b64,
            'samples_used': len(X_sample),
            'total_samples': len(X)
        }
        
    except Exception as e:
        logger.error(f"Error generating simple SHAP plot: {str(e)}")
        raise


# Import numpy for the feature importance calculation
