# explain.py

import os
import sys
import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from model_utils import get_feature_columns

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def explain_with_shap(model, X: pd.DataFrame, max_display: int = 10):
    """
    Visualizes SHAP values for a fitted tree-based model.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        A trained model that supports SHAP (e.g. tree-based).
    
    X : pd.DataFrame
        Feature matrix to compute SHAP values on.
    
    max_display : int
        Max number of top features to display in summary plot.
    """
    print("Generating SHAP summary plot...")

    # TreeExplainer is efficient for tree-based models
    explainer = shap.TreeExplainer(model)
    X_clean = X.dropna()
    
    shap_values = explainer.shap_values(X_clean)

    # Voor binaire classificatie (True klasse is klasse-index 1):
    if isinstance(shap_values, list):
        # Oudere SHAP-versies geven lijst van arrays terug
        shap_values_class = shap_values[1]
    else:
        # Nieuwere SHAP-versies geven 3D-array terug
        shap_values_class = shap_values[:, :, 1]

    shap.summary_plot(shap_values_class, X_clean)
    plt.savefig("../shap_summary_plot.png", bbox_inches='tight')

if __name__ == "__main__":
    # Optional usage: run directly to test SHAP explanations
    # Load model and data manually
    from feature_engineering import add_features
    from preprocessor import read_and_preprocess
    from stop_detector_rb import rule_based_detection

    # Load and preprocess CSV data
    with open("../data/raw/gps_traces.csv", "rb") as f:
        df = read_and_preprocess(f.read())

    df = rule_based_detection(df)
    df = add_features(df)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]

    # Load model
    model = joblib.load("../models/stop_model.pkl")

    # Run SHAP explainability
    explain_with_shap(model, X)
