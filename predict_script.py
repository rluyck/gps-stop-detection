# predict_script.py
from src.preprocessor import read_and_preprocess
from src.feature_engineering import add_features
from src.stop_detector_ml import apply_stop_classifier

# Load and preprocess new traces
with open("data/new_traces.csv", "rb") as f:
    preprocessed_gdf = read_and_preprocess(f.read())

# Generate features
enriched_df = add_features(preprocessed_gdf)

# Predict with trained model
predicted_df = apply_stop_classifier(enriched_df, model_path="models/stop_model.pkl")

# Save to file
predicted_df.to_csv("data/predicted_traces.csv", index=False)
