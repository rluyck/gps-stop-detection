# train_script.py
from src.preprocessor import read_and_preprocess  #  Reads and preprocesses the GPS data
from src.stop_detector_rb import rule_based_detection  # Applies rule-based logic to label stops
from src.stop_detector_ml import train_stop_classifier
from src.model_utils import split_by_trace, evaluate_model, get_feature_columns

# Load and preprocess data
with open("data/raw/gps_traces.csv", "rb") as f:
    preprocessed_gdf = read_and_preprocess(f.read())

# Label data using rule-based logic
labeled_df = rule_based_detection(preprocessed_gdf)

# Split into train, val, test sets by trace
df_train, df_val, df_test = split_by_trace(labeled_df)

# Train classifier on training set only
clf = train_stop_classifier(df_train, model_path="models/stop_model.pkl")

# Evaluate on validation and test sets
feature_cols = get_feature_columns(df_val)

print("\n📊 Validation Set Performance:")
evaluate_model(clf, df_val[feature_cols], df_val['stopped'])

print("\n🧪 Test Set Performance:")
evaluate_model(clf, df_test[feature_cols], df_test['stopped'])
