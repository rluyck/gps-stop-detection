import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def split_by_trace(df: pd.DataFrame, val_size=0.15, test_size=0.15, random_state=42):
    """
    Splits the DataFrame into train/val/test sets based on unique (device_id, trace_number) pairs.

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrame
    """
    trace_pairs = df[['device_id', 'trace_number']].drop_duplicates()

    trace_trainval, trace_test = train_test_split(
        trace_pairs,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    trace_train, trace_val = train_test_split(
        trace_trainval,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        shuffle=True
    )

    def subset(df, pairs):
        return df.merge(pairs, on=['device_id', 'trace_number'])

    df_train = subset(df, trace_train)
    df_val   = subset(df, trace_val)
    df_test  = subset(df, trace_test)

    return df_train, df_val, df_test


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Returns the list of feature columns used for model training and prediction.
    Adjust this list based on your engineered features.
    """
    return [
        'distance_m',
        # 'time_diff_s',  # the model can infer speed from distance and time combined
        'lat',
        'lon'
        # 'speed_kmh',  # 1-1 with label 'stopped' 
        # 'stop_group_duration_s'  # 1-1 with label 'stopped' 
    ]



def evaluate_model(model, X, y_true) -> None:
    """
    Prints standard classification metrics for a model.

    Metrics explained:
    - Precision: Of the predicted positives, how many were correct?
    - Recall: Of the actual positives, how many were found?
    - F1-score: Harmonic mean of precision and recall (balance between them)
    - Support: Number of true instances per class in the test data
    - Accuracy: Overall percentage of correct predictions

    Also includes macro and weighted averages for class-wise aggregation:
    - Macro avg: Average over classes, unweighted
    - Weighted avg: Average over classes, weighted by support (class size)
    
    
    """
    y_pred = model.predict(X)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
