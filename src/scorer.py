import pandas as pd
import joblib
import json
import os
from sklearn.preprocessing import MinMaxScaler


def load_features(feature_path: str):
    df = pd.read_csv(feature_path)

    features_to_use = [
        'repayment_efficiency', 'total_repay', 'num_repay', 'deposit_frequency',
        'repay_borrow_ratio', 'total_deposit', 'net_borrow', 'activity_score'
    ]
    return df, df[features_to_use], features_to_use


def get_model_path(model_name: str) -> str:
    """
    Map user choice to actual model path.
    """
    model_map = {
        "random_forest": "outputs/models/random_forest_model.pkl",
        "xgboost": "outputs/models/xgboost_model.pkl"
    }
    path = model_map.get(model_name.lower())
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model file not found for selection: '{model_name}'")
    return path


def predict_scores(model_path: str, feature_path: str, output_path: str):
    df, X, features_to_use = load_features(feature_path)

    # Load trained model
    model = joblib.load(model_path)

    # Predict and normalize scores
    predictions = model.predict(X)
    predictions_scaled = MinMaxScaler(feature_range=(0, 1000)).fit_transform(predictions.reshape(-1, 1))
    df["predicted_credit_score"] = predictions_scaled.round(2)

    # Prepare JSON output
    scores_json = {
        row["wallet"]: row["predicted_credit_score"]
        for _, row in df[["wallet", "predicted_credit_score"]].iterrows()
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores_json, f, indent=2)

    print(f"✅ Credit scores saved to {output_path}")
