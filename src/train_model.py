import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error


def main():
    # Load features
    df = pd.read_csv("data/processed/features_per_wallet.csv")

    # Select features used for scoring and training
    features_to_use = [
        'repayment_efficiency', 'total_repay', 'num_repay', 'deposit_frequency',
        'repay_borrow_ratio', 'total_deposit', 'net_borrow', 'activity_score'
    ]
    X = df[features_to_use]

    # Generate synthetic credit score
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features_to_use)

    df['credit_score'] = (
        0.25 * X_scaled['repayment_efficiency'] +
        0.20 * X_scaled['total_repay'] +
        0.15 * X_scaled['num_repay'] +
        0.10 * X_scaled['deposit_frequency'] +
        0.10 * X_scaled['repay_borrow_ratio'] +
        0.10 * X_scaled['total_deposit'] +
        0.05 * (1 - X_scaled['net_borrow']) +
        0.05 * X_scaled['activity_score']
    )

    # Rescale credit score to 0–1000
    df['credit_score'] = MinMaxScaler(feature_range=(0, 1000)).fit_transform(df[['credit_score']])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df[features_to_use], df['credit_score'], test_size=0.2, random_state=42
    )

    # Train models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Evaluate
    rf_preds = rf_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)

    rf_rmse = root_mean_squared_error(y_test, rf_preds)
    xgb_rmse = root_mean_squared_error(y_test, xgb_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)

    tolerance = 5  # Acceptable error range
    
    correct1 = np.abs(y_test - rf_preds) <= tolerance
    accuracy1 = correct1.sum() / len(correct1)

    correct2 = np.abs(y_test - xgb_preds) <= tolerance
    accuracy2 = correct2.sum() / len(correct2)
    
    print(f"Random Forest Mean Squared Error:", rf_rmse)
    print(f"XGBoost Mean Squared Error:", xgb_rmse)
    print(f"Random Forest R²: {rf_r2:.2f}")
    print(f"XGBoost R²: {xgb_r2:.2f}")
    print(f"Approximate Accuracy rf within ±{tolerance} points: {accuracy1:.2%}")
    print(f"Approximate Accuracy xgb within ±{tolerance} points: {accuracy2:.2%}")
    
    # Save models
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(rf_model, "outputs/models/random_forest_model.pkl")
    joblib.dump(xgb_model, "outputs/models/xgboost_model.pkl")
    print("✅ Models saved in outputs/models/")


if __name__ == "__main__":
    main()
    