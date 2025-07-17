import argparse
import os
from scorer import predict_scores
from explain_model_shap import explain_model
from analyze_scored_wallets import analyze_wallet_scores

# Main function to generate wallet credit scores
def main():
    parser = argparse.ArgumentParser(description="Generate wallet credit scores using a selected trained model.")
    parser.add_argument(
        "--features", type=str, default="data/processed/features_per_wallet.csv",
        help="Path to the processed features CSV"
    )
    args = parser.parse_args()

    # Model selection
    print("\nSelect model to use:")
    print("1: Random Forest")
    print("2: XGBoost")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        model_name = "random_forest"
        model_path = "outputs/models/random_forest_model.pkl"
        output_path = "outputs/scores/rf_wallet_scores.json"
    elif choice == "2":
        model_name = "xgboost"
        model_path = "outputs/models/xgboost_model.pkl"
        output_path = "outputs/scores/xgb_wallet_scores.json"
    else:
        print("❌ Invalid choice. Exiting.")
        return

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    # Run prediction
    predict_scores(
        model_path=model_path,
        feature_path=args.features,
        output_path=output_path
    )
    #generate SHAP explanation
    explain_model(
        model_path=model_path,
        feature_path=args.features,
        model_name=model_name
    )
    # Analyze scores distribution and wallet behavior
    analyze_wallet_scores(
        scored_file_path=output_path,
        model_name=model_name
    )

if __name__ == "__main__":
    main()
