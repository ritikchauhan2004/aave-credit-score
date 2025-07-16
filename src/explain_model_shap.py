import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os


def explain_model(model_path: str, feature_path: str, model_name: str):
    # Load data
    df = pd.read_csv(feature_path)
    features = [
        'repayment_efficiency', 'total_repay', 'num_repay', 'deposit_frequency',
        'repay_borrow_ratio', 'total_deposit', 'net_borrow', 'activity_score'
    ]
    X = df[features]

    # Load model
    model = joblib.load(model_path)

    # Create SHAP explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Plot and save
    os.makedirs("outputs/shap", exist_ok=True)
    output_path = f"outputs/shap/{model_name}_shap_summary_plot.png"
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ SHAP summary plot saved to {output_path}")

if __name__ == "__main__":
    print("Choose model to explain:")
    print("1: Random Forest")
    print("2: XGBoost")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        model_path = "outputs/models/random_forest_model.pkl"
        model_name = "rf"
    elif choice == "2":
        model_path = "outputs/models/xgboost_model.pkl"
        model_name = "xgb"
    else:
        print("❌ Invalid selection.")
        exit()

    explain_model(
        model_path=model_path,
        feature_path="data/processed/features_per_wallet.csv",
        model_name=model_name
    )
