import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


def analyze_wallet_scores(scored_file_path: str, model_name: str, score_column: str = "predicted_credit_score"):
    if not os.path.exists(scored_file_path):
        print(f"❌ File not found: {scored_file_path}")
        return

    # Load wallet features
    df = pd.read_csv("data/processed/features_per_wallet.csv")

    # Load credit scores from JSON
    with open(scored_file_path, "r") as f:
        score_dict = json.load(f)

    scores_df = pd.DataFrame(list(score_dict.items()), columns=["wallet", score_column])
    df = df.merge(scores_df, on="wallet", how="left")

    # Bin into score ranges (0–100, ..., 900–1000)
    bins = list(range(0, 1100, 100))
    labels = [f"{i}-{i+99}" for i in bins[:-1]]
    df["score_range"] = pd.cut(df[score_column], bins=bins, labels=labels, right=False)

    # Plot and save
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="score_range", palette="coolwarm")
    plt.title(f"Wallet Credit Score Distribution ({model_name})")
    plt.xlabel("Credit Score Range")
    plt.ylabel("Number of Wallets")
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("outputs/scores", exist_ok=True)
    filename = f"outputs/scores/{model_name[:3]}_credit_score_distribution.png"
    plt.savefig(filename)
    plt.close()
    print(f"📊 Distribution plot saved: {filename}")

    # Behavior summaries
    low_df = df[df[score_column] < 300]
    high_df = df[df[score_column] >= 700]

    print("\n🔻 Behavior of Low-Score Wallets (<300):")
    print(low_df.describe().T[["mean", "std"]].round(2))

    print("\n🔺 Behavior of High-Score Wallets (≥700):")
    print(high_df.describe().T[["mean", "std"]].round(2))


if __name__ == "__main__":
    print("Select model score file to analyze:")
    print("1: Random Forest")
    print("2: XGBoost")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        score_file = "outputs/scores/rf_wallet_scores.json"
        model_name = "RandomForest"
    elif choice == "2":
        score_file = "outputs/scores/xgb_wallet_scores.json"
        model_name = "XGBoost"
    else:
        print("❌ Invalid selection. Exiting.")
        exit()

    analyze_wallet_scores(scored_file_path=score_file, model_name=model_name)
