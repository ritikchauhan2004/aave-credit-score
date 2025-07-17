import pandas as pd
import numpy as np
from json_loader import stream_transactions, convert_timestamp

# aggregate wallet features
def aggregate_wallet_features(json_path: str) -> pd.DataFrame:
    records = []

    for tx in stream_transactions(json_path):
        try:
            wallet = tx.get("userWallet")
            action = tx.get("action", "").lower()
            timestamp = convert_timestamp(tx.get("timestamp"))
            amount = float(tx.get("actionData", {}).get("amount", 0)) / 1e6
            asset_price = float(tx.get("actionData", {}).get("assetPriceUSD", 1))
            amount_usd = amount * asset_price

            records.append({
                "wallet": wallet,
                "action": action,
                "timestamp": timestamp,
                "amount_usd": amount_usd
            })

        except Exception as e:
            print(f"Skipping tx due to error: {e}")
            continue

    df = pd.DataFrame(records)
    df.dropna(subset=["wallet"], inplace=True)

    # Count actions per wallet
    count_df = df.groupby(["wallet", "action"]).size().unstack(fill_value=0)
    count_df.columns = [f"num_{col}" for col in count_df.columns]

    # Sum of USD amounts per action per wallet
    sum_df = df.groupby(["wallet", "action"])["amount_usd"].sum().unstack(fill_value=0)
    sum_df.columns = [f"total_{col}" for col in sum_df.columns]

    agg_df = pd.concat([count_df, sum_df], axis=1).reset_index()

    # Time-based features
    gap_features = []
    for wallet, group in df.groupby("wallet"):
        times = group.sort_values("timestamp")["timestamp"].tolist()
        if len(times) >= 2:
            gaps = [(times[i+1] - times[i]).total_seconds() for i in range(len(times) - 1)]
            avg_gap = sum(gaps) / len(gaps)
        else:
            avg_gap = 0
        gap_features.append({"wallet": wallet, "avg_tx_gap_sec": avg_gap})

    gap_df = pd.DataFrame(gap_features)
    agg_df = agg_df.merge(gap_df, on="wallet", how="left")

    # Fallback fill for missing features
    for col in ['num_borrow', 'num_deposit', 'num_repay', 'num_redeemunderlying',
                'total_borrow', 'total_deposit', 'total_repay', 'total_redeemunderlying']:
        if col not in agg_df.columns:
            agg_df[col] = 0

    # Ratios
    agg_df["total_transactions"] = df.groupby("wallet").size().values

    # Drop columns we don’t want
    agg_df.drop(columns=['num_liquidationcall', 'total_liquidationcall'], inplace=True, errors='ignore')

    # Log-transform monetary features
    monetary_cols = ['total_borrow', 'total_deposit', 'total_redeemunderlying', 'total_repay']
    for col in monetary_cols:
        agg_df[f"log_{col}"] = np.log1p(agg_df[col])

    # Additional engineered features
    agg_df['activity_score'] = agg_df['total_transactions'] / (agg_df['avg_tx_gap_sec'] + 1)
    agg_df['net_borrow'] = agg_df['total_borrow'] - agg_df['total_repay']
    agg_df['borrow_to_deposit_ratio'] = agg_df['total_borrow'] / (agg_df['total_deposit'] + 1)
    agg_df['deposit_frequency'] = agg_df['num_deposit'] / (agg_df['avg_tx_gap_sec'] + 1)
    agg_df['repay_frequency'] = agg_df['num_repay'] / (agg_df['total_transactions'] + 1)
    agg_df['repayment_efficiency'] = agg_df['total_repay'] / (agg_df['total_borrow'] + 1)
    agg_df['avg_repay_per_tx'] = agg_df['total_repay'] / (agg_df['num_repay'] + 1)

    # Final selected features
    features = [
        'wallet', 'num_borrow', 'num_deposit', 'num_redeemunderlying', 'num_repay',
        'total_borrow', 'total_deposit', 'total_redeemunderlying', 'total_repay',
        'avg_tx_gap_sec', 'total_transactions',
        'activity_score', 'net_borrow', 'borrow_to_deposit_ratio',
        'deposit_frequency', 'repay_frequency', 'repayment_efficiency',
        'avg_repay_per_tx'
    ]

    return agg_df[features]


# Run and save
if __name__ == "__main__":
    path = "data/raw/user_transactions.json"
    df = aggregate_wallet_features(path)
    df.to_csv("data/processed/features_per_wallet.csv", index=False)
    print("✅ Features saved to data/processed/features_per_wallet.csv")
