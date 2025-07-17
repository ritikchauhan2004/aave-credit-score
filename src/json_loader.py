import ijson
from datetime import datetime, UTC
import os

# Efficiently stream and process large JSON files
def stream_transactions(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} does not exist.")

    with open(json_path, 'r') as file:
        # Each item represents one transaction object
        for record in ijson.items(file, 'item'):
            yield record

# Convert timestamp to datetime object
def convert_timestamp(ts):
    try:
        return datetime.utcfromtimestamp(int(ts))
    except Exception as e:
        print(f"[!] Timestamp conversion error for '{ts}': {e}")
        return None


# Test block (remove in production)
if __name__ == "__main__":
    sample_path = "data/raw/user_transactions.json"
    count = 0
    for tx in stream_transactions(sample_path):
        print(f"{tx['userWallet']} at {convert_timestamp(tx['timestamp'])}")
        count += 1
        if count >= 3:
            break
