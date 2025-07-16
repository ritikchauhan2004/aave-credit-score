import ijson
from datetime import datetime
import os

def stream_transactions(json_path):
    """
    Lazily stream records from a large JSON array stored in a file.
    Each item in the array is yielded as a dictionary.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} does not exist.")

    with open(json_path, 'r') as file:
        # Each item represents one transaction object
        for record in ijson.items(file, 'item'):
            yield record


def convert_timestamp(ts):
    """
    Converts a UNIX timestamp to a UTC datetime object.
    """
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
