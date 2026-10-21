"""Simulate arrival of a new batch; commit/push the resulting CSV to trigger CI."""
import argparse

from pipeline import append_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/users.csv")
    parser.add_argument("--batch", default="data/incoming_users.csv")
    args = parser.parse_args()
    print(f"Added {append_batch(args.data, args.batch)} users. Commit/push the changed CSV to trigger GitHub Actions.")
