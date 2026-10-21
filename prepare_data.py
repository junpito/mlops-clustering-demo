"""One-time snapshot preparation from the source used by the supplied notebook."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from pipeline import aggregate_users

SOURCE = ("https://docs.google.com/spreadsheets/d/e/"
          "2PACX-1vTbg5WVW6W3c8SPNUGc3A3AL-AG32TPEQGpdzARfNICMsLFI0LQj0jporhsLCeVhkN5AoRsTkn08AYl/"
          "pub?output=csv")


def prepare(source, destination):
    destination = Path(destination)
    if (destination / "users.csv").exists():
        raise ValueError("Snapshot exists; choose a fresh destination to avoid resetting training data")
    raw = pd.read_csv(source)[[
        "AccountID", "TransactionID", "TransactionAmount",
        "TransactionDuration", "LoginAttempts", "AccountBalance",
    ]]
    duplicates = int(raw.duplicated().sum())
    clean = raw.drop_duplicates()
    missing = int(clean.isna().any(axis=1).sum())
    clean = clean.dropna()
    frame = aggregate_users(clean)
    # Hash order avoids the notebook's row order influencing the split.
    order = frame.user_id.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    frame = frame.assign(_order=order).sort_values("_order").drop(columns="_order")
    n = len(frame)
    if n < 100:
        raise ValueError("Need at least 100 accounts for this demonstration")
    a, b, c = int(n * .6), int(n * .8), int(n * .9)
    pieces = {
        "baseline_users.csv": frame.iloc[:a],
        "users.csv": frame.iloc[:a],
        "incoming_users.csv": frame.iloc[a:b],
        "validation_users.csv": frame.iloc[b:c],
        "test_users.csv": frame.iloc[c:],
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, piece in pieces.items():
        piece.sort_values("user_id").to_csv(destination / name, index=False)
    manifest = {
        "source_url": SOURCE,
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "cleaning": {"source_rows": len(raw), "duplicate_rows_removed": duplicates,
                     "missing_rows_removed": missing, "accepted_transactions": len(clean)},
        "unit": "one account (AccountID), treated as user for this demo",
        "split": "SHA256(AccountID), sorted; 60% baseline, 20% incoming, 10% validation, 10% test",
        "user_counts": {name: len(piece) for name, piece in pieces.items()},
        "snapshot_sha256": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in pieces
        },
        "note": "Numeric behavior features only; no IP/device identifiers, demographics, or cluster targets.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["user_counts"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/source-transactions.csv")
    parser.add_argument("--destination", default="data")
    args = parser.parse_args()
    prepare(args.source, args.destination)
