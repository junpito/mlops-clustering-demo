"""Account-level clustering functions shared by training and inference."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "transaction_count", "mean_transaction_amount", "mean_transaction_duration",
    "mean_login_attempts", "mean_account_balance",
]


def aggregate_users(raw):
    columns = ["AccountID", "TransactionID", "TransactionAmount",
               "TransactionDuration", "LoginAttempts", "AccountBalance"]
    if not set(columns).issubset(raw.columns):
        raise ValueError("Missing transaction columns")
    raw = raw[columns].copy()
    if raw.isna().any().any() or raw["TransactionID"].duplicated().any():
        raise ValueError("Missing values or duplicate transaction IDs")
    for col in columns[2:]:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
        if not np.isfinite(raw[col]).all() or (raw[col] < 0).any():
            raise ValueError(f"Invalid transaction feature: {col}")
    result = raw.groupby("AccountID", sort=True).agg(
        transaction_count=("TransactionID", "count"),
        mean_transaction_amount=("TransactionAmount", "mean"),
        mean_transaction_duration=("TransactionDuration", "mean"),
        mean_login_attempts=("LoginAttempts", "mean"),
        mean_account_balance=("AccountBalance", "mean"),
    ).reset_index().rename(columns={"AccountID": "user_id"})
    return validate_users(result)


def append_batch(path, batch):
    path = Path(path)
    current, incoming = load_users(path), load_users(batch)
    left, right = current.set_index("user_id"), incoming.set_index("user_id")
    overlap = left.index.intersection(right.index)
    if len(overlap) and not np.allclose(
        left.loc[overlap, FEATURES], right.loc[overlap, FEATURES], rtol=0, atol=1e-9
    ):
        raise ValueError("Existing user changed; this demo only supports new users")
    fresh = incoming[~incoming.user_id.isin(current.user_id)]
    if not fresh.empty:
        combined = pd.concat([current, fresh], ignore_index=True).sort_values("user_id")
        temporary = path.with_suffix(".csv.tmp")
        combined.to_csv(temporary, index=False)
        temporary.replace(path)
    return len(fresh)


def build_model(k, seed):
    return make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=seed, n_init=10))


def check_gate(metrics, baseline_score, stability, k):
    # Demonstration thresholds, not evidence of business usefulness.
    reasons = []
    if not np.isfinite(metrics["silhouette"]) or metrics["silhouette"] < 0.05:
        reasons.append("test silhouette < 0.05 or nonfinite")
    if not np.isfinite(baseline_score) or metrics["silhouette"] < baseline_score - 0.02:
        reasons.append("test silhouette regressed by more than 0.02")
    if metrics["min_cluster_fraction"] < 0.05 or metrics["observed_clusters"] != k:
        reasons.append("missing cluster or cluster fraction < 0.05")
    if not np.isfinite(stability) or stability < 0.60:
        reasons.append("validation seed stability ARI < 0.60")
    return reasons


def load_users(path):
    return validate_users(pd.read_csv(path, dtype={"user_id": "string"}))


def validate_users(frame):
    expected = ["user_id", *FEATURES]
    if set(frame.columns) != set(expected) or frame.empty:
        raise ValueError(f"Expected nonempty data with columns {expected}")
    frame = frame[expected].copy()
    if frame.user_id.isna().any():
        raise ValueError("Missing user_id")
    frame["user_id"] = frame.user_id.astype(str).str.strip()
    if frame.user_id.eq("").any() or frame.user_id.duplicated().any():
        raise ValueError("Blank or duplicate user_id")
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    values = frame[FEATURES].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Features must be finite, nonnegative numbers")
    count = frame.transaction_count
    if (count < 1).any() or (count != np.floor(count)).any():
        raise ValueError("transaction_count must be a positive integer")
    return frame


def validate_splits(base, train, valid, test):
    for a, b in combinations([train, valid, test], 2):
        if set(a.user_id) & set(b.user_id):
            raise ValueError("Training/validation/test user overlap")
    if not set(base.user_id).issubset(train.user_id):
        raise ValueError("Baseline users must remain in training data")
    original = base.set_index("user_id")[FEATURES].sort_index()
    current = train.set_index("user_id").loc[original.index, FEATURES]
    if not np.allclose(original, current, rtol=0, atol=1e-9):
        raise ValueError("Baseline user features changed")


def evaluate(model, frame, reference_scaler):
    """Compare partitions in ONE distance space fitted only on baseline users."""
    labels = model.predict(frame[FEATURES])
    counts = np.bincount(labels, minlength=model[-1].n_clusters)
    observed = int(np.count_nonzero(counts))
    score = -1.0
    if 1 < observed < len(frame):
        score = silhouette_score(
            reference_scaler.transform(frame[FEATURES]), labels,
            sample_size=min(2000, len(frame)), random_state=42,
        )
    return {"silhouette": float(score), "min_cluster_fraction": float(counts.min() / len(frame)),
            "observed_clusters": observed}
