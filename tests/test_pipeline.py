"""Behavior checks: no ID features/leakage, safe append, portable inference."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline import (
    FEATURES, aggregate_users, append_batch, build_model, check_gate,
    load_users, validate_splits,
)


def users(n=30):
    rng = np.random.default_rng(10)
    frame = pd.DataFrame(
        rng.uniform(1, 100, (n, len(FEATURES))), columns=FEATURES
    )
    frame["user_id"] = [f"USER-{i:03d}" for i in range(n)]
    frame["transaction_count"] = 2.0
    return frame[["user_id", *FEATURES]]


class PipelineTests(unittest.TestCase):
    def test_snapshot_preparation_records_dropped_missing_and_duplicate_rows(self):
        from prepare_data import prepare
        import json
        raw = pd.DataFrame({
            "AccountID": [f"A{i}" for i in range(100)],
            "TransactionID": [f"T{i}" for i in range(100)],
            "TransactionAmount": 10, "TransactionDuration": 2,
            "LoginAttempts": 1, "AccountBalance": 100,
        })
        missing = raw.iloc[:1].copy()
        missing["TransactionAmount"] = np.nan
        raw = pd.concat([raw, raw.iloc[:1], missing], ignore_index=True)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "raw.csv"
            raw.to_csv(source, index=False)
            prepare(source, Path(folder) / "data")
            report = json.loads((Path(folder) / "data/manifest.json").read_text())
            self.assertEqual(report["cleaning"]["duplicate_rows_removed"], 1)
            self.assertEqual(report["cleaning"]["missing_rows_removed"], 1)
            self.assertEqual(report["user_counts"]["baseline_users.csv"], 60)

    def test_transactions_become_one_row_per_account_not_per_transaction(self):
        raw = pd.DataFrame({
            "AccountID": ["A", "A", "B"], "TransactionID": ["T1", "T2", "T3"],
            "TransactionAmount": [10, 30, 80], "TransactionDuration": [2, 4, 6],
            "LoginAttempts": [1, 3, 1], "AccountBalance": [100, 200, 300],
        })
        result = aggregate_users(raw).set_index("user_id")
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc["A", "transaction_count"], 2)
        self.assertEqual(result.loc["A", "mean_transaction_amount"], 20)
        self.assertEqual(result.loc["A", "mean_account_balance"], 150)
        self.assertNotIn("TransactionID", result.columns)

    def test_invalid_values_and_duplicate_users_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "users.csv"
            for column, bad in [
                ("user_id", None), ("mean_account_balance", np.inf),
                ("mean_transaction_amount", -1), ("transaction_count", 0.5),
            ]:
                with self.subTest(column=column, bad=bad):
                    frame = users()
                    frame.loc[0, column] = bad
                    frame.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        load_users(path)
            frame = users()
            frame.loc[1, "user_id"] = frame.loc[0, "user_id"]
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_users(path)

    def test_append_is_idempotent_and_rejects_changed_existing_user(self):
        with tempfile.TemporaryDirectory() as folder:
            path, batch = Path(folder) / "users.csv", Path(folder) / "batch.csv"
            frame = users()
            frame.iloc[:20].to_csv(path, index=False)
            frame.iloc[20:].to_csv(batch, index=False)
            self.assertEqual(append_batch(path, batch), 10)
            before = path.read_bytes()
            self.assertEqual(append_batch(path, batch), 0)
            self.assertEqual(path.read_bytes(), before)
            changed = frame.iloc[:1].copy()
            changed["mean_transaction_amount"] = 999
            changed.to_csv(batch, index=False)
            with self.assertRaises(ValueError):
                append_batch(path, batch)
            self.assertEqual(path.read_bytes(), before)

    def test_evaluation_overlap_or_changed_baseline_fails_before_training(self):
        frame = users()
        base, train, valid, test = (
            frame.iloc[:10], frame.iloc[:20], frame.iloc[20:25], frame.iloc[25:]
        )
        validate_splits(base, train, valid, test)
        with self.assertRaises(ValueError):
            validate_splits(base, train, train.iloc[:5], test)
        changed = train.copy()
        changed.loc[0, FEATURES[1]] = 1234
        with self.assertRaises(ValueError):
            validate_splits(base, changed, valid, test)

    def test_model_roundtrip_keeps_preprocessing_and_ignores_id(self):
        frame = users()
        model = build_model(3, 42).fit(frame[FEATURES])
        expected = model.predict(frame[FEATURES])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(expected, restored.predict(frame[FEATURES]))
            self.assertEqual(list(restored.feature_names_in_), FEATURES)

    def test_quality_gate_rejects_regression_small_cluster_and_instability(self):
        good = {"silhouette": 0.4, "min_cluster_fraction": 0.2,
                "observed_clusters": 3}
        self.assertEqual(check_gate(good, 0.4, 0.9, 3), [])
        self.assertTrue(check_gate(good, 0.6, 0.9, 3))
        self.assertTrue(check_gate({**good, "min_cluster_fraction": 0.01}, 0.4, 0.9, 3))
        self.assertTrue(check_gate(good, 0.4, 0.1, 3))
        self.assertTrue(check_gate({**good, "silhouette": float("nan")}, 0.4, 0.9, 3))


if __name__ == "__main__":
    unittest.main()
