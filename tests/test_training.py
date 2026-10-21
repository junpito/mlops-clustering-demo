import json
import shutil
import tempfile
import unittest
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from pipeline import FEATURES
from train import run
from view_runs import relocate


class TrainingTests(unittest.TestCase):
    def test_full_run_logs_candidates_and_loads_portable_winner(self):
        # Three well-separated synthetic groups; labels are never training inputs.
        values, _ = make_blobs(n_samples=160, centers=3, n_features=5,
                               cluster_std=.2, random_state=42)
        frame = pd.DataFrame(values + 20, columns=FEATURES)
        frame["transaction_count"] = 3
        frame["user_id"] = [f"U{i}" for i in range(len(frame))]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {}
            for name, rows in [
                ("baseline", frame.iloc[:60]), ("users", frame.iloc[:100]),
                ("validation", frame.iloc[100:130]), ("test", frame.iloc[130:]),
            ]:
                paths[name] = root / f"{name}.csv"
                rows.to_csv(paths[name], index=False)
            output = root / "output"
            report = run(paths["users"], paths["baseline"], paths["validation"],
                         paths["test"], output, root / "mlruns", ks=[2, 3])
            self.assertEqual(report["counts"]["new_users"], 40)
            self.assertEqual(len(report["candidates"]), 4)
            self.assertEqual(report["status"], "READY_FOR_INTEGRATION")
            self.assertEqual(report["selected"]["k"], 3)
            model = mlflow.sklearn.load_model(str(output / "model"))
            assignments = pd.read_csv(output / "assignments.csv")
            np.testing.assert_array_equal(assignments.cluster, model.predict(frame.iloc[:100][FEATURES]))
            self.assertTrue((output / "summary.md").exists())
            self.assertTrue((output / "cluster_profiles.csv").exists())
            self.assertTrue(report["dataset_sha256"]["train"])
            # Exercise the actual download scenario, not only a synthetic meta.yaml.
            archive_store = root / "downloaded" / "mlruns"
            shutil.copytree(root / "mlruns", archive_store)
            relocate(archive_store)
            mlflow.set_tracking_uri(archive_store.resolve().as_uri())
            client = mlflow.tracking.MlflowClient()
            self.assertTrue(client.list_artifacts(report["selected"]["run_id"], "model"))
            downloaded = mlflow.sklearn.load_model(
                f"runs:/{report['selected']['run_id']}/model"
            )
            np.testing.assert_array_equal(
                model.predict(frame.iloc[:100][FEATURES]),
                downloaded.predict(frame.iloc[:100][FEATURES]),
            )
            with self.assertRaises(FileExistsError):
                run(paths["users"], paths["baseline"], paths["validation"],
                    paths["test"], output, root / "mlruns", ks=[2, 3])

    def test_archived_mlflow_paths_are_rebased_to_download_location(self):
        import yaml
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "mlruns"
            exp = root / "123"
            run_path = exp / ("a" * 32)
            (run_path / "artifacts").mkdir(parents=True)
            (exp / "meta.yaml").write_text(
                yaml.safe_dump({"artifact_location": "file:///old/path", "experiment_id": "123"})
            )
            (run_path / "meta.yaml").write_text(
                yaml.safe_dump({"artifact_uri": "file:///old/run", "run_id": "a" * 32})
            )
            relocate(root)
            self.assertEqual(yaml.safe_load((run_path / "meta.yaml").read_text())["artifact_uri"],
                             (run_path / "artifacts").resolve().as_uri())
            self.assertEqual(yaml.safe_load((exp / "meta.yaml").read_text())["artifact_location"],
                             exp.resolve().as_uri())


if __name__ == "__main__":
    unittest.main()
