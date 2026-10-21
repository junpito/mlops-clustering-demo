"""Rebase downloaded MLflow file-store metadata before opening its local UI."""
import argparse
from pathlib import Path
import subprocess
import sys

import yaml


def relocate(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for experiment in root.iterdir():
        if not experiment.is_dir() or not experiment.name.isdigit():
            continue
        files = [(experiment / "meta.yaml", "artifact_location", experiment)]
        files.extend(
            (run / "meta.yaml", "artifact_uri", run / "artifacts")
            for run in experiment.iterdir() if run.is_dir() and (run / "meta.yaml").is_file()
        )
        for path, key, target in files:
            if path.is_file():
                metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
                metadata[key] = target.as_uri()
                path.write_text(yaml.safe_dump(metadata), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", default="mlruns")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--repair-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.tracking).resolve()
    relocate(root)
    if not args.repair_only:
        subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri",
                        root.as_uri(), "--host", "127.0.0.1", "--port", str(args.port)], check=True)
