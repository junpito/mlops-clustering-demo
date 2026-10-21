"""Assign users to an existing candidate model; never fit/retrain at inference."""
import argparse
from pathlib import Path

import mlflow.sklearn

from pipeline import FEATURES, load_users

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Trusted model/ folder from this demo")
    parser.add_argument("--data", default="data/incoming_users.csv")
    parser.add_argument("--output", default="results/predictions.csv")
    args = parser.parse_args()
    users = load_users(args.data)
    model = mlflow.sklearn.load_model(args.model)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a new output file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    users.assign(cluster=model.predict(users[FEATURES]))[["user_id", "cluster"]].to_csv(output, index=False)
    print(f"Assigned {len(users)} users without retraining: {output}")
