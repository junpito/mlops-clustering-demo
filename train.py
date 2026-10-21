"""Train candidate user clusters, record MLflow runs, and produce a handoff."""
import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from pipeline import FEATURES, build_model, check_gate, evaluate, load_users, validate_splits


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(data, baseline, validation, test, output, tracking, ks=None):
    output, tracking = Path(output), Path(tracking).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    ks = ks or [2, 3, 4]
    base, train, valid, heldout = [load_users(p) for p in [baseline, data, validation, test]]
    validate_splits(base, train, valid, heldout)
    if min(len(base), len(valid), len(heldout)) <= max(ks):
        raise ValueError("Each split needs more users than the largest cluster count")
    output.mkdir(parents=True)
    tracking.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking.as_uri())
    mlflow.set_experiment("user-clustering-demo")
    pip_requirements = [
        f"{name}=={version(name)}" for name in
        ["mlflow", "scikit-learn", "pandas", "numpy", "scipy", "cloudpickle"]
    ]
    hashes = {name: sha256(path) for name, path in [
        ("baseline", baseline), ("train", data), ("validation", validation), ("test", test)
    ]}
    report = {
        "counts": {"baseline_users": len(base), "training_users": len(train),
                   "new_users": len(train) - len(base), "validation_users": len(valid),
                   "test_users": len(heldout)},
        "dataset_sha256": hashes,
        "code_commit": os.getenv("GITHUB_SHA", "local-uncommitted"),
        "python": platform.python_version(),
        "evaluation_space": "StandardScaler fitted on baseline users; common to all candidates",
        "gate_thresholds": {"min_silhouette": .05, "max_regression": .02,
                            "min_cluster_fraction": .05, "min_seed_ari": .60},
        "candidates": [],
    }
    models = {}
    # Fixed thread count bounds this small CPU demo on both laptops and CI runners.
    with threadpool_limits(limits=1), mlflow.start_run(run_name=f"users-{len(train)}") as parent:
        report["run_id"] = parent.info.run_id
        mlflow.log_params({**report["counts"], **{f"sha256_{k}": v for k, v in hashes.items()}})
        mlflow.set_tags({"code_commit": report["code_commit"], "purpose": "staging-demo"})
        reference = StandardScaler().fit(base[FEATURES])
        baseline_model = build_model(3, 42).fit(base[FEATURES])
        baseline_test = evaluate(baseline_model, heldout, reference)
        with mlflow.start_run(run_name="frozen-baseline-k3-seed42", nested=True) as baseline_run:
            report["baseline_run_id"] = baseline_run.info.run_id
            mlflow.log_params({"k": 3, "seed": 42, "training_users": len(base)})
            mlflow.log_metrics({f"test_{k}": v for k, v in baseline_test.items()})
            mlflow.sklearn.log_model(
                baseline_model, "model", pip_requirements=pip_requirements,
                input_example=base[FEATURES].head(3),
                signature=mlflow.models.infer_signature(base[FEATURES], baseline_model.predict(base[FEATURES])),
            )

        for k in ks:
            for seed in [42, 43]:
                with mlflow.start_run(run_name=f"k{k}-seed{seed}", nested=True) as child:
                    model = build_model(k, seed).fit(train[FEATURES])
                    metrics = evaluate(model, valid, reference)
                    candidate = {"k": k, "seed": seed, "run_id": child.info.run_id,
                                 **{f"validation_{name}": value for name, value in metrics.items()}}
                    report["candidates"].append(candidate)
                    models[(k, seed)] = model
                    mlflow.log_params({"k": k, "seed": seed, "n_init": 10,
                                       "training_users": len(train), "data_sha256": hashes["train"]})
                    mlflow.log_metrics({f"validation_{name}": value for name, value in metrics.items()})
                    mlflow.sklearn.log_model(
                        model, "model", pip_requirements=pip_requirements,
                        input_example=train[FEATURES].head(3),
                        signature=mlflow.models.infer_signature(train[FEATURES], model.predict(train[FEATURES])),
                    )

        eligible = [c for c in report["candidates"]
                    if c["validation_observed_clusters"] == c["k"]
                    and c["validation_min_cluster_fraction"] >= .05]
        # If all fail the validation screen, retain a diagnostic candidate but reject the release.
        selected = max(eligible or report["candidates"], key=lambda c: c["validation_silhouette"])
        best = models[(selected["k"], selected["seed"])]
        stability = float(adjusted_rand_score(
            models[(selected["k"], 42)].predict(valid[FEATURES]),
            models[(selected["k"], 43)].predict(valid[FEATURES]),
        ))
        test_metrics = evaluate(best, heldout, reference)
        reasons = check_gate(test_metrics, baseline_test["silhouette"], stability, selected["k"])
        if not eligible:
            reasons.append("No candidate passed validation cluster coverage/size screen")
        report.update({
            "selected": selected, "baseline_test": baseline_test, "selected_test": test_metrics,
            "seed_stability_ari": stability, "gate_failures": reasons,
            "status": "REJECTED" if reasons else "READY_FOR_INTEGRATION",
            "production_deployed": False,
        })

        mlflow.sklearn.save_model(
            best, str(output / "model"), pip_requirements=pip_requirements,
            input_example=train[FEATURES].head(3),
            signature=mlflow.models.infer_signature(train[FEATURES], best.predict(train[FEATURES])),
        )
        restored = mlflow.sklearn.load_model(str(output / "model"))
        np.testing.assert_array_equal(
            best.predict(heldout[FEATURES]), restored.predict(heldout[FEATURES])
        )
        report["model_reload_verified"] = True
        assigned = train.assign(cluster=best.predict(train[FEATURES]))
        assigned[["user_id", "cluster"]].to_csv(output / "assignments.csv", index=False)
        assigned.groupby("cluster")[FEATURES].agg(["count", "mean", "min", "max"]).to_csv(
            output / "cluster_profiles.csv"
        )
        pca = PCA(n_components=2).fit(reference.transform(base[FEATURES]))
        xy = pca.transform(reference.transform(train[FEATURES]))
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=assigned.cluster, s=15, cmap="viridis")
        ax.set(xlabel="PC 1", ylabel="PC 2", title="User clusters (PCA for visualization only)")
        fig.colorbar(scatter, ax=ax, label="Cluster ID (local to this model)")
        fig.tight_layout()
        fig.savefig(output / "clusters.png", dpi=130)
        plt.close(fig)
        contract = {
            "input": {"unit": "one user/account", "id": "user_id (excluded from model)",
                      "features": FEATURES, "rules": "finite, nonnegative; positive integer transaction_count"},
            "output": "user_id, cluster",
            "cluster_ids": "Unordered labels local to one model version; not stable business segment names.",
            "artifact": "model/ contains StandardScaler + KMeans and environment requirements",
        }
        (output / "input_output_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        summary = (
            f"# Clustering demo: {report['status']}\n\n"
            f"- Training users: {len(train)} ({len(train)-len(base)} new versus baseline).\n"
            f"- Selected: k={selected['k']}, seed={selected['seed']} using validation only.\n"
            f"- Test silhouette: {test_metrics['silhouette']:.4f}; "
            f"baseline: {baseline_test['silhouette']:.4f} (same test users and distance space).\n"
            f"- Seed stability on validation (ARI): {stability:.4f}.\n"
            f"- Test minimum cluster fraction: {test_metrics['min_cluster_fraction']:.4f}.\n"
            f"- Model reload verified: true. Production deployed: false.\n"
            f"- MLflow parent run: `{report['run_id']}`.\n\n"
            "## Gate failures\n\n" + ("\n".join(f"- {r}" for r in reasons) or "None.") +
            "\n\nThis is a technical demonstration. Business usefulness needs domain review. "
            "Repeated evaluation on this fixed test set is not a substitute for future unseen data.\n"
        )
        (output / "summary.md").write_text(summary, encoding="utf-8")
        mlflow.log_metrics({
            "selected_test_silhouette": test_metrics["silhouette"],
            "baseline_test_silhouette": baseline_test["silhouette"],
            "seed_stability_ari": stability, "quality_gate_passed": int(not reasons),
        })
        mlflow.set_tag("release_status", report["status"])
        mlflow.log_artifacts(str(output), artifact_path="handoff")
    print(summary)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/users.csv")
    parser.add_argument("--baseline", default="data/baseline_users.csv")
    parser.add_argument("--validation", default="data/validation_users.csv")
    parser.add_argument("--test", default="data/test_users.csv")
    parser.add_argument("--output", default="results/current")
    parser.add_argument("--tracking", default="mlruns")
    args = parser.parse_args()
    report = run(args.data, args.baseline, args.validation, args.test, args.output, args.tracking)
    raise SystemExit(0 if report["status"] == "READY_FOR_INTEGRATION" else 2)
