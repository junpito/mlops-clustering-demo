# Data-triggered clustering demo

**Goal:** Reuse the supplied BMLP KMeans project and SMSML workflow pattern to
demonstrate new users triggering GitHub Actions and producing MLflow artifacts.

**Architecture:** Standalone demo folder, copied into its own repository root
when published. Account-level features from the notebook's public transaction
source; fixed validation/test users and a separate incoming batch. GitHub Actions
reacts to committed `data/users.csv`, never to an uncommitted local edit.
MLflow file tracking is packaged with each workflow run; no external service or
automatic production deployment.

**Stack:** Python 3.12, pandas, scikit-learn, MLflow, unittest, GitHub Actions.

**Agreed design:** Prior chat approves execution of a CSV/push-triggered KMeans
demo; reuse Downloads project; integration team owns production.

## Constraints and decisions

- Preserve source Downloads and retrieval application; only write this demo.
- Do not publish a repository or data without a destination and authorization.
- Preserve attribution to both supplied projects and the source dataset.
- Aggregate real AccountIDs, never use IDs or previous cluster labels as features.
- Pin a baseline training snapshot and fixed validation/test sets.
- Score all candidate partitions in a common feature space fitted on baseline
  training data. Choose candidates on validation, reserve test for final gate.
- New rows create a candidate; neither passing CI nor a better score automatically
  replaces a production model.
- Upload MLflow logs and diagnostics even if the quality gate rejects the model.

## Tasks

- [x] Prepare data snapshot and deterministic account splits; record provenance.
- [x] Write behavior tests for unique user append, invalid data rejection,
  disjoint splits, model roundtrip, and quality gate.
- [x] Implement account feature pipeline, candidate runs, comparison, reports,
  and portable saved preprocessing/model.
- [x] Implement idempotent batch append CLI, prediction CLI, MLflow archive viewer.
- [x] Add workflow, dependency pins, and Indonesian walkthrough.
- [x] Run all demo tests and both baseline/expanded end-to-end experiments;
  verify downloaded-style artifact restoration and workflow YAML.

## Review focus

Duplicate/new IDs, nonfinite features, overlap between training and evaluation,
stale artifacts after a failed run, and portability of local MLflow file URIs.

## Execution notes

Implementation proceeds inline as requested. No worktree is needed for a new
standalone deliverable in `outputs/`; no existing tracked source is edited.
