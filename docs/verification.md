# Verification record

Final local checks: 21 October 2026, Asia/Jakarta. Python 3.12, Windows.

- `python -m unittest discover -s tests -v`: 9 tests passed. Includes actual
  MLflow model logging, pipeline roundtrip, and model load via run URI after
  copying/rebasing a tracking directory (download simulation).
- `python -m pip check`: no broken requirements.
- `predict.py`: assigned all 99 incoming users from the saved expanded model
  without fitting again; output in local `results/verified-assignments.csv`.
- Workflow YAML parsed successfully; push filter includes `data/users.csv`.
- Independent source review found no actionable defects.

## Real-data experiments

Baseline-data experiment: 296 training users, selected k=4/seed=43.
Test silhouette 0.1559, baseline reference 0.1358, validation seed ARI 0.9055.
Rejected because the smallest test cluster contained 4%, below the preset 5%.
Run: `933812ab928a4af0abc3578a523f7439`.

Expanded-data experiment: 395 training users, selected k=3/seed=42.
Test silhouette 0.1766, baseline reference 0.1358, validation seed ARI 1.0000,
minimum test cluster fraction 20%. All demonstration gates passed.
Run: `4bf56e10799344c8a4a17caf0886414d`.

Both experiments used the same 49 validation and 50 test users, and the same
baseline-fitted distance space. Original `data/users.csv` still contains 296
users so the user can demonstrate the append/commit trigger. Expanded training
used `results/simulation/users.csv`.

## Limits

GitHub Actions has not run remotely: no destination repository was supplied.
Local YAML parsing is not a GitHub runner execution. No push, publication, or
production deployment was performed. The source ZIP must be unpacked into a
separate repository root for workflow discovery.

MLflow 2.19 emits a non-fatal Python datetime deprecation warning. Test/model
operations passed. Technical gate success is not proof of business usefulness.
The full retrieval-Manirent test suite was not run: this standalone deliverable
does not edit its application code or dependencies.
