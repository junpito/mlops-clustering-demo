# Source and adaptations

This demo reuses the approach and source data from the user-supplied archive
`2026072313213913967c4982b546519ad7e5ed66d1fde2`.

## Clustering reference

`BMLP_Himawan-Prastawa/[Clustering]_Submission_Akhir_BMLP_Himawan_Prastawa.ipynb`
provides the transaction dataset URL, StandardScaler/KMeans approach, silhouette
evaluation, PCA visualizations, and cluster profiling.

Adaptation: aggregate transactions by AccountID before clustering; use numeric
behavior features; preserve scaler together with model; fixed validation/test
users; record all candidates in MLflow. Original `.h5` files are joblib exports,
not used by this demo. Prior cluster `Target` labels are not features.

Dataset source as recorded by that notebook:
https://docs.google.com/spreadsheets/d/e/2PACX-1vTbg5WVW6W3c8SPNUGc3A3AL-AG32TPEQGpdzARfNICMsLFI0LQj0jporhsLCeVhkN5AoRsTkn08AYl/pub?output=csv

The exact downloaded content hash, data cleaning counts, split policy, and
initial snapshot hashes are recorded in `data/manifest.json`. AccountID is a
dataset identifier, not a verified real-world identity. IP addresses and device
identifiers are not included in the packaged snapshots. The raw downloaded CSV
is ignored by Git and excluded from the deliverable ZIP.

No redistribution license was supplied in the archive; this attribution is not
a license grant. Check source permissions before publishing a public copy.

## MLflow and workflow reference

`SMSML_Vika-Putri-Ariyanti/Membangun_model/modelling.py`,
`modelling_tuning.py`, and `Workflow-CI.txt`.

Workflow linked by that file:
https://github.com/Vputri/Workflow-CI/blob/main/.github/workflows/ci.yml

Training reference:
https://github.com/Vputri/Workflow-CI/blob/main/MLProject/modelling.py

Adaptation: retain local MLflow tracking + GitHub artifact upload; replace
RandomForest classification with clustering experiments; add dataset path
trigger, data contracts, quality gate, inference check, and portable archive UI.
Docker publishing and production deployment are deliberately not part of this demo.
