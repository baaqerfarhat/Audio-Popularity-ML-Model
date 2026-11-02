# Same-date KNN musical-attribute experiment

Adding key, mode, and time signature did not improve the final blend in this paired
five-seed run. Weighted AUC fell by 0.000077, effectively a tie at this precision.
Keep audio-only as the default; the musical option and all submissions remain available.

## Change and comparison

The experimental KNN retains the ten standardized audio features and appends
one-hot encodings of `key`, `mode`, and `time_signature`, scaled by `1/sqrt(2)`.
Each category mismatch contributes 1 to squared Euclidean distance. Missing
attributes receive an unknown category. Category codes do not imply ordering.

Both variants use the same exact-release-date candidate restriction, five neighbors,
distance weights, smoothing, model hyperparameters, and label-feature construction.
They use grouped five-fold cross-validation by `uri`, seeds 0 through 4, and six CPU
threads per model. Both were run locally in the same Python 3.11.6 environment.
There are six model variants and 150 fitted models per run, 300 fits in total.

The new attributes changed the validation KNN label-mean feature for 2,615 of
3,863 rows (67.7%) in seed 0. Of those 3,863 rows, 2,658 had same-date training
candidates. See `feature_diagnostics.json`.

## Results

Scores below are means of the five per-seed OOF AUCs, matching `summary.json`.
They are not AUCs recomputed from the seed-averaged prediction file.

| Metric | Audio only | Audio + musical | Difference |
|---|---:|---:|---:|
| Selected blend OOF AUC | 0.859361 | 0.859335 | -0.000025 |
| Selected blend weighted AUC | 0.843170 | 0.843093 | -0.000077 |
| Equal six-model blend OOF AUC | 0.859189 | 0.859120 | -0.000070 |
| Equal six-model blend weighted AUC | 0.842952 | 0.842859 | -0.000093 |

Both runs selected equal weights of 0.25 on tuned LightGBM, ExtraTrees, base
CatBoost, and tuned XGBoost. XGBoost's weighted AUC increased by 0.000256; all
other individual models decreased slightly. The largest decrease was tuned
CatBoost at -0.000505. Full precision comparisons are in `comparison.csv`.

This is one feature configuration, with no new hyperparameter or distance-weight
search. The blend is selected and scored on the same OOF predictions, as in the
existing pipeline; this is not an independent assessment of blend selection.
The weighted score only adjusts duplicate-track prevalence and is not a measured
leaderboard score. No statistical significance claim is made. The fresh baseline
differs slightly from the historical saved run, so use this paired comparison
to assess the feature change.

## Reproduce

From the repository root, create an isolated Python 3.11 environment and install
the recorded dependencies:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r final_pipeline/requirements-lock.txt
.venv/Scripts/python.exe -m unittest discover -s final_pipeline -p test_knn_coordinates.py -v
```

The runs used these settings (use fresh output paths to preserve this experiment):

```powershell
.venv/Scripts/python.exe -u final_pipeline/train.py --knn-features audio --seeds 0,1,2,3,4 --threads 6 --out final_pipeline/experiments/knn_musical_rerun/audio
.venv/Scripts/python.exe -u final_pipeline/train.py --knn-features audio-musical --seeds 0,1,2,3,4 --threads 6 --out final_pipeline/experiments/knn_musical_rerun/audio_musical
```

`run_metadata.json` records input hashes, the starting Git revision, and run
settings. `audio.log` and `audio_musical.log` contain the full training output.
Each result directory includes model and blend submissions, OOF predictions,
150 fold metrics, feature importances, ROC curves, and a summary.

## Validation

Four distance tests passed: original audio geometry is preserved in audio-only
mode; category mismatches have equal bounded cost; missing and unseen categories
produce finite coordinates; musical matches can change the selected neighbor
without crossing the exact-date boundary.

Both runs exited successfully. All 16 submissions have the expected 863 rows,
correct unique IDs and ordering, and finite predictions between 0 and 1. OOF rows
match the training labels and IDs, all fold metrics are present and finite, and
the recorded input fingerprints are unchanged. See `validation.json`.

The new blend submission is
[`audio_musical/submission_blend.csv`](audio_musical/submission_blend.csv).
The paired baseline is [`audio/submission_blend.csv`](audio/submission_blend.csv).
The original `final_pipeline/results/` artifacts were preserved.
