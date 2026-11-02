# Final pipeline — what changed and why

The [November 1 bias–variance diagnostics](experiments/bias_variance_2026_11_01/README.md)
compare training and validation scores, learning curves, and prediction stability.
They preserve the production model settings and existing submissions.

The follow-up [gentle regularization comparison](experiments/gentle_regularization_2026_11_01/README.md)
tests six single-parameter changes over five seeds. L2=0.1 is a tentative candidate;
its fixed-blend gain is tiny, so the default configuration remains unchanged.

The [L1 and combined L1 + L2 follow-up](experiments/l1_l2_regularization_2026_11_01/README.md)
tests four additional penalty settings. None beat the L2-only candidate, and all
four slightly lowered the fixed-weight blend score relative to the baseline.

The [track-error analysis](experiments/track_errors_2026_11_01/README.md) identifies
407 consistently misclassified unique tracks, release-period and instrumentalness
patterns, and 32 separate URIs with identical inputs but conflicting labels.

`train.py` is self-contained (needs pandas, numpy, scipy, scikit-learn, lightgbm, xgboost, catboost, matplotlib).

```
cd final_pipeline
python train.py --data-dir ../data --out results --seeds 0,1,2,3,4        # ~10 min on 2 CPU cores
python train.py --task-type GPU ...                                        # CatBoost on GPU (optional)
```

Outputs in `results/`: `submission_blend.csv` (recommended), `submission_equal_rank_avg.csv`,
`submission_<model>.csv` per model, `oof_predictions.csv`, `fold_metrics.csv`,
`feature_importance_<model>.csv`, `roc_oof.png`, `summary.json`.

## Musical attributes in same-date KNN (November 1, 2026)

The KNN can now include `key`, `mode`, and `time_signature` alongside the ten standardized
audio features. Each attribute is one-hot encoded and scaled by `1/sqrt(2)`: a mismatch adds
exactly 1 to squared distance, regardless of the numeric category code or category frequency.
Missing values use an unknown category. The exact-release-date candidate restriction, five
neighbors, smoothing, and existing label-feature construction are unchanged. For training
rows, the KNN excludes zero-distance neighbors; only the separate date target encoding uses
inner grouped cross-validation.

Use `--knn-features audio` (the default) to reproduce the old distance calculation, or
`--knn-features audio-musical` for the new one. `--threads` controls CPU threads per model
(default 6). Specify a separate `--out` folder to preserve earlier results.

The paired five-seed experiment and results are in
[`experiments/knn_musical_2026_11_01`](experiments/knn_musical_2026_11_01/README.md).
Its package versions are recorded in `requirements-lock.txt` (Python 3.11 on Windows).
The musical variant did not improve the paired blend score (weighted AUC 0.843093 versus
0.843170 for audio-only), so the original distance remains the default.
From the repository root, run the distance checks with:

```
.venv/Scripts/python.exe -m unittest discover -s final_pipeline -p test_knn_coordinates.py -v
```

The sections below describe the original audio-only experiments, not the new musical-attribute run.

## Audit of the existing code
* Grouped `StratifiedGroupKFold` by `uri`, date parsing, fold-averaged test predictions, submission format: all correct.
  Public scores (0.831 / 0.834 / 0.838) match CV (0.831–0.837), so nothing was broken.
* The gap to the 0.86–0.88 cluster is partly real modelling headroom and partly selection luck:
  the public board is ~345 rows, so each public score has a standard error of about ±0.02 and teams pick
  their best of 8–20 entries.

## Three findings that move the score
1. **`uri_count` – track multiplicity.** 424 of 3,863 training rows are repeated `uri`s (same track, same
   features). Tracks that appear twice are 77% High, three times 96% High, once 30%. This is a
   "how many playlists/charts the track sits in" signal and is computable without labels.
   In CV it is worth +0.017 AUC, but the test set has far fewer duplicates (22 rows vs 11% of train): the TAs did a random
   row split and then removed test rows whose `uri` also occurs in train (963 original test ids, 103 gaps – matches the
   expected number of split duplicate pairs). So the realistic leaderboard gain is ≈+0.006.
2. **Album effect.** Tracks sharing the exact release date are album-mates; their labels agree 71% of the time versus
   63% for random same-year tracks (corr 0.32 vs 0.12). Encoded as (a) an out-of-fold target encoding of the date
   string (+count) and (b) the distance-weighted label mean of the 5 nearest same-date tracks in standardized audio
   space, both computed from each outer fold's training part. The date target encoding uses inner
   5-fold splits grouped by uri; training-row KNN excludes zero-distance neighbors. Worth ≈+0.004.
3. **"Test-like" CV.** Because duplicates are over-represented in train, plain OOF AUC overstates leaderboard AUC.
   We report a weighted OOF AUC that down-weights duplicate rows to the test prevalence (2.5%); it tracks the public
   board far better and is what the blend weights are optimised on.

Other features: day-of-week of release (Sunday releases 15% High, Tuesday 46%), release precision, age, log-transforms,
`date_group_size` (number of distinct tracks sharing the date in train+test), a few audio interactions.
Leak checks done and negative: test `ID` order, Spotify id characters, near-duplicate recordings, numeric-precision/provenance patterns.

## Grouped 5-fold CV, mean over seeds (plain / test-like)
| model | OOF AUC | test-like AUC |
|---|---|---|
| your best single model (CatBoost/LightGBM, date feats) | 0.830 | ≈0.828 |
| your advanced ensemble | 0.837 | – |
| LightGBM, +uri_count +album feats | 0.854 | 0.838 |
| LightGBM tuned (Optuna, 60 trials) | 0.857 | 0.841 |
| XGBoost tuned (45 trials) | 0.856 | 0.841 |
| CatBoost tuned (30 trials) | 0.857 | 0.840 |
| ExtraTrees | 0.853 | 0.836 |
| **rank-average blend** | **0.861** | **0.845** |

Things tried that did not help (so you can list them): training with test-like sample weights, fuzzy-date kNN
target encoding, CatBoost native date categorical, DART, pairwise-ranking objective, SVM / kNN / spline-logistic
(0.81–0.83, too weak to enter the blend), dropping any of the feature groups.

## What to expect
True AUC of the blend ≈0.845. On a 345-row public board that is 0.845 ± 0.02, so anything from 0.83 to 0.87 is
"normal"; 0.90 would need +2.5 standard errors of luck. The final ranking uses the other 60% (518 rows, ±0.017).
Select two uncorrelated-as-possible but strong entries: `submission_blend.csv` and `submission_equal_rank_avg.csv`
(or the best public-scoring single model).
