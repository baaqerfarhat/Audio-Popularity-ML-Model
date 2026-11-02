# Final pipeline — what changed and why

`train.py` is self-contained (needs pandas, numpy, scipy, scikit-learn, lightgbm, xgboost, catboost, matplotlib).

```
cd final_pipeline
python train.py --data-dir ../data --out results --seeds 0,1,2,3,4        # ~10 min on 2 CPU cores
python train.py --task-type GPU ...                                        # CatBoost on GPU (optional)
```

Outputs in `results/`: `submission_blend.csv` (recommended), `submission_equal_rank_avg.csv`,
`submission_<model>.csv` per model, `oof_predictions.csv`, `fold_metrics.csv`,
`feature_importance_<model>.csv`, `roc_oof.png`, `summary.json`.

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
   space, both computed strictly from each fold's training part (inner 5-fold, grouped by uri). Worth ≈+0.004.
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

## Release-period specialists (integrated 2026-11-02, off by default)

`train.py --specialists lgb_tuned` additionally trains one LightGBM per release period (year < 2020 / >= 2020,
same hyperparameters; label-derived features rebuilt from the period's own training rows with grouped inner
cross-fitting; unknown year -> global) and writes `<model>__specialist` (routed) and `<model>__mix`
(0.5 global + 0.5 specialist) variants. `--feature-set v2 --et-config base` reproduce the 0.870-public
configuration exactly. `evaluate_period_specialists.py` makes the paired comparison;
`test_period_specialists.py` covers the 2020 boundary, missing-date fallback, routing, group separation and
output alignment (`python -m unittest test_period_specialists -v`).

Matched result on the 10-fold x 5-seed pipeline (`experiments/period_specialists_10fold_2026_11_02/RESULTS.md`):
the 50/50 mix lifts the LightGBM component by +0.0009 weighted OOF AUC (4/5 seeds; +0.0020 on 2020-onward
tracks, -0.0012 on older tracks, and the recent-High-vs-old-Low pair block gets worse), specialists alone are
-0.0027 (0/5). With the four ensemble weights held fixed at 0.25 the gain shrinks to +0.0002 (mix) /
+0.0001 (specialist alone). That is far inside leaderboard noise, so the baseline stays the default;
`submission_fixed_blend_mix.csv` is kept as a candidate. Exploratory: validation splits were reused.
