# Era-relative audio features — November 1, 2026

Keep as an exploratory candidate; do not change production defaults.

Added exactly four features: within-release-decade percentiles of loudness, duration, energy and acousticness. Calendar decades and a 30-reference-track minimum were fixed before evaluation. Each reference contains distinct URIs from the outer training fold only; the query URI is excluded even for training rows. Percentiles use half credit for ties. Unknown/sparse decades use the global training reference. Missing audio stays NaN. No labels or test statistics enter the new features.

Both variants retain the current LightGBM settings and original features. Existing transductive counts/scaling and the training KNN zero-distance exclusion remain unchanged, so this is an isolated feature addition, not a preprocessing overhaul. Five seeds, five grouped folds, 50 fits. Conflicting labels remain unchanged and grouped by URI. No boundary, feature-set, or hyperparameter search.

## Mean per-seed OOF metrics

```
      candidate      auc  weighted_auc  logloss    brier
       baseline 0.856664      0.840284 0.439530 0.143570
era_percentiles 0.856971      0.840803 0.439458 0.143423
```

Mean weighted AUC change: **+0.000520**. Improved **4/5 seeds**; range -0.000673 to +0.001831.

```
 seed  baseline  era_percentiles     delta
    0  0.841938         0.842489  0.000552
    1  0.839487         0.840049  0.000562
    2  0.840811         0.842642  0.001831
    3  0.842108         0.841434 -0.000673
    4  0.837077         0.837404  0.000327
```

## Fixed ensemble check

Equal rank weights on LightGBM, ExtraTrees, base CatBoost and tuned XGBoost; only LightGBM changes. Average each model's probabilities over seeds before ranking. This is a different aggregation from mean per-seed AUC above. Rank scores are not calibrated probabilities; their logloss/Brier exports should not be interpreted as calibration measurements.

```
      candidate      auc  weighted_auc  logloss    brier
       baseline 0.863475      0.847765 0.500715 0.167564
era_percentiles 0.863655      0.848012 0.500609 0.167498
```

Weighted ensemble change: **+0.000248**. Existing default parameters and submissions remain unchanged. These reused observations make the results exploratory; five seeds are not independent replications, and a small gain does not establish improvement on untouched data.

## Reference coverage

Counts below sum all 25 fold transformations, so rows repeat across fits.

```
            fallback_rows   rows  fallback_fraction
role
test                  100  21575           0.004635
train                 140  77260           0.001812
validation             35  19315           0.001812
```

Baseline OOF and test predictions reproduce the previous regularization baseline to 1e-12. All input/source hashes, per-seed weighted AUCs, source-row alignment, URI grouping, finite predictions and submission IDs were checked. Three unit tests cover decade boundaries, self-exclusion, deduplication, ties, fallback, missing audio and query-label/statistic isolation.

Run from repository root: `.venv/Scripts/python.exe final_pipeline/try_era_features.py --out <fresh-directory>`. Test with `.venv/Scripts/python.exe -m unittest discover -s final_pipeline -p test_era_features.py -v`. Dependencies remain in `final_pipeline/requirements-lock.txt`. The summary script targets this dated directory. Saved arrays retain every seed's OOF/test predictions; CSVs include folds, diagnostics, feature importance, scores, paired deltas and candidate submissions.
