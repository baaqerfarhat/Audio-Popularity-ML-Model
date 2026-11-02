"""Verify and summarize the fixed era-feature experiment."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    here=Path(__file__).resolve().parent; out=here/'experiments/era_features_2026_11_01'
    d=np.load(out/'predictions.npz'); s=pd.read_csv(out/'seed_metrics.csv')
    summary=pd.read_csv(out/'summary.csv'); paired=pd.read_csv(out/'paired_deltas.csv')
    blend=pd.read_csv(out/'fixed_blend_comparison.csv'); diag=pd.read_csv(out/'feature_diagnostics.csv')
    metadata=json.loads((out/'metadata.json').read_text())
    for path,digest in metadata['sha256'].items():
        assert hashlib.sha256((here.parent/path).read_bytes()).hexdigest()==digest,path
    for row in s.itertuples():
        p=d['oof_'+row.candidate][row.seed]
        assert abs(roc_auc_score(d['y'],p,sample_weight=d['weights'])-row.weighted_auc)<1e-12
    raw=pd.read_csv(here.parent/'data/train.csv'); clean=raw.dropna(subset=['uri'])
    np.testing.assert_array_equal(d['source_row'],clean.index)
    members=pd.read_csv(out/'fold_membership.csv')
    for _,part in members.groupby('seed'):
        assert len(part)==len(clean) and part.source_row.nunique()==len(clean)
        assert part.groupby('uri').fold.nunique().max()==1
    test=pd.read_csv(here.parent/'data/test.csv')
    for path in out.glob('submission_*.csv'):
        frame=pd.read_csv(path); np.testing.assert_array_equal(frame.ID,test.ID)
        assert np.isfinite(frame.Popularity_Type).all() and frame.Popularity_Type.between(0,1).all()
    importance=pd.read_csv(out/'feature_importance.csv').groupby('feature').gain.mean().sort_values(ascending=False)
    importance.to_csv(out/'mean_feature_importance.csv')
    base=blend.set_index('candidate').loc['baseline','weighted_auc']
    gain=blend.set_index('candidate').loc['era_percentiles','weighted_auc']-base
    fallback=diag.groupby('role')[['fallback_rows','rows']].sum()
    fallback['fallback_fraction']=fallback.fallback_rows/fallback.rows
    verdict='Keep as an exploratory candidate; do not change production defaults.' if paired.delta.mean()>0 and gain>0 else 'Do not promote this feature set to the default pipeline.'
    report=f'''# Era-relative audio features — November 1, 2026

{verdict}

Added exactly four features: within-release-decade percentiles of loudness, duration, energy and acousticness. Calendar decades and a 30-reference-track minimum were fixed before evaluation. Each reference contains distinct URIs from the outer training fold only; the query URI is excluded even for training rows. Percentiles use half credit for ties. Unknown/sparse decades use the global training reference. Missing audio stays NaN. No labels or test statistics enter the new features.

Both variants retain the current LightGBM settings and original features. Existing transductive counts/scaling and the training KNN zero-distance exclusion remain unchanged, so this is an isolated feature addition, not a preprocessing overhaul. Five seeds, five grouped folds, 50 fits. Conflicting labels remain unchanged and grouped by URI. No boundary, feature-set, or hyperparameter search.

## Mean per-seed OOF metrics

```
{summary.to_string(index=False,float_format=lambda x:f'{x:.6f}')}
```

Mean weighted AUC change: **{paired.delta.mean():+.6f}**. Improved **{int((paired.delta>0).sum())}/5 seeds**; range {paired.delta.min():+.6f} to {paired.delta.max():+.6f}.

```
{paired.to_string(index=False,float_format=lambda x:f'{x:.6f}')}
```

## Fixed ensemble check

Equal rank weights on LightGBM, ExtraTrees, base CatBoost and tuned XGBoost; only LightGBM changes. Average each model's probabilities over seeds before ranking. This is a different aggregation from mean per-seed AUC above. Rank scores are not calibrated probabilities; their logloss/Brier exports should not be interpreted as calibration measurements.

```
{blend.to_string(index=False,float_format=lambda x:f'{x:.6f}')}
```

Weighted ensemble change: **{gain:+.6f}**. Existing default parameters and submissions remain unchanged. These reused observations make the results exploratory; five seeds are not independent replications, and a small gain does not establish improvement on untouched data.

## Reference coverage

Counts below sum all 25 fold transformations, so rows repeat across fits.

```
{fallback.to_string(float_format=lambda x:f'{x:.6f}')}
```

Baseline OOF and test predictions reproduce the previous regularization baseline to 1e-12. All input/source hashes, per-seed weighted AUCs, source-row alignment, URI grouping, finite predictions and submission IDs were checked. Three unit tests cover decade boundaries, self-exclusion, deduplication, ties, fallback, missing audio and query-label/statistic isolation.

Run from repository root: `.venv/Scripts/python.exe final_pipeline/try_era_features.py --out <fresh-directory>`. Test with `.venv/Scripts/python.exe -m unittest discover -s final_pipeline -p test_era_features.py -v`. Dependencies remain in `final_pipeline/requirements-lock.txt`. The summary script targets this dated directory. Saved arrays retain every seed's OOF/test predictions; CSVs include folds, diagnostics, feature importance, scores, paired deltas and candidate submissions.
'''
    (out/'README.md').write_text('\n'.join(line.rstrip() for line in report.splitlines())+'\n',encoding='utf-8')
    validation=json.loads((out/'validation.json').read_text()); validation.update(source_hashes_verified=True,all_seed_auc_recomputed=True,original_row_alignment_verified=True,all_submission_ids_verified=True)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(summary.to_string(index=False)); print(paired.to_string(index=False)); print(blend.to_string(index=False)); print(fallback.to_string())


if __name__=='__main__':
    main()
