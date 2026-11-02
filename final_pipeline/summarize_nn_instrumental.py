"""Verify saved NN predictions and generate a compact experiment report."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_bias_variance import metrics


def main():
    here=Path(__file__).resolve().parent
    out=here/'experiments/nn_instrumental_2026_11_01'
    metadata=json.loads((out/'metadata.json').read_text())
    metadata['final_runner_sha256']=hashlib.sha256((here/'nn_instrumental.py').read_bytes()).hexdigest()
    metadata['postprocessing_note']='Final runner corrects submission ID formatting and exports fixed test blends; fitting code is unchanged from the recorded run-time hash. Report script regenerates these exports from saved predictions.'
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    d=np.load(out/'predictions.npz'); seeds=pd.read_csv(out/'seed_metrics.csv')
    folds=pd.read_csv(out/'fold_metrics.csv'); pairs=pd.read_csv(out/'pairwise_auc.csv')
    membership=pd.read_csv(out/'split_membership.csv')
    for _,frame in membership.groupby(['seed','fold']):
        sets={role:set(part.uri) for role,part in frame.groupby('role')}
        assert not sets['outer_valid'] & (sets['inner_train']|sets['inner_valid'])
        assert not sets['inner_train'] & sets['inner_valid']
    for (seed,model),frame in pairs.groupby(['seed','model']):
        score=np.average(frame.auc,weights=frame.pair_mass)
        expected=seeds.query('seed==@seed and model==@model and group=="overall"').weighted_auc.iloc[0]
        assert abs(score-expected)<1e-10
    mask=d['mask']==0
    np.testing.assert_array_equal(d['instrumental_head_oof'][:,mask],d['shared_general'][:,mask])
    test=pd.read_csv(here.parent/'data/test.csv')
    base_dir=here/'experiments/knn_musical_2026_11_01/audio'
    rank=lambda x:rankdata(x)/len(x)
    trees=[pd.read_csv(base_dir/f'submission_{v}.csv') for v in ['lgb_tuned','extra_trees','cat_base','xgb_tuned']]
    assert all(np.array_equal(t.ID,test.ID) for t in trees)
    bt=np.mean([rank(t.Popularity_Type) for t in trees],axis=0)
    for v in ['global','instrumental_head']:
        p=d[v+'_test'].mean(0)
        assert p.shape==(len(test),) and np.isfinite(p).all()
        pd.DataFrame({'ID':test.ID,'Popularity_Type':p}).to_csv(out/f'submission_{v}.csv',index=False)
        pd.DataFrame({'ID':test.ID,'Popularity_Type':.9*bt+.1*rank(p)}).to_csv(out/f'submission_trees_plus_{v}.csv',index=False)
    summary=seeds.groupby(['model','group'])[['auc','weighted_auc','logloss','brier','high_miss_rate']].mean()
    gaps=folds.groupby('model')[['train_auc','validation_auc','epochs']].mean()
    gaps['auc_gap']=gaps.train_auc-gaps.validation_auc
    gaps.to_csv(out/'training_gaps.csv')
    paired=seeds.pivot(index=['seed','group'],columns='model',values='weighted_auc')
    paired['delta']=paired.instrumental_head-paired['global']; paired.to_csv(out/'paired_seed_deltas.csv')
    ensemble=pd.read_csv(out/'ensemble_metrics.csv')
    ensemble['weighted_auc_delta']=ensemble.weighted_auc-ensemble.weighted_auc.iloc[0]
    ensemble.to_csv(out/'ensemble_metrics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for group,color in [('overall','#2460a8'),('instrumental','#c96b28')]:
        part=paired.xs(group,level='group')
        axes[0].plot(part.index,part.delta,marker='o',label=group,color=color)
    axes[0].axhline(0,color='gray',lw=1); axes[0].set(xlabel='Split seed',ylabel='Head NN − global NN weighted AUC',xticks=range(5))
    axes[0].legend(); axes[0].set_title('Does the instrumental head help?')
    axes[1].scatter(['Trees','Trees +\nglobal NN','Trees +\nhead NN'],ensemble.weighted_auc,c=['#999999','#2460a8','#c96b28'],s=70)
    axes[1].set(ylim=(min(ensemble.weighted_auc)-.002,max(ensemble.weighted_auc)+.001),ylabel='Weighted AUC of seed-averaged rank blend')
    axes[1].set_title('Fixed 90% trees + 10% neural network')
    fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)
    delta=paired.xs('overall',level='group').delta
    report=f'''# Global and instrumental-head neural networks

Recommendation: retain the existing tree ensemble. The instrumental head improves its subgroup on average, but worsens overall weighted AUC in all five seeds. Neither fixed 10% NN blend improves the tree baseline. This is evidence against adopting this configuration, not against every possible neural architecture.

Both networks use a 64/32 ReLU backbone, dropout 0.1 and AdamW (learning rate 0.001, weight decay 0.0001). The specialist adds a 32/8/1 residual branch with a zero-initialized last layer. Instrumentalness >= 0.5 activates it; missing values use the general head.

The general BCE loss uses all rows. The specialist adds 0.25 times instrumental routed BCE plus 0.01 times mean squared instrumental residual logits. Each minibatch averages its own eligible rows. No class weighting or oversampling. Settings were fixed before examining results.

Five seeds, five URI-grouped outer folds per seed, with a separate grouped inner split selecting epochs (maximum 150, patience 15, minimum logloss improvement 0.0001). Each model is then freshly trained on all outer-training rows for its selected epoch count: 50 selection fits and 50 final fits. Outer labels never select epochs. Identical conflicting-label tracks remain grouped and retain their labels.

All imputation, scaling and category learning use permitted training rows only. Date target encodings and audio-only same-date KNN label features use five-fold URI-grouped cross-fitting for training rows. Validation/test features use the corresponding training references only. Musical attributes are one-hot NN inputs. File-level track/date counts are omitted. These preprocessing differences mean comparisons with older trees are not architecture-only comparisons.

## Mean per-seed OOF results

```
{summary.to_string(float_format=lambda x:f'{x:.6f}')}
```

Overall head-minus-global weighted AUC: {delta.mean():+.6f}; wins in {int((delta>0).sum())}/5 seeds. These overlapping repeated splits are not five independent experiments. Results are exploratory because the outer splits have been reused. High miss rate uses a fixed 0.5 threshold and is secondary to ranking quality.

## Training gaps

```
{gaps.to_string(float_format=lambda x:f'{x:.6f}')}
```

Training AUC uses the cross-fitted training features seen during fitting; validation uses full training references. These gaps are descriptive, not a formal bias–variance decomposition.

## Fixed ensemble comparison

```
{ensemble.to_string(index=False,float_format=lambda x:f'{x:.6f}')}
```

The existing four tree models retain equal rank weights. Average probabilities over seeds first, rank each model, then blend 90% existing tree ranks and 10% NN ranks. No weight search. This aggregation differs from mean per-seed AUC above; rank blends are ranking scores, not calibrated probabilities. Existing default models and submissions are unchanged.

Artifacts: predictions.npz contains per-seed OOF/test arrays, labels, weights, source rows and shared-head components; CSV files contain row-aligned predictions, split memberships, all epoch losses, paired deltas and four positive/negative subgroup AUC contributions. Submission IDs match the original test IDs. Model weights are not persisted; rerun the script to refit.

Reproduce from repository root with `.venv/Scripts/python.exe final_pipeline/nn_instrumental.py --out <fresh-directory>`. Dependencies are pinned in `final_pipeline/nn-requirements-lock.txt`; install CPU torch from https://download.pytorch.org/whl/cpu. The report script targets the dated experiment directory. Four focused unit tests and post-run checks cover group separation, query-label/statistic isolation, prediction ordering, zero-initialized residuals, masked routing, gradients, finite outputs and pairwise AUC reconciliation.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(summary.to_string()); print(ensemble.to_string(index=False)); print(gaps.to_string())


if __name__=='__main__':
    main()
