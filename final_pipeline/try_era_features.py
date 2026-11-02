"""Fixed decade-relative audio percentile experiment; no tuning or default changes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
import train as pipeline
from analyze_bias_variance import metrics

COLUMNS=['loudness','duration_ms','energy','acousticness']
MIN_REFERENCE=30


def decades(df):
    year=pd.to_numeric(df.track_album_release_date.astype(str).str.extract(r'^(\d{4})(?:-|$)',expand=False),errors='coerce')
    return np.floor(year.to_numpy()/10)*10


class EraFeatures:
    """Empirical midrank percentiles, deduplicated by URI and excluding query URI.

    Unknown/sparse decades fall back to the global reference. Missing audio stays
    missing. Nothing is fitted to query rows or popularity labels.
    """
    def fit(self, df):
        self.reference=df.drop_duplicates('uri').copy()
        self.eras=decades(self.reference)
        self.ids=self.reference.uri.to_numpy()
        self.audio=self.reference[COLUMNS].to_numpy(dtype=float)
        return self

    def transform(self, df):
        out=np.full((len(df),len(COLUMNS)),np.nan); fallback=[]; sizes=[]
        for i,(uri,era,values) in enumerate(zip(df.uri,decades(df),df[COLUMNS].to_numpy(dtype=float))):
            eligible=self.ids!=uri
            local=eligible & (self.eras==era)
            use_global=int(local.sum())<MIN_REFERENCE
            reference=self.audio[eligible if use_global else local]
            fallback.append(use_global); sizes.append(len(reference))
            for j,value in enumerate(values):
                if not np.isfinite(value):
                    continue
                ref=reference[:,j]; ref=ref[np.isfinite(ref)]
                if len(ref):
                    out[i,j]=((ref<value).sum()+.5*(ref==value).sum())/len(ref)
        return pd.DataFrame(out,index=df.index,columns=['era_pct_'+c for c in COLUMNS]),np.array(fallback),np.array(sizes)


def main():
    here=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=here/'experiments/era_features_2026_11_01')
    ap.add_argument('--threads',type=int,default=6); args=ap.parse_args()
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'metadata.json').exists():
        raise FileExistsError('Choose a fresh output directory')
    raw=pd.read_csv(here.parent/'data/train.csv'); tr=raw.dropna(subset=['uri']).reset_index(drop=True)
    source_rows=raw.index[raw.uri.notna()].to_numpy(); te=pd.read_csv(here.parent/'data/test.csv')
    y=tr[pipeline.TARGET].eq('High').astype(int).to_numpy(); X,Xt=pipeline.build_static_features(tr,te)
    dup=X.uri_count.to_numpy()>1; weights=np.where(dup,(22/863)/(1-22/863)*((~dup).sum()/dup.sum()),1.)
    params=json.loads((here/'tuned_lgb.json').read_text()); fit=pipeline.lgb_model(params,args.threads)
    sources=[Path(__file__),here/'train.py',here/'analyze_bias_variance.py',here/'tuned_lgb.json',here/'requirements-lock.txt',here.parent/'data/train.csv',here.parent/'data/test.csv']
    metadata=dict(seeds=list(range(5)),folds=5,parameters=params,threads=args.threads,
        features=COLUMNS,era='calendar decade: floor(year / 10) * 10',minimum_reference=MIN_REFERENCE,
        reference='outer training distinct URIs; query URI excluded; sparse/unknown decade uses global training reference',
        missing_audio='NaN',percentile='fraction strictly below + half fraction equal',
        primary_metric='mean per-seed weighted OOF AUC',
        baseline='unchanged audio KNN pipeline, including existing transductive counts/scaling and zero-distance training KNN exclusion',
        ensemble='fixed equal ranks of LightGBM, ExtraTrees, base CatBoost, tuned XGBoost; seed-average then rank',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=here,text=True).strip(),
        sha256={str(p.relative_to(here.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    variants=['baseline','era_percentiles']; oof={v:np.full((5,len(tr)),np.nan) for v in variants}; test={v:np.zeros((5,len(te))) for v in variants}
    folds_log=[]; seed_log=[]; membership=[]; diagnostics=[]; importance=[]
    for seed in range(5):
        folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,y,tr.uri))
        mats=pipeline.fold_matrices(X,Xt,tr,te,y,folds,seed,knn_musical=False)
        for fi,((a,b),(A,B,T)) in enumerate(zip(folds,mats)):
            assert not set(tr.uri.iloc[a]) & set(tr.uri.iloc[b])
            membership.extend(dict(seed=seed,fold=fi,source_row=int(source_rows[i]),uri=tr.uri.iloc[i]) for i in b)
            era=EraFeatures().fit(tr.iloc[a]); added=[]
            for role,frame in [('train',tr.iloc[a]),('validation',tr.iloc[b]),('test',te)]:
                feats,fallback,sizes=era.transform(frame); added.append(feats)
                diagnostics.append(dict(seed=seed,fold=fi,role=role,rows=len(frame),fallback_rows=int(fallback.sum()),min_reference=int(sizes.min()),median_reference=float(np.median(sizes))))
            for v in variants:
                aa,bb,tt=(A,B,T) if v=='baseline' else tuple(pd.concat([frame,extra],axis=1) for frame,extra in zip([A,B,T],added))
                pv,pt,model=fit(aa,y[a],bb,tt,seed*100+fi)
                oof[v][seed,b]=pv; test[v][seed]+=pt/5
                row=dict(candidate=v,seed=seed,fold=fi)
                for prefix,yy,pp,ww in [('train',y[a],model.predict_proba(aa)[:,1],weights[a]),('validation',y[b],pv,weights[b])]:
                    row.update({prefix+'_'+k:val for k,val in metrics(yy,pp,ww).items()})
                folds_log.append(row)
                if v=='era_percentiles':
                    importance.extend(dict(seed=seed,fold=fi,feature=c,gain=float(g)) for c,g in zip(aa.columns,model.booster_.feature_importance(importance_type='gain')))
            print(f'seed {seed} fold {fi} complete',flush=True)
        for v in variants:
            seed_log.append(dict(candidate=v,seed=seed,**metrics(y,oof[v][seed],weights)))
        pd.DataFrame(folds_log).to_csv(out/'fold_metrics.csv',index=False)
        print(pd.DataFrame(seed_log).tail(2).to_string(index=False),flush=True)
    for name,rows in [('seed_metrics',seed_log),('fold_membership',membership),('feature_diagnostics',diagnostics),('feature_importance',importance)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    np.savez_compressed(out/'predictions.npz',y=y,weights=weights,source_row=source_rows,**{'oof_'+v:oof[v] for v in variants},**{'test_'+v:test[v] for v in variants})
    old=np.load(here/'experiments/gentle_regularization_2026_11_01/predictions.npz')
    np.testing.assert_allclose(oof['baseline'],old['oof_baseline'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(test['baseline'],old['test_baseline'],atol=1e-12,rtol=0)
    base_dir=here/'experiments/knn_musical_2026_11_01/audio'; base=pd.read_csv(base_dir/'oof_predictions.csv')
    assert np.array_equal(base.uri,tr.uri) and np.array_equal(base.y,y)
    rank=lambda p:rankdata(p)/len(p)
    names=['extra_trees','cat_base','xgb_tuned']; others=[rank(base[n+'_prob']) for n in names]; other_test=[]
    for n in names:
        frame=pd.read_csv(base_dir/f'submission_{n}.csv'); assert np.array_equal(frame.ID,te.ID)
        other_test.append(rank(frame.Popularity_Type))
    ensemble=[]; pred=pd.DataFrame(dict(source_row=source_rows,uri=tr.uri,y=y))
    for v in variants:
        assert np.isfinite(oof[v]).all() and np.isfinite(test[v]).all()
        pred[v]=oof[v].mean(0)
        p=np.mean([rank(oof[v].mean(0)),*others],axis=0); t=np.mean([rank(test[v].mean(0)),*other_test],axis=0)
        ensemble.append(dict(candidate=v,**metrics(y,p,weights)))
        for label,values in [(v,test[v].mean(0)),('fixed_blend_'+v,t)]:
            pd.DataFrame({'ID':te.ID,pipeline.TARGET:values}).to_csv(out/f'submission_{label}.csv',index=False)
    pred.to_csv(out/'oof_predictions.csv',index=False); pd.DataFrame(ensemble).to_csv(out/'fixed_blend_comparison.csv',index=False)
    s=pd.DataFrame(seed_log); summary=s.groupby('candidate')[['auc','weighted_auc','logloss','brier']].mean(); summary.to_csv(out/'summary.csv')
    paired=s.pivot(index='seed',columns='candidate',values='weighted_auc'); paired['delta']=paired.era_percentiles-paired.baseline; paired.to_csv(out/'paired_deltas.csv')
    (out/'validation.json').write_text(json.dumps(dict(fits=len(folds_log),baseline_reproduced=True,group_boundaries_checked=True,predictions_finite=True,test_ids_checked=True),indent=2)+'\n')
    print(summary.to_string()); print(pd.DataFrame(ensemble).to_string(index=False)); print(paired.to_string())


if __name__=='__main__':
    main()
