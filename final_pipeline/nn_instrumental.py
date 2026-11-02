"""Matched, nested grouped validation of a global NN and instrumental residual NN.

Run from the repository: .venv/Scripts/python.exe final_pipeline/nn_instrumental.py
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss
from scipy.stats import rankdata
import train as pipeline
from analyze_bias_variance import metrics
from try_period_specialists import instrumental_groups, pairwise_auc

CONFIG = dict(hidden=[64,32], dropout=.1, learning_rate=.001, weight_decay=.0001,
              batch_size=256, max_epochs=150, patience=15, min_delta=.0001,
              instrumental_loss_weight=.25, residual_penalty=.01, ensemble_nn_weight=.1)


def splits(df, y, seed):
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(df,y,df.uri))
    for a,b in folds:
        assert not set(df.uri.iloc[a]) & set(df.uri.iloc[b])
    return folds


def label_features(fit, y, query):
    """No query labels or query-fitted statistics are accessed."""
    def audio(df):
        x = df[pipeline.KNN_SPACE].copy()
        x['duration_ms'] = np.log1p(x.duration_ms.clip(lower=0))
        x['instrumentalness'] = np.log10(x.instrumentalness.clip(lower=0)+1e-6)
        return x.replace([np.inf,-np.inf],np.nan)
    imp = SimpleImputer(strategy='median',keep_empty_features=True)
    a = imp.fit_transform(audio(fit)); b = imp.transform(audio(query))
    scale = StandardScaler().fit(a)
    a,b = scale.transform(a),scale.transform(b)
    dates = fit.track_album_release_date.astype(str).to_numpy()
    qdates = query.track_album_release_date.astype(str).to_numpy()
    te = pipeline.te_map(dates,y,qdates)
    mean,dist = pipeline.knn_date(a,dates,y,b,qdates)
    return np.column_stack([te,mean,dist])


class Features:
    def numeric(self, df, labels):
        x = pd.concat([df[pipeline.AUDIO],pipeline.audio_interactions(df),pipeline.date_features(df)],axis=1)
        return np.column_stack([x.replace([np.inf,-np.inf],np.nan).to_numpy(),labels])

    def fit_transform(self, df, y, seed):
        self.reference,self.y = df.copy(),np.array(y,copy=True)
        labels = np.empty((len(df),4)); self.crossfit_fold = np.full(len(df),-1)
        for fold,(a,b) in enumerate(splits(df,y,seed)):
            labels[b] = label_features(df.iloc[a],y[a],df.iloc[b])
            self.crossfit_fold[b] = fold
        self.imputer = SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
        num = self.imputer.fit_transform(self.numeric(df,labels))
        self.scaler = StandardScaler().fit(num)
        self.categories = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
        cat = self.categories.fit_transform(df[pipeline.CATS].fillna(-1))
        return np.column_stack([self.scaler.transform(num),cat]).astype('float32')

    def transform(self, df):
        labels = label_features(self.reference,self.y,df)
        num = self.scaler.transform(self.imputer.transform(self.numeric(df,labels)))
        cat = self.categories.transform(df[pipeline.CATS].fillna(-1))
        return np.column_stack([num,cat]).astype('float32')


class Network(nn.Module):
    def __init__(self, size, specialist):
        super().__init__()
        self.specialist = specialist
        self.backbone = nn.Sequential(nn.Linear(size,64),nn.ReLU(),nn.Dropout(.1),
                                      nn.Linear(64,32),nn.ReLU(),nn.Dropout(.1))
        self.general = nn.Linear(32,1)
        self.residual = nn.Sequential(nn.Linear(32,8),nn.ReLU(),nn.Linear(8,1))
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x, mask):
        h = self.backbone(x); g = self.general(h).squeeze(-1)
        r = self.residual(h).squeeze(-1) if self.specialist else torch.zeros_like(g)
        return g + mask*r, g, r


def losses(model, x, y, mask):
    routed,g,r = model(x,mask)
    general = nn.functional.binary_cross_entropy_with_logits(g,y)
    selected = mask.bool()
    inst = nn.functional.binary_cross_entropy_with_logits(routed[selected],y[selected]) if selected.any() else g.sum()*0
    penalty = r[selected].square().mean() if selected.any() else r.sum()*0
    total = general + CONFIG['instrumental_loss_weight']*inst + CONFIG['residual_penalty']*penalty if model.specialist else general
    return total,general,inst,penalty


def predict(model, x, mask):
    model.eval()
    with torch.no_grad():
        p,g,r = model(torch.as_tensor(x),torch.as_tensor(mask,dtype=torch.float32))
        return torch.sigmoid(p).numpy(),torch.sigmoid(g).numpy(),r.numpy()


def fit_network(x,y,mask,specialist,seed,epochs,validation=None):
    torch.manual_seed(seed)
    model = Network(x.shape[1],specialist)
    optimizer = torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
    xt,yt,mt = torch.as_tensor(x),torch.as_tensor(y,dtype=torch.float32),torch.as_tensor(mask,dtype=torch.float32)
    rng = np.random.default_rng(seed); history=[]; best=float('inf'); best_epoch=epochs
    for epoch in range(1,epochs+1):
        model.train()
        for batch in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/CONFIG['batch_size']))):
            optimizer.zero_grad()
            loss = losses(model,xt[batch],yt[batch],mt[batch])[0]
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            values = [float(v) for v in losses(model,xt,yt,mt)]
        row = dict(epoch=epoch, total_loss=values[0],general_loss=values[1],instrumental_loss=values[2],residual_squared=values[3])
        if validation is not None:
            vx,vy,vm = validation
            score = log_loss(vy,predict(model,vx,vm)[0],labels=[0,1])
            row['inner_logloss'] = score
            if score < best-CONFIG['min_delta']:
                best,best_epoch = score,epoch
        history.append(row)
        if validation is not None and epoch-best_epoch >= CONFIG['patience']:
            break
    return model,best_epoch,history


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=here/'experiments/nn_instrumental_2026_11_01')
    args = parser.parse_args(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'metadata.json').exists():
        raise FileExistsError('Use a fresh output directory')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    raw = pd.read_csv(here.parent/'data/train.csv'); tr=raw.dropna(subset=['uri']).reset_index(drop=True)
    source_rows=raw.index[raw.uri.notna()].to_numpy()
    te=pd.read_csv(here.parent/'data/test.csv'); y=tr[pipeline.TARGET].eq('High').astype(int).to_numpy()
    mask=instrumental_groups(tr.instrumentalness); tmask=instrumental_groups(te.instrumentalness)
    dup=tr.groupby('uri').uri.transform('count').to_numpy()>1
    weights=np.where(dup,(22/863)/(1-22/863)*((~dup).sum()/dup.sum()),1.)
    variants=['global','instrumental_head']
    oof={v:np.full((5,len(tr)),np.nan) for v in variants}; test={v:np.zeros((5,len(te))) for v in variants}
    general_oof=np.full((5,len(tr)),np.nan); residual_oof=general_oof.copy()
    files=[Path(__file__),here/'train.py',here/'nn-requirements-lock.txt',here.parent/'data/train.csv',here.parent/'data/test.csv']
    metadata=dict(config=CONFIG,seeds=list(range(5)),folds=5,torch_version=torch.__version__,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=here,text=True).strip(),
        sha256={str(p.relative_to(here.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        primary_metric='mean per-seed weighted OOF AUC',early_stopping='first inner grouped 5-fold split; fresh full outer-train refit',
        features='audio, interactions, dates, one-hot musical attributes, grouped crossfit date TE and audio-only date KNN; no file-level counts',
        preprocessing='fit exclusively on permitted training rows; no test statistics',instrumental_threshold=.5,
        caveat='Exploratory: these outer splits have been reused for previous experiments.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    pd.DataFrame([dict(group=g,rows=int((mask==g).sum()),high=int(y[mask==g].sum()),
        unique_tracks=tr.loc[mask==g,'uri'].nunique(),test_rows=int((tmask==g).sum())) for g in [0,1]]).to_csv(out/'cohorts.csv',index=False)
    histories=[]; fold_rows=[]; memberships=[]; seed_rows=[]; pairs=[]
    for seed in range(5):
        for fold,(a,b) in enumerate(splits(tr,y,seed)):
            fit_seed=seed*100+fold; df=tr.iloc[a]
            ia,ib=splits(df,y[a],fit_seed+1000)[0]
            inner=Features(); ix=inner.fit_transform(df.iloc[ia],y[a][ia],fit_seed+2000); iv=inner.transform(df.iloc[ib])
            full=Features(); fx=full.fit_transform(df,y[a],fit_seed+3000); vx=full.transform(tr.iloc[b]); tx=full.transform(te)
            for role,indices in [('outer_valid',b),('inner_train',a[ia]),('inner_valid',a[ib])]:
                memberships.extend(dict(seed=seed,fold=fold,role=role,source_row=int(source_rows[i]),uri=tr.uri.iloc[i]) for i in indices)
            for v in variants:
                specialist=v=='instrumental_head'
                _,epochs,hist=fit_network(ix,y[a][ia],mask[a][ia],specialist,fit_seed,CONFIG['max_epochs'],(iv,y[a][ib],mask[a][ib]))
                histories.extend(dict(seed=seed,fold=fold,model=v,stage='selection',**r) for r in hist)
                model,_,hist=fit_network(fx,y[a],mask[a],specialist,fit_seed,epochs)
                histories.extend(dict(seed=seed,fold=fold,model=v,stage='refit',**r) for r in hist)
                p,g,r=predict(model,vx,mask[b]); oof[v][seed,b]=p; test[v][seed]+=predict(model,tx,tmask)[0]/5
                if specialist:
                    assert np.array_equal(p[mask[b]==0],g[mask[b]==0])
                    general_oof[seed,b]=g; residual_oof[seed,b]=r
                row=dict(seed=seed,fold=fold,model=v,epochs=epochs,train_rows=len(a),instrumental_high_train=int(y[a][mask[a]==1].sum()))
                for prefix,yy,pp,ww in [('train',y[a],predict(model,fx,mask[a])[0],weights[a]),('validation',y[b],p,weights[b])]:
                    row.update({prefix+'_'+k:val for k,val in metrics(yy,pp,ww).items()})
                fold_rows.append(row)
                print(f'seed={seed} fold={fold} {v}: epochs={epochs} validation AUC={row["validation_auc"]:.5f}',flush=True)
            pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv',index=False)
            pd.DataFrame(histories).to_csv(out/'training_history.csv',index=False)
        for v in variants:
            for group,select in [('overall',np.ones(len(y),dtype=bool)),('instrumental',mask==1),('other',mask==0)]:
                p=oof[v][seed,select]; yy=y[select]
                seed_rows.append(dict(seed=seed,model=v,group=group,high_miss_rate=float((p[yy==1]<.5).mean()),**metrics(yy,p,weights[select])))
            for pos in [0,1]:
                for neg in [0,1]:
                    auc,mass=pairwise_auc(y,oof[v][seed],weights,pos,neg,mask)
                    pairs.append(dict(seed=seed,model=v,positive_group=pos,negative_group=neg,auc=auc,pair_mass=mass))
    pd.DataFrame(seed_rows).to_csv(out/'seed_metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'pairwise_auc.csv',index=False)
    pd.DataFrame(memberships).to_csv(out/'split_membership.csv',index=False)
    np.savez_compressed(out/'predictions.npz',y=y,mask=mask,weights=weights,source_row=source_rows,
        shared_general=general_oof,shared_residual=residual_oof,**{v+'_oof':oof[v] for v in variants},**{v+'_test':test[v] for v in variants})
    predictions=pd.DataFrame(dict(source_row=source_rows,uri=tr.uri,y=y))
    for v in variants:
        assert np.isfinite(oof[v]).all()
        predictions[v]=oof[v].mean(0)
        pd.DataFrame({'ID':te.ID,'Popularity_Type':test[v].mean(0)}).to_csv(out/f'submission_{v}.csv',index=False)
    predictions.to_csv(out/'oof_predictions.csv',index=False)
    base_dir=here/'experiments/knn_musical_2026_11_01/audio'
    base=pd.read_csv(base_dir/'oof_predictions.csv')
    assert np.array_equal(base.uri.to_numpy(),tr.uri.to_numpy()) and np.array_equal(base.y,y)
    names=['lgb_tuned','extra_trees','cat_base','xgb_tuned']
    rank=lambda x:rankdata(x)/len(x)
    bp=np.mean([rank(base[n+'_prob']) for n in names],axis=0)
    baseline_test=[pd.read_csv(base_dir/f'submission_{n}.csv') for n in names]
    assert all(np.array_equal(frame.ID,te.ID) for frame in baseline_test)
    bt=np.mean([rank(frame.Popularity_Type) for frame in baseline_test],axis=0)
    ensemble=[dict(model='trees',**metrics(y,bp,weights))]
    for v in variants:
        blend=.9*bp+.1*rank(oof[v].mean(0))
        ensemble.append(dict(model='trees_plus_'+v,**metrics(y,blend,weights)))
        pd.DataFrame({'ID':te.ID,'Popularity_Type':.9*bt+.1*rank(test[v].mean(0))}).to_csv(out/f'submission_trees_plus_{v}.csv',index=False)
    pd.DataFrame(ensemble).to_csv(out/'ensemble_metrics.csv',index=False)
    summary=pd.DataFrame(seed_rows).groupby(['model','group'])[['auc','weighted_auc','logloss','brier','high_miss_rate']].mean()
    summary.to_csv(out/'summary.csv'); print(summary.to_string(),flush=True)
    print(pd.DataFrame(ensemble).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
