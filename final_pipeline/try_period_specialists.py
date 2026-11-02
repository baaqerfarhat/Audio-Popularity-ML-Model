"""Fixed release-period or instrumental specialists versus global LightGBM."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from analyze_bias_variance import metrics
import train as pipeline


def release_periods(dates):
    years = pd.to_numeric(pd.Series(dates).astype(str).str.extract(r"^(\d{4})(?:-|$)", expand=False), errors="coerce")
    return np.select([years.lt(2020), years.ge(2020)], [0, 1], default=-1)


def instrumental_groups(values):
    """Missing values fall back to global predictions with the other tracks."""
    return pd.to_numeric(pd.Series(values), errors="coerce").ge(.5).astype(int).to_numpy()


def route_predictions(global_p, older_p, recent_p, periods):
    return np.where(periods == 0, older_p, np.where(periods == 1, recent_p, global_p))


def pairwise_auc(y, p, weights, positive_period, negative_period, periods):
    pos = (y == 1) & (periods == positive_period)
    neg = (y == 0) & (periods == negative_period)
    ps, pw = p[pos], weights[pos]
    order = np.argsort(p[neg])
    ns, nw = p[neg][order], weights[neg][order]
    mass = pw.sum() * nw.sum()
    if mass == 0:
        return float("nan"), 0.
    cumulative = np.r_[0., nw.cumsum()]
    left = np.searchsorted(ns, ps, side="left")
    right = np.searchsorted(ns, ps, side="right")
    concordant = cumulative[left] + .5 * (cumulative[right] - cumulative[left])
    return float(np.dot(pw, concordant) / mass), float(mass)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--specialization", choices=["period", "instrumental"], default="period")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    instrumental = args.specialization == "instrumental"
    experiment = "instrumental_specialist_2026_11_01" if instrumental else "period_specialists_2026_11_01"
    out = (args.out or here / "experiments" / experiment).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists():
        raise FileExistsError("Choose a fresh output directory")
    params = json.loads((here / "tuned_lgb.json").read_text())
    tr = pd.read_csv(here.parent / "data/train.csv").dropna(subset=["uri"]).reset_index(drop=True)
    te = pd.read_csv(here.parent / "data/test.csv")
    y = tr[pipeline.TARGET].eq("High").astype(int).to_numpy()
    period = instrumental_groups(tr.instrumentalness) if instrumental else release_periods(tr.track_album_release_date)
    period_test = instrumental_groups(te.instrumentalness) if instrumental else release_periods(te.track_album_release_date)
    cohort_names = {0:"non_instrumental_or_missing",1:"instrumental"} if instrumental else {0:"before_2020",1:"2020_onward"}
    trained_regions = [1] if instrumental else [0,1]
    X, Xt = pipeline.build_static_features(tr, te)
    dup = X.uri_count.to_numpy() > 1
    weights = np.where(dup, (22/863)/(1-22/863) * ((~dup).sum()/dup.sum()), 1.)
    variants = ["global", "specialist", "half_global_half_specialist"]
    oof = {name: np.full((5,len(tr)), np.nan) for name in variants}
    test = {name: np.zeros((5,len(te))) for name in variants}
    sources = [here.parent / "data/train.csv", here.parent / "data/test.csv", here / "train.py",
               here / "analyze_bias_variance.py", here / "tuned_lgb.json", here / "requirements-lock.txt", Path(__file__)]
    metadata = dict(seeds=list(range(5)), folds=5, specialization=args.specialization,
                    split_rule="instrumentalness >= 0.5" if instrumental else "release year >= 2020",
                    trained_regions=trained_regions, cohort_names=cohort_names, blend_weight_global=.5,
                    missing_value_policy="global fallback", knn_features="audio", threads=args.threads,
                    parameters=params, primary_metric="mean per-seed weighted OOF AUC",
                    specialist_labels="only same-group outer training rows; inner grouped target encoding rebuilt",
                    label_free_features="same full-file counts and train+test audio scaling as original pipeline",
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=here,text=True).strip(),
                    sha256={str(p.relative_to(here.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out / "metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    cohort = []
    for value, label in [(-1,"unknown"),*cohort_names.items()]:
        mask = period == value
        cohort.append(dict(period=label, train_rows=int(mask.sum()), unique_tracks=int(tr.loc[mask,"uri"].nunique()),
                           high_rows=int(y[mask].sum()), test_rows=int((period_test==value).sum())))
    pd.DataFrame(cohort).to_csv(out / "cohorts.csv",index=False)
    fit = pipeline.lgb_model(params,args.threads)
    fits, seed_rows, pair_rows, membership = [], [], [], []
    for seed in range(5):
        folds = list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X,y,tr.uri))
        global_mats = pipeline.fold_matrices(X,Xt,tr,te,y,folds,seed,knn_musical=False)
        regional_folds = {region:[(tri[period[tri]==region],vai) for tri,vai in folds] for region in trained_regions}
        regional_mats = {region:pipeline.fold_matrices(X,Xt,tr,te,y,regional_folds[region],seed,knn_musical=False)
                         for region in trained_regions}
        for fi, ((tri,vai),(Xtr,Xva,Xtest)) in enumerate(zip(folds,global_mats)):
            assert not set(tr.uri.iloc[tri]) & set(tr.uri.iloc[vai])
            membership.extend(dict(seed=seed,fold=fi,source_row=int(i)) for i in vai)
            pg, tg, mg = fit(Xtr,y[tri],Xva,Xtest,seed*100+fi)
            train_global=metrics(y[tri],mg.predict_proba(Xtr)[:,1],weights[tri])
            fits.append(dict(seed=seed,fold=fi,learner="global",train_rows=len(tri),validation_rows=len(vai),
                             train_auc=train_global["auc"],validation_auc=metrics(y[vai],pg,weights[vai])["auc"]))
            pv, pt = {0:pg}, {0:tg}
            for region in trained_regions:
                regional_tri,_ = regional_folds[region][fi]
                assert (period[regional_tri]==region).all() and len(np.unique(y[regional_tri]))==2
                assert not set(tr.uri.iloc[regional_tri]) & set(tr.uri.iloc[vai])
                A,B,C = regional_mats[region][fi]
                pv[region],pt[region],model = fit(A,y[regional_tri],B,C,seed*100+fi)
                vm = period[vai]==region
                fits.append(dict(seed=seed,fold=fi,learner=cohort_names[region],
                                 train_rows=len(regional_tri),validation_rows=int(vm.sum()),
                                 train_auc=metrics(y[regional_tri],model.predict_proba(A)[:,1],weights[regional_tri])["auc"],
                                 validation_auc=metrics(y[vai][vm],pv[region][vm],weights[vai][vm])["auc"]))
            ps = route_predictions(pg,pv[0],pv[1],period[vai])
            ts = route_predictions(tg,pt[0],pt[1],period_test)
            for name,p,t in [("global",pg,tg),("specialist",ps,ts),("half_global_half_specialist",.5*(pg+ps),.5*(tg+ts))]:
                oof[name][seed,vai]=p; test[name][seed]+=t/5
        for name in variants:
            p=oof[name][seed]
            for label,mask in [("overall",np.ones(len(y),dtype=bool)),(cohort_names[0],period==0),(cohort_names[1],period==1)]:
                scores=metrics(y[mask],p[mask],weights[mask])
                high=mask & (y==1); low=mask & (y==0)
                seed_rows.append(dict(variant=name,seed=seed,cohort=label,rows=int(mask.sum()),**scores,
                                      high_miss_rate=float((p[high]<.5).mean()),low_false_high_rate=float((p[low]>=.5).mean()),
                                      observed_high_rate=float(y[mask].mean()),mean_p_high=float(p[mask].mean())))
            for positive,negative in [(0,0),(1,1),(0,1),(1,0)]:
                auc,mass=pairwise_auc(y,p,weights,positive,negative,period)
                pair_rows.append(dict(variant=name,seed=seed,positive_period=positive,negative_period=negative,
                                      weighted_pair_auc=auc,pair_weight=mass))
            overall=seed_rows[-3]
            print(f"seed {seed} {name}: AUC={overall['auc']:.6f}, weighted={overall['weighted_auc']:.6f}, logloss={overall['logloss']:.6f}",flush=True)
        pd.DataFrame(fits).to_csv(out / "fit_metrics.csv",index=False)
    seed_df=pd.DataFrame(seed_rows)
    seed_df.to_csv(out / "seed_metrics.csv",index=False)
    pair_df=pd.DataFrame(pair_rows)
    pair_df.to_csv(out / "pair_metrics.csv",index=False)
    pd.DataFrame(membership).to_csv(out / "fold_membership.csv",index=False)
    np.savez_compressed(out / "predictions.npz",y=y,weights=weights,period=period,period_test=period_test,
                        **{"oof_"+n:v for n,v in oof.items()},**{"test_"+n:v for n,v in test.items()})
    summary=[]
    for (name,cohort),frame in seed_df.groupby(["variant","cohort"],sort=False):
        base=seed_df[(seed_df.variant=="global")&(seed_df.cohort==cohort)].set_index("seed")
        delta=frame.set_index("seed").weighted_auc-base.weighted_auc
        summary.append(dict(variant=name,cohort=cohort,auc=frame.auc.mean(),weighted_auc=frame.weighted_auc.mean(),
                            weighted_delta=delta.mean(),seed_wins=int((delta>0).sum()),logloss=frame.logloss.mean(),
                            high_miss_rate=frame.high_miss_rate.mean(),low_false_high_rate=frame.low_false_high_rate.mean()))
    table=pd.DataFrame(summary)
    table.to_csv(out / "summary.csv",index=False)
    pair_df.groupby(["variant","positive_period","negative_period"],sort=False).weighted_pair_auc.mean().reset_index().to_csv(out / "pair_summary.csv",index=False)
    for name in variants:
        pd.DataFrame({"ID":te.ID,pipeline.TARGET:test[name].mean(0)}).to_csv(out / f"submission_{name}.csv",index=False)
    print(table.to_string(index=False,float_format=lambda v:f"{v:.6f}"),flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.4,4.8))
    labels=["All tracks","Other tracks","Instrumentalness ≥0.5"] if instrumental else ["All tracks","Before 2020","2020 onward"]
    for name,offset,color,marker,label in [("global",-.16,"#28649B","o","Global"),("specialist",0,"#B07113","s","Routed specialist"),
                                          ("half_global_half_specialist",.16,"#8A517E","^","50/50 average")]:
        d=table[table.variant.eq(name)].set_index("cohort").loc[["overall",cohort_names[0],cohort_names[1]]]
        ax.scatter(np.arange(3)+offset,d.weighted_auc,color=color,marker=marker,s=65,label=label)
    ax.set_xticks(np.arange(3),labels)
    ax.set(ylabel="Mean weighted OOF AUC (higher is better)",
           title="Instrumental specialist versus global LightGBM" if instrumental else "Release-period specialists versus global LightGBM")
    ax.grid(axis="y",alpha=.2); ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="best",frameon=False)
    fig.text(.5,.01,"Five seeds × five grouped folds. Fixed instrumentalness cutoff 0.5; other tracks use global predictions." if instrumental
             else "Five seeds × five grouped folds. Fixed 2020 boundary and model settings; exploratory validation.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1)); fig.savefig(out / "comparison.png",dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
