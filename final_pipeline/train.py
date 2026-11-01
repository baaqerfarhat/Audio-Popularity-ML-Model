"""CS155 MP1 — final pipeline (self-contained).

Grouped (by uri) stratified 5-fold CV repeated over several seeds; feature engineering that adds
(1) release-date features, (2) uri_count (track multiplicity), (3) album-level signal via
out-of-fold target encoding of the exact release date and a same-date kNN label mean;
GBDT model zoo + ExtraTrees; rank-average blend. Writes submissions + diagnostics.

Run:  python train.py --data-dir ../data --out results --seeds 0,1,2,3,4
"""
from __future__ import annotations
import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

TARGET = "Popularity_Type"
AUDIO = ["speechiness", "danceability", "duration_ms", "energy", "instrumentalness", "valence", "tempo",
         "loudness", "acousticness", "liveness"]
CATS = ["key", "mode", "time_signature"]
KNN_SPACE = ["speechiness", "danceability", "energy", "instrumentalness", "valence", "loudness", "acousticness",
             "liveness", "tempo", "duration_ms"]

# ----------------------------------------------------------------------------- features

def date_features(df):
    s = df["track_album_release_date"].astype(str)
    out = pd.DataFrame(index=df.index)
    out["rel_year"] = pd.to_numeric(s.str[:4], errors="coerce")
    out["rel_month"] = pd.to_numeric(s.str.extract(r"^\d{4}-(\d{2})", expand=False), errors="coerce")
    out["rel_day"] = pd.to_numeric(s.str.extract(r"^\d{4}-\d{2}-(\d{2})", expand=False), errors="coerce")
    prec = np.select([s.str.match(r"^\d{4}$"), s.str.match(r"^\d{4}-\d{2}$")], [0, 1], default=2)
    out["rel_precision"] = prec
    dt = pd.to_datetime(s.where(prec == 2), format="%Y-%m-%d", errors="coerce")
    ym = pd.to_datetime(s.where(prec == 1) + "-15", format="%Y-%m-%d", errors="coerce")
    yy = pd.to_datetime(s.where(prec == 0) + "-07-01", format="%Y-%m-%d", errors="coerce")
    dt_fill = dt.fillna(ym).fillna(yy)
    out["rel_ordinal"] = (dt_fill - pd.Timestamp("1950-01-01")).dt.days
    out["age_days"] = (pd.Timestamp("2024-12-31") - dt_fill).dt.days
    out["log_age"] = np.log1p(out["age_days"].clip(lower=0))
    out["rel_dow"] = dt.dt.dayofweek
    out["is_friday"] = (out["rel_dow"] == 4).astype(float).where(out["rel_dow"].notna())
    out["rel_doy"] = dt.dt.dayofyear
    return out


def audio_interactions(df):
    out = pd.DataFrame(index=df.index)
    out["log_duration"] = np.log1p(df["duration_ms"].clip(lower=0))
    out["loud_energy"] = df["loudness"] * df["energy"]
    out["energy_minus_acoustic"] = df["energy"] - df["acousticness"]
    out["dance_valence"] = df["danceability"] * df["valence"]
    out["dance_energy"] = df["danceability"] * df["energy"]
    out["inst_log"] = np.log10(df["instrumentalness"] + 1e-6)
    out["speech_log"] = np.log10(df["speechiness"] + 1e-4)
    out["acoustic_log"] = np.log10(df["acousticness"] + 1e-6)
    out["live_log"] = np.log10(df["liveness"] + 1e-4)
    out["tempo_dev"] = np.abs(df["tempo"] - 120)
    out["loud_per_energy"] = df["loudness"] / (df["energy"] + 1e-3)
    return out


def build_static_features(tr, te):
    parts_tr = [tr[AUDIO + CATS].copy(), audio_interactions(tr), date_features(tr)]
    parts_te = [te[AUDIO + CATS].copy(), audio_interactions(te), date_features(te)]
    # track multiplicity inside its own file (train/test uris never overlap)
    parts_tr.append(tr.groupby("uri")["uri"].transform("count").rename("uri_count").to_frame())
    parts_te.append(te.groupby("uri")["uri"].transform("count").rename("uri_count").to_frame())
    # number of distinct tracks (train+test) sharing the exact release-date string (album-size proxy)
    alld = pd.concat([tr[["uri", "track_album_release_date"]], te[["uri", "track_album_release_date"]]]).drop_duplicates("uri")
    cnt = alld.groupby("track_album_release_date")["uri"].count()
    parts_tr.append(tr["track_album_release_date"].map(cnt).rename("date_group_size").to_frame())
    parts_te.append(te["track_album_release_date"].map(cnt).rename("date_group_size").to_frame())
    return pd.concat(parts_tr, axis=1), pd.concat(parts_te, axis=1)


def te_map(fit_keys, fit_y, apply_keys, prior_w=3.0):
    fit_keys = pd.Series(fit_keys).astype(str).values; apply_keys = pd.Series(apply_keys).astype(str).values
    gm = float(np.mean(fit_y))
    g = pd.DataFrame({"k": fit_keys, "y": fit_y}).groupby("k")["y"].agg(["sum", "count"])
    m = (g["sum"] + prior_w * gm) / (g["count"] + prior_w)
    return np.column_stack([pd.Series(apply_keys).map(m).fillna(gm).values,
                            pd.Series(apply_keys).map(g["count"]).fillna(0).values])


def te_inner_oof(keys, yv, groups, seed, prior_w=3.0):
    """Target encoding of `keys` for the rows themselves, out-of-fold (inner 5-fold grouped by uri)."""
    out = np.zeros((len(yv), 2))
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for itr, iva in skf.split(keys, yv, groups=groups):
        out[iva] = te_map(keys[itr], yv[itr], keys[iva], prior_w)
    return out


def knn_date(Xa_fit, date_fit, y_fit, Xa_ap, date_ap, exclude_self=False, k=5, prior=2.0):
    """Mean label (prior-smoothed, distance-weighted) of the k nearest same-release-date tracks in audio space."""
    gm = y_fit.mean()
    out_mean = np.full(len(Xa_ap), gm); out_dist = np.full(len(Xa_ap), np.nan)
    groups = {}
    for i, d in enumerate(date_fit):
        groups.setdefault(d, []).append(i)
    for j, d in enumerate(date_ap):
        idx = groups.get(d)
        if not idx:
            continue
        idx = np.array(idx)
        dist = np.sqrt(((Xa_fit[idx] - Xa_ap[j]) ** 2).sum(1))
        if exclude_self:
            keep = dist > 1e-9
            idx, dist = idx[keep], dist[keep]
            if len(idx) == 0:
                continue
        order = np.argsort(dist)[:k]
        w = 1.0 / (1.0 + dist[order])
        out_mean[j] = (np.sum(w * y_fit[idx[order]]) + prior * gm) / (np.sum(w) + prior)
        out_dist[j] = dist[order][0]
    return out_mean, out_dist


def fold_matrices(X, Xte, tr, te, y, folds, seed):
    """Per fold: (Xtr, Xva, Xtest) with label-derived features computed strictly from the fold's training part."""
    dates = tr["track_album_release_date"].astype(str).values; dates_te = te["track_album_release_date"].astype(str).values
    uris = tr["uri"].values
    Xa_all = pd.concat([tr[KNN_SPACE], te[KNN_SPACE]])
    Xa_all = Xa_all.assign(duration_ms=np.log1p(Xa_all["duration_ms"]), instrumentalness=np.log10(Xa_all["instrumentalness"] + 1e-6))
    mu, sd = Xa_all.mean(), Xa_all.std()
    Xa = ((Xa_all.iloc[:len(tr)] - mu) / sd).fillna(0).values; Xa_te = ((Xa_all.iloc[len(tr):] - mu) / sd).fillna(0).values
    mats = []
    for fi, (tri, vai) in enumerate(folds):
        Xtr, Xva, Xt = X.iloc[tri].copy(), X.iloc[vai].copy(), Xte.copy()
        inner = te_inner_oof(dates[tri], y[tri], uris[tri], seed * 10 + fi)
        Xtr["date_te"], Xtr["date_te_n"] = inner[:, 0], inner[:, 1]
        v = te_map(dates[tri], y[tri], dates[vai]); Xva["date_te"], Xva["date_te_n"] = v[:, 0], v[:, 1]
        t = te_map(dates[tri], y[tri], dates_te); Xt["date_te"], Xt["date_te_n"] = t[:, 0], t[:, 1]
        Xtr["knn_date_mean"], Xtr["knn_date_dist"] = knn_date(Xa[tri], dates[tri], y[tri], Xa[tri], dates[tri], exclude_self=True)
        Xva["knn_date_mean"], Xva["knn_date_dist"] = knn_date(Xa[tri], dates[tri], y[tri], Xa[vai], dates[vai])
        Xt["knn_date_mean"], Xt["knn_date_dist"] = knn_date(Xa[tri], dates[tri], y[tri], Xa_te, dates_te)
        mats.append((Xtr, Xva, Xt))
    return mats

# ----------------------------------------------------------------------------- models

def lgb_model(params):
    import lightgbm as lgb
    def f(Xtr, ytr, Xva, Xte, seed):
        m = lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1, **params).fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], m
    return f


def xgb_model(params):
    from xgboost import XGBClassifier
    def f(Xtr, ytr, Xva, Xte, seed):
        m = XGBClassifier(random_state=seed, n_jobs=-1, tree_method="hist", **params).fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], m
    return f


def cat_model(params, task_type="CPU"):
    from catboost import CatBoostClassifier
    def f(Xtr, ytr, Xva, Xte, seed):
        Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
        for c in CATS:
            for d in (Xtr, Xva, Xte):
                d[c] = d[c].fillna(-1).astype(int).astype(str)
        m = CatBoostClassifier(random_seed=seed, verbose=False, allow_writing_files=False, task_type=task_type,
                               loss_function="Logloss", **params)
        m.fit(Xtr, ytr, cat_features=CATS)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], m
    return f


def et_model(params):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    def f(Xtr, ytr, Xva, Xte, seed):
        imp = SimpleImputer(strategy="median").fit(Xtr)
        m = ExtraTreesClassifier(random_state=seed, n_jobs=-1, **params).fit(imp.transform(Xtr), ytr)
        return m.predict_proba(imp.transform(Xva))[:, 1], m.predict_proba(imp.transform(Xte))[:, 1], m
    return f


def get_models(task_type):
    return {
        "lgb_tuned": lgb_model(LGB_TUNED),
        "lgb_deep": lgb_model(dict(n_estimators=500, learning_rate=0.02, num_leaves=31, max_depth=-1, min_child_samples=15,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_alpha=0.0, reg_lambda=2.0)),
        "xgb_tuned": xgb_model(XGB_TUNED),
        "cat_tuned": cat_model(CAT_TUNED, task_type),
        "cat_base": cat_model(dict(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=5, random_strength=1,
                                   bootstrap_type="Bayesian", bagging_temperature=0.5, border_count=128), task_type),
        "extra_trees": et_model(dict(n_estimators=800, min_samples_leaf=3, max_features=0.5)),
    }


# Tuned by Optuna on grouped 5-fold CV (see tuning logs); placeholders are overwritten below if tune_*.json exist.
LGB_TUNED = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, max_depth=5, min_child_samples=20,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0)
XGB_TUNED = dict(n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=3, subsample=0.8,
                 colsample_bytree=0.7, reg_lambda=2.0, gamma=0.1)
CAT_TUNED = dict(iterations=1200, learning_rate=0.03, depth=4, l2_leaf_reg=3, random_strength=1,
                 bootstrap_type="Bernoulli", subsample=0.8, border_count=254)

# ----------------------------------------------------------------------------- main

def rank01(v):
    return rankdata(v) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--models", default="lgb_tuned,lgb_deep,xgb_tuned,cat_tuned,cat_base,extra_trees")
    ap.add_argument("--task-type", default="CPU", choices=["CPU", "GPU"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    for fam, store in (("lgb", LGB_TUNED), ("xgb", XGB_TUNED), ("cat", CAT_TUNED)):
        p = os.path.join(here, f"tuned_{fam}.json")
        if os.path.exists(p):
            store.clear(); store.update(json.load(open(p)))
            print(f"loaded tuned params for {fam}: {store}")

    tr = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    te = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    ss = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    tr = tr.dropna(subset=["uri"]).reset_index(drop=True)
    y = (tr[TARGET] == "High").astype(int).values
    seeds = [int(s) for s in args.seeds.split(",")]
    X, Xte = build_static_features(tr, te)
    print(f"train {X.shape}, test {Xte.shape}, positives {y.mean():.3f}")

    # 'test-like' evaluation weights: duplicate-uri rows are 11% of train but ~2.5% of test (TAs removed test rows
    # whose uri appears in train), so we down-weight them when estimating leaderboard performance
    cnt = X["uri_count"].values; dup = cnt > 1
    W = np.where(dup, (22 / 863) / (1 - 22 / 863) * (np.sum(~dup) / np.sum(dup)), 1.0)

    models = {k: v for k, v in get_models(args.task_type).items() if k in args.models.split(",")}
    oof = {m: np.zeros((len(seeds), len(y))) for m in models}
    tst = {m: np.zeros((len(seeds), len(te))) for m in models}
    importances = {}
    fold_rows = []
    for si, seed in enumerate(seeds):
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups=tr["uri"]))
        mats = fold_matrices(X, Xte, tr, te, y, folds, seed)
        for name, fp in models.items():
            t0 = time.time()
            for fi, (tri, vai) in enumerate(folds):
                Xtr, Xva, Xt = mats[fi]
                p_va, p_te, m = fp(Xtr, y[tri], Xva, Xt, seed * 100 + fi)
                oof[name][si, vai] = p_va; tst[name][si] += p_te / len(folds)
                fold_rows.append(dict(model=name, seed=seed, fold=fi, auc=roc_auc_score(y[vai], p_va)))
                if hasattr(m, "feature_importances_"):
                    importances.setdefault(name, []).append(pd.Series(m.feature_importances_, index=Xtr.columns))
            a = roc_auc_score(y, oof[name][si]); aw = roc_auc_score(y, oof[name][si], sample_weight=W)
            print(f"seed {seed} {name:12s} OOF AUC={a:.4f} test-like={aw:.4f} [{time.time()-t0:.0f}s]", flush=True)

    # ---- per-model summary
    summary = {"seeds": seeds, "models": {}}
    for name in models:
        summary["models"][name] = dict(
            oof_auc=float(np.mean([roc_auc_score(y, oof[name][s]) for s in range(len(seeds))])),
            test_like_auc=float(np.mean([roc_auc_score(y, oof[name][s], sample_weight=W) for s in range(len(seeds))])))
        sub = ss.copy(); sub["ID"] = te["ID"].values; sub[TARGET] = tst[name].mean(0)
        sub.to_csv(os.path.join(args.out, f"submission_{name}.csv"), index=False)

    # ---- blend: rank-average, weights by hill-climbing on test-like AUC (averaged over seeds)
    names = list(models)
    R = {n: np.array([rank01(oof[n][s]) for s in range(len(seeds))]) for n in names}
    def score(pred, w=None):
        return float(np.mean([roc_auc_score(y, pred[s], sample_weight=w) for s in range(len(seeds))]))
    chosen, cur, best = [], None, -1
    for _ in range(25):
        cand = None
        for n in names:
            trial = R[n] if cur is None else (cur * len(chosen) + R[n]) / (len(chosen) + 1)
            s = score(trial, W)
            if s > best + 1e-6:
                best, cand, cand_pred = s, n, trial
        if cand is None:
            break
        chosen.append(cand); cur = cand_pred
    weights = pd.Series(chosen).value_counts(normalize=True).to_dict()
    equal = np.mean([R[n] for n in names], axis=0)
    summary["blend"] = dict(weights=weights, oof_auc=score(cur), test_like_auc=best,
                            equal_weight_oof_auc=score(equal), equal_weight_test_like_auc=score(equal, W))
    print("blend weights:", weights); print("blend OOF AUC", round(score(cur), 4), "test-like", round(best, 4))

    test_rank = {n: rank01(tst[n].mean(0)) for n in names}
    blend_test = sum(w * test_rank[n] for n, w in weights.items())
    sub = ss.copy(); sub["ID"] = te["ID"].values; sub[TARGET] = blend_test
    sub.to_csv(os.path.join(args.out, "submission_blend.csv"), index=False)
    equal_test = np.mean([test_rank[n] for n in names], axis=0)
    sub[TARGET] = equal_test; sub.to_csv(os.path.join(args.out, "submission_equal_rank_avg.csv"), index=False)

    # ---- diagnostics
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.out, "fold_metrics.csv"), index=False)
    oof_df = pd.DataFrame({"source_row": np.arange(len(y)), "y": y, "uri": tr["uri"]})
    for n in names:
        oof_df[f"{n}_prob"] = oof[n].mean(0)
    oof_df["blend_rank"] = cur.mean(0)
    oof_df.to_csv(os.path.join(args.out, "oof_predictions.csv"), index=False)
    for n, imps in importances.items():
        imp = pd.concat(imps, axis=1).mean(1).sort_values(ascending=False)
        imp.to_csv(os.path.join(args.out, f"feature_importance_{n}.csv"), header=["importance"])
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        for n in names:
            fpr, tpr, _ = roc_curve(y, oof[n].mean(0)); ax.plot(fpr, tpr, alpha=0.6, label=f"{n} ({summary['models'][n]['oof_auc']:.3f})")
        fpr, tpr, _ = roc_curve(y, cur.mean(0)); ax.plot(fpr, tpr, "k", lw=2.5, label=f"blend ({summary['blend']['oof_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", c="gray"); ax.set(xlabel="FPR", ylabel="TPR", title="Grouped 5-fold OOF ROC (mean over seeds)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(args.out, "roc_oof.png"), dpi=160); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
