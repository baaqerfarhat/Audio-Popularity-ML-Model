"""Train/validation gaps, repeated OOF stability, and grouped learning curves.

Diagnostics of the current audio-only pipeline, not an exact bias/variance
decomposition. Run from any directory; no existing submissions are overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import train as pipeline


def metrics(y, p, weights):
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probability predictions")
    return dict(auc=roc_auc_score(y, p), weighted_auc=roc_auc_score(y, p, sample_weight=weights),
                logloss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p))


def nested_training_subsets(indices, X, y, groups, seed):
    """Nested quarter/half/full pools, preserving complete track groups."""
    parts = [indices[va] for _, va in StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=seed).split(X.iloc[indices], y[indices], groups[indices])]
    return {0.25: np.sort(parts[0]), 0.5: np.sort(np.concatenate(parts[:2])), 1.0: indices}


def summarize(out):
    folds = pd.read_csv(out / "fold_metrics.csv")
    full = folds[folds.fraction == 1.0]
    rows = []
    for name, frame in full.groupby("model", sort=False):
        row = dict(model=name, fits=len(frame))
        for metric in ["auc", "weighted_auc", "logloss", "brier"]:
            row[f"train_{metric}"] = frame[f"train_{metric}"].mean()
            row[f"validation_{metric}"] = frame[f"validation_{metric}"].mean()
        row["auc_gap"] = row["train_auc"] - row["validation_auc"]
        row["validation_fold_auc_sd"] = frame.validation_auc.std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "model_summary.csv", index=False)
    curve = folds[folds.model == "lgb_tuned"].groupby("fraction", as_index=False).agg(
        fits=("fold", "size"), train_rows=("train_rows", "mean"),
        train_auc=("train_auc", "mean"), validation_auc=("validation_auc", "mean"),
        train_logloss=("train_logloss", "mean"), validation_logloss=("validation_logloss", "mean"))
    curve.to_csv(out / "learning_curve.csv", index=False)
    baseline = full[full.model == "lgb_tuned"]
    regularized = full[full.model == "lgb_regularized"]
    paired = baseline.merge(regularized, on=["seed", "fold"], suffixes=("_baseline", "_regularized"), validate="one_to_one")
    paired["auc_delta"] = paired.validation_auc_regularized - paired.validation_auc_baseline
    paired["logloss_delta"] = paired.validation_logloss_regularized - paired.validation_logloss_baseline
    paired[["seed", "fold", "auc_delta", "logloss_delta"]].to_csv(out / "regularization_paired.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"), flush=True)
    print(curve.to_string(index=False, float_format=lambda v: f"{v:.6f}"), flush=True)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=here / "experiments/bias_variance_2026_11_01")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "fold_metrics.csv").exists():
        raise FileExistsError("Use a fresh output directory to preserve earlier diagnostic runs")
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least two distinct seeds for the stability diagnostic")
    for fam, store in [("lgb", pipeline.LGB_TUNED), ("xgb", pipeline.XGB_TUNED), ("cat", pipeline.CAT_TUNED)]:
        store.clear()
        store.update(json.loads((here / f"tuned_{fam}.json").read_text()))
    tr = pd.read_csv(here.parent / "data/train.csv").dropna(subset=["uri"]).reset_index(drop=True)
    te = pd.read_csv(here.parent / "data/test.csv")
    y = (tr[pipeline.TARGET] == "High").astype(int).to_numpy()
    groups = tr.uri.to_numpy()
    X, Xt = pipeline.build_static_features(tr, te)
    dup = X.uri_count.to_numpy() > 1
    weights = np.where(dup, (22 / 863) / (1 - 22 / 863) * ((~dup).sum() / dup.sum()), 1.)
    models = pipeline.get_models("CPU", args.threads)
    reg = dict(pipeline.LGB_TUNED)
    reg.update(num_leaves=10, max_depth=4, min_child_samples=60, reg_lambda=10.0)
    models["lgb_regularized"] = pipeline.lgb_model(reg, args.threads)
    metadata = dict(python=platform.python_version(), seeds=seeds, folds=5,
                    knn_features="audio", threads=args.threads, learning_curve_model="lgb_tuned",
                    fractions=[0.25, 0.5, 1.0], regularized_lgb_parameters=reg,
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip(),
                    sha256={str(p.relative_to(here.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [here.parent / "data/train.csv", here.parent / "data/test.csv",
                                      here / "train.py", Path(__file__), here / "requirements-lock.txt",
                                      here / "tuned_lgb.json", here / "tuned_xgb.json", here / "tuned_cat.json"]})
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows, seed_rows, prediction_rows, membership = [], [], [], []
    oof = {name: np.full((len(seeds), len(y)), np.nan) for name in models}
    for si, seed in enumerate(seeds):
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, groups))
        mats = pipeline.fold_matrices(X, Xt, tr, te, y, folds, seed, knn_musical=False)
        for fi, (tri, vai) in enumerate(folds):
            assert not set(groups[tri]) & set(groups[vai])
            membership.extend(dict(seed=seed, fold=fi, source_row=int(i)) for i in vai)
        for name, fit in models.items():
            started = time.time()
            for fi, ((tri, vai), (Xtr, Xva, _)) in enumerate(zip(folds, mats)):
                # The third prediction request uses the actual training matrix.
                # Factories retain CatBoost category conversion and ExtraTrees imputation.
                pv, pt, _ = fit(Xtr, y[tri], Xva, Xtr, seed * 100 + fi)
                oof[name][si, vai] = pv
                row = dict(model=name, seed=seed, fold=fi, fraction=1., train_rows=len(tri), validation_rows=len(vai))
                row.update({"train_" + k: v for k, v in metrics(y[tri], pt, weights[tri]).items()})
                row.update({"validation_" + k: v for k, v in metrics(y[vai], pv, weights[vai]).items()})
                rows.append(row)
            seed_rows.append(dict(model=name, seed=seed, **metrics(y, oof[name][si], weights)))
            pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
            print(f"seed {seed} {name}: train AUC={np.mean([r['train_auc'] for r in rows if r['model']==name and r['seed']==seed]):.4f}, OOF AUC={seed_rows[-1]['auc']:.4f} [{time.time()-started:.0f}s]", flush=True)
        for fi, (tri, vai) in enumerate(folds):
            subsets = nested_training_subsets(tri, X, y, groups, seed * 100 + fi)
            for fraction in [0.25, 0.5]:
                subset = subsets[fraction]
                assert not set(groups[subset]) & set(groups[vai])
                # Recompute label-derived features using ONLY the reduced training pool.
                Xtr, Xva, _ = pipeline.fold_matrices(X, Xt, tr, te, y, [(subset, vai)], seed,
                                                    knn_musical=False)[0]
                pv, pt, _ = models["lgb_tuned"](Xtr, y[subset], Xva, Xtr, seed * 100 + fi)
                row = dict(model="lgb_tuned", seed=seed, fold=fi, fraction=fraction,
                           train_rows=len(subset), validation_rows=len(vai))
                row.update({"train_" + k: v for k, v in metrics(y[subset], pt, weights[subset]).items()})
                row.update({"validation_" + k: v for k, v in metrics(y[vai], pv, weights[vai]).items()})
                rows.append(row)
        pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
        print(f"seed {seed}: learning curve complete", flush=True)
    stability = []
    for name, pred in oof.items():
        assert np.isfinite(pred).all()
        sd = pred.std(axis=0, ddof=1)
        stability.append(dict(model=name, mean_oof_auc=np.mean([roc_auc_score(y, p) for p in pred]),
                              seed_oof_auc_sd=np.std([roc_auc_score(y, p) for p in pred], ddof=1),
                              mean_prediction_sd=sd.mean(), p90_prediction_sd=np.quantile(sd, .9),
                              averaged_prediction_auc=roc_auc_score(y, pred.mean(axis=0))))
        for si, seed in enumerate(seeds):
            prediction_rows.append(pd.DataFrame(dict(model=name, seed=seed, source_row=np.arange(len(y)),
                                                     y=y, probability=pred[si])))
    pd.concat(prediction_rows).to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(stability).to_csv(out / "prediction_stability.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "seed_metrics.csv", index=False)
    pd.DataFrame(membership).to_csv(out / "fold_membership.csv", index=False)
    summarize(out)
    print("Diagnostics complete", flush=True)


if __name__ == "__main__":
    main()
