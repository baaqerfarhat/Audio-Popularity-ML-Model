"""Paired, one-control-at-a-time LightGBM regularization experiment."""
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


CHANGES = {
    "baseline": {}, "leaves_16": {"num_leaves": 16}, "depth_5": {"max_depth": 5},
    "min_leaf_20": {"min_child_samples": 20}, "min_leaf_30": {"min_child_samples": 30},
    "l2_0p1": {"reg_lambda": .1}, "l2_1": {"reg_lambda": 1.},
}


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=here / "experiments/gentle_regularization_2026_11_01")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists():
        raise FileExistsError("Choose a fresh output directory")
    base = json.loads((here / "tuned_lgb.json").read_text())
    configs = {name: dict(base, **change) for name, change in CHANGES.items()}
    assert all(sum(config[k] != base[k] for k in base) == (name != "baseline")
               for name, config in configs.items())
    seeds = list(range(5))
    files = [here.parent / "data/train.csv", here.parent / "data/test.csv", here / "train.py",
             here / "analyze_bias_variance.py", Path(__file__), here / "tuned_lgb.json", here / "requirements-lock.txt"]
    metadata = dict(seeds=seeds, folds=5, threads=args.threads, knn_features="audio", configurations=configs,
                    selection_metric="mean per-seed weighted OOF AUC (existing duplicate-prevalence adjustment)",
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip(),
                    sha256={str(p.relative_to(here.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    tr = pd.read_csv(here.parent / "data/train.csv").dropna(subset=["uri"]).reset_index(drop=True)
    te = pd.read_csv(here.parent / "data/test.csv")
    y = (tr[pipeline.TARGET] == "High").astype(int).to_numpy()
    X, Xt = pipeline.build_static_features(tr, te)
    dup = X.uri_count.to_numpy() > 1
    weights = np.where(dup, (22/863)/(1-22/863) * ((~dup).sum()/dup.sum()), 1.)
    oof = {name: np.full((5, len(tr)), np.nan) for name in configs}
    test = {name: np.zeros((5, len(te))) for name in configs}
    fold_rows, seed_rows, membership = [], [], []
    for seed in seeds:
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, tr.uri))
        mats = pipeline.fold_matrices(X, Xt, tr, te, y, folds, seed, knn_musical=False)
        for fi, (tri, vai) in enumerate(folds):
            assert not set(tr.uri.iloc[tri]) & set(tr.uri.iloc[vai])
            membership.extend(dict(seed=seed, fold=fi, source_row=int(i)) for i in vai)
        for name, params in configs.items():
            fit = pipeline.lgb_model(params, args.threads)
            for fi, ((tri, vai), (Xtr, Xva, Xtest)) in enumerate(zip(folds, mats)):
                pv, ptest, model = fit(Xtr, y[tri], Xva, Xtest, seed*100+fi)
                pt = model.predict_proba(Xtr)[:, 1]
                oof[name][seed, vai] = pv
                test[name][seed] += ptest / 5
                record = dict(candidate=name, seed=seed, fold=fi)
                record.update({"train_"+k: v for k,v in metrics(y[tri], pt, weights[tri]).items()})
                record.update({"validation_"+k: v for k,v in metrics(y[vai], pv, weights[vai]).items()})
                fold_rows.append(record)
            scores = metrics(y, oof[name][seed], weights)
            seed_rows.append(dict(candidate=name, seed=seed, **scores))
            print(f"seed {seed} {name}: AUC={scores['auc']:.6f}, weighted={scores['weighted_auc']:.6f}", flush=True)
        pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    fold_df, seed_df = pd.DataFrame(fold_rows), pd.DataFrame(seed_rows)
    seed_df.to_csv(out / "seed_metrics.csv", index=False)
    pd.DataFrame(membership).to_csv(out / "fold_membership.csv", index=False)
    np.savez_compressed(out / "predictions.npz", y=y, weights=weights,
                        **{f"oof_{n}": v for n,v in oof.items()}, **{f"test_{n}": v for n,v in test.items()})
    summary = []
    baseline = seed_df[seed_df.candidate == "baseline"].set_index("seed")
    for name in configs:
        current = seed_df[seed_df.candidate == name].set_index("seed")
        f = fold_df[fold_df.candidate == name]
        delta = current.weighted_auc - baseline.weighted_auc
        row = dict(candidate=name, oof_auc=current.auc.mean(), weighted_oof_auc=current.weighted_auc.mean(),
                   weighted_delta=delta.mean(), weighted_seed_wins=int((delta > 0).sum()),
                   min_seed_weighted_delta=delta.min(), max_seed_weighted_delta=delta.max(),
                   train_auc=f.train_auc.mean(), validation_fold_auc=f.validation_auc.mean(),
                   auc_gap=(f.train_auc-f.validation_auc).mean(), logloss=current.logloss.mean())
        summary.append(row)
        pd.DataFrame({"ID": te.ID, pipeline.TARGET: test[name].mean(0)}).to_csv(out / f"submission_{name}.csv", index=False)
    table = pd.DataFrame(summary)
    table.to_csv(out / "summary.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"), flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Current baseline", "Leaves: 22 → 16", "Depth: 6 → 5", "Min. leaf rows: 11 → 20",
              "Min. leaf rows: 11 → 30", "L2: 0.0029 → 0.1", "L2: 0.0029 → 1"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    yaxis = np.arange(len(table))
    ax.scatter(table.weighted_oof_auc, yaxis, color="#28649B", s=55, zorder=3)
    ax.axvline(table.iloc[0].weighted_oof_auc, color="#777777", ls="--", lw=1)
    for i, row in table.iterrows():
        ax.annotate(f"{row.weighted_oof_auc:.6f} ({row.weighted_delta:+.6f})", (row.weighted_oof_auc, i),
                    xytext=(9, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(yaxis, labels)
    ax.invert_yaxis()
    ax.set(xlabel="Mean weighted OOF AUC across five seeds (higher is better)",
           title="Gentle LightGBM regularization: one change at a time")
    lo, hi = table.weighted_oof_auc.min(), table.weighted_oof_auc.max()
    span = max(hi-lo, .001)
    ax.set_xlim(lo-.15*span, hi+1.3*span)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.2)
    fig.text(.5, .01, "Same five grouped folds per seed. Parentheses show changes from baseline; exploratory CV, not a new test set.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
