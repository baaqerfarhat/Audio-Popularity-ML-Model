"""Matched evaluation of release-period specialists from a train.py run made with --specialists ... --save-predictions.

Writes, next to predictions.npz: seed_metrics.csv, summary.csv, period_metrics.csv, pair_summary.csv,
fixed_blend_comparison.csv, fixed_blend_seed_metrics.csv, candidate submissions, metadata.json, RESULTS.md.

  python evaluate_period_specialists.py --run experiments/period_specialists_10fold_2026_11_02 --model lgb_tuned \
         --blend lgb_tuned,xgb_tuned,cat_base,extra_trees
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, log_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T


def pairwise_auc(y, p, w, pos_period, neg_period, periods):
    """Weighted fraction of correctly ordered (positive from pos_period, negative from neg_period) pairs; ties = 1/2."""
    pos = (y == 1) & (periods == pos_period); neg = (y == 0) & (periods == neg_period)
    ps, pw = p[pos], w[pos]
    order = np.argsort(p[neg]); ns, nw = p[neg][order], w[neg][order]
    mass = pw.sum() * nw.sum()
    if mass == 0:
        return float("nan"), 0.0
    cum = np.r_[0.0, nw.cumsum()]
    left = np.searchsorted(ns, ps, side="left"); right = np.searchsorted(ns, ps, side="right")
    concordant = cum[left] + 0.5 * (cum[right] - cum[left])
    return float(np.dot(pw, concordant) / mass), float(mass)


def rank01(v):
    return rankdata(v) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="lgb_tuned", help="the specialised component")
    ap.add_argument("--blend", default="lgb_tuned,xgb_tuned,cat_base,extra_trees",
                    help="fixed equal-weight rank-average members (the 0.870-public ensemble)")
    args = ap.parse_args()
    run = os.path.abspath(args.run)
    d = np.load(os.path.join(run, "predictions.npz"), allow_pickle=True)
    y, W, period, period_te, seeds = d["y"], d["W"], d["period"], d["period_te"], d["seeds"]
    test_ids = d["test_ids"]
    cut = int(json.load(open(os.path.join(run, "summary.json")))["period_cut"])
    labels = {0: f"before_{cut}", 1: f"{cut}_onward", -1: "unknown"}
    variants = {"global": args.model, "specialist": f"{args.model}__specialist", "mix": f"{args.model}__mix"}
    oof = {k: d[f"oof_{v}"] for k, v in variants.items()}; tst = {k: d[f"test_{v}"] for k, v in variants.items()}
    cohorts = {"overall": np.ones(len(y), bool), labels[0]: period == 0, labels[1]: period == 1}

    # ---- per-seed metrics (pooled OOF per seed), overall and within each period
    rows, pair_rows = [], []
    for k in variants:
        for si, seed in enumerate(seeds):
            p = oof[k][si]
            for cname, mask in cohorts.items():
                rows.append(dict(variant=k, seed=int(seed), cohort=cname, rows=int(mask.sum()),
                                 auc=roc_auc_score(y[mask], p[mask]), weighted_auc=roc_auc_score(y[mask], p[mask], sample_weight=W[mask]),
                                 logloss=log_loss(y[mask], np.clip(p[mask], 1e-6, 1 - 1e-6), sample_weight=W[mask]),
                                 mean_p=float(p[mask].mean()), observed_high_rate=float(y[mask].mean())))
            for a in (0, 1):
                for b in (0, 1):
                    auc, mass = pairwise_auc(y, p, W, a, b, period)
                    pair_rows.append(dict(variant=k, seed=int(seed), positive_period=labels[a], negative_period=labels[b],
                                          weighted_pair_auc=auc, pair_weight=mass))
    seed_df = pd.DataFrame(rows); seed_df.to_csv(os.path.join(run, "seed_metrics.csv"), index=False)
    pair_df = pd.DataFrame(pair_rows); pair_df.to_csv(os.path.join(run, "pair_metrics.csv"), index=False)

    summary = []
    for (k, c), f in seed_df.groupby(["variant", "cohort"], sort=False):
        base = seed_df[(seed_df.variant == "global") & (seed_df.cohort == c)].set_index("seed")
        f = f.set_index("seed"); delta = f.weighted_auc - base.weighted_auc; delta_plain = f.auc - base.auc
        summary.append(dict(variant=k, cohort=c, auc=f.auc.mean(), weighted_auc=f.weighted_auc.mean(),
                            weighted_delta_vs_global=delta.mean(), weighted_delta_sd=delta.std(ddof=1), seed_wins=int((delta > 0).sum()),
                            plain_delta_vs_global=delta_plain.mean(), logloss=f.logloss.mean(), mean_p=f.mean_p.mean()))
    summary = pd.DataFrame(summary); summary.to_csv(os.path.join(run, "summary.csv"), index=False)
    pair_summary = pair_df.groupby(["variant", "positive_period", "negative_period"], sort=False).weighted_pair_auc.mean().reset_index()
    pair_summary.to_csv(os.path.join(run, "pair_summary.csv"), index=False)

    # ---- fixed-weight ensemble: equal rank average of the four members; only the specialised component changes
    members = args.blend.split(",")
    blend_rows, blend_seed_rows = [], []
    for k in variants:
        comp_oof = {m: d[f"oof_{m}"] for m in members}; comp_tst = {m: d[f"test_{m}"] for m in members}
        comp_oof[args.model] = oof[k]; comp_tst[args.model] = tst[k]
        # (a) seed-averaged probabilities -> rank -> equal-weight average (the aggregation used for submissions)
        blend_oof = np.mean([rank01(comp_oof[m].mean(0)) for m in members], axis=0)
        blend_tst = np.mean([rank01(comp_tst[m].mean(0)) for m in members], axis=0)
        blend_rows.append(dict(variant=k, auc=roc_auc_score(y, blend_oof), weighted_auc=roc_auc_score(y, blend_oof, sample_weight=W)))
        # (b) per-seed rank blends -> mean per-seed AUC (comparable with the single-model table above)
        for si, seed in enumerate(seeds):
            bo = np.mean([rank01(comp_oof[m][si]) for m in members], axis=0)
            blend_seed_rows.append(dict(variant=k, seed=int(seed), auc=roc_auc_score(y, bo), weighted_auc=roc_auc_score(y, bo, sample_weight=W)))
        T.build_submission(pd.DataFrame({"ID": test_ids, T.TARGET: 0.5}), test_ids, blend_tst).to_csv(
            os.path.join(run, f"submission_fixed_blend_{k}.csv"), index=False)
        T.build_submission(pd.DataFrame({"ID": test_ids, T.TARGET: 0.5}), test_ids, tst[k].mean(0)).to_csv(
            os.path.join(run, f"submission_{args.model}_{k}.csv"), index=False)
    blend_df = pd.DataFrame(blend_rows); blend_df["weighted_delta_vs_global"] = blend_df.weighted_auc - blend_df.weighted_auc.iloc[0]
    blend_df["auc_delta_vs_global"] = blend_df.auc - blend_df.auc.iloc[0]
    blend_df.to_csv(os.path.join(run, "fixed_blend_comparison.csv"), index=False)
    bs = pd.DataFrame(blend_seed_rows); bs.to_csv(os.path.join(run, "fixed_blend_seed_metrics.csv"), index=False)
    bs_sum = []
    for k, f in bs.groupby("variant", sort=False):
        base = bs[bs.variant == "global"].set_index("seed"); f = f.set_index("seed"); delta = f.weighted_auc - base.weighted_auc
        bs_sum.append(dict(variant=k, mean_auc=f.auc.mean(), mean_weighted_auc=f.weighted_auc.mean(),
                           weighted_delta_vs_global=delta.mean(), seed_wins=int((delta > 0).sum())))
    bs_sum = pd.DataFrame(bs_sum)

    # ---- metadata for reproducibility
    here = os.path.dirname(os.path.abspath(__file__))
    srcs = ["train.py", "evaluate_period_specialists.py", "test_period_specialists.py", "tuned_lgb.json", "tuned_xgb.json", "tuned_cat.json"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
    except Exception:
        commit = "unavailable"
    meta = dict(run=os.path.basename(run), seeds=[int(s) for s in seeds], n_folds=int(json.load(open(os.path.join(run, "summary.json")))["n_folds"]),
                feature_set=json.load(open(os.path.join(run, "summary.json")))["feature_set"],
                et_config=json.load(open(os.path.join(run, "summary.json")))["et_config"], specialised_model=args.model, period_cut=cut,
                specialist_weight_global=json.load(open(os.path.join(run, "summary.json")))["specialist_weight_global"],
                unknown_date_policy="global fallback", fixed_blend_members=members, fixed_blend_weights=[0.25] * len(members),
                primary_metric="mean per-seed weighted (test-like) OOF AUC", early_stopping="none (fixed iteration counts)",
                source_commit=commit, sha256={s: hashlib.sha256(open(os.path.join(here, s), "rb").read()).hexdigest() for s in srcs},
                cohorts={labels[r]: dict(train_rows=int((period == r).sum()), high_rows=int(y[period == r].sum()),
                                         test_rows=int((period_te == r).sum())) for r in (0, 1, -1)})
    json.dump(meta, open(os.path.join(run, "metadata.json"), "w"), indent=2)

    # ---- report
    fmt = lambda v: f"{v:+.6f}" if isinstance(v, float) else str(v)
    L = [f"# Release-period specialists on the 10-fold pipeline ({os.path.basename(run)})", "",
         f"Specialised component: `{args.model}`; periods: release year < {cut} / >= {cut}; unknown year -> global. "
         f"{len(seeds)} seeds x {meta['n_folds']} grouped folds; feature set `{meta['feature_set']}`, ExtraTrees `{meta['et_config']}`. "
         "Primary metric: mean per-seed weighted (test-like) OOF AUC. Exploratory: the validation splits were reused.", "",
         "## Single component (mean of per-seed pooled OOF scores)", "",
         "| variant | cohort | AUC | weighted AUC | weighted delta vs global | seed wins |", "|---|---|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        L.append(f"| {r.variant} | {r.cohort} | {r.auc:.6f} | {r.weighted_auc:.6f} | {r.weighted_delta_vs_global:+.6f} | {r.seed_wins}/{len(seeds)} |")
    L += ["", "## Cross-period score compatibility (weighted pair AUC, mean over seeds)", "",
          "| High period / Low period | " + " | ".join(variants) + " |", "|---|" + "---:|" * len(variants)]
    for a in (0, 1):
        for b in (0, 1):
            vals = [pair_summary[(pair_summary.variant == k) & (pair_summary.positive_period == labels[a]) & (pair_summary.negative_period == labels[b])].weighted_pair_auc.iloc[0] for k in variants]
            L.append(f"| {labels[a]} / {labels[b]} | " + " | ".join(f"{v:.6f}" for v in vals) + " |")
    mp = summary[summary.cohort != "overall"].pivot(index="variant", columns="cohort", values="mean_p")
    L += ["", "Mean predicted probability by period (score-scale drift check):", "", mp.round(4).to_markdown(), "",
          "## Fixed four-member ensemble (equal rank average; only the specialised component changes)", "",
          "Seed-averaged probabilities -> rank -> average (submission aggregation):", "",
          "| component variant | AUC | weighted AUC | weighted delta |", "|---|---:|---:|---:|"]
    for _, r in blend_df.iterrows():
        L.append(f"| {r.variant} | {r.auc:.6f} | {r.weighted_auc:.6f} | {r.weighted_delta_vs_global:+.6f} |")
    L += ["", "Mean per-seed AUC of per-seed rank blends (paired):", "", "| component variant | mean AUC | mean weighted AUC | weighted delta | seed wins |", "|---|---:|---:|---:|---:|"]
    for _, r in bs_sum.iterrows():
        L.append(f"| {r.variant} | {r.mean_auc:.6f} | {r.mean_weighted_auc:.6f} | {r.weighted_delta_vs_global:+.6f} | {r.seed_wins}/{len(seeds)} |")
    L += ["", "Candidate submissions: `submission_fixed_blend_mix.csv`, `submission_fixed_blend_specialist.csv`, "
          f"`submission_{args.model}_mix.csv` (baseline files: `submission_fixed_blend_global.csv`, `submission_{args.model}_global.csv`)."]
    open(os.path.join(run, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
