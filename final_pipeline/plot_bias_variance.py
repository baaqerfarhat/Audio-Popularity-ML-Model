"""Render reproducible scientific figures from completed diagnostic tables."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "experiments/bias_variance_2026_11_01")
    args = ap.parse_args()
    out = args.results
    data = pd.read_csv(out / "model_summary.csv")
    curve = pd.read_csv(out / "learning_curve.csv")
    names = {"lgb_tuned": "LightGBM tuned", "lgb_deep": "LightGBM deep", "xgb_tuned": "XGBoost tuned",
             "cat_tuned": "CatBoost tuned", "cat_base": "CatBoost base", "extra_trees": "ExtraTrees",
             "lgb_regularized": "LightGBM regularized"}
    blue, gold = "#28649B", "#B07113"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ypos = np.arange(len(data))
    for i, row in data.iterrows():
        ax.plot([row.validation_auc, row.train_auc], [i, i], color="#B3B3B3", lw=1.5, zorder=1)
    ax.scatter(data.train_auc, ypos, label="Training", color=gold, marker="s", s=52, zorder=3)
    ax.scatter(data.validation_auc, ypos, label="Validation", color=blue, marker="o", s=52, zorder=3)
    ax.set_yticks(ypos, [names[n] for n in data.model])
    ax.invert_yaxis()
    ax.set(xlim=(.78, 1.01), xlabel="Mean fold AUC (higher is better)", title="Training–validation gaps")
    ax.grid(axis="x", alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .96), ncol=2, frameon=False)
    ax = axes[1]
    ax.plot(curve.train_rows, curve.train_auc, color=gold, marker="s", lw=2, label="Training")
    ax.plot(curve.train_rows, curve.validation_auc, color=blue, marker="o", lw=2, label="Validation")
    for _, row in curve.iterrows():
        ax.annotate(f"{row.validation_auc:.3f}", (row.train_rows, row.validation_auc),
                    xytext=(0, -18), textcoords="offset points", ha="center", color=blue)
    ax.set(xlabel="Mean training rows per fold", ylabel="Mean fold AUC", ylim=(.78, 1.01),
           title="LightGBM tuned: learning curve")
    ax.set_xticks(curve.train_rows, [f"{n:,.0f}" for n in curve.train_rows])
    ax.grid(alpha=.2)
    fig.suptitle("Audio-only pipeline: bias–variance diagnostics", fontsize=16, y=1.02)
    fig.text(.5, .01, "Means over 15 grouped folds (3 seeds). Training scores are in-sample; this is not an exact bias–variance decomposition.",
             ha="center", fontsize=9, color="#505050")
    fig.tight_layout(rect=(0, .06, 1, .88))
    fig.savefig(out / "bias_variance.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "bias_variance.svg", bbox_inches="tight")
    plt.close(fig)
    print(out / "bias_variance.png")


if __name__ == "__main__":
    main()
