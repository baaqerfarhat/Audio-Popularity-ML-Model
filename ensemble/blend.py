from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold


TARGET = "Popularity_Type"
RAW_FEATURES = [
    "speechiness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "valence",
    "tempo",
    "loudness",
    "acousticness",
    "liveness",
    "mode",
    "key",
    "time_signature",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catboost-results",
        type=Path,
        default=project_root / "catboost" / "results",
    )
    parser.add_argument(
        "--lightgbm-results",
        type=Path,
        default=project_root / "lightgbm" / "results",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=project_root / "data" / "train.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def rank_percentile(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def load_oof(
    catboost_results: Path,
    lightgbm_results: Path,
) -> pd.DataFrame:
    catboost = pd.read_csv(catboost_results / "oof_predictions.csv").rename(
        columns={"high_probability": "catboost_probability"}
    )
    lightgbm = pd.read_csv(lightgbm_results / "oof_predictions.csv").rename(
        columns={"high_probability": "lightgbm_probability"}
    )
    merged = catboost.merge(
        lightgbm,
        on=["source_row", "actual_label"],
        how="inner",
        validate="one_to_one",
    ).sort_values("source_row")
    if len(merged) != len(catboost) or len(merged) != len(lightgbm):
        raise ValueError("OOF prediction rows do not align.")
    return merged.reset_index(drop=True)


def make_fold_ids(
    train_path: Path,
    source_rows: pd.Series,
    seed: int,
) -> np.ndarray:
    train = pd.read_csv(train_path)
    valid_mask = train[RAW_FEATURES].notna().any(axis=1)
    clean = train.loc[valid_mask].copy()
    y = clean[TARGET].map({"Low": 0, "High": 1}).reset_index(drop=True)
    groups = clean["uri"].reset_index(drop=True)
    X = clean[RAW_FEATURES].reset_index(drop=True)
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )
    fold_ids = np.zeros(len(clean), dtype=int)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        fold_ids[validation_indices] = fold
    mapping = dict(zip(clean.index.astype(int), fold_ids))
    return source_rows.map(mapping).to_numpy()


def score_candidate(
    y: np.ndarray,
    probability: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[float, float, float]:
    overall_auc = roc_auc_score(y, probability)
    fold_scores = [
        roc_auc_score(y[fold_ids == fold], probability[fold_ids == fold])
        for fold in sorted(np.unique(fold_ids))
    ]
    return overall_auc, float(np.mean(fold_scores)), float(np.std(fold_scores))


def search_weights(
    oof: pd.DataFrame,
    fold_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    y = (oof["actual_label"] == "High").astype(int).to_numpy()
    cat_probability = oof["catboost_probability"].to_numpy()
    lgb_probability = oof["lightgbm_probability"].to_numpy()
    cat_rank = rank_percentile(cat_probability)
    lgb_rank = rank_percentile(lgb_probability)
    rows = []
    predictions = {}

    for method, cat_values, lgb_values in [
        ("probability", cat_probability, lgb_probability),
        ("rank", cat_rank, lgb_rank),
    ]:
        for catboost_weight in np.linspace(0, 1, 101):
            blended = (
                catboost_weight * cat_values
                + (1 - catboost_weight) * lgb_values
            )
            overall_auc, mean_fold_auc, std_fold_auc = score_candidate(
                y,
                blended,
                fold_ids,
            )
            key = (method, round(float(catboost_weight), 2))
            predictions[key] = blended
            rows.append(
                {
                    "method": method,
                    "catboost_weight": catboost_weight,
                    "lightgbm_weight": 1 - catboost_weight,
                    "overall_auc": overall_auc,
                    "mean_fold_auc": mean_fold_auc,
                    "std_fold_auc": std_fold_auc,
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["overall_auc", "mean_fold_auc"],
        ascending=False,
    )
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    best = results.iloc[0]
    key = (str(best["method"]), round(float(best["catboost_weight"]), 2))
    return results, best, predictions[key]


def build_submission(
    catboost_results: Path,
    lightgbm_results: Path,
    best: pd.Series,
    results_dir: Path,
) -> None:
    catboost = pd.read_csv(catboost_results / "submission.csv").rename(
        columns={TARGET: "catboost_probability"}
    )
    lightgbm = pd.read_csv(lightgbm_results / "submission.csv").rename(
        columns={TARGET: "lightgbm_probability"}
    )
    merged = catboost.merge(
        lightgbm,
        on="ID",
        how="inner",
        validate="one_to_one",
    )
    cat_values = merged["catboost_probability"].to_numpy()
    lgb_values = merged["lightgbm_probability"].to_numpy()
    if best["method"] == "rank":
        cat_values = rank_percentile(cat_values)
        lgb_values = rank_percentile(lgb_values)
    catboost_weight = float(best["catboost_weight"])
    merged[TARGET] = (
        catboost_weight * cat_values
        + (1 - catboost_weight) * lgb_values
    )
    merged[["ID", TARGET]].to_csv(
        results_dir / "submission.csv",
        index=False,
    )


def save_fold_metrics(
    oof: pd.DataFrame,
    ensemble_probability: np.ndarray,
    fold_ids: np.ndarray,
    results_dir: Path,
) -> pd.DataFrame:
    y = (oof["actual_label"] == "High").astype(int).to_numpy()
    rows = []
    for fold in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold
        rows.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "catboost_auc": roc_auc_score(
                    y[mask],
                    oof.loc[mask, "catboost_probability"],
                ),
                "lightgbm_auc": roc_auc_score(
                    y[mask],
                    oof.loc[mask, "lightgbm_probability"],
                ),
                "ensemble_auc": roc_auc_score(
                    y[mask],
                    ensemble_probability[mask],
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    return metrics


def plot_weight_search(
    results: pd.DataFrame,
    best: pd.Series,
    results_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in results.groupby("method"):
        group = group.sort_values("catboost_weight")
        ax.plot(
            group["catboost_weight"],
            group["overall_auc"],
            label=method,
        )
    ax.scatter(
        [best["catboost_weight"]],
        [best["overall_auc"]],
        color="black",
        zorder=5,
        label="selected",
    )
    ax.set(
        xlabel="CatBoost weight",
        ylabel="OOF ROC-AUC",
        title="CatBoost and LightGBM blend search",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "auc_by_weight.png", dpi=180)
    plt.close(fig)


def plot_roc_comparison(
    oof: pd.DataFrame,
    ensemble_probability: np.ndarray,
    results_dir: Path,
) -> None:
    y = (oof["actual_label"] == "High").astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    series = [
        ("CatBoost", oof["catboost_probability"].to_numpy()),
        ("LightGBM", oof["lightgbm_probability"].to_numpy()),
        ("Ensemble", ensemble_probability),
    ]
    for name, probability in series:
        fpr, tpr, _ = roc_curve(y, probability)
        auc = roc_auc_score(y, probability)
        width = 2.5 if name == "Ensemble" else 1.5
        ax.plot(fpr, tpr, linewidth=width, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="OOF ROC comparison",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(results_dir / "roc_comparison.png", dpi=180)
    plt.close(fig)


def save_summary(
    oof: pd.DataFrame,
    best: pd.Series,
    ensemble_probability: np.ndarray,
    fold_metrics: pd.DataFrame,
    results_dir: Path,
) -> None:
    y = (oof["actual_label"] == "High").astype(int).to_numpy()
    summary = {
        "models": ["CatBoost", "LightGBM"],
        "method": str(best["method"]),
        "catboost_weight": float(best["catboost_weight"]),
        "lightgbm_weight": float(best["lightgbm_weight"]),
        "catboost_oof_auc": float(
            roc_auc_score(y, oof["catboost_probability"])
        ),
        "lightgbm_oof_auc": float(
            roc_auc_score(y, oof["lightgbm_probability"])
        ),
        "ensemble_oof_auc": float(
            roc_auc_score(y, ensemble_probability)
        ),
        "mean_fold_auc": float(fold_metrics["ensemble_auc"].mean()),
        "std_fold_auc": float(fold_metrics["ensemble_auc"].std(ddof=1)),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Ensemble Results",
        "",
        f"- Method: `{summary['method']}`",
        f"- CatBoost weight: `{summary['catboost_weight']:.2f}`",
        f"- LightGBM weight: `{summary['lightgbm_weight']:.2f}`",
        f"- CatBoost OOF AUC: `{summary['catboost_oof_auc']:.6f}`",
        f"- LightGBM OOF AUC: `{summary['lightgbm_oof_auc']:.6f}`",
        f"- Ensemble OOF AUC: `{summary['ensemble_oof_auc']:.6f}`",
    ]
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    oof = load_oof(args.catboost_results, args.lightgbm_results)
    fold_ids = make_fold_ids(args.train_path, oof["source_row"], args.seed)
    weight_results, best, ensemble_probability = search_weights(oof, fold_ids)
    weight_results.to_csv(args.results_dir / "weight_search.csv", index=False)
    oof_output = oof.copy()
    oof_output["ensemble_probability"] = ensemble_probability
    oof_output["fold"] = fold_ids
    oof_output.to_csv(args.results_dir / "oof_predictions.csv", index=False)
    fold_metrics = save_fold_metrics(
        oof,
        ensemble_probability,
        fold_ids,
        args.results_dir,
    )
    build_submission(
        args.catboost_results,
        args.lightgbm_results,
        best,
        args.results_dir,
    )
    plot_weight_search(weight_results, best, args.results_dir)
    plot_roc_comparison(oof, ensemble_probability, args.results_dir)
    save_summary(
        oof,
        best,
        ensemble_probability,
        fold_metrics,
        args.results_dir,
    )
    print(
        f"Selected {best['method']} blend: "
        f"CatBoost={best['catboost_weight']:.2f}, "
        f"LightGBM={best['lightgbm_weight']:.2f}"
    )
    print(f"Ensemble OOF AUC: {best['overall_auc']:.6f}")
    print(f"Results written to: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
