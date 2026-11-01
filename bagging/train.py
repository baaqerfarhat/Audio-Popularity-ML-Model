from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


TARGET = "Popularity_Type"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--seeds", default="42,123,2026,31415,9001")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_best_parameters(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary["best_parameters"]


def train_catboost_bag(
    module: ModuleType,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    parameters: dict,
    seeds: list[int],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y))
    test_predictions = []
    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        fold_validation = []
        for seed in seeds:
            model = module.build_model(parameters, seed, -1, "GPU", device)
            model.fit(
                X.iloc[train_indices],
                y.iloc[train_indices],
                cat_features=module.CATEGORICAL_FEATURES,
            )
            fold_validation.append(
                model.predict_proba(X.iloc[validation_indices])[:, 1]
            )
            test_predictions.append(model.predict_proba(X_test)[:, 1])
        oof[validation_indices] = np.mean(fold_validation, axis=0)
        print(f"CatBoost fold {fold}/5 complete")
    return oof, np.mean(test_predictions, axis=0)


def train_lightgbm_bag(
    module: ModuleType,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    parameters: dict,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y))
    test_predictions = []
    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        fold_validation = []
        for seed in seeds:
            model = module.build_model(parameters, seed, -1)
            model.fit(
                X.iloc[train_indices],
                y.iloc[train_indices],
                categorical_feature=module.CATEGORICAL_FEATURES,
            )
            fold_validation.append(
                model.predict_proba(X.iloc[validation_indices])[:, 1]
            )
            test_predictions.append(model.predict_proba(X_test)[:, 1])
        oof[validation_indices] = np.mean(fold_validation, axis=0)
        print(f"LightGBM fold {fold}/5 complete")
    return oof, np.mean(test_predictions, axis=0)


def rank_percentile(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def search_blend(
    y: np.ndarray,
    catboost_oof: np.ndarray,
    lightgbm_oof: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    rows = []
    predictions = {}
    for method, cat_values, lgb_values in [
        ("probability", catboost_oof, lightgbm_oof),
        (
            "rank",
            rank_percentile(catboost_oof),
            rank_percentile(lightgbm_oof),
        ),
    ]:
        for step in range(101):
            catboost_weight = step / 100
            blended = (
                catboost_weight * cat_values
                + (1 - catboost_weight) * lgb_values
            )
            fold_scores = [
                roc_auc_score(
                    y[fold_ids == fold],
                    blended[fold_ids == fold],
                )
                for fold in sorted(np.unique(fold_ids))
            ]
            rows.append(
                {
                    "method": method,
                    "catboost_weight": catboost_weight,
                    "lightgbm_weight": 1 - catboost_weight,
                    "overall_auc": roc_auc_score(y, blended),
                    "mean_fold_auc": np.mean(fold_scores),
                    "std_fold_auc": np.std(fold_scores),
                }
            )
            predictions[(method, catboost_weight)] = blended
    results = pd.DataFrame(rows).sort_values(
        ["overall_auc", "mean_fold_auc"],
        ascending=False,
    )
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    best = results.iloc[0]
    prediction = predictions[
        (str(best["method"]), float(best["catboost_weight"]))
    ]
    return results, best, prediction


def make_fold_ids(
    length: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    fold_ids = np.zeros(length, dtype=int)
    for fold, (_, validation_indices) in enumerate(splits, start=1):
        fold_ids[validation_indices] = fold
    return fold_ids


def save_fold_metrics(
    y: np.ndarray,
    catboost_oof: np.ndarray,
    lightgbm_oof: np.ndarray,
    ensemble_oof: np.ndarray,
    fold_ids: np.ndarray,
    results_dir: Path,
) -> pd.DataFrame:
    rows = []
    for fold in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold
        rows.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "catboost_auc": roc_auc_score(y[mask], catboost_oof[mask]),
                "lightgbm_auc": roc_auc_score(y[mask], lightgbm_oof[mask]),
                "ensemble_auc": roc_auc_score(y[mask], ensemble_oof[mask]),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    return metrics


def build_ensemble_test(
    catboost_test: np.ndarray,
    lightgbm_test: np.ndarray,
    best: pd.Series,
) -> np.ndarray:
    if best["method"] == "rank":
        catboost_test = rank_percentile(catboost_test)
        lightgbm_test = rank_percentile(lightgbm_test)
    catboost_weight = float(best["catboost_weight"])
    return (
        catboost_weight * catboost_test
        + (1 - catboost_weight) * lightgbm_test
    )


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
        label="selected",
        zorder=5,
    )
    ax.set(
        xlabel="CatBoost weight",
        ylabel="OOF ROC-AUC",
        title="Bagged ensemble weight search",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "auc_by_weight.png", dpi=180)
    plt.close(fig)


def plot_roc(
    y: np.ndarray,
    catboost_oof: np.ndarray,
    lightgbm_oof: np.ndarray,
    ensemble_oof: np.ndarray,
    results_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probability in [
        ("Bagged CatBoost", catboost_oof),
        ("Bagged LightGBM", lightgbm_oof),
        ("Bagged ensemble", ensemble_oof),
    ]:
        fpr, tpr, _ = roc_curve(y, probability)
        auc = roc_auc_score(y, probability)
        width = 2.5 if name == "Bagged ensemble" else 1.5
        ax.plot(fpr, tpr, linewidth=width, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Seed and fold bagging OOF ROC",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(results_dir / "roc_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    cat_module = load_module(
        "catboost_training",
        args.project_root / "catboost" / "train.py",
    )
    lgb_module = load_module(
        "lightgbm_training",
        args.project_root / "lightgbm" / "train.py",
    )

    cat_train, cat_test, sample_submission = cat_module.load_data(
        args.project_root / "data"
    )
    cat_X, y, groups, source_rows, _ = cat_module.prepare_training_data(
        cat_train
    )
    cat_X_test = cat_module.prepare_features(cat_test)
    splits = cat_module.make_splits(cat_X, y, groups, seeds[0], 5)

    lgb_train, lgb_test, _ = lgb_module.load_data(args.project_root / "data")
    lgb_X, lgb_y, lgb_groups, lgb_source_rows, _ = (
        lgb_module.prepare_training_data(lgb_train)
    )
    lgb_X_test = lgb_module.prepare_features(lgb_test)
    lgb_X, lgb_X_test = lgb_module.align_categories(lgb_X, lgb_X_test)
    if not y.equals(lgb_y) or not source_rows.equals(lgb_source_rows):
        raise ValueError("CatBoost and LightGBM training rows do not align.")
    if not groups.equals(lgb_groups):
        raise ValueError("CatBoost and LightGBM groups do not align.")

    cat_parameters = load_best_parameters(
        args.project_root / "catboost" / "results" / "summary.json"
    )
    lgb_parameters = load_best_parameters(
        args.project_root / "lightgbm" / "results" / "summary.json"
    )
    catboost_oof, catboost_test = train_catboost_bag(
        cat_module,
        cat_X,
        y,
        cat_X_test,
        splits,
        cat_parameters,
        seeds,
        args.device,
    )
    lightgbm_oof, lightgbm_test = train_lightgbm_bag(
        lgb_module,
        lgb_X,
        y,
        lgb_X_test,
        splits,
        lgb_parameters,
        seeds,
    )

    y_array = y.to_numpy()
    fold_ids = make_fold_ids(len(y), splits)
    weight_results, best, ensemble_oof = search_blend(
        y_array,
        catboost_oof,
        lightgbm_oof,
        fold_ids,
    )
    ensemble_test = build_ensemble_test(
        catboost_test,
        lightgbm_test,
        best,
    )
    weight_results.to_csv(
        args.results_dir / "weight_search.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "source_row": source_rows.astype(int),
            "actual_label": np.where(y_array == 1, "High", "Low"),
            "catboost_probability": catboost_oof,
            "lightgbm_probability": lightgbm_oof,
            "ensemble_probability": ensemble_oof,
            "fold": fold_ids,
        }
    ).to_csv(args.results_dir / "oof_predictions.csv", index=False)

    for name, probability in [
        ("catboost_submission.csv", catboost_test),
        ("lightgbm_submission.csv", lightgbm_test),
        ("submission.csv", ensemble_test),
    ]:
        submission = sample_submission.copy()
        submission["ID"] = cat_test["ID"].to_numpy()
        submission[TARGET] = probability
        submission.to_csv(args.results_dir / name, index=False)

    fold_metrics = save_fold_metrics(
        y_array,
        catboost_oof,
        lightgbm_oof,
        ensemble_oof,
        fold_ids,
        args.results_dir,
    )
    plot_weight_search(weight_results, best, args.results_dir)
    plot_roc(
        y_array,
        catboost_oof,
        lightgbm_oof,
        ensemble_oof,
        args.results_dir,
    )
    summary = {
        "seeds": seeds,
        "folds": 5,
        "models_per_family": len(seeds) * 5,
        "method": str(best["method"]),
        "catboost_weight": float(best["catboost_weight"]),
        "lightgbm_weight": float(best["lightgbm_weight"]),
        "catboost_oof_auc": float(roc_auc_score(y_array, catboost_oof)),
        "lightgbm_oof_auc": float(roc_auc_score(y_array, lightgbm_oof)),
        "ensemble_oof_auc": float(roc_auc_score(y_array, ensemble_oof)),
        "mean_fold_auc": float(fold_metrics["ensemble_auc"].mean()),
        "std_fold_auc": float(fold_metrics["ensemble_auc"].std(ddof=1)),
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"Selected {best['method']} blend: "
        f"CatBoost={best['catboost_weight']:.2f}, "
        f"LightGBM={best['lightgbm_weight']:.2f}"
    )
    print(f"Bagged ensemble OOF AUC: {best['overall_auc']:.6f}")
    print(f"Results written to: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
