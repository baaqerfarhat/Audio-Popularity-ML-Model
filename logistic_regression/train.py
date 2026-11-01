"""Train and evaluate an L2-regularized logistic regression model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
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
]
BINARY_FEATURES = ["mode"]
CATEGORICAL_FEATURES = ["key", "time_signature"]
MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
DROPPED_FEATURES = [
    "analysis_url",
    "track_href",
    "uri",
    "type",
    "track_album_release_date",
]
TARGET = "Popularity_Type"
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Tune an L2 logistic regression model with grouped 5-fold CV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data",
        help="Directory containing train.csv, test.csv, and sample_submission.csv.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory where tables, plots, and predictions are written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used by GridSearchCV; -1 uses all available cores.",
    )
    return parser.parse_args()


def build_pipeline(seed: int) -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=seed,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    for path in (train_path, test_path, sample_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)

    required_train = set(MODEL_FEATURES + DROPPED_FEATURES + [TARGET])
    required_test = set(MODEL_FEATURES + DROPPED_FEATURES + ["ID"])
    missing_train = sorted(required_train - set(train.columns))
    missing_test = sorted(required_test - set(test.columns))
    if missing_train:
        raise ValueError(f"Training data is missing columns: {missing_train}")
    if missing_test:
        raise ValueError(f"Test data is missing columns: {missing_test}")

    return train, test, sample_submission


def prepare_training_data(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[int]]:
    valid_mask = train[MODEL_FEATURES].notna().any(axis=1)
    removed_source_rows = train.index[~valid_mask].astype(int).tolist()
    clean = train.loc[valid_mask].copy()

    unknown_labels = sorted(set(clean[TARGET].dropna()) - {"Low", "High"})
    if unknown_labels or clean[TARGET].isna().any():
        raise ValueError(
            f"Expected only Low/High target labels; found unexpected values: "
            f"{unknown_labels}"
        )

    X = clean[MODEL_FEATURES].reset_index(drop=True)
    y = clean[TARGET].map({"Low": 0, "High": 1}).reset_index(drop=True)
    groups = clean["uri"].reset_index(drop=True)
    source_rows = pd.Series(clean.index.to_numpy(), name="source_row")

    if groups.isna().any():
        missing = groups.isna()
        groups.loc[missing] = [f"missing_uri_{i}" for i in groups.index[missing]]

    return X, y, groups, source_rows, removed_source_rows


def make_grouped_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )
    splits = list(splitter.split(X, y, groups))

    for train_indices, validation_indices in splits:
        train_groups = set(groups.iloc[train_indices])
        validation_groups = set(groups.iloc[validation_indices])
        if train_groups & validation_groups:
            raise RuntimeError("A song URI appeared in both training and validation.")

    return splits


def save_cv_results(search: GridSearchCV, results_dir: Path) -> pd.DataFrame:
    raw = pd.DataFrame(search.cv_results_)
    cv_results = pd.DataFrame(
        {
            "C": raw["param_model__C"].astype(float),
            "mean_validation_auc": raw["mean_test_score"],
            "std_validation_auc": raw["std_test_score"],
            "mean_training_auc": raw["mean_train_score"],
            "std_training_auc": raw["std_train_score"],
            "mean_fit_time_seconds": raw["mean_fit_time"],
            "rank": raw["rank_test_score"],
        }
    ).sort_values("C")
    cv_results.to_csv(results_dir / "cv_results.csv", index=False)
    return cv_results


def evaluate_best_model(
    best_estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    results_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    oof_probabilities = np.full(len(y), np.nan)
    fold_rows: list[dict[str, float | int]] = []
    fold_curves: list[tuple[np.ndarray, np.ndarray, float]] = []

    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        estimator = clone(best_estimator)
        estimator.fit(X.iloc[train_indices], y.iloc[train_indices])

        probabilities = estimator.predict_proba(X.iloc[validation_indices])[:, 1]
        oof_probabilities[validation_indices] = probabilities
        fold_auc = roc_auc_score(y.iloc[validation_indices], probabilities)
        fpr, tpr, _ = roc_curve(y.iloc[validation_indices], probabilities)
        fold_curves.append((fpr, tpr, fold_auc))

        fold_rows.append(
            {
                "fold": fold,
                "training_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "training_groups": groups.iloc[train_indices].nunique(),
                "validation_groups": groups.iloc[validation_indices].nunique(),
                "training_high_fraction": y.iloc[train_indices].mean(),
                "validation_high_fraction": y.iloc[validation_indices].mean(),
                "validation_auc": fold_auc,
            }
        )

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("Some rows did not receive an out-of-fold prediction.")

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    plot_roc_curves(fold_curves, results_dir)
    return fold_metrics, oof_probabilities


def plot_roc_curves(
    fold_curves: list[tuple[np.ndarray, np.ndarray, float]],
    results_dir: Path,
) -> None:
    mean_fpr = np.linspace(0, 1, 501)
    interpolated_tprs = []

    fig, ax = plt.subplots(figsize=(8, 6))
    for fold, (fpr, tpr, auc) in enumerate(fold_curves, start=1):
        ax.plot(fpr, tpr, alpha=0.45, label=f"Fold {fold} (AUC={auc:.3f})")
        interpolated = np.interp(mean_fpr, fpr, tpr)
        interpolated[0] = 0
        interpolated_tprs.append(interpolated)

    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_tpr[-1] = 1
    mean_auc = np.trapezoid(mean_tpr, mean_fpr)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="black",
        linewidth=2.5,
        label=f"Mean ROC (AUC={mean_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Grouped 5-fold validation ROC curves",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "roc_curves.png", dpi=180)
    plt.close(fig)


def plot_cv_results(cv_results: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        cv_results["C"],
        cv_results["mean_validation_auc"],
        yerr=cv_results["std_validation_auc"],
        marker="o",
        capsize=4,
        label="Validation AUC",
    )
    ax.plot(
        cv_results["C"],
        cv_results["mean_training_auc"],
        marker="s",
        linestyle="--",
        label="Training AUC",
    )
    ax.set_xscale("log")
    ax.set(
        xlabel="C (inverse L2 regularization strength)",
        ylabel="ROC-AUC",
        title="Logistic regression regularization search",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "auc_by_c.png", dpi=180)
    plt.close(fig)


def save_coefficients(
    fitted_estimator: Pipeline,
    results_dir: Path,
) -> pd.DataFrame:
    preprocessor = fitted_estimator.named_steps["preprocessor"]
    model = fitted_estimator.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]

    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "odds_ratio": np.exp(np.clip(coefficients, -700, 700)),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficient_table.insert(
        0, "importance_rank", np.arange(1, len(coefficient_table) + 1)
    )
    coefficient_table.to_csv(results_dir / "coefficients.csv", index=False)

    top = coefficient_table.head(20).sort_values("coefficient")
    colors = np.where(top["coefficient"] >= 0, "#2a9d8f", "#e76f51")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel="Logistic regression coefficient",
        ylabel="Transformed feature",
        title="Top coefficients by absolute magnitude",
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "top_coefficients.png", dpi=180)
    plt.close(fig)

    return coefficient_table


def plot_class_distribution(y: pd.Series, results_dir: Path) -> None:
    counts = pd.Series(
        {"Low": int((y == 0).sum()), "High": int((y == 1).sum())}
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=["#457b9d", "#e76f51"])
    ax.bar_label(bars)
    ax.set(
        xlabel="Popularity type",
        ylabel="Number of training songs",
        title="Training class distribution",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "class_distribution.png", dpi=180)
    plt.close(fig)


def save_summary(
    results_dir: Path,
    best_c: float,
    best_cv_auc: float,
    fold_metrics: pd.DataFrame,
    y: pd.Series,
    removed_source_rows: list[int],
    coefficient_table: pd.DataFrame,
) -> None:
    summary = {
        "model": "L2-regularized logistic regression",
        "loss": "binary cross-entropy",
        "class_weight": "balanced",
        "validation": "5-fold StratifiedGroupKFold grouped by uri",
        "best_C": best_c,
        "grid_search_mean_validation_auc": best_cv_auc,
        "fold_auc_mean": float(fold_metrics["validation_auc"].mean()),
        "fold_auc_std": float(fold_metrics["validation_auc"].std(ddof=1)),
        "training_rows_used": int(len(y)),
        "class_counts": {
            "Low": int((y == 0).sum()),
            "High": int((y == 1).sum()),
        },
        "removed_zero_feature_source_rows": removed_source_rows,
        "top_10_transformed_features_by_absolute_coefficient": coefficient_table[
            "feature"
        ]
        .head(10)
        .tolist(),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Logistic Regression Results",
        "",
        f"- Best C: `{best_c:g}`",
        f"- Mean grouped 5-fold validation AUC: `{best_cv_auc:.6f}`",
        f"- Fold AUC standard deviation: "
        f"`{fold_metrics['validation_auc'].std(ddof=1):.6f}`",
        f"- Training rows used: `{len(y)}`",
        f"- Class counts: Low `{int((y == 0).sum())}`, "
        f"High `{int((y == 1).sum())}`",
        "- Class weighting: `balanced`",
        "",
        "See `cv_results.csv`, `fold_metrics.csv`, and `coefficients.csv` "
        "for the complete result tables.",
    ]
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train, test, sample_submission = load_data(args.data_dir)
    X, y, groups, source_rows, removed_source_rows = prepare_training_data(train)
    splits = make_grouped_splits(X, y, groups, args.seed)

    pipeline = build_pipeline(args.seed)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"model__C": C_VALUES},
        scoring="roc_auc",
        cv=splits,
        n_jobs=args.n_jobs,
        return_train_score=True,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)

    cv_results = save_cv_results(search, args.results_dir)
    plot_cv_results(cv_results, args.results_dir)
    fold_metrics, oof_probabilities = evaluate_best_model(
        search.best_estimator_,
        X,
        y,
        groups,
        splits,
        args.results_dir,
    )

    oof_table = pd.DataFrame(
        {
            "source_row": source_rows.astype(int),
            "actual_label": np.where(y == 1, "High", "Low"),
            "high_probability": oof_probabilities,
        }
    )
    oof_table.to_csv(args.results_dir / "oof_predictions.csv", index=False)

    final_estimator = search.best_estimator_
    coefficient_table = save_coefficients(final_estimator, args.results_dir)
    plot_class_distribution(y, args.results_dir)

    test_probabilities = final_estimator.predict_proba(test[MODEL_FEATURES])[:, 1]
    if len(sample_submission) != len(test):
        raise ValueError("sample_submission.csv and test.csv have different lengths.")
    submission = sample_submission.copy()
    submission["ID"] = test["ID"].to_numpy()
    submission[TARGET] = test_probabilities
    submission.to_csv(args.results_dir / "submission.csv", index=False)

    save_summary(
        args.results_dir,
        float(search.best_params_["model__C"]),
        float(search.best_score_),
        fold_metrics,
        y,
        removed_source_rows,
        coefficient_table,
    )

    print(f"Best C: {search.best_params_['model__C']:g}")
    print(f"Mean grouped 5-fold validation AUC: {search.best_score_:.6f}")
    print(f"Results written to: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
