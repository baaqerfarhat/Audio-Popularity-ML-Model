from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from xgboost import XGBClassifier


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
RAW_CATEGORICAL_FEATURES = ["mode", "key", "time_signature"]
ENGINEERED_NUMERIC_FEATURES = [
    "release_year",
    "release_month",
    "release_day",
    "release_ordinal",
]
ENGINEERED_CATEGORICAL_FEATURES = ["release_precision"]
ENGINEERED_FEATURES = (
    ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
)
CATEGORICAL_FEATURES = (
    RAW_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
)
RAW_MODEL_FEATURES = NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES
MODEL_FEATURES = (
    NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES + CATEGORICAL_FEATURES
)
TARGET = "Popularity_Type"
BASELINE_PARAMETERS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.0,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "scale_pos_weight": 1.84,
    "max_bin": 256,
}
PARAMETER_SPACE = {
    "n_estimators": [300, 500, 700, 1000],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5, 10, 20],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "gamma": [0.0, 0.1, 0.5, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0, 5.0],
    "reg_lambda": [1.0, 5.0, 10.0, 20.0],
    "scale_pos_weight": [1.0, 1.84],
    "max_bin": [128, 256, 512],
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--n-configs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    required = set(
        RAW_MODEL_FEATURES
        + ["track_album_release_date", "uri", TARGET]
    )
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")
    return train, test, sample_submission


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[RAW_MODEL_FEATURES].copy()
    release_date = frame["track_album_release_date"].astype(str)
    X["release_year"] = pd.to_numeric(
        release_date.str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )
    X["release_month"] = pd.to_numeric(
        release_date.str.extract(r"^\d{4}-(\d{2})", expand=False),
        errors="coerce",
    )
    X["release_day"] = pd.to_numeric(
        release_date.str.extract(r"^\d{4}-\d{2}-(\d{2})", expand=False),
        errors="coerce",
    )
    parsed_release_date = pd.to_datetime(
        release_date,
        format="mixed",
        errors="coerce",
    )
    X["release_ordinal"] = (
        parsed_release_date - pd.Timestamp("1950-01-01")
    ).dt.days
    X["release_precision"] = np.select(
        [
            release_date.str.match(r"^\d{4}$"),
            release_date.str.match(r"^\d{4}-\d{2}$"),
        ],
        ["year", "month"],
        default="day",
    )
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].fillna("missing").astype(str).astype("category")
    return X[MODEL_FEATURES]


def align_categories(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for column in CATEGORICAL_FEATURES:
        categories = sorted(
            set(train_features[column].astype(str))
            | set(test_features[column].astype(str))
        )
        train_features[column] = pd.Categorical(
            train_features[column].astype(str),
            categories=categories,
        )
        test_features[column] = pd.Categorical(
            test_features[column].astype(str),
            categories=categories,
        )
    return train_features, test_features


def prepare_training_data(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[int]]:
    valid_mask = train[RAW_MODEL_FEATURES].notna().any(axis=1)
    removed_source_rows = train.index[~valid_mask].astype(int).tolist()
    clean = train.loc[valid_mask].copy()
    X = prepare_features(clean).reset_index(drop=True)
    y = clean[TARGET].map({"Low": 0, "High": 1}).reset_index(drop=True)
    groups = clean["uri"].reset_index(drop=True)
    source_rows = pd.Series(clean.index.to_numpy(), name="source_row")
    return X, y, groups, source_rows, removed_source_rows


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seed: int,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(X, y, groups))


def build_model(
    parameters: dict[str, int | float],
    seed: int,
    device: str,
    n_jobs: int,
) -> XGBClassifier:
    return XGBClassifier(
        **parameters,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device=device,
        enable_categorical=True,
        random_state=seed,
        n_jobs=n_jobs,
    )


def select_configurations(
    n_configs: int,
    seed: int,
) -> list[dict[str, int | float]]:
    sampled = list(
        ParameterSampler(
            PARAMETER_SPACE,
            n_iter=max(n_configs - 1, 1),
            random_state=seed,
        )
    )
    return [BASELINE_PARAMETERS.copy()] + sampled


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    configurations: list[dict[str, int | float]],
    seed: int,
    device: str,
    n_jobs: int,
    results_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    rows = []
    for number, parameters in enumerate(configurations, start=1):
        started = time.perf_counter()
        validation_scores = []
        training_scores = []
        for train_indices, validation_indices in splits:
            model = build_model(parameters, seed, device, n_jobs)
            model.fit(X.iloc[train_indices], y.iloc[train_indices])
            train_probability = model.predict_proba(X.iloc[train_indices])[:, 1]
            validation_probability = model.predict_proba(
                X.iloc[validation_indices]
            )[:, 1]
            training_scores.append(
                roc_auc_score(y.iloc[train_indices], train_probability)
            )
            validation_scores.append(
                roc_auc_score(y.iloc[validation_indices], validation_probability)
            )
        row = {"configuration": number, **parameters}
        row.update(
            {
                "mean_validation_auc": float(np.mean(validation_scores)),
                "std_validation_auc": float(np.std(validation_scores, ddof=0)),
                "mean_training_auc": float(np.mean(training_scores)),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        rows.append(row)
        print(
            f"{number}/{len(configurations)} "
            f"AUC={np.mean(validation_scores):.6f}"
        )

    results = pd.DataFrame(rows).sort_values(
        "mean_validation_auc",
        ascending=False,
    )
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    results.to_csv(results_dir / "tuning_results.csv", index=False)
    best_number = int(results.iloc[0]["configuration"])
    return results, configurations[best_number - 1]


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    source_rows: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    parameters: dict[str, int | float],
    seed: int,
    device: str,
    n_jobs: int,
    results_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    oof_probability = np.full(len(y), np.nan)
    fold_rows = []
    curves = []
    for fold, (train_indices, validation_indices) in enumerate(splits, start=1):
        model = build_model(parameters, seed, device, n_jobs)
        model.fit(X.iloc[train_indices], y.iloc[train_indices])
        probability = model.predict_proba(X.iloc[validation_indices])[:, 1]
        oof_probability[validation_indices] = probability
        auc = roc_auc_score(y.iloc[validation_indices], probability)
        fpr, tpr, _ = roc_curve(y.iloc[validation_indices], probability)
        curves.append((fpr, tpr, auc))
        fold_rows.append(
            {
                "fold": fold,
                "training_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "training_groups": groups.iloc[train_indices].nunique(),
                "validation_groups": groups.iloc[validation_indices].nunique(),
                "validation_auc": auc,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "source_row": source_rows.astype(int),
            "actual_label": np.where(y == 1, "High", "Low"),
            "high_probability": oof_probability,
        }
    ).to_csv(results_dir / "oof_predictions.csv", index=False)
    plot_roc_curves(curves, results_dir)
    return fold_metrics, oof_probability


def plot_roc_curves(
    curves: list[tuple[np.ndarray, np.ndarray, float]],
    results_dir: Path,
) -> None:
    mean_fpr = np.linspace(0, 1, 501)
    tprs = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for fold, (fpr, tpr, auc) in enumerate(curves, start=1):
        ax.plot(fpr, tpr, alpha=0.45, label=f"Fold {fold} (AUC={auc:.3f})")
        interpolated = np.interp(mean_fpr, fpr, tpr)
        interpolated[0] = 0
        tprs.append(interpolated)
    mean_tpr = np.mean(tprs, axis=0)
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
        title="XGBoost grouped 5-fold validation ROC curves",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "roc_curves.png", dpi=180)
    plt.close(fig)


def save_feature_importance(
    model: XGBClassifier,
    results_dir: Path,
) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": MODEL_FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    importance.to_csv(results_dir / "feature_importance.csv", index=False)
    ordered = importance.sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(ordered["feature"], ordered["importance"], color="#2a9d8f")
    ax.set(
        xlabel="XGBoost feature importance",
        ylabel="Feature",
        title="XGBoost feature importance",
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "feature_importance.png", dpi=180)
    plt.close(fig)
    return importance


def plot_tuning(results: pd.DataFrame, results_dir: Path) -> None:
    ordered = results.sort_values("mean_validation_auc")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(
        ordered["configuration"].astype(str),
        ordered["mean_validation_auc"],
        xerr=ordered["std_validation_auc"],
        color="#457b9d",
    )
    ax.set(
        xlabel="Mean grouped validation ROC-AUC",
        ylabel="Configuration",
        title="XGBoost hyperparameter search",
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "tuning_results.png", dpi=180)
    plt.close(fig)


def save_summary(
    parameters: dict[str, int | float],
    tuning_results: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    device: str,
    results_dir: Path,
) -> None:
    summary = {
        "model": "XGBClassifier",
        "device": device,
        "engineered_features": ENGINEERED_FEATURES,
        "tuning": "3-fold StratifiedGroupKFold grouped by uri",
        "validation": "5-fold StratifiedGroupKFold grouped by uri",
        "best_parameters": parameters,
        "mean_validation_auc": float(fold_metrics["validation_auc"].mean()),
        "std_validation_auc": float(
            fold_metrics["validation_auc"].std(ddof=1)
        ),
        "configurations_tested": int(len(tuning_results)),
        "top_features": feature_importance["feature"].head(10).tolist(),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# XGBoost Results",
        "",
        f"- Mean grouped 5-fold validation AUC: "
        f"`{fold_metrics['validation_auc'].mean():.6f}`",
        f"- Best parameters: `{json.dumps(parameters)}`",
    ]
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train, test, sample_submission = load_data(args.data_dir)
    X, y, groups, source_rows, _ = prepare_training_data(train)
    X_test = prepare_features(test)
    X, X_test = align_categories(X, X_test)
    tuning_splits = make_splits(X, y, groups, args.seed, 3)
    evaluation_splits = make_splits(X, y, groups, args.seed, 5)
    configurations = select_configurations(args.n_configs, args.seed)
    tuning_results, best_parameters = tune_model(
        X,
        y,
        tuning_splits,
        configurations,
        args.seed,
        args.device,
        args.n_jobs,
        args.results_dir,
    )
    plot_tuning(tuning_results, args.results_dir)
    fold_metrics, _ = evaluate_model(
        X,
        y,
        groups,
        source_rows,
        evaluation_splits,
        best_parameters,
        args.seed,
        args.device,
        args.n_jobs,
        args.results_dir,
    )
    final_model = build_model(
        best_parameters,
        args.seed,
        args.device,
        args.n_jobs,
    )
    final_model.fit(X, y)
    feature_importance = save_feature_importance(final_model, args.results_dir)
    submission = sample_submission.copy()
    submission["ID"] = test["ID"].to_numpy()
    submission[TARGET] = final_model.predict_proba(X_test)[:, 1]
    submission.to_csv(args.results_dir / "submission.csv", index=False)
    save_summary(
        best_parameters,
        tuning_results,
        fold_metrics,
        feature_importance,
        args.device,
        args.results_dir,
    )
    print(f"Best parameters: {best_parameters}")
    print(
        "Mean grouped 5-fold validation AUC: "
        f"{fold_metrics['validation_auc'].mean():.6f}"
    )
    print(f"Results written to: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
