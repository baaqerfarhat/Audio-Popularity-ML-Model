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
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier


TARGET = "Popularity_Type"
CATBOOST_PARAMETERS = {
    "iterations": 600,
    "learning_rate": 0.025,
    "depth": 6,
    "l2_leaf_reg": 5,
    "auto_class_weights": "Balanced",
    "random_strength": 1,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0,
    "boosting_type": "Plain",
    "border_count": 128,
    "one_hot_max_size": 2,
    "leaf_estimation_iterations": 3,
}
LIGHTGBM_PARAMETERS = {
    "n_estimators": 700,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 10,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "class_weight": "balanced",
}
EBM_PARAMETERS = {
    "interactions": 30,
    "max_leaves": 3,
    "learning_rate": 0.03,
    "min_samples_leaf": 4,
    "max_bins": 256,
    "max_interaction_bins": 32,
    "outer_bags": 8,
    "max_rounds": 10000,
    "early_stopping_rounds": 100,
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def add_audio_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    result["log_duration"] = np.log1p(result["duration_ms"].clip(lower=0))
    result["tempo_half_distance"] = np.minimum(
        np.abs(result["tempo"] - 60),
        np.abs(result["tempo"] - 120),
    )
    result["loud_energy"] = result["loudness"] * result["energy"]
    result["energy_acoustic"] = result["energy"] - result["acousticness"]
    result["nonvocal"] = result["instrumentalness"] * (
        1 - result["speechiness"]
    )
    result["dance_energy"] = result["danceability"] * result["energy"]
    return result


def train_catboost(
    module: ModuleType,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int],
    task_type: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y))
    test_predictions = []
    for fold, (train_indices, validation_indices) in enumerate(splits, 1):
        fold_predictions = []
        for seed in seeds:
            model = module.build_model(
                CATBOOST_PARAMETERS,
                seed,
                -1,
                task_type,
                device,
            )
            model.fit(
                X.iloc[train_indices],
                y.iloc[train_indices],
                cat_features=module.CATEGORICAL_FEATURES,
            )
            fold_predictions.append(
                model.predict_proba(X.iloc[validation_indices])[:, 1]
            )
            test_predictions.append(model.predict_proba(X_test)[:, 1])
        oof[validation_indices] = np.mean(fold_predictions, axis=0)
        print(f"Enhanced CatBoost fold {fold}/5 complete")
    return oof, np.mean(test_predictions, axis=0)


def train_lightgbm(
    module: ModuleType,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y))
    test_predictions = []
    for fold, (train_indices, validation_indices) in enumerate(splits, 1):
        fold_predictions = []
        for seed in seeds:
            model = module.build_model(LIGHTGBM_PARAMETERS, seed, -1)
            model.fit(
                X.iloc[train_indices],
                y.iloc[train_indices],
                categorical_feature=module.CATEGORICAL_FEATURES,
            )
            fold_predictions.append(
                model.predict_proba(X.iloc[validation_indices])[:, 1]
            )
            test_predictions.append(model.predict_proba(X_test)[:, 1])
        oof[validation_indices] = np.mean(fold_predictions, axis=0)
        print(f"Enhanced LightGBM fold {fold}/5 complete")
    return oof, np.mean(test_predictions, axis=0)


def prepare_extra_trees(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([X, X_test], ignore_index=True)
    combined = pd.get_dummies(
        combined,
        columns=categorical_features,
        dummy_na=True,
        dtype=float,
    )
    return (
        combined.iloc[: len(X)].reset_index(drop=True),
        combined.iloc[len(X) :].reset_index(drop=True),
    )


def train_extra_trees(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y))
    test_predictions = []
    for fold, (train_indices, validation_indices) in enumerate(splits, 1):
        imputer = SimpleImputer(strategy="median")
        train_values = imputer.fit_transform(X.iloc[train_indices])
        validation_values = imputer.transform(X.iloc[validation_indices])
        test_values = imputer.transform(X_test)
        fold_predictions = []
        for seed in seeds:
            model = ExtraTreesClassifier(
                n_estimators=700,
                min_samples_leaf=2,
                max_features=0.8,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(train_values, y.iloc[train_indices])
            fold_predictions.append(
                model.predict_proba(validation_values)[:, 1]
            )
            test_predictions.append(model.predict_proba(test_values)[:, 1])
        oof[validation_indices] = np.mean(fold_predictions, axis=0)
        print(f"ExtraTrees fold {fold}/5 complete")
    return oof, np.mean(test_predictions, axis=0)


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y))
    test_predictions = []
    for fold, (train_indices, validation_indices) in enumerate(splits, 1):
        imputer = SimpleImputer(strategy="median")
        train_values = imputer.fit_transform(X.iloc[train_indices])
        validation_values = imputer.transform(X.iloc[validation_indices])
        test_values = imputer.transform(X_test)
        fold_predictions = []
        for seed in seeds:
            model = RandomForestClassifier(
                n_estimators=700,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features=0.8,
                max_depth=15,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(train_values, y.iloc[train_indices])
            fold_predictions.append(
                model.predict_proba(validation_values)[:, 1]
            )
            test_predictions.append(model.predict_proba(test_values)[:, 1])
        oof[validation_indices] = np.mean(fold_predictions, axis=0)
        print(f"Random Forest fold {fold}/5 complete")
    return oof, np.mean(test_predictions, axis=0)


def train_histogram_boosting(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y))
    test_predictions = []
    for fold, (train_indices, validation_indices) in enumerate(splits, 1):
        imputer = SimpleImputer(strategy="median")
        train_values = imputer.fit_transform(X.iloc[train_indices])
        validation_values = imputer.transform(X.iloc[validation_indices])
        test_values = imputer.transform(X_test)
        model = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1,
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(train_values, y.iloc[train_indices])
        oof[validation_indices] = model.predict_proba(validation_values)[:, 1]
        test_predictions.append(model.predict_proba(test_values)[:, 1])
        print(f"Histogram Boosting fold {fold}/5 complete")
    return oof, np.mean(test_predictions, axis=0)


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y))
    test_predictions = []
    for fold, (train_indices, validation_indices) in enumerate(splits, 1):
        fold_predictions = []
        for seed in seeds:
            model = XGBClassifier(
                n_estimators=260,
                learning_rate=0.02,
                max_depth=10,
                subsample=0.815,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                device="cuda" if device != "cpu" else "cpu",
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X.iloc[train_indices], y.iloc[train_indices])
            fold_predictions.append(
                model.predict_proba(X.iloc[validation_indices])[:, 1]
            )
            test_predictions.append(model.predict_proba(X_test)[:, 1])
        oof[validation_indices] = np.mean(fold_predictions, axis=0)
        print(f"One-hot XGBoost fold {fold}/5 complete")
    return oof, np.mean(test_predictions, axis=0)


def prepare_ebm(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = X.copy()
    test = X_test.copy()
    for column in categorical_features:
        train[column] = train[column].astype(str)
        test[column] = test[column].astype(str)
    return train, test


def train_ebm(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros(len(y))
    test_predictions = []
    for fold, (train_indices, validation_indices) in enumerate(splits, 1):
        fold_predictions = []
        for seed in seeds:
            model = ExplainableBoostingClassifier(
                **EBM_PARAMETERS,
                n_jobs=-1,
                random_state=seed,
            )
            model.fit(X.iloc[train_indices], y.iloc[train_indices])
            fold_predictions.append(
                model.predict_proba(X.iloc[validation_indices])[:, 1]
            )
            test_predictions.append(model.predict_proba(X_test)[:, 1])
        oof[validation_indices] = np.mean(fold_predictions, axis=0)
        print(f"EBM fold {fold}/5 complete")
    return oof, np.mean(test_predictions, axis=0)


def load_bagged_predictions(
    project_root: Path,
    source_rows: pd.Series,
    test_ids: pd.Series,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    results = project_root / "bagging" / "results"
    oof = pd.read_csv(results / "oof_predictions.csv")
    if not np.array_equal(oof["source_row"], source_rows.to_numpy()):
        raise ValueError("Bagged OOF rows do not match the training rows.")
    train_predictions = {
        "bagged_catboost": oof["catboost_probability"].to_numpy(),
        "bagged_lightgbm": oof["lightgbm_probability"].to_numpy(),
    }
    test_predictions = {}
    for name, filename in [
        ("bagged_catboost", "catboost_submission.csv"),
        ("bagged_lightgbm", "lightgbm_submission.csv"),
    ]:
        submission = pd.read_csv(results / filename)
        if not np.array_equal(submission["ID"], test_ids.to_numpy()):
            raise ValueError(f"{filename} IDs do not match the test data.")
        test_predictions[name] = submission[TARGET].to_numpy()
    return train_predictions, test_predictions


def forward_weights(
    y: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    scores = [
        roc_auc_score(y, predictions[:, column])
        for column in range(predictions.shape[1])
    ]
    weights = np.zeros(predictions.shape[1])
    weights[int(np.argmax(scores))] = 1
    current = predictions @ weights
    current_score = roc_auc_score(y, current)
    for _ in range(8):
        best_score = current_score
        best_candidate = None
        for column in range(predictions.shape[1]):
            for amount in np.linspace(0.01, 0.5, 50):
                candidate = (1 - amount) * current + amount * predictions[:, column]
                score = roc_auc_score(y, candidate)
                if score > best_score:
                    best_score = score
                    best_candidate = (column, amount, candidate)
        if best_candidate is None or best_score - current_score < 1e-5:
            break
        column, amount, current = best_candidate
        weights *= 1 - amount
        weights[column] += amount
        current_score = best_score
    return weights


def crossfit_weights(
    y: np.ndarray,
    predictions: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crossfit_prediction = np.zeros(len(y))
    fold_weights = []
    for fold in sorted(np.unique(fold_ids)):
        training_mask = fold_ids != fold
        validation_mask = fold_ids == fold
        weights = forward_weights(
            y[training_mask],
            predictions[training_mask],
        )
        crossfit_prediction[validation_mask] = (
            predictions[validation_mask] @ weights
        )
        fold_weights.append(weights)
    average_weights = np.mean(fold_weights, axis=0)
    average_weights /= average_weights.sum()
    return crossfit_prediction, average_weights, np.vstack(fold_weights)


def make_fold_ids(
    length: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    fold_ids = np.zeros(length, dtype=int)
    for fold, (_, validation_indices) in enumerate(splits, 1):
        fold_ids[validation_indices] = fold
    return fold_ids


def save_plots(
    y: np.ndarray,
    model_predictions: dict[str, np.ndarray],
    ensemble_prediction: np.ndarray,
    weights: pd.DataFrame,
    results_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, prediction in model_predictions.items():
        fpr, tpr, _ = roc_curve(y, prediction)
        ax.plot(
            fpr,
            tpr,
            alpha=0.6,
            label=f"{name} ({roc_auc_score(y, prediction):.4f})",
        )
    fpr, tpr, _ = roc_curve(y, ensemble_prediction)
    ax.plot(
        fpr,
        tpr,
        color="black",
        linewidth=2.5,
        label=f"ensemble ({roc_auc_score(y, ensemble_prediction):.4f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Advanced ensemble grouped OOF ROC",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "roc_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(weights["model"], weights["average_weight"], color="#2a9d8f")
    ax.set(
        xlabel="Model",
        ylabel="Cross-fitted average weight",
        title="Advanced ensemble model weights",
    )
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "ensemble_weights.png", dpi=180)
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

    train, test, sample_submission = cat_module.load_data(
        args.project_root / "data"
    )
    base_X, y, groups, source_rows, _ = cat_module.prepare_training_data(train)
    base_X_test = cat_module.prepare_features(test)
    X = add_audio_features(base_X)
    X_test = add_audio_features(base_X_test)
    splits = cat_module.make_splits(X, y, groups, seeds[0], 5)
    fold_ids = make_fold_ids(len(y), splits)

    model_oof, model_test = load_bagged_predictions(
        args.project_root,
        source_rows,
        test["ID"],
    )
    model_oof["enhanced_catboost"], model_test["enhanced_catboost"] = (
        train_catboost(
            cat_module,
            X,
            y,
            X_test,
            splits,
            seeds,
            args.task_type,
            args.device,
        )
    )

    lgb_X = X.copy()
    lgb_X_test = X_test.copy()
    lgb_X, lgb_X_test = lgb_module.align_categories(lgb_X, lgb_X_test)
    model_oof["enhanced_lightgbm"], model_test["enhanced_lightgbm"] = (
        train_lightgbm(
            lgb_module,
            lgb_X,
            y,
            lgb_X_test,
            splits,
            seeds,
        )
    )

    extra_X, extra_X_test = prepare_extra_trees(
        X,
        X_test,
        cat_module.CATEGORICAL_FEATURES,
    )
    model_oof["extra_trees"], model_test["extra_trees"] = train_extra_trees(
        extra_X,
        y,
        extra_X_test,
        splits,
        seeds,
    )
    model_oof["random_forest"], model_test["random_forest"] = (
        train_random_forest(
            extra_X,
            y,
            extra_X_test,
            splits,
            seeds,
        )
    )
    model_oof["histogram_boosting"], model_test["histogram_boosting"] = (
        train_histogram_boosting(
            extra_X,
            y,
            extra_X_test,
            splits,
            seeds[0],
        )
    )
    model_oof["onehot_xgboost"], model_test["onehot_xgboost"] = (
        train_xgboost(
            extra_X,
            y,
            extra_X_test,
            splits,
            seeds,
            "cpu" if args.task_type == "CPU" else args.device,
        )
    )

    ebm_X, ebm_X_test = prepare_ebm(
        X,
        X_test,
        cat_module.CATEGORICAL_FEATURES,
    )
    model_oof["ebm"], model_test["ebm"] = train_ebm(
        ebm_X,
        y,
        ebm_X_test,
        splits,
        seeds,
    )

    names = list(model_oof)
    oof_matrix = np.column_stack([model_oof[name] for name in names])
    test_matrix = np.column_stack([model_test[name] for name in names])
    y_array = y.to_numpy()
    crossfit_prediction, average_weights, fold_weights = crossfit_weights(
        y_array,
        oof_matrix,
        fold_ids,
    )
    full_weights = forward_weights(y_array, oof_matrix)
    selected_prediction = oof_matrix @ average_weights
    test_prediction = test_matrix @ average_weights

    model_scores = pd.DataFrame(
        {
            "model": names,
            "oof_auc": [
                roc_auc_score(y_array, model_oof[name]) for name in names
            ],
        }
    ).sort_values("oof_auc", ascending=False)
    model_scores.to_csv(args.results_dir / "model_scores.csv", index=False)

    weights = pd.DataFrame(
        {
            "model": names,
            "average_weight": average_weights,
            "full_oof_weight": full_weights,
        }
    )
    for fold, values in enumerate(fold_weights, 1):
        weights[f"fold_{fold}_weight"] = values
    weights.to_csv(args.results_dir / "ensemble_weights.csv", index=False)

    oof_output = pd.DataFrame(
        {
            "source_row": source_rows.astype(int),
            "actual_label": np.where(y_array == 1, "High", "Low"),
            "fold": fold_ids,
        }
    )
    for name in names:
        oof_output[f"{name}_probability"] = model_oof[name]
    oof_output["ensemble_probability"] = selected_prediction
    oof_output["crossfit_ensemble_probability"] = crossfit_prediction
    oof_output.to_csv(args.results_dir / "oof_predictions.csv", index=False)

    test_output = pd.DataFrame({"ID": test["ID"].to_numpy()})
    for name in names:
        test_output[f"{name}_probability"] = model_test[name]
    test_output["ensemble_probability"] = test_prediction
    test_output.to_csv(
        args.results_dir / "test_predictions.csv",
        index=False,
    )

    fold_rows = []
    for fold in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold
        fold_rows.append(
            {
                "fold": int(fold),
                "rows": int(mask.sum()),
                "ensemble_auc": roc_auc_score(
                    y_array[mask],
                    selected_prediction[mask],
                ),
                "crossfit_ensemble_auc": roc_auc_score(
                    y_array[mask],
                    crossfit_prediction[mask],
                ),
            }
        )
    pd.DataFrame(fold_rows).to_csv(
        args.results_dir / "fold_metrics.csv",
        index=False,
    )

    submission = sample_submission.copy()
    submission["ID"] = test["ID"].to_numpy()
    submission[TARGET] = test_prediction
    submission.to_csv(args.results_dir / "submission.csv", index=False)

    full_submission = sample_submission.copy()
    full_submission["ID"] = test["ID"].to_numpy()
    full_submission[TARGET] = test_matrix @ full_weights
    full_submission.to_csv(
        args.results_dir / "submission_full_weights.csv",
        index=False,
    )

    save_plots(
        y_array,
        model_oof,
        selected_prediction,
        weights,
        args.results_dir,
    )
    summary = {
        "seeds": seeds,
        "folds": 5,
        "engineered_features": [
            "log_duration",
            "tempo_half_distance",
            "loud_energy",
            "energy_acoustic",
            "nonvocal",
            "dance_energy",
        ],
        "model_auc": dict(zip(model_scores["model"], model_scores["oof_auc"])),
        "selected_weights": dict(zip(names, average_weights)),
        "crossfit_ensemble_auc": float(
            roc_auc_score(y_array, crossfit_prediction)
        ),
        "selected_weight_oof_auc": float(
            roc_auc_score(y_array, selected_prediction)
        ),
        "full_weight_oof_auc": float(
            roc_auc_score(y_array, oof_matrix @ full_weights)
        ),
        "equal_vote_oof_auc": float(
            roc_auc_score(y_array, oof_matrix.mean(axis=1))
        ),
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(model_scores.to_string(index=False))
    print(
        "Cross-fitted ensemble AUC: "
        f"{summary['crossfit_ensemble_auc']:.6f}"
    )
    print(
        "Selected-weight OOF AUC: "
        f"{summary['selected_weight_oof_auc']:.6f}"
    )
    print(f"Results written to: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
