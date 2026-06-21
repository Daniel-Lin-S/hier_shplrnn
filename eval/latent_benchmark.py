"""Latent benchmark orchestration and evaluation utilities."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import logsumexp
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

from models.feature_extractors import create_feature_extractor
from visualisation.eval_plotter import plot_subject_feature_pca

DEFAULT_RIDGE_ALPHAS: tuple[float, ...] = tuple(float(v) for v in np.logspace(-4.0, 4.0, num=17))


@dataclass(frozen=True)
class ExtractorSpec:
    """Specification for one extractor benchmark run."""

    name: str
    extractor_type: str
    params: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration values for latent-space benchmarking."""

    dataset_group: str
    save_path: str
    train_data_path: str
    train_labels_path: str
    test_data_path: str
    test_labels_path: str
    total_test_samples: int
    balanced_test_selection: bool
    nested_cv_folds: int
    ridge_alphas: Sequence[float]
    random_state: int
    cache_baselines_once_per_dataset_group: bool
    evaluate_baselines: bool
    model_spec: ExtractorSpec
    baseline_specs: Sequence[ExtractorSpec]
    create_pca_plot: bool
    pca_plot_prefix: str
    pca_plot_show_arrows: bool


def benchmark_config_from_files(
    shared_config_path: str,
    model_config_path: str,
    save_path_override: str | None = None,
) -> BenchmarkConfig:
    """Build benchmark config from shared and per-model YAML files.

    Parameters
    ----------
    shared_config_path : str
        Path to shared benchmark YAML containing dataset and baseline settings.
    model_config_path : str
        Path to per-model YAML containing only model extractor details.
    save_path_override : str | None, optional
        Optional output directory override from CLI.

    Returns
    -------
    BenchmarkConfig
        Parsed benchmark configuration.
    """
    shared_yaml = _load_yaml_file(shared_config_path)
    model_yaml = _load_yaml_file(model_config_path)

    shared_benchmark = _require_mapping(shared_yaml.get("benchmark"), "benchmark")

    model_overrides = model_yaml.get("benchmark_overrides", {})
    model_overrides = _require_mapping(model_overrides, "benchmark_overrides")

    model_mapping = _require_mapping(model_yaml.get("model"), "model")
    model_spec = _parse_extractor_spec(model_mapping, "model")

    dataset_group = _read_with_override(shared_benchmark, model_overrides, "dataset_group")
    save_path = save_path_override if save_path_override is not None else _read_with_override(
        shared_benchmark,
        model_overrides,
        "save_path",
    )

    train_data_path = _read_with_override(shared_benchmark, model_overrides, "train_data_path")
    train_labels_path = _read_with_override(shared_benchmark, model_overrides, "train_labels_path")
    test_data_path = _read_with_override(shared_benchmark, model_overrides, "test_data_path")
    test_labels_path = _read_with_override(shared_benchmark, model_overrides, "test_labels_path")

    total_test_samples = _read_int_with_override(shared_benchmark, model_overrides, "total_test_samples")
    balanced_test_selection = bool(
        _read_with_default(shared_benchmark, model_overrides, "balanced_test_selection", True)
    )

    nested_cv_folds = int(_read_with_default(shared_benchmark, model_overrides, "nested_cv_folds", 4))
    if nested_cv_folds < 2:
        raise ValueError(f"Expected nested_cv_folds >= 2, but got {nested_cv_folds}.")

    ridge_values = _read_with_default(shared_benchmark, model_overrides, "ridge_alphas", list(DEFAULT_RIDGE_ALPHAS))
    ridge_alphas = _parse_float_sequence(ridge_values, "benchmark.ridge_alphas")

    random_state = int(_read_with_default(shared_benchmark, model_overrides, "random_state", 42))
    cache_baselines = bool(
        _read_with_default(shared_benchmark, model_overrides, "cache_baselines_once_per_dataset_group", True)
    )
    evaluate_baselines = bool(_read_with_default(shared_benchmark, model_overrides, "evaluate_baselines", True))

    baseline_specs_raw = _read_with_default(shared_benchmark, model_overrides, "baselines", [])
    baseline_specs = _parse_extractor_specs(
        baseline_specs_raw,
        "benchmark.baselines",
        allow_empty=True,
    )

    pca_cfg = _read_with_default(shared_benchmark, model_overrides, "pca_plot", {})
    pca_cfg = _require_mapping(pca_cfg, "benchmark.pca_plot")
    create_pca_plot = bool(pca_cfg.get("enabled", True))
    pca_plot_prefix = str(pca_cfg.get("prefix", "subject_feature_pca"))
    pca_plot_show_arrows = bool(pca_cfg.get("show_arrows", False))

    return BenchmarkConfig(
        dataset_group=str(dataset_group),
        save_path=str(save_path),
        train_data_path=str(train_data_path),
        train_labels_path=str(train_labels_path),
        test_data_path=str(test_data_path),
        test_labels_path=str(test_labels_path),
        total_test_samples=total_test_samples,
        balanced_test_selection=balanced_test_selection,
        nested_cv_folds=nested_cv_folds,
        ridge_alphas=ridge_alphas,
        random_state=random_state,
        cache_baselines_once_per_dataset_group=cache_baselines,
        evaluate_baselines=evaluate_baselines,
        model_spec=model_spec,
        baseline_specs=baseline_specs,
        create_pca_plot=create_pca_plot,
        pca_plot_prefix=pca_plot_prefix,
        pca_plot_show_arrows=pca_plot_show_arrows,
    )


def run_latent_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    """Run latent-space benchmarking.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark runtime configuration.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per evaluated extractor.
    """
    train_signals = load_signal_tensor(config.train_data_path)
    test_signals = load_signal_tensor(config.test_data_path)
    train_labels = load_labels(config.train_labels_path, expected_length=train_signals.shape[0])
    test_labels = load_labels(config.test_labels_path, expected_length=test_signals.shape[0])

    selected_test_signals, selected_test_labels, _selected_indices = select_test_subset(
        test_signals,
        test_labels,
        total_samples=config.total_test_samples,
        balanced_binary=config.balanced_test_selection,
    )

    dataset_root = Path(config.save_path) / _safe_token(config.dataset_group)
    models_root = dataset_root / "models"
    baselines_root = dataset_root / "baselines"
    dataset_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    baselines_root.mkdir(parents=True, exist_ok=True)

    legacy_selection_path = dataset_root / "selected_test_indices.csv"
    if legacy_selection_path.exists():
        legacy_selection_path.unlink()
        print(
            f"Removed legacy test-selection artifact '{legacy_selection_path}'.",
            flush=True,
        )

    _save_benchmark_config(dataset_root, config)

    summary_rows: list[dict[str, float | str]] = []

    model_output_dir = models_root / _safe_token(config.model_spec.name)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running model benchmark '{config.model_spec.name}'.", flush=True)

    model_metrics = _run_single_extractor(
        spec=config.model_spec,
        train_signals=train_signals,
        train_labels=train_labels,
        test_signals=selected_test_signals,
        test_labels=selected_test_labels,
        nested_cv_folds=config.nested_cv_folds,
        ridge_alphas=config.ridge_alphas,
        random_state=config.random_state,
        output_dir=model_output_dir,
        create_pca_plot=config.create_pca_plot,
        pca_plot_prefix=config.pca_plot_prefix,
        pca_plot_show_arrows=config.pca_plot_show_arrows,
    )
    model_metrics["extractor_group"] = "model"
    summary_rows.append(model_metrics)

    if config.evaluate_baselines:
        for spec in config.baseline_specs:
            output_dir = baselines_root / _safe_token(spec.name)
            output_dir.mkdir(parents=True, exist_ok=True)

            metrics_path = output_dir / "classification_metrics.csv"
            if config.cache_baselines_once_per_dataset_group and metrics_path.exists():
                print(
                    f"Using cached baseline metrics for '{spec.name}' from '{metrics_path}'.",
                    flush=True,
                )
                cached_metrics = _load_single_row_metrics(metrics_path)
                cached_metrics["extractor_group"] = "baseline"
                summary_rows.append(cached_metrics)
                continue

            print(f"Running baseline benchmark '{spec.name}'.", flush=True)
            baseline_metrics = _run_single_extractor(
                spec=spec,
                train_signals=train_signals,
                train_labels=train_labels,
                test_signals=selected_test_signals,
                test_labels=selected_test_labels,
                nested_cv_folds=config.nested_cv_folds,
                ridge_alphas=config.ridge_alphas,
                random_state=config.random_state,
                output_dir=output_dir,
                create_pca_plot=config.create_pca_plot,
                pca_plot_prefix=config.pca_plot_prefix,
                pca_plot_show_arrows=config.pca_plot_show_arrows,
            )
            baseline_metrics["extractor_group"] = "baseline"
            summary_rows.append(baseline_metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = dataset_root / "latent_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved benchmark summary to {summary_path}", flush=True)
    return summary_df


def _run_single_extractor(
    spec: ExtractorSpec,
    train_signals: np.ndarray,
    train_labels: np.ndarray,
    test_signals: np.ndarray,
    test_labels: np.ndarray,
    nested_cv_folds: int,
    ridge_alphas: Sequence[float],
    random_state: int,
    output_dir: Path,
    create_pca_plot: bool,
    pca_plot_prefix: str,
    pca_plot_show_arrows: bool,
) -> dict[str, float | str]:
    """Run one extractor benchmark and save artifacts.

    Parameters
    ----------
    spec : ExtractorSpec
        Extractor configuration.
    train_signals : np.ndarray
        Training EEG tensor.
    train_labels : np.ndarray
        Training labels.
    test_signals : np.ndarray
        Selected test EEG tensor.
    test_labels : np.ndarray
        Selected test labels.
    nested_cv_folds : int
        Number of folds for nested cross-validation.
    ridge_alphas : Sequence[float]
        Ridge alpha search candidates.
    random_state : int
        Random seed.
    output_dir : Path
        Directory where artifacts are saved.
    create_pca_plot : bool
        Whether PCA plot output is enabled.
    pca_plot_prefix : str
        PCA plot filename prefix.
    pca_plot_show_arrows : bool
        Whether PCA loadings arrows are shown.

    Returns
    -------
    dict[str, float | str]
        Metrics dictionary.
    """
    extractor = create_feature_extractor(spec.extractor_type, spec.params)
    extractor.set_runtime_output_dir(str(output_dir))
    extractor.fit(train_signals, train_labels)

    train_features = extractor.transform(train_signals, split_name="train")
    test_features = extractor.transform(test_signals, split_name="test")

    return evaluate_feature_set(
        model_name=spec.name,
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        nested_cv_folds=nested_cv_folds,
        ridge_alphas=ridge_alphas,
        random_state=random_state,
        output_dir=output_dir,
        create_pca_plot=create_pca_plot,
        pca_plot_prefix=pca_plot_prefix,
        pca_plot_show_arrows=pca_plot_show_arrows,
    )


def evaluate_feature_set(
    model_name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    nested_cv_folds: int,
    ridge_alphas: Sequence[float],
    random_state: int,
    output_dir: Path,
    create_pca_plot: bool,
    pca_plot_prefix: str,
    pca_plot_show_arrows: bool,
) -> dict[str, float | str]:
    """Train and evaluate ridge classifier plus latent diagnostics.

    Parameters
    ----------
    model_name : str
        Benchmark row name.
    train_features : np.ndarray
        Training feature matrix.
    train_labels : np.ndarray
        Training labels.
    test_features : np.ndarray
        Test feature matrix.
    test_labels : np.ndarray
        Test labels.
    nested_cv_folds : int
        Number of folds for nested cross-validation.
    ridge_alphas : Sequence[float]
        Ridge alpha search candidates.
    random_state : int
        Random seed.
    output_dir : Path
        Output directory.
    create_pca_plot : bool
        Whether PCA plot output is enabled.
    pca_plot_prefix : str
        PCA plot filename prefix.
    pca_plot_show_arrows : bool
        Whether PCA loadings arrows are shown.

    Returns
    -------
    dict[str, float | str]
        One-row summary metrics dictionary.
    """
    _validate_feature_matrix(train_features, f"{model_name} train")
    _validate_feature_matrix(test_features, f"{model_name} test")

    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError(
            f"Train feature count ({train_features.shape[0]}) does not match train labels ({train_labels.shape[0]})."
        )
    if test_features.shape[0] != test_labels.shape[0]:
        raise ValueError(
            f"Test feature count ({test_features.shape[0]}) does not match test labels ({test_labels.shape[0]})."
        )

    label_encoder = LabelEncoder()
    train_targets = np.asarray(label_encoder.fit_transform(train_labels.astype(str)), dtype=np.int64)
    test_targets = np.asarray(label_encoder.transform(test_labels.astype(str)), dtype=np.int64)

    outer_df, alpha_df, selected_alpha = nested_cv_alpha_selection(
        features=train_features,
        targets=train_targets,
        alphas=ridge_alphas,
        n_splits=nested_cv_folds,
        random_state=random_state,
    )

    model = _build_ridge_pipeline(selected_alpha)
    model.fit(train_features, train_targets)
    predicted_targets = model.predict(test_features)
    decision_scores = model.decision_function(test_features)

    accuracy = float(accuracy_score(test_targets, predicted_targets))
    auroc = float(compute_auroc(test_targets, decision_scores, num_classes=len(label_encoder.classes_)))

    ridge = model.named_steps["ridgeclassifier"]
    if not isinstance(ridge, RidgeClassifier):
        raise TypeError(
            "Ridge pipeline unexpectedly returned a non-RidgeClassifier final estimator: "
            f"type={type(ridge).__name__}."
        )

    n_parameters = int(np.asarray(ridge.coef_).size + np.asarray(ridge.intercept_).size)
    aic, bic = compute_aic_bic(test_targets, decision_scores, n_parameters)

    mi_train_values = mutual_info_classif(
        train_features,
        train_targets,
        discrete_features=False,
        random_state=random_state,
    )
    mi_test_values = mutual_info_classif(
        test_features,
        test_targets,
        discrete_features=False,
        random_state=random_state,
    )

    if not np.all(np.isfinite(mi_train_values)):
        raise ValueError(f"Train mutual information contains non-finite values for '{model_name}'.")
    if not np.all(np.isfinite(mi_test_values)):
        raise ValueError(f"Test mutual information contains non-finite values for '{model_name}'.")

    mi_df = pd.DataFrame(
        {
            "feature_index": np.arange(train_features.shape[1]),
            "mutual_information_train": mi_train_values,
            "mutual_information_test": mi_test_values,
        }
    )

    outer_df.to_csv(output_dir / "nested_cv_outer_folds.csv", index=False)
    alpha_df.to_csv(output_dir / "nested_cv_alpha_scores.csv", index=False)
    mi_df.to_csv(output_dir / "mutual_information.csv", index=False)

    _save_feature_matrix(output_dir / "train_features.csv", train_features, train_labels)
    _save_feature_matrix(output_dir / "test_features.csv", test_features, test_labels)

    if create_pca_plot:
        if test_features.shape[1] < 2:
            print(
                f"Skipping PCA plot for '{model_name}' because feature_dim={test_features.shape[1]} < 2.",
                flush=True,
            )
        else:
            pca_plot_path = output_dir / f"{pca_plot_prefix}.png"
            plot_subject_feature_pca(
                feature_vectors=test_features,
                labels=test_labels,
                run_path=model_name,
                output_path=str(pca_plot_path),
                show_arrows=pca_plot_show_arrows,
            )

    nested_mean = float(outer_df["outer_accuracy"].mean())
    nested_std = float(outer_df["outer_accuracy"].std(ddof=0))
    metrics = {
        "model_name": model_name,
        "train_samples": float(train_features.shape[0]),
        "test_samples": float(test_features.shape[0]),
        "feature_dim": float(train_features.shape[1]),
        "selected_alpha": float(selected_alpha),
        "nested_cv_accuracy_mean": nested_mean,
        "nested_cv_accuracy_std": nested_std,
        "test_accuracy": accuracy,
        "test_auroc": auroc,
        "test_aic": float(aic),
        "test_bic": float(bic),
        "train_mutual_information_mean": float(np.mean(mi_train_values)),
        "train_mutual_information_max": float(np.max(mi_train_values)),
        "test_mutual_information_mean": float(np.mean(mi_test_values)),
        "test_mutual_information_max": float(np.max(mi_test_values)),
    }

    metrics_path = output_dir / "classification_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    accuracy_path = output_dir / "classification_accuracy.txt"
    accuracy_path.write_text(f"{accuracy:.8f}\n", encoding="utf-8")

    print(f"Saved metrics for '{model_name}' to {metrics_path}", flush=True)
    return metrics


def nested_cv_alpha_selection(
    features: np.ndarray,
    targets: np.ndarray,
    alphas: Sequence[float],
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Select ridge alpha with nested cross-validation.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    targets : np.ndarray
        Encoded labels.
    alphas : Sequence[float]
        Ridge alpha candidates.
    n_splits : int
        Fold count.
    random_state : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, float]
        Outer fold summary, alpha ranking, selected alpha.
    """
    if n_splits < 2:
        raise ValueError(f"Expected n_splits >= 2 for nested CV, but got {n_splits}.")

    candidate_alphas = sorted(float(alpha) for alpha in alphas)
    if not candidate_alphas:
        raise ValueError("At least one ridge alpha value is required for nested CV.")

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_rows: list[dict[str, float]] = []

    for fold_index, (outer_train_idx, outer_valid_idx) in enumerate(outer_cv.split(features, targets), start=1):
        x_outer_train = features[outer_train_idx]
        y_outer_train = targets[outer_train_idx]
        x_outer_valid = features[outer_valid_idx]
        y_outer_valid = targets[outer_valid_idx]

        inner_scores: list[tuple[float, float]] = []
        for alpha in candidate_alphas:
            inner_accuracy = cross_validated_accuracy(
                features=x_outer_train,
                targets=y_outer_train,
                alpha=alpha,
                n_splits=n_splits,
                random_state=random_state + fold_index,
            )
            inner_scores.append((alpha, inner_accuracy))

        best_alpha, best_inner_accuracy = max(inner_scores, key=lambda pair: pair[1])
        fold_model = _build_ridge_pipeline(best_alpha)
        fold_model.fit(x_outer_train, y_outer_train)
        outer_predictions = fold_model.predict(x_outer_valid)
        outer_accuracy = float(accuracy_score(y_outer_valid, outer_predictions))

        outer_rows.append(
            {
                "fold": float(fold_index),
                "best_alpha": float(best_alpha),
                "inner_accuracy": float(best_inner_accuracy),
                "outer_accuracy": outer_accuracy,
            }
        )

    alpha_rows: list[dict[str, float]] = []
    for alpha in candidate_alphas:
        mean_accuracy = cross_validated_accuracy(
            features=features,
            targets=targets,
            alpha=alpha,
            n_splits=n_splits,
            random_state=random_state + 100,
        )
        alpha_rows.append(
            {
                "alpha": float(alpha),
                "mean_cv_accuracy": float(mean_accuracy),
            }
        )

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df = alpha_df.sort_values(["mean_cv_accuracy", "alpha"], ascending=[False, True]).reset_index(drop=True)
    selected_alpha = float(alpha_df["alpha"].to_numpy(dtype=float)[0])

    outer_df = pd.DataFrame(outer_rows)
    return outer_df, alpha_df, selected_alpha


def cross_validated_accuracy(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    n_splits: int,
    random_state: int,
) -> float:
    """Compute mean CV accuracy for one alpha value.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    targets : np.ndarray
        Encoded labels.
    alpha : float
        Ridge regularization strength.
    n_splits : int
        Fold count.
    random_state : int
        Random seed.

    Returns
    -------
    float
        Mean validation accuracy.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: list[float] = []

    for train_index, valid_index in cv.split(features, targets):
        x_train = features[train_index]
        y_train = targets[train_index]
        x_valid = features[valid_index]
        y_valid = targets[valid_index]

        model = _build_ridge_pipeline(alpha)
        model.fit(x_train, y_train)
        predictions = model.predict(x_valid)
        scores.append(float(accuracy_score(y_valid, predictions)))

    return float(np.mean(scores))


def compute_auroc(test_targets: np.ndarray, decision_scores: np.ndarray, num_classes: int) -> float:
    """Compute AUROC for binary or multiclass ridge scores.

    Parameters
    ----------
    test_targets : np.ndarray
        Encoded test labels.
    decision_scores : np.ndarray
        Ridge decision scores.
    num_classes : int
        Number of encoded classes.

    Returns
    -------
    float
        AUROC value.
    """
    if num_classes < 2:
        raise ValueError(f"AUROC requires at least two classes, but got {num_classes}.")

    if np.unique(test_targets).size < 2:
        print(
            "AUROC is undefined because the evaluation subset contains only one class. "
            "Returning NaN for AUROC.",
            flush=True,
        )
        return float("nan")

    if num_classes == 2:
        if decision_scores.ndim == 2:
            if decision_scores.shape[1] == 1:
                score_vector = decision_scores[:, 0]
            else:
                score_vector = decision_scores[:, 1]
        else:
            score_vector = decision_scores
        return float(roc_auc_score(test_targets, score_vector))

    if decision_scores.ndim != 2 or decision_scores.shape[1] != num_classes:
        raise ValueError(
            "For multiclass AUROC, decision scores must be a 2D matrix with one column per class: "
            f"expected shape (*, {num_classes}), got {decision_scores.shape}."
        )

    one_hot_targets = label_binarize(test_targets, classes=np.arange(num_classes))
    return float(roc_auc_score(one_hot_targets, decision_scores, average="macro", multi_class="ovr"))


def compute_aic_bic(
    test_targets: np.ndarray,
    decision_scores: np.ndarray,
    num_parameters: int,
) -> tuple[float, float]:
    """Compute AIC and BIC from pseudo-log-likelihood.

    Parameters
    ----------
    test_targets : np.ndarray
        Encoded test labels.
    decision_scores : np.ndarray
        Decision score matrix or vector.
    num_parameters : int
        Number of fitted classifier parameters.

    Returns
    -------
    tuple[float, float]
        AIC and BIC values.
    """
    if num_parameters <= 0:
        raise ValueError(f"Expected at least one trainable parameter, but got {num_parameters}.")

    if decision_scores.ndim == 1:
        logits = np.column_stack((np.zeros_like(decision_scores), decision_scores))
    elif decision_scores.ndim == 2:
        logits = decision_scores
    else:
        raise ValueError(
            "Decision scores must be 1D or 2D for AIC/BIC computation, "
            f"but got shape {decision_scores.shape}."
        )

    num_samples = logits.shape[0]
    if test_targets.shape[0] != num_samples:
        raise ValueError(
            "Target and score sample counts differ for AIC/BIC computation: "
            f"targets={test_targets.shape[0]}, scores={num_samples}."
        )

    log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
    log_likelihood = float(np.sum(log_probabilities[np.arange(num_samples), test_targets]))

    aic = 2.0 * float(num_parameters) - 2.0 * log_likelihood
    bic = np.log(float(num_samples)) * float(num_parameters) - 2.0 * log_likelihood
    return float(aic), float(bic)


def select_test_subset(
    signals: np.ndarray,
    labels: np.ndarray,
    total_samples: int,
    balanced_binary: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select test subset according to benchmark policy.

    Parameters
    ----------
    signals : np.ndarray
        Full test signal tensor.
    labels : np.ndarray
        Full test labels.
    total_samples : int
        Number of test samples to keep.
    balanced_binary : bool
        Whether to select half samples from each class in first-occurrence order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Selected signals, selected labels, selected source indices.
    """
    if total_samples <= 0:
        raise ValueError(f"total_samples must be positive, but got {total_samples}.")

    if signals.shape[0] != labels.shape[0]:
        raise ValueError(
            "Signal and label counts differ for test subset selection: "
            f"signals={signals.shape[0]}, labels={labels.shape[0]}."
        )

    if not balanced_binary:
        count = min(total_samples, labels.shape[0])
        indices = np.arange(count, dtype=np.int64)
        return signals[indices], labels[indices], indices

    indices = select_balanced_binary_indices(labels, total_samples)
    return signals[indices], labels[indices], indices


def select_balanced_binary_indices(labels: np.ndarray, total_samples: int) -> np.ndarray:
    """Select first-occurrence balanced binary indices.

    Parameters
    ----------
    labels : np.ndarray
        One-dimensional label array.
    total_samples : int
        Total subset size. Must be even.

    Returns
    -------
    np.ndarray
        Selected source indices.
    """
    if total_samples % 2 != 0:
        raise ValueError(
            "Balanced binary selection requires an even total sample count, "
            f"but got total_samples={total_samples}."
        )

    labels_str = labels.astype(str)
    unique_labels = np.unique(labels_str)
    if unique_labels.size != 2:
        raise ValueError(
            "Balanced binary selection requires exactly two classes, "
            f"but got labels={unique_labels.tolist()}."
        )

    per_class = total_samples // 2
    ordered_labels = unique_labels.tolist()
    if set(ordered_labels) == {"0", "1"}:
        ordered_labels = ["1", "0"]

    selected_parts: list[np.ndarray] = []
    for label in ordered_labels:
        label_indices = np.flatnonzero(labels_str == label)
        if label_indices.size < per_class:
            raise ValueError(
                "Insufficient samples for balanced selection: "
                f"label='{label}' has {label_indices.size}, required={per_class}."
            )
        selected_parts.append(label_indices[:per_class])

    return np.concatenate(selected_parts).astype(np.int64)


def load_signal_tensor(path: str) -> np.ndarray:
    """Load and validate EEG tensor with shape (samples, timesteps, channels).

    Parameters
    ----------
    path : str
        Path to ``.pt`` tensor file.

    Returns
    -------
    np.ndarray
        Signal array in float64.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Signal file '{path}' does not exist.")

    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Expected a torch.Tensor in '{path}', but got type '{type(tensor).__name__}'."
        )

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)

    if tensor.ndim != 3:
        raise ValueError(
            "Expected signal tensor with 2 or 3 dimensions and channel-last layout, "
            f"but got shape {tuple(tensor.shape)} from '{path}'."
        )

    array = tensor.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Signal tensor at '{path}' contains non-finite values.")
    return array


def load_labels(path: str, expected_length: int | None = None) -> np.ndarray:
    """Load and validate one-dimensional label arrays.

    Parameters
    ----------
    path : str
        Path to label ``.npy`` file.
    expected_length : int | None, optional
        Optional expected number of labels.

    Returns
    -------
    np.ndarray
        Label array converted to string dtype.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file '{path}' does not exist.")

    labels = np.asarray(np.load(path))
    if labels.ndim != 1:
        raise ValueError(
            "Expected labels to be one-dimensional, "
            f"but got shape {labels.shape} from '{path}'."
        )

    if expected_length is not None and labels.shape[0] != expected_length:
        raise ValueError(
            "Label length does not match expected sample count: "
            f"labels={labels.shape[0]}, expected={expected_length}, path='{path}'."
        )

    return labels.astype(str)


def _build_ridge_pipeline(alpha: float) -> Pipeline:
    """Create standard scaler + ridge classifier pipeline.

    Parameters
    ----------
    alpha : float
        Ridge regularization strength.

    Returns
    -------
    Pipeline
        Configured sklearn pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridgeclassifier", RidgeClassifier(alpha=float(alpha))),
        ]
    )


def _save_feature_matrix(path: Path, features: np.ndarray, labels: np.ndarray) -> None:
    """Save feature matrix with a trailing label column.

    Parameters
    ----------
    path : Path
        Output CSV path.
    features : np.ndarray
        Feature matrix.
    labels : np.ndarray
        Label array.
    """
    columns = [f"feature_{index + 1}" for index in range(features.shape[1])]
    df = pd.DataFrame(features, columns=columns)
    df["label"] = labels.astype(str)
    df.to_csv(path, index=False)


def _validate_feature_matrix(features: np.ndarray, name: str) -> None:
    """Validate extracted feature matrices.

    Parameters
    ----------
    features : np.ndarray
        Candidate feature matrix.
    name : str
        Matrix identifier for error messages.
    """
    if features.ndim != 2:
        raise ValueError(f"Feature matrix '{name}' must be 2D, but got shape {features.shape}.")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"Feature matrix '{name}' is empty with shape {features.shape}.")
    if not np.all(np.isfinite(features)):
        raise ValueError(f"Feature matrix '{name}' contains non-finite values.")


def _save_benchmark_config(dataset_root: Path, config: BenchmarkConfig) -> None:
    """Persist benchmark config snapshot.

    Parameters
    ----------
    dataset_root : Path
        Dataset-group output root.
    config : BenchmarkConfig
        Active benchmark configuration.
    """
    config_path = dataset_root / "benchmark_config_snapshot.json"
    config_dict = asdict(config)
    config_path.write_text(json.dumps(config_dict, indent=2, sort_keys=True), encoding="utf-8")


def _load_single_row_metrics(path: Path) -> dict[str, float | str]:
    """Load one-row metric CSV as dictionary.

    Parameters
    ----------
    path : Path
        Metrics CSV path.

    Returns
    -------
    dict[str, float | str]
        Metric dictionary.
    """
    df = pd.read_csv(path)
    if df.shape[0] != 1:
        raise ValueError(f"Expected one-row metrics CSV at '{path}', but got {df.shape[0]} rows.")
    return dict(df.iloc[0].to_dict())


def _parse_extractor_spec(spec_mapping: Mapping[str, Any], config_path: str) -> ExtractorSpec:
    """Parse one extractor specification.

    Parameters
    ----------
    spec_mapping : Mapping[str, Any]
        Raw mapping for one extractor.
    config_path : str
        Human-readable path for error messages.

    Returns
    -------
    ExtractorSpec
        Parsed extractor specification.
    """
    name = _require_str(spec_mapping, "name", config_path)
    extractor_type = _require_str(spec_mapping, "extractor", config_path)
    params_raw = spec_mapping.get("params", {})
    params = _require_mapping(params_raw, f"{config_path}.params")
    return ExtractorSpec(name=name, extractor_type=extractor_type, params=dict(params))


def _parse_extractor_specs(
    specs: Any,
    config_path: str,
    allow_empty: bool = False,
) -> list[ExtractorSpec]:
    """Parse extractor specs from YAML values.

    Parameters
    ----------
    specs : Any
        Raw YAML value.
    config_path : str
        Human-readable location string.
    allow_empty : bool, optional
        Whether an empty list is acceptable.

    Returns
    -------
    list[ExtractorSpec]
        Parsed extractor specifications.
    """
    if specs is None and allow_empty:
        return []

    if not isinstance(specs, list):
        raise TypeError(f"Expected '{config_path}' to be a list, but got {type(specs).__name__}.")

    parsed: list[ExtractorSpec] = []
    for index, item in enumerate(specs):
        item_path = f"{config_path}[{index}]"
        item_mapping = _require_mapping(item, item_path)
        parsed.append(_parse_extractor_spec(item_mapping, item_path))

    if not parsed and not allow_empty:
        raise ValueError(f"'{config_path}' must contain at least one extractor specification.")
    return parsed


def _parse_float_sequence(value: Any, path: str) -> list[float]:
    """Parse a float sequence from YAML values.

    Parameters
    ----------
    value : Any
        Raw YAML value.
    path : str
        Human-readable location string.

    Returns
    -------
    list[float]
        Parsed float list.
    """
    if not isinstance(value, list):
        raise TypeError(f"Expected '{path}' to be a list, but got {type(value).__name__}.")

    parsed = [float(v) for v in value]
    if not parsed:
        raise ValueError(f"'{path}' must contain at least one value.")
    return parsed


def _read_with_override(base: Mapping[str, Any], override: Mapping[str, Any], key: str) -> Any:
    """Read required key from override mapping first, then base mapping.

    Parameters
    ----------
    base : Mapping[str, Any]
        Base mapping.
    override : Mapping[str, Any]
        Override mapping.
    key : str
        Required key.

    Returns
    -------
    Any
        Resolved value.
    """
    if key in override:
        return override[key]
    if key in base:
        return base[key]
    raise KeyError(f"Missing required benchmark key '{key}' in shared and override mappings.")


def _read_int_with_override(base: Mapping[str, Any], override: Mapping[str, Any], key: str) -> int:
    """Read integer value from override/base mappings.

    Parameters
    ----------
    base : Mapping[str, Any]
        Base mapping.
    override : Mapping[str, Any]
        Override mapping.
    key : str
        Required key.

    Returns
    -------
    int
        Parsed integer value.
    """
    value = _read_with_override(base, override, key)
    if not isinstance(value, int):
        raise TypeError(f"Expected benchmark key '{key}' to be int, but got {type(value).__name__}.")
    return value


def _read_with_default(base: Mapping[str, Any], override: Mapping[str, Any], key: str, default: Any) -> Any:
    """Read key from override/base mappings with a fallback default.

    Parameters
    ----------
    base : Mapping[str, Any]
        Base mapping.
    override : Mapping[str, Any]
        Override mapping.
    key : str
        Key name.
    default : Any
        Default value if key is missing in both mappings.

    Returns
    -------
    Any
        Resolved value.
    """
    if key in override:
        return override[key]
    if key in base:
        return base[key]
    return default


def _load_yaml_file(path: str) -> Mapping[str, Any]:
    """Load YAML file and verify mapping root.

    Parameters
    ----------
    path : str
        YAML file path.

    Returns
    -------
    Mapping[str, Any]
        Parsed mapping.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file '{path}' does not exist.")

    with file_path.open("r", encoding="utf-8") as file:
        parsed = yaml.safe_load(file)

    if not isinstance(parsed, Mapping):
        raise TypeError(f"YAML file '{path}' must contain a mapping at root.")
    return parsed


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Require a mapping value.

    Parameters
    ----------
    value : Any
        Candidate value.
    path : str
        Human-readable location string.

    Returns
    -------
    Mapping[str, Any]
        Verified mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{path}' to be a mapping, but got {type(value).__name__}.")
    return value


def _require_str(mapping: Mapping[str, Any], key: str, path: str | None = None) -> str:
    """Require a non-empty string key from mapping.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Mapping to read.
    key : str
        Required key.
    path : str | None, optional
        Optional path prefix.

    Returns
    -------
    str
        Parsed string value.
    """
    if key not in mapping:
        source = path if path is not None else "mapping"
        raise KeyError(f"Missing required key '{key}' in '{source}'.")

    value = mapping[key]
    if not isinstance(value, str) or value.strip() == "":
        source = path if path is not None else "mapping"
        raise ValueError(f"Expected '{source}.{key}' to be non-empty string, but got {value!r}.")
    return value


def _safe_token(value: str) -> str:
    """Convert arbitrary string to filesystem-safe token.

    Parameters
    ----------
    value : str
        Input string.

    Returns
    -------
    str
        Sanitized token.
    """
    token = value.strip().replace(" ", "_")
    token = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in token)
    return token if token else "token"
