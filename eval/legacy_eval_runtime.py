"""Legacy checkpoint evaluation runtime helpers."""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
import re
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
import torch

from config_loader import apply_main_eval_config
from data_io.pt_tensor import DEFAULT_SUBSAMPLE_SEED
from models.hier_shplrnn import shallowPLRNN
from multitasking import get_current_gpu_utilization
from trainers.bptt import load_from_path, read_hypers
from visualisation.eval_plotter import plot_subject_feature_pca


def legacy_get_device(args: argparse.Namespace) -> argparse.Namespace:
    """Set device string for legacy hierarchical evaluation mode.

    Parameters
    ----------
    args : argparse.Namespace
        Legacy model argument namespace.

    Returns
    -------
    argparse.Namespace
        Namespace with ``device`` field populated.
    """
    args.device = "cpu"
    if args.use_gpu:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        args.device = f"{args.device}:{args.device_id}"
    print(f"Using device: {args.device}", flush=True)
    return args


def legacy_get_dataset(args: argparse.Namespace) -> Any:
    """Construct the training/evaluation dataset for legacy mode.

    Parameters
    ----------
    args : argparse.Namespace
        Legacy model argument namespace.

    Returns
    -------
    Any
        Instantiated ``MultiSubjectDataset`` object.
    """
    base_path = Path(__file__).resolve().parent.parent
    candidate_paths = [
        base_path / "data_io" / "dataset.py",
        base_path / "io" / "dataset.py",
    ]
    dataset_module_path: Path | None = None
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            dataset_module_path = candidate_path
            break

    if dataset_module_path is None:
        raise FileNotFoundError(
            "Could not locate a dataset module for legacy evaluation. "
            f"Checked paths: {[str(path) for path in candidate_paths]}."
        )

    spec = importlib.util.spec_from_file_location("hierarchicaldsr_io_dataset", str(dataset_module_path))
    if spec is None or spec.loader is None:
        raise ImportError(
            "Failed to create import specification for legacy dataset module "
            f"'{dataset_module_path}'."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset_class = getattr(module, "MultiSubjectDataset", None)
    if dataset_class is None:
        raise AttributeError(
            "Expected class 'MultiSubjectDataset' in legacy dataset module, "
            f"but it was not found in '{dataset_module_path}'."
        )

    return dataset_class(
        args.data_path,
        args.seq_len,
        args.train_set_size,
        args.subjects_per_batch,
        args.num_workers,
        args.device,
        getattr(args, "subsample_size", None),
        getattr(args, "subsample_seed", DEFAULT_SUBSAMPLE_SEED),
    )


def handle_model_paths(args: argparse.Namespace) -> list[str]:
    """Resolve all run directories from a model root path.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed evaluation arguments.

    Returns
    -------
    list[str]
        Sorted list of discovered run directories.
    """
    if args.model_path is None:
        raise ValueError("Model path must be specified.")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path '{args.model_path}' does not exist.")

    run_ids = {str(i).zfill(3) for i in range(1, 1000)}
    discovered_runs: list[str] = []
    for root, _, files in os.walk(args.model_path):
        if os.path.basename(root) in run_ids and any(name.endswith(".pt") for name in files):
            discovered_runs.append(root)

    if not discovered_runs and os.path.isdir(args.model_path):
        root_files = os.listdir(args.model_path)
        if any(name.endswith(".pt") for name in root_files):
            discovered_runs.append(args.model_path)

    if not discovered_runs:
        raise ValueError(
            "Could not find any run directories with checkpoint files (*.pt). "
            f"Searched under '{args.model_path}'."
        )

    return sorted(discovered_runs)


def safe_run_name(run_path: str) -> str:
    """Convert a run path to a filename-safe identifier."""
    normalized = os.path.normpath(run_path)
    parts = [part for part in normalized.split(os.sep) if part not in {"", "."}]
    run_tag = "__".join(parts)
    run_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", run_tag)
    return run_tag if run_tag else "run"


def resolve_subject_labels_path(args: argparse.Namespace) -> str | None:
    """Resolve an optional subject-label path for visualization."""
    if args.subject_labels_path is not None:
        return args.subject_labels_path
    if args.eval_data_path is None:
        return None
    candidate_path = f"{os.path.splitext(args.eval_data_path)[0]}_labels.npy"
    if os.path.exists(candidate_path):
        return candidate_path
    return None


def load_subject_labels(labels_path: str, expected_num_subjects: int) -> np.ndarray:
    """Load and validate subject labels."""
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Subject label file '{labels_path}' does not exist.")

    labels = np.asarray(np.load(labels_path))
    if labels.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional subject label array, "
            f"but got shape {labels.shape} from '{labels_path}'."
        )
    if labels.shape[0] != expected_num_subjects:
        raise ValueError(
            "Subject label count does not match feature vector count: "
            f"got {labels.shape[0]} labels from '{labels_path}' and "
            f"{expected_num_subjects} subjects in the checkpoint."
        )
    return labels


def extract_subject_feature_vectors(model: shallowPLRNN) -> np.ndarray:
    """Extract per-subject feature vectors from a loaded model."""
    if not hasattr(model, "p_vector"):
        raise ValueError(
            "Feature extraction currently expects projection hierarchy vectors ('p_vector'), "
            "but this checkpoint does not expose that attribute."
        )

    feature_vectors = cast(torch.Tensor, model.p_vector).detach().cpu().numpy()
    if feature_vectors.ndim != 2:
        raise ValueError(
            "Expected subject feature vectors with shape (num_subjects, feature_dim), "
            f"but got shape {feature_vectors.shape}."
        )
    return feature_vectors


EvaluationResult = tuple[str, np.ndarray, np.ndarray, np.ndarray]


def evaluate_model(
    args_and_path: tuple[argparse.Namespace, str],
) -> EvaluationResult:
    """Evaluate a single model run."""
    args, model_path = args_and_path
    print(f"\nEvaluating model at path {model_path}", flush=True)

    worker_args = argparse.Namespace(**vars(args))
    worker_args.model_path = model_path
    worker_args.finetune = False

    modelargs = read_hypers(worker_args)
    modelargs.use_gpu = args.use_gpu
    modelargs.compile = args.compile
    if args.eval_data_path is not None:
        modelargs.eval_data_path = args.eval_data_path

    if args.use_gpu:
        _, mem_dict = get_current_gpu_utilization()
        min_mem = 100.0
        device_id = None
        for gpu_id in args.free_gpus:
            if mem_dict[gpu_id] < min_mem:
                min_mem = mem_dict[gpu_id]
                device_id = gpu_id
        if device_id is None:
            raise RuntimeError(
                "No suitable GPU found for this worker. "
                f"Available candidates were: {args.free_gpus}."
            )
        modelargs.device_id = device_id

    modelargs.kl_bins = args.kl_bins
    modelargs.pse_smooth = args.pse_smooth

    modelargs = legacy_get_device(modelargs)
    dataset = legacy_get_dataset(modelargs)
    model = shallowPLRNN(modelargs, dataset)
    load_from_path(model, worker_args)
    feature_vectors = extract_subject_feature_vectors(model)
    if modelargs.compile:
        model = cast(shallowPLRNN, torch.compile(model))
    model.eval()
    model.evaluator.compute_expensive(["dstsp", "pse"])
    dstsp = np.atleast_1d(model.evaluator.get_state_space_divergence().cpu().squeeze().numpy())
    pse = np.atleast_1d(model.evaluator.get_pse().squeeze())
    return model_path, dstsp, pse, feature_vectors


def _prepare_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply legacy evaluation config and defaults."""
    # Capture CLI overrides before they are potentially overwritten by config
    cli_subsample_size = getattr(args, "subsample_size", None)
    cli_subsample_seed = getattr(args, "subsample_seed", DEFAULT_SUBSAMPLE_SEED)

    args = apply_main_eval_config(args)

    # Re-apply CLI overrides if they were provided
    if cli_subsample_size is not None:
        args.subsample_size = cli_subsample_size
    if cli_subsample_seed != DEFAULT_SUBSAMPLE_SEED:
        args.subsample_seed = cli_subsample_seed

    if args.save_path is None:
        args.save_path = "./results/experiment"
    return args


def _resolve_available_gpus(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve available GPUs for worker processes."""
    if not args.use_gpu:
        return args

    util_dict, mem_dict = get_current_gpu_utilization()
    args.free_gpus = [
        int(gpu_id)
        for gpu_id in util_dict.keys()
        if util_dict[gpu_id] < 0.05 and mem_dict[gpu_id] < 0.1
    ]
    if len(args.free_gpus) == 0:
        raise RuntimeError(
            "No free GPUs found (criteria: utilization < 5% and memory < 10%). "
            "Use CPU mode or free up a GPU."
        )

    print(
        "Can use GPUs:",
        args.free_gpus,
        ". Keep num_workers in a safe range for available memory.",
        flush=True,
    )
    return args


def _run_parallel_evaluation(args: argparse.Namespace, paths: Sequence[str]) -> list[EvaluationResult]:
    """Run model evaluation across all discovered paths."""
    worker_count = min(args.num_workers, len(paths))
    if worker_count <= 0:
        raise ValueError(
            "Cannot start evaluation workers because worker_count <= 0. "
            f"num_workers={args.num_workers}, paths={len(paths)}."
        )

    jobs = [(args, path) for path in paths]
    with multiprocessing.Pool(worker_count) as pool:
        results = pool.map(evaluate_model, jobs, chunksize=1)
    return list(results)


def _ensure_save_directory(save_path: str) -> None:
    """Create output directory if it does not exist."""
    os.makedirs(save_path, exist_ok=True)


def _build_metric_results_dataframe(results_list: Sequence[EvaluationResult]) -> pd.DataFrame:
    """Build deterministic metric dataframe from per-run results."""
    results: dict[tuple[str, int], dict[str, float]] = {}
    sorted_results = sorted(results_list, key=lambda item: item[0])
    for run_path, dstsp, pse, _ in sorted_results:
        if dstsp.shape[0] != pse.shape[0]:
            raise ValueError(
                "Metric arrays must have matching lengths for each run. "
                f"run_path='{run_path}', dstsp={dstsp.shape[0]}, pse={pse.shape[0]}."
            )

        for subject_index, (dstsp_value, pse_value) in enumerate(zip(dstsp, pse)):
            results[(run_path, subject_index)] = {
                "dstsp": float(dstsp_value),
                "pse": float(pse_value),
            }

    if not results:
        raise ValueError("No metric results were produced by evaluation workers.")
    return pd.DataFrame(results)


def _save_metric_results(df: pd.DataFrame, save_path: str) -> None:
    """Save metric results CSV."""
    result_path = os.path.join(save_path, "results.csv")
    df.to_csv(result_path)
    print(f"Results saved to {result_path}", flush=True)


def _resolve_labels_for_features(
    args: argparse.Namespace,
    results_list: Sequence[EvaluationResult],
) -> np.ndarray | None:
    """Resolve optional subject labels for feature export and plotting."""
    labels_path = resolve_subject_labels_path(args)
    if labels_path is None:
        print(
            "No subject label path provided or inferred. "
            "Subject features will be exported but label-aware PCA plots are skipped.",
            flush=True,
        )
        return None

    if len(results_list) == 0:
        raise ValueError("Cannot resolve labels because no evaluation results were returned.")

    expected_num_subjects = int(results_list[0][3].shape[0])
    labels = load_subject_labels(labels_path, expected_num_subjects)
    print(f"Loaded subject labels from {labels_path}", flush=True)
    return labels


def _build_feature_rows(
    results_list: Sequence[EvaluationResult],
    labels: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Build deterministic rows for subject feature export."""
    feature_rows: list[dict[str, Any]] = []
    sorted_results = sorted(results_list, key=lambda item: item[0])
    for run_path, _, _, feature_vectors in sorted_results:
        if labels is not None and labels.shape[0] != feature_vectors.shape[0]:
            raise ValueError(
                "Label count does not match feature vectors for run "
                f"'{run_path}': labels={labels.shape[0]}, features={feature_vectors.shape[0]}."
            )

        for subject_index, subject_features in enumerate(feature_vectors):
            row: dict[str, Any] = {
                "run_path": run_path,
                "subject_index": int(subject_index),
            }
            for dim_index, dim_value in enumerate(subject_features, start=1):
                row[f"feature_{dim_index}"] = float(dim_value)
            if labels is not None:
                row["label"] = str(labels[subject_index])
            feature_rows.append(row)

    if not feature_rows:
        raise ValueError(
            "No subject feature vectors were collected from evaluated runs. "
            "Cannot write feature export CSV."
        )
    return feature_rows


def _save_feature_rows(args: argparse.Namespace, feature_rows: Sequence[dict[str, Any]]) -> None:
    """Save subject feature vectors to CSV."""
    feature_df = pd.DataFrame(list(feature_rows))
    feature_csv_path = os.path.join(args.save_path, args.subject_feature_csv)
    feature_df.to_csv(feature_csv_path, index=False)
    print(f"Saved subject feature vectors to {feature_csv_path}", flush=True)


def _save_feature_pca_plots(
    save_path: str,
    results_list: Sequence[EvaluationResult],
    labels: np.ndarray,
) -> None:
    """Save per-run PCA plots for extracted subject features."""
    sorted_results = sorted(results_list, key=lambda item: item[0])
    for run_path, _, _, feature_vectors in sorted_results:
        if feature_vectors.shape[1] < 2:
            print(
                f"Skipping PCA plot for run '{run_path}' because feature_dim={feature_vectors.shape[1]} < 2.",
                flush=True,
            )
            continue

        plot_filename = f"subject_feature_pca_{safe_run_name(run_path)}.png"
        plot_path = os.path.join(save_path, plot_filename)
        plot_subject_feature_pca(
            feature_vectors=feature_vectors,
            labels=labels,
            run_path=run_path,
            output_path=plot_path,
            show_arrows=False,
        )


def run_legacy_evaluation(args: argparse.Namespace) -> None:
    """Run legacy checkpoint evaluation workflow."""
    args = _prepare_legacy_args(args)
    args = _resolve_available_gpus(args)

    paths = handle_model_paths(args)
    results_list = _run_parallel_evaluation(args, paths)

    _ensure_save_directory(args.save_path)
    results_df = _build_metric_results_dataframe(results_list)
    _save_metric_results(results_df, args.save_path)

    labels = _resolve_labels_for_features(args, results_list)
    feature_rows = _build_feature_rows(results_list, labels)
    _save_feature_rows(args, feature_rows)

    if labels is None:
        print("Skipping subject feature PCA plots because labels are unavailable.", flush=True)
        return

    _save_feature_pca_plots(args.save_path, results_list, labels)
