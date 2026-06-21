"""Evaluate trained hierarchical PLRNN checkpoints and export summary metrics.

Outputs
-------
This entry script writes metrics and feature artifacts under ``{save_path}``.

Output files include:
- ``results.csv``: one column per ``(run_path, subject_index)`` pair and includes:
- ``dstsp``: state-space divergence value.
- ``pse``: power-spectrum error value.
- ``subject_features.csv``: extracted per-subject feature vectors.
- ``subject_feature_pca_<run>.png``: 2D PCA scatter with principal-component arrows.
"""

import argparse
import multiprocessing
import os
import re
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from trainers.bptt import load_from_path, read_hypers
from config_loader import apply_main_eval_config
from visualisation.eval_plotter import plot_subject_feature_pca
from main import get_device, get_dataset
from models.hier_shplrnn import shallowPLRNN
from multitasking import get_current_gpu_utilization

torch.set_num_threads(1)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained hierarchical PLRNN model.")

    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="Path to the hierarchical YAML configuration file.",
    )

    io_group = parser.add_argument_group("Model And Data")
    io_group.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model run or experiment root.",
    )
    io_group.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="Optional override path to evaluation data (.pt).",
    )
    io_group.add_argument(
        "--save_path",
        type=str,
        default="./results/experiment",
        help="Directory to save evaluation outputs.",
    )
    io_group.add_argument(
        "--subject_labels_path",
        type=str,
        default=None,
        help="Optional path to subject labels (.npy) used for label-aware PCA plotting.",
    )

    output_group = parser.add_argument_group("Feature Export")
    output_group.add_argument(
        "--subject_feature_csv",
        type=str,
        default="subject_features.csv",
        help="Filename for exported subject feature vectors.",
    )
    output_group.add_argument(
        "--subject_feature_pca_prefix",
        type=str,
        default="subject_feature_pca",
        help="Filename prefix for per-run PCA plots.",
    )

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--use_gpu", action="store_true", help="Use GPU when available.")

    return parser.parse_args()


def handle_path(args: argparse.Namespace) -> list[str]:
    """Resolve all run directories from a model root path.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed evaluation arguments.

    Returns
    -------
    list[str]
        Sorted list of discovered run directories.

    Raises
    ------
    FileNotFoundError
        If ``args.model_path`` does not exist.
    ValueError
        If no valid run directories or checkpoints are found.
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


def _safe_run_name(run_path: str) -> str:
    """Convert a run path to a filename-safe identifier.

    Parameters
    ----------
    run_path : str
        Run directory path.

    Returns
    -------
    str
        Sanitized name that can safely be used in filenames.
    """
    normalized = os.path.normpath(run_path)
    parts = [part for part in normalized.split(os.sep) if part not in {"", "."}]
    run_tag = "__".join(parts)
    run_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", run_tag)
    return run_tag if run_tag else "run"


def resolve_subject_labels_path(args: argparse.Namespace) -> str | None:
    """Resolve an optional subject-label path for visualization.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed evaluation arguments.

    Returns
    -------
    str | None
        Explicit or inferred label path when available, otherwise ``None``.
    """
    if args.subject_labels_path is not None:
        return args.subject_labels_path
    if args.eval_data_path is None:
        return None
    candidate_path = f"{os.path.splitext(args.eval_data_path)[0]}_labels.npy"
    if os.path.exists(candidate_path):
        return candidate_path
    return None


def load_subject_labels(labels_path: str, expected_num_subjects: int) -> np.ndarray:
    """Load and validate subject labels.

    Parameters
    ----------
    labels_path : str
        Path to a ``.npy`` file containing one label per subject.
    expected_num_subjects : int
        Number of subject feature vectors expected.

    Returns
    -------
    np.ndarray
        One-dimensional label array.

    Raises
    ------
    FileNotFoundError
        If ``labels_path`` does not exist.
    ValueError
        If labels are not one-dimensional or count does not match expected subjects.
    """
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
    """Extract per-subject feature vectors from a loaded model.

    Parameters
    ----------
    model : shallowPLRNN
        Loaded hierarchical PLRNN model.

    Returns
    -------
    np.ndarray
        Array with shape ``(num_subjects, feature_dim)``.

    Raises
    ------
    ValueError
        If the loaded model does not expose projection vectors as subject features.
    """
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


def evaluate_model(
    args_and_path: tuple[argparse.Namespace, str],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a single model run.

    Parameters
    ----------
    args_and_path : tuple[argparse.Namespace, str]
        Pair of global evaluation arguments and run directory path.

    Returns
    -------
    tuple[str, np.ndarray, np.ndarray, np.ndarray]
        Run path, state-space divergence values, PSE values, and subject feature vectors.
    """
    args, model_path = args_and_path
    print("\nEvaluating model at path", model_path, flush=True)

    worker_args = argparse.Namespace(**vars(args))
    worker_args.model_path = model_path
    worker_args.finetune = False

    modelargs = read_hypers(worker_args)
    # overwrite some arguments
    modelargs.use_gpu = args.use_gpu
    modelargs.compile = args.compile
    if args.eval_data_path is not None:
        modelargs.eval_data_path = args.eval_data_path

    # get the GPU with the most free memory
    if args.use_gpu:
        _, mem_dict = get_current_gpu_utilization()
        min_mem = 100.0
        device_id = None
        for g in args.free_gpus:
            if mem_dict[g] < min_mem:
                min_mem = mem_dict[g]
                device_id = g
        if device_id is None:
            raise RuntimeError(
                "No suitable GPU found for this worker. "
                f"Available candidates were: {args.free_gpus}."
            )
        modelargs.device_id = device_id

    # change pse smoothing and kl bins if necessary
    modelargs.kl_bins = args.kl_bins
    modelargs.pse_smooth = args.pse_smooth

    modelargs = get_device(modelargs)
    dataset = get_dataset(modelargs)
    model = shallowPLRNN(modelargs, dataset)
    load_from_path(model, worker_args)
    feature_vectors = extract_subject_feature_vectors(model)
    if modelargs.compile:
        model = cast(shallowPLRNN, torch.compile(model))
    model.eval()
    model.evaluator.compute_expensive(['dstsp', 'pse'])
    dstsp = np.atleast_1d(model.evaluator.get_state_space_divergence().cpu().squeeze().numpy())
    pse = np.atleast_1d(model.evaluator.get_pse().squeeze())
    return model_path, dstsp, pse, feature_vectors


def main() -> None:
    """Run multi-process evaluation over discovered model runs."""
    args = parse_args()
    args = apply_main_eval_config(args)

    # get all free GPUs
    if args.use_gpu:
        util_dict, mem_dict = get_current_gpu_utilization()
        args.free_gpus = [int(g) for g in util_dict.keys() if util_dict[g] < 0.05 and mem_dict[g] < 0.1]
        if len(args.free_gpus) == 0:
            raise RuntimeError(
                "No free GPUs found (criteria: utilization < 5% and memory < 10%). "
                "Use CPU mode or free up a GPU."
            )
        print(
            "Can use GPUs:",
            args.free_gpus,
            ". Keep num_workers in a safe range for available memory.",
            flush=True
        )

    paths = handle_path(args)

    # split up the evaluation into multiple processes
    worker_count = min(args.num_workers, len(paths))
    with multiprocessing.Pool(worker_count) as pool:
        results_list = pool.map(evaluate_model, [(args, p) for p in paths], chunksize=1)

    # store the results in a dictionary
    results = {}
    for (p, dstsp, pse, _) in results_list:
        for s, (kl, hel) in enumerate(zip(dstsp, pse)):
            results[(p, s)] = {'dstsp': kl, 'pse': hel}

    # generate folders if necessary
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # save metric results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.save_path, 'results.csv'))
    print("Results saved to ", os.path.join(args.save_path, 'results.csv'), flush=True)

    # export subject feature vectors and optionally generate PCA plots
    labels_path = resolve_subject_labels_path(args)
    labels: np.ndarray | None = None
    if labels_path is None:
        print(
            "No subject label path provided or inferred. "
            "Subject features will be exported but label-aware PCA plots are skipped.",
            flush=True,
        )

    feature_rows: list[dict[str, Any]] = []
    for p, _, _, feature_vectors in results_list:
        if labels_path is not None and labels is None:
            labels = load_subject_labels(labels_path, feature_vectors.shape[0])
            print(f"Loaded subject labels from {labels_path}", flush=True)

        if labels is not None and labels.shape[0] != feature_vectors.shape[0]:
            raise ValueError(
                "Label count does not match feature vectors for run "
                f"'{p}': got {labels.shape[0]} labels and "
                f"{feature_vectors.shape[0]} feature vectors."
            )

        for subject_index, subject_features in enumerate(feature_vectors):
            row: dict[str, Any] = {
                "run_path": p,
                "subject_index": subject_index,
            }
            for dim_index, dim_value in enumerate(subject_features):
                row[f"feature_{dim_index + 1}"] = float(dim_value)
            if labels is not None:
                row["label"] = str(labels[subject_index])
            feature_rows.append(row)

        if labels is not None:
            run_name = _safe_run_name(p)
            pca_path = os.path.join(args.save_path, f"{args.subject_feature_pca_prefix}_{run_name}.png")
            plot_subject_feature_pca(feature_vectors, labels, p, pca_path)

    if not feature_rows:
        raise ValueError(
            "No subject feature vectors were collected from evaluated runs. "
            "Cannot write feature export CSV."
        )
    feature_df = pd.DataFrame(feature_rows)
    feature_csv_path = os.path.join(args.save_path, args.subject_feature_csv)
    feature_df.to_csv(feature_csv_path, index=False)
    print(f"Saved subject feature vectors to {feature_csv_path}", flush=True)

if __name__ == "__main__":
    main()
