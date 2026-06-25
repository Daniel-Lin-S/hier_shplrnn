"""Evaluate hierarchical checkpoints and run latent-space benchmarks.

Outputs
-------
This entry script writes metrics and feature artifacts under ``{save_path}``.

Output files include:
- ``results.csv``: one column per ``(run_path, subject_index)`` pair and includes:
- ``dstsp``: state-space divergence value.
- ``pse``: power-spectrum error value.
- ``subject_features.csv``: extracted per-subject feature vectors.

If ``--run_latent_benchmark`` is enabled, this script evaluates latent-space
classifiability and information content for configured feature extractors and
writes one result folder per extractor under the configured benchmark root.
"""

import argparse
import importlib.util
import multiprocessing
import os
import re
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import torch

from trainers.bptt import load_from_path, read_hypers
from config_loader import apply_main_eval_config, load_config
from eval.latent_benchmark import benchmark_config_from_files, run_latent_benchmark
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
        default=None,
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
        default=None,
        help=(
            "Directory to save evaluation outputs in legacy mode. "
            "In benchmark mode, this optionally overrides evaluation.benchmark.save_path from config."
        ),
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
    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--use_gpu", action="store_true", help="Use GPU when available.")

    benchmark_group = parser.add_argument_group("Latent Benchmark")
    benchmark_group.add_argument(
        "--run_latent_benchmark",
        action="store_true",
        help=(
            "Run latent-space benchmark for a model using separate shared/model YAML files."
        ),
    )
    benchmark_group.add_argument(
        "--benchmark_config",
        type=str,
        default=None,
        help=(
            "Path to shared benchmark YAML (dataset settings, CV, baseline specs). "
            "Overrides evaluation.benchmark.shared_config_path from --config."
        ),
    )
    benchmark_group.add_argument(
        "--benchmark_model_config",
        type=str,
        default=None,
        help=(
            "Path to benchmark model YAML (single `model` or list `models` extractor specs, "
            "optionally with `model_defaults`). "
            "Overrides evaluation.benchmark.model_config_path from --config."
        ),
    )

    return parser.parse_args()


def _legacy_get_device(args: argparse.Namespace) -> argparse.Namespace:
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


def _legacy_get_dataset(args: argparse.Namespace) -> Any:
    """Construct the training/evaluation dataset for legacy mode.

    This loader avoids importing ``main.py`` directly, because ``io.dataset``
    can collide with Python's built-in ``io`` module in some environments.

    Parameters
    ----------
    args : argparse.Namespace
        Legacy model argument namespace.

    Returns
    -------
    Any
        Instantiated ``MultiSubjectDataset`` object.
    """
    base_path = Path(__file__).resolve().parent
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
    )


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


def resolve_benchmark_config_paths(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve shared/model benchmark YAML paths.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI namespace.

    Returns
    -------
    tuple[str, str]
        Shared benchmark config path and per-model config path.

    Raises
    ------
    ValueError
        If one or both benchmark config paths are missing.
    """
    shared_path = args.benchmark_config
    model_path = args.benchmark_model_config

    if shared_path is not None and model_path is not None:
        return shared_path, model_path

    config_mapping = load_config(args.config)
    benchmark_mapping = _read_benchmark_mapping(config_mapping)

    if shared_path is None:
        shared_path = _read_optional_non_empty_string(benchmark_mapping, "shared_config_path")
    if model_path is None:
        model_path = _read_optional_non_empty_string(benchmark_mapping, "model_config_path")

    if shared_path is None or model_path is None:
        raise ValueError(
            "Latent benchmark requires both shared and model YAML paths. "
            "Provide --benchmark_config and --benchmark_model_config, or set "
            "evaluation.benchmark.shared_config_path and evaluation.benchmark.model_config_path "
            f"in '{args.config}'."
        )
    return shared_path, model_path


def _read_benchmark_mapping(config_mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read optional evaluation.benchmark mapping from a root config.

    Parameters
    ----------
    config_mapping : Mapping[str, Any]
        Parsed root YAML mapping.

    Returns
    -------
    Mapping[str, Any]
        Benchmark mapping or empty mapping if missing.
    """
    evaluation_mapping = config_mapping.get("evaluation")
    if not isinstance(evaluation_mapping, Mapping):
        return {}

    benchmark_mapping = evaluation_mapping.get("benchmark")
    if benchmark_mapping is None:
        return {}
    if not isinstance(benchmark_mapping, Mapping):
        raise TypeError(
            "Expected 'evaluation.benchmark' to be a mapping in main config, "
            f"but got type '{type(benchmark_mapping).__name__}'."
        )
    return benchmark_mapping


def _read_optional_non_empty_string(mapping: Mapping[str, Any], key: str) -> str | None:
    """Read an optional non-empty string key.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Source mapping.
    key : str
        Key name.

    Returns
    -------
    str | None
        String value when present and non-empty, otherwise ``None``.
    """
    if key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Expected '{key}' to be string when provided, but got {type(value).__name__}.")
    if value.strip() == "":
        return None
    return value


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

    modelargs = _legacy_get_device(modelargs)
    dataset = _legacy_get_dataset(modelargs)
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

    if args.run_latent_benchmark:
        shared_config_path, model_config_path = resolve_benchmark_config_paths(args)
        benchmark_config = benchmark_config_from_files(
            shared_config_path,
            model_config_path,
            save_path_override=args.save_path,
        )
        run_latent_benchmark(benchmark_config)
        return

    args = apply_main_eval_config(args)

    if args.save_path is None:
        args.save_path = "./results/experiment"

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

    # export subject feature vectors
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
