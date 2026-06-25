"""Run YAML-driven progressive cohort training via repeated ``main.py`` calls.

This utility reads a ``progressive_training`` section from YAML, samples random
subject subsets for increasing cohort sizes, and invokes ``main.py`` once per
cohort/run combination.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class ProgressiveTrainingPlan:
    """Configuration for progressive subset training.

    Attributes
    ----------
    main_config_path : str
        Path to ``main.py`` training config YAML.
    source_data_path : str
        Path to source tensor file with shape ``(subjects, timesteps, channels)``.
    subset_output_dir : str
        Directory where sampled subset tensors and metadata are stored.
    subset_prefix : str
        Filename prefix for emitted subset artifacts.
    cohort_sizes : tuple[int, ...]
        Strictly increasing cohort sizes used for progressive stages.
    random_seed : int
        Base RNG seed for reproducible subject sampling.
    nested_subsets : bool
        Whether larger cohorts include smaller ones for the same run.
    runs_per_size : int
        Number of independent random subset runs to train per cohort size.
    run_start : int
        First run id (inclusive).
    save_path : str
        Root output directory passed to ``main.py --save_path``.
    experiment : str
        Experiment folder name passed to ``main.py --experiment``.
    model_name_prefix : str
        Prefix used to name each cohort model as ``{prefix}_{cohort_size}``.
    force_retrain : bool
        Whether to retrain even when a run directory already exists.
    use_gpu : bool
        Whether to pass ``--use_gpu`` to ``main.py``.
    device_id : int
        CUDA device id forwarded to ``main.py --device_id`` when GPU is used.
    """

    main_config_path: str
    source_data_path: str
    subset_output_dir: str
    subset_prefix: str
    cohort_sizes: tuple[int, ...]
    random_seed: int
    nested_subsets: bool
    runs_per_size: int
    run_start: int
    save_path: str
    experiment: str
    model_name_prefix: str
    force_retrain: bool
    use_gpu: bool
    device_id: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run progressive subset training by repeatedly invoking main.py."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/progressive_train_epileptic.yaml",
        help="Path to YAML containing top-level progressive_training settings.",
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Override plan.force_retrain and always rerun all stages.",
    )
    return parser.parse_args()


def load_progressive_plan(config_path: str) -> ProgressiveTrainingPlan:
    """Load and validate progressive-training plan.

    Parameters
    ----------
    config_path : str
        Path to YAML file containing ``progressive_training``.

    Returns
    -------
    ProgressiveTrainingPlan
        Parsed training plan.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Progressive config file '{path}' does not exist.")

    with path.open("r", encoding="utf-8") as file:
        config_raw = yaml.safe_load(file)

    config = _require_mapping(config_raw, "root")
    plan_mapping = _require_mapping(config.get("progressive_training"), "progressive_training")
    run_root = Path.cwd().resolve()

    main_config = _require_str(plan_mapping, "main_config_path", "progressive_training")
    source_data = _require_str(plan_mapping, "source_data_path", "progressive_training")
    subset_dir = _require_str(plan_mapping, "subset_output_dir", "progressive_training")
    save_path = _require_str(plan_mapping, "save_path", "progressive_training")

    plan = ProgressiveTrainingPlan(
        main_config_path=_resolve_path(run_root, main_config),
        source_data_path=_resolve_path(run_root, source_data),
        subset_output_dir=_resolve_path(run_root, subset_dir),
        subset_prefix=str(plan_mapping.get("subset_prefix", "progressive_subset")),
        cohort_sizes=_parse_cohort_sizes(plan_mapping.get("cohort_sizes")),
        random_seed=_require_int_with_default(plan_mapping, "random_seed", 42),
        nested_subsets=_require_bool_with_default(plan_mapping, "nested_subsets", True),
        runs_per_size=_require_positive_int_with_default(plan_mapping, "runs_per_size", 1),
        run_start=_require_positive_int_with_default(plan_mapping, "run_start", 1),
        save_path=_resolve_path(run_root, save_path),
        experiment=_require_str(plan_mapping, "experiment", "progressive_training"),
        model_name_prefix=_require_str(plan_mapping, "model_name_prefix", "progressive_training"),
        force_retrain=_require_bool_with_default(plan_mapping, "force_retrain", False),
        use_gpu=_require_bool_with_default(plan_mapping, "use_gpu", True),
        device_id=_require_int_with_default(plan_mapping, "device_id", 0),
    )

    if plan.subset_prefix.strip() == "":
        raise ValueError("Expected progressive_training.subset_prefix to be non-empty when provided.")

    return plan


def run_progressive_training(plan: ProgressiveTrainingPlan) -> None:
    """Execute progressive subset training according to a plan.

    Parameters
    ----------
    plan : ProgressiveTrainingPlan
        Parsed progressive training plan.
    """
    source_tensor = _load_source_tensor(plan.source_data_path)
    total_subjects = int(source_tensor.shape[0])

    for size in plan.cohort_sizes:
        if size > total_subjects:
            raise ValueError(
                "Cohort size exceeds available subjects in source tensor: "
                f"requested={size}, available={total_subjects}, source='{plan.source_data_path}'."
            )

    subset_output_dir = Path(plan.subset_output_dir)
    subset_output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent

    for run_offset in range(plan.runs_per_size):
        run_id = plan.run_start + run_offset
        rng = np.random.default_rng(plan.random_seed + run_id)
        permutation: np.ndarray | None = None
        if plan.nested_subsets:
            permutation = np.asarray(rng.permutation(total_subjects), dtype=np.int64)

        for cohort_size in plan.cohort_sizes:
            subject_indices = _sample_subject_indices(
                rng=rng,
                total_subjects=total_subjects,
                cohort_size=cohort_size,
                nested_subsets=plan.nested_subsets,
                permutation=permutation,
            )

            subset_filename = f"{plan.subset_prefix}_{cohort_size}_run{run_id:03d}.pt"
            subset_path = subset_output_dir / subset_filename
            metadata_path = subset_output_dir / subset_filename.replace(".pt", ".json")

            subset_tensor = source_tensor[torch.as_tensor(subject_indices, dtype=torch.long)]
            torch.save(subset_tensor.to(dtype=torch.float32), subset_path)
            _save_subset_metadata(
                metadata_path=metadata_path,
                source_data_path=plan.source_data_path,
                run_id=run_id,
                cohort_size=cohort_size,
                random_seed=plan.random_seed,
                nested_subsets=plan.nested_subsets,
                subject_indices=subject_indices,
            )

            _run_main_training_for_subset(
                plan=plan,
                run_id=run_id,
                cohort_size=cohort_size,
                subset_path=subset_path,
                repo_root=repo_root,
            )


def _sample_subject_indices(
    rng: np.random.Generator,
    total_subjects: int,
    cohort_size: int,
    nested_subsets: bool,
    permutation: np.ndarray | None,
) -> np.ndarray:
    """Sample subject indices for one progressive stage.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    total_subjects : int
        Number of available subjects.
    cohort_size : int
        Number of subjects to sample.
    nested_subsets : bool
        Whether to use prefix sampling from one fixed permutation.
    permutation : np.ndarray | None
        Shared permutation used when ``nested_subsets`` is True.

    Returns
    -------
    np.ndarray
        Sampled subject indices.
    """
    if nested_subsets:
        if permutation is None:
            raise ValueError("Expected a shared permutation when nested_subsets=True.")
        return np.asarray(permutation[:cohort_size], dtype=np.int64)

    sampled = rng.choice(total_subjects, size=cohort_size, replace=False)
    return np.asarray(sampled, dtype=np.int64)


def _run_main_training_for_subset(
    plan: ProgressiveTrainingPlan,
    run_id: int,
    cohort_size: int,
    subset_path: Path,
    repo_root: Path,
) -> None:
    """Invoke main.py for one sampled subset.

    Parameters
    ----------
    plan : ProgressiveTrainingPlan
        Progressive training plan.
    run_id : int
        Run identifier.
    cohort_size : int
        Cohort size for this stage.
    subset_path : Path
        Path to saved subset tensor.
    repo_root : Path
        Repository root where main.py is located.
    """
    model_name = f"{plan.model_name_prefix}_{cohort_size}"
    run_dir = Path(plan.save_path) / plan.experiment / model_name / f"{run_id:03d}"
    hypers_path = run_dir / "hypers.txt"

    if hypers_path.exists() and not plan.force_retrain:
        print(
            "Skipping training because run already exists and force_retrain is disabled: "
            f"run_dir='{run_dir}'.",
            flush=True,
        )
        return

    command = [
        sys.executable,
        "main.py",
        "--config",
        plan.main_config_path,
        "--data_path",
        str(subset_path),
        "--eval_data_path",
        str(subset_path),
        "--save_path",
        plan.save_path,
        "--experiment",
        plan.experiment,
        "--name",
        model_name,
        "--run",
        str(run_id),
    ]

    if plan.use_gpu:
        command.append("--use_gpu")
        command.extend(["--device_id", str(plan.device_id)])

    print(
        "Launching main.py for progressive stage: "
        f"run={run_id:03d}, cohort_size={cohort_size}, model_name='{model_name}'.",
        flush=True,
    )
    subprocess.run(command, check=True, cwd=str(repo_root))


def _save_subset_metadata(
    metadata_path: Path,
    source_data_path: str,
    run_id: int,
    cohort_size: int,
    random_seed: int,
    nested_subsets: bool,
    subject_indices: np.ndarray,
) -> None:
    """Save subset selection metadata for reproducibility.

    Parameters
    ----------
    metadata_path : Path
        Output metadata path.
    source_data_path : str
        Original source data path.
    run_id : int
        Run identifier.
    cohort_size : int
        Number of selected subjects.
    random_seed : int
        Base random seed from plan.
    nested_subsets : bool
        Whether this run uses nested subset prefixes.
    subject_indices : np.ndarray
        Selected subject indices.
    """
    payload = {
        "source_data_path": source_data_path,
        "run_id": run_id,
        "cohort_size": cohort_size,
        "random_seed": random_seed,
        "nested_subsets": nested_subsets,
        "subject_indices": [int(index) for index in subject_indices.tolist()],
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_source_tensor(path: str) -> torch.Tensor:
    """Load and validate source tensor.

    Parameters
    ----------
    path : str
        Tensor file path.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(subjects, timesteps, channels)`` in float32.
    """
    tensor_path = Path(path)
    if not tensor_path.exists():
        raise FileNotFoundError(f"Source data tensor '{path}' does not exist.")

    tensor = torch.load(tensor_path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Expected source data '{path}' to contain torch.Tensor, but got {type(tensor).__name__}."
        )

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)

    if tensor.ndim != 3:
        raise ValueError(
            "Expected source tensor to be 2D or 3D, "
            f"but got shape {tuple(tensor.shape)} from '{path}'."
        )

    if tensor.shape[0] <= 0:
        raise ValueError(f"Expected source tensor to include at least one subject, but got shape {tuple(tensor.shape)}.")

    if not torch.isfinite(tensor).all():
        raise ValueError(f"Expected source tensor '{path}' to contain only finite values.")

    return tensor.to(dtype=torch.float32)


def _parse_cohort_sizes(value: Any) -> tuple[int, ...]:
    """Parse and validate progressive cohort sizes.

    Parameters
    ----------
    value : Any
        Raw YAML value.

    Returns
    -------
    tuple[int, ...]
        Strictly increasing, positive cohort sizes.
    """
    if not isinstance(value, list):
        raise TypeError(
            "Expected progressive_training.cohort_sizes to be a list of positive integers, "
            f"but got {type(value).__name__}."
        )
    if not value:
        raise ValueError("Expected progressive_training.cohort_sizes to be non-empty.")

    parsed: list[int] = []
    for index, raw_item in enumerate(value):
        if not isinstance(raw_item, int):
            raise TypeError(
                "Expected each progressive cohort size to be int, "
                f"but got {type(raw_item).__name__} at index {index}."
            )
        if raw_item <= 0:
            raise ValueError(f"Expected progressive cohort sizes to be positive, but got {raw_item} at index {index}.")
        if parsed and raw_item <= parsed[-1]:
            raise ValueError(
                "Expected progressive_training.cohort_sizes to be strictly increasing, "
                f"but got previous={parsed[-1]} and current={raw_item}."
            )
        parsed.append(raw_item)

    return tuple(parsed)


def _resolve_path(base_dir: Path, raw_path: str) -> str:
    """Resolve path values relative to a run directory.

    Parameters
    ----------
    base_dir : Path
        Current run directory from which the command is invoked.
    raw_path : str
        Raw path from YAML.

    Returns
    -------
    str
        Absolute resolved path.
    """
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return str(candidate.resolve())


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Require mapping value with contextual error.

    Parameters
    ----------
    value : Any
        Raw value.
    path : str
        Human-readable location.

    Returns
    -------
    Mapping[str, Any]
        Validated mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{path}' to be a mapping, but got {type(value).__name__}.")
    return value


def _require_str(mapping: Mapping[str, Any], key: str, path: str) -> str:
    """Read required string key from mapping.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Source mapping.
    key : str
        Required key.
    path : str
        Human-readable location.

    Returns
    -------
    str
        Non-empty string value.
    """
    if key not in mapping:
        raise KeyError(f"Missing required key '{path}.{key}'.")
    value = mapping[key]
    if not isinstance(value, str):
        raise TypeError(f"Expected '{path}.{key}' to be string, but got {type(value).__name__}.")
    if value.strip() == "":
        raise ValueError(f"Expected '{path}.{key}' to be non-empty.")
    return value


def _require_int_with_default(mapping: Mapping[str, Any], key: str, default: int) -> int:
    """Read integer key with fallback.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Source mapping.
    key : str
        Optional key.
    default : int
        Fallback value.

    Returns
    -------
    int
        Parsed integer value.
    """
    value = mapping.get(key, default)
    if not isinstance(value, int):
        raise TypeError(f"Expected 'progressive_training.{key}' to be int, but got {type(value).__name__}.")
    return value


def _require_positive_int_with_default(mapping: Mapping[str, Any], key: str, default: int) -> int:
    """Read positive integer key with fallback.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Source mapping.
    key : str
        Optional key.
    default : int
        Fallback value.

    Returns
    -------
    int
        Parsed positive integer value.
    """
    value = _require_int_with_default(mapping, key, default)
    if value <= 0:
        raise ValueError(f"Expected 'progressive_training.{key}' to be > 0, but got {value}.")
    return value


def _require_bool_with_default(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    """Read boolean key with fallback.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Source mapping.
    key : str
        Optional key.
    default : bool
        Fallback value.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"Expected 'progressive_training.{key}' to be bool, but got {type(value).__name__}.")
    return value


def main() -> None:
    """Entrypoint for progressive subset training."""
    args = parse_args()
    plan = load_progressive_plan(args.config)
    if args.force_retrain:
        plan = replace(plan, force_retrain=True)

    print(
        "Loaded progressive training plan: "
        f"sizes={list(plan.cohort_sizes)}, runs_per_size={plan.runs_per_size}, "
        f"experiment='{plan.experiment}', model_prefix='{plan.model_name_prefix}'.",
        flush=True,
    )
    run_progressive_training(plan)


if __name__ == "__main__":
    main()
