from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

import yaml

_MAIN_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "obs_size": ("model", "dimensions", "obs_size"),
    "latent_size": ("model", "dimensions", "latent_size"),
    "hidden_size": ("model", "dimensions", "hidden_size"),
    "forcing_size": ("model", "dimensions", "forcing_size"),
    "obs_model": ("model", "observation", "obs_model"),
    "hierarchisation_scheme": ("model", "hierarchy", "scheme"),
    "num_individual_params": ("model", "hierarchy", "num_individual_params"),
    "lam": ("model", "hierarchy", "lambda_regularization"),
    "clipped": ("model", "observation", "clipped"),
    "learn_noise_cov": ("model", "observation", "learn_noise_cov"),
    "seq_len": ("training", "sequence", "seq_len"),
    "train_set_size": ("training", "sequence", "train_set_size"),
    "tf_alpha_start": ("training", "teacher_forcing", "alpha_start"),
    "tf_alpha_end": ("training", "teacher_forcing", "alpha_end"),
    "num_epochs": ("training", "optimization", "num_epochs"),
    "batch_size": ("training", "optimization", "batch_size"),
    "batches_per_epoch": ("training", "optimization", "batches_per_epoch"),
    "subjects_per_batch": ("training", "optimization", "subjects_per_batch"),
    "num_workers": ("training", "optimization", "num_workers"),
    "learning_rate": ("training", "optimization", "learning_rate", "shared"),
    "individual_learning_rate": ("training", "optimization", "learning_rate", "individual"),
    "weight_decay": ("training", "optimization", "weight_decay"),
    "clip_grad_norm": ("training", "optimization", "clip_grad_norm"),
    "metrics": ("evaluation", "metrics", "enabled"),
    "kl_bins": ("evaluation", "metrics", "kl_bins"),
    "pse_smooth": ("evaluation", "metrics", "pse_smooth"),
    "plots": ("evaluation", "plots", "enabled"),
    "compile": ("runtime", "compile"),
}


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a structured YAML configuration.

    Parameters
    ----------
    config_path : str
        Path to a YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the given configuration file does not exist.
    TypeError
        If the file does not parse to a mapping.
    KeyError
        If required top-level sections are missing.
    ValueError
        If the configuration is empty.
    """
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' does not exist.")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file '{path}' is empty.")
    if not isinstance(config, dict):
        raise TypeError(
            "The configuration root must be a mapping, "
            f"but got type '{type(config).__name__}'."
        )

    for section in ("model", "training", "evaluation", "runtime"):
        if section not in config:
            raise KeyError(
                f"Missing required top-level section '{section}' in configuration '{path}'."
            )

    return config


def apply_main_config(args: Namespace) -> Namespace:
    """Populate all model/training/evaluation parameters from YAML.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for the training entry script.

    Returns
    -------
    Namespace
        Updated namespace with configuration-backed parameters.
    """
    config = load_config(args.config)

    for arg_name, keys in _MAIN_CONFIG_KEYS.items():
        setattr(args, arg_name, _read_nested_key(config, keys, args.config))

    _validate_main_args(args)
    return args


def apply_main_eval_config(args: Namespace) -> Namespace:
    """Populate evaluation parameters from YAML.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for the evaluation entry script.

    Returns
    -------
    Namespace
        Updated namespace with configuration-backed parameters.
    """
    config = load_config(args.config)

    args.compile = _read_nested_key(config, ("runtime", "compile"), args.config)
    workers = _read_nested_key(config, ("training", "optimization", "num_workers"), args.config)
    args.num_workers = 10 if workers is None else workers
    args.kl_bins = _read_nested_key(config, ("evaluation", "metrics", "kl_bins"), args.config)
    args.pse_smooth = _read_nested_key(config, ("evaluation", "metrics", "pse_smooth"), args.config)

    if args.num_workers < 1:
        raise ValueError(
            f"Expected 'num_workers' to be at least 1, but got {args.num_workers}."
        )

    if not isinstance(args.compile, bool):
        raise TypeError(
            f"Expected 'runtime.compile' to be bool, but got type '{type(args.compile).__name__}'."
        )

    if not isinstance(args.kl_bins, int) or args.kl_bins < 0:
        raise ValueError(
            f"Expected 'evaluation.metrics.kl_bins' to be an int >= 0, but got {args.kl_bins}."
        )

    if not isinstance(args.pse_smooth, int) or args.pse_smooth < 0:
        raise ValueError(
            "Expected 'evaluation.metrics.pse_smooth' to be an int >= 0, "
            f"but got {args.pse_smooth}."
        )

    return args


def _read_nested_key(config: Mapping[str, Any], keys: tuple[str, ...], config_path: str) -> Any:
    """Read a nested key from a mapping and raise clear errors.

    Parameters
    ----------
    config : Mapping[str, Any]
        Configuration mapping.
    keys : tuple[str, ...]
        Ordered key path to read.
    config_path : str
        Human-readable path used for error messages.

    Returns
    -------
    Any
        The value at the requested key path, or None if the key is missing.

    Raises
    ------
    TypeError
        If an intermediate value is not a mapping.
    """
    node: Any = config
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(node, Mapping):
            prefix = ".".join(traversed[:-1])
            raise TypeError(
                f"Expected a mapping at '{prefix}' while reading '{'.'.join(keys)}' "
                f"in '{config_path}', but got type '{type(node).__name__}'."
            )
        if key not in node:
            return None
        node = node[key]
    return node


def _validate_main_args(args: Namespace) -> None:
    """Validate training arguments after configuration expansion.

    Parameters
    ----------
    args : Namespace
        Training argument namespace.

    Raises
    ------
    ValueError
        If one or more argument constraints are violated.
    TypeError
        If collection arguments have invalid types.
    """
    if args.obs_model not in {"identity", "linear"}:
        raise ValueError(
            f"Expected 'obs_model' to be one of ['identity', 'linear'], got '{args.obs_model}'."
        )

    for name in ("seq_len", "train_set_size", "num_epochs", "batch_size"):
        value = getattr(args, name)
        if value is None or value <= 0:
            raise ValueError(
                f"Expected '{name}' to be a positive integer, but got {value}."
            )

    if args.learning_rate is None or args.learning_rate <= 0:
        raise ValueError(
            f"Expected 'learning_rate' to be a positive float, but got {args.learning_rate}."
        )

    if args.individual_learning_rate is not None and args.individual_learning_rate <= 0:
        raise ValueError(
            "Expected 'individual_learning_rate' to be positive when provided, "
            f"but got {args.individual_learning_rate}."
        )

    if not isinstance(args.metrics, list):
        raise TypeError(
            f"Expected 'metrics' to be a list, but got type '{type(args.metrics).__name__}'."
        )

    if not isinstance(args.plots, list):
        raise TypeError(
            f"Expected 'plots' to be a list, but got type '{type(args.plots).__name__}'."
        )
