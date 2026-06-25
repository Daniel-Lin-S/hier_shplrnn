import yaml
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a structured YAML configuration, using default.yaml as base.

    Parameters
    ----------
    config_path : str
        Path to a user-provided YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Merged configuration dictionary.
    """
    default_path = Path(__file__).parent / "configs" / "default.yaml"
    if not default_path.exists():
        # Fallback if scripts are run from different directories
        default_path = Path("./configs/default.yaml")
    
    with default_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    user_path = Path(config_path).expanduser()
    if user_path.resolve() != default_path.resolve():
        if not user_path.exists():
            raise FileNotFoundError(f"Configuration file '{user_path}' does not exist.")
        
        with user_path.open("r", encoding="utf-8") as file:
            user_config = yaml.safe_load(file) or {}
        
        if not isinstance(user_config, dict):
            raise TypeError(f"User configuration '{user_path}' must be a mapping.")
        
        _check_unknown_keys(config, user_config)
        config = _deep_merge(config, user_config)

    for section in ("model", "training", "evaluation", "runtime"):
        if section not in config:
            raise KeyError(
                f"Missing required top-level section '{section}' in configuration."
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

    flattened = _flatten_dict(config)
    
    # Special mappings for legacy names used in codebase
    special_mappings = {
        "model_dimensions_obs_size": "obs_size",
        "model_dimensions_latent_size": "latent_size",
        "model_dimensions_hidden_size": "hidden_size",
        "model_dimensions_forcing_size": "forcing_size",
        "model_observation_obs_model": "obs_model",
        "model_observation_clipped": "clipped",
        "model_observation_learn_noise_cov": "learn_noise_cov",
        "model_hierarchy_scheme": "hierarchisation_scheme",
        "model_hierarchy_num_individual_params": "num_individual_params",
        "model_hierarchy_lambda_regularization": "lam",
        "training_sequence_seq_len": "seq_len",
        "training_sequence_train_set_size": "train_set_size",
        "training_optimization_num_epochs": "num_epochs",
        "training_optimization_batch_size": "batch_size",
        "training_optimization_batches_per_epoch": "batches_per_epoch",
        "training_optimization_subjects_per_batch": "subjects_per_batch",
        "training_optimization_num_workers": "num_workers",
        "training_optimization_learning_rate_shared": "learning_rate",
        "training_optimization_learning_rate_individual": "individual_learning_rate",
        "training_optimization_weight_decay": "weight_decay",
        "training_optimization_clip_grad_norm": "clip_grad_norm",
        "training_optimization_checkpoint_interval": "checkpoint_interval",
        "training_teacher_forcing_alpha_start": "tf_alpha_start",
        "training_teacher_forcing_alpha_end": "tf_alpha_end",
        "evaluation_metrics_enabled": "metrics",
        "evaluation_metrics_kl_bins": "kl_bins",
        "evaluation_metrics_pse_smooth": "pse_smooth",
        "evaluation_plots_enabled": "plots",
        "evaluation_intervals_cheap": "cheap_eval_interval",
        "evaluation_intervals_expensive": "expensive_eval_interval",
        "runtime_compile": "compile"
    }

    for key, value in flattened.items():
        # Get the internal name (either from mapping or by stripping prefixes)
        internal_name = special_mappings.get(key, key)
        setattr(args, internal_name, value)

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
    flattened = _flatten_dict(config)

    args.compile = flattened.get("runtime_compile")
    workers = flattened.get("optimization_num_workers")
    args.num_workers = 10 if workers is None else workers
    args.kl_bins = flattened.get("metrics_kl_bins")
    args.pse_smooth = flattened.get("metrics_pse_smooth")

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

def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _check_unknown_keys(base: Mapping[str, Any], overlay: Mapping[str, Any], path: str = "") -> None:
    """Warn about keys in overlay that are not present in base."""
    for key, value in overlay.items():
        current_path = f"{path}.{key}" if path else key
        if key not in base:
            warnings.warn(f"Unknown configuration key: '{current_path}' provided in user config is not in default.yaml.")
        elif isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _check_unknown_keys(base[key], value, current_path)

def _flatten_dict(d: dict[str, Any], parent_key: str = '', sep: str = '_') -> dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
