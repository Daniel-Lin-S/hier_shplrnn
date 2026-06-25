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


_PATH_SUFFIX_ALIASES: dict[tuple[str, ...], str] = {
    ("learning_rate", "shared"): "learning_rate",
    ("learning_rate", "individual"): "individual_learning_rate",
    ("teacher_forcing", "alpha_start"): "tf_alpha_start",
    ("teacher_forcing", "alpha_end"): "tf_alpha_end",
    ("metrics", "enabled"): "metrics",
    ("plots", "enabled"): "plots",
    ("evaluation", "intervals", "cheap"): "cheap_eval_interval",
    ("evaluation", "intervals", "expensive"): "expensive_eval_interval",
    ("hierarchy", "scheme"): "hierarchisation_scheme",
    ("hierarchy", "lambda_regularization"): "lam",
}


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
    _apply_config_to_namespace(args, config)

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
    _apply_config_to_namespace(args, config)
    
    # Handle overrides/defaults specific to evaluation
    if getattr(args, "num_workers", None) is None:
        args.num_workers = 10

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


def _apply_config_to_namespace(args: Namespace, config: Mapping[str, Any]) -> None:
    """Recursively copy configuration leaves into the argument namespace.

    The primary target name is the leaf key itself. A small set of path-suffix
    aliases preserves legacy argument names for existing training/eval code.

    Parameters
    ----------
    args : Namespace
        Argument namespace to update.
    config : Mapping[str, Any]
        Nested configuration tree.
    """

    def visit(node: Mapping[str, Any], path: tuple[str, ...]) -> None:
        for key, value in node.items():
            new_path = (*path, key)
            if isinstance(value, Mapping):
                visit(value, new_path)
                continue

            attr_name = _resolve_attr_name(new_path)
            setattr(args, attr_name, value)

    visit(config, ())


def _resolve_attr_name(path: tuple[str, ...]) -> str:
    """Resolve argument name from configuration path.

    Parameters
    ----------
    path : tuple[str, ...]
        Path segments from config root to a leaf value.

    Returns
    -------
    str
        Namespace attribute name.
    """
    for suffix, alias in _PATH_SUFFIX_ALIASES.items():
        if len(path) >= len(suffix) and path[-len(suffix):] == suffix:
            return alias
    return path[-1]
