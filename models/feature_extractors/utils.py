"""Shared helper utilities for latent feature extractors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

BAND_LIMITS_HZ: tuple[tuple[str, tuple[float, float]], ...] = (
    ("delta", (0.5, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 13.0)),
    ("beta", (13.0, 30.0)),
    ("gamma", (30.0, 45.0)),
)


def cast_tensor(value: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a float64 NumPy array.

    Parameters
    ----------
    value : torch.Tensor
        Input tensor.

    Returns
    -------
    np.ndarray
        Converted float64 array.
    """
    return value.detach().cpu().numpy().astype(np.float64)


def resolve_checkpoint_path(model_path: str) -> str:
    """Resolve a model checkpoint path from run folder or direct file.

    Parameters
    ----------
    model_path : str
        Path to a run directory or checkpoint file.

    Returns
    -------
    str
        Resolved checkpoint path.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

    if path.is_file() and path.suffix == ".pt":
        return str(path)

    if not path.is_dir():
        raise ValueError(
            f"Model path '{model_path}' must be a directory with checkpoints or a .pt file."
        )

    candidates: list[tuple[int, Path]] = []
    for candidate in path.iterdir():
        if candidate.is_file() and candidate.suffix == ".pt" and candidate.stem.startswith("model_"):
            try:
                epoch = int(candidate.stem.split("_")[-1])
            except ValueError:
                continue
            candidates.append((epoch, candidate))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find checkpoint files named 'model_<epoch>.pt' in '{model_path}'."
        )

    _, latest = max(candidates, key=lambda pair: pair[0])
    return str(latest)


def clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove wrapper prefixes from checkpoint keys.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        Raw checkpoint state dictionary.

    Returns
    -------
    dict[str, torch.Tensor]
        Normalized state dictionary.
    """
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = key.replace("_orig_mod.", "").replace("module.", "")
        cleaned[normalized] = value
    return cleaned


def flatten_signals(signals: np.ndarray) -> np.ndarray:
    """Flatten channel-last tensors to 2D matrices.

    Parameters
    ----------
    signals : np.ndarray
        Input tensor in ``(samples, timesteps, channels)`` format.

    Returns
    -------
    np.ndarray
        Flattened matrix in ``(samples, features)`` format.
    """
    flattened = signals.reshape(signals.shape[0], -1)
    validate_feature_matrix(flattened, "flattened_signals")
    return flattened


def prepare_cbramod_input(
    signals: np.ndarray,
    input_rate_hz: int,
    target_rate_hz: int,
    patch_size: int,
) -> torch.Tensor:
    """Resample and patchify EEG windows for CBraMod.

    Parameters
    ----------
    signals : np.ndarray
        Input EEG tensor in ``(samples, timesteps, channels)`` format.
    input_rate_hz : int
        Sampling rate of input windows.
    target_rate_hz : int
        Target sampling rate for CBraMod.
    patch_size : int
        Points per patch.

    Returns
    -------
    torch.Tensor
        Patchified tensor in ``(samples, channels, segments, patch_size)`` format.
    """
    if input_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError(
            f"Sampling rates must be positive, but got input={input_rate_hz}, target={target_rate_hz}."
        )
    if patch_size <= 0:
        raise ValueError(f"Patch size must be positive, but got {patch_size}.")

    x = torch.as_tensor(signals, dtype=torch.float32)
    if x.ndim != 3:
        raise ValueError(
            "CBraMod preprocessing expects channel-last signals of shape "
            f"(samples, timesteps, channels), but got shape {tuple(x.shape)}."
        )

    x = x.permute(0, 2, 1).contiguous()
    num_samples, num_channels, num_timesteps = x.shape
    target_timesteps = int(round(num_timesteps * float(target_rate_hz) / float(input_rate_hz)))
    if target_timesteps <= 0:
        raise ValueError(
            "Resampling produced a non-positive target length: "
            f"source_timesteps={num_timesteps}, input_rate={input_rate_hz}, "
            f"target_rate={target_rate_hz}, target_timesteps={target_timesteps}."
        )

    if target_timesteps != num_timesteps:
        x = x.view(num_samples * num_channels, 1, num_timesteps)
        x = F.interpolate(x, size=target_timesteps, mode="linear", align_corners=False)
        x = x.view(num_samples, num_channels, target_timesteps)

    segments = int(np.ceil(target_timesteps / float(patch_size)))
    padded_timesteps = segments * patch_size
    pad_right = padded_timesteps - target_timesteps
    if pad_right > 0:
        x = F.pad(x, (0, pad_right), mode="constant", value=0.0)

    return x.view(num_samples, num_channels, segments, patch_size)


def validate_feature_matrix(features: np.ndarray, name: str) -> None:
    """Validate extracted feature matrices.

    Parameters
    ----------
    features : np.ndarray
        Candidate feature matrix.
    name : str
        Extractor identifier for error messages.
    """
    if features.ndim != 2:
        raise ValueError(
            f"Feature extractor '{name}' must return a 2D matrix, but got shape {features.shape}."
        )
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(
            f"Feature extractor '{name}' returned an empty matrix with shape {features.shape}."
        )
    if not np.all(np.isfinite(features)):
        raise ValueError(f"Feature extractor '{name}' returned non-finite values.")


def resolve_individual_learning_rate(learning_rate: Any) -> float:
    """Resolve individual-parameter learning rate from mixed config formats.

    Parameters
    ----------
    learning_rate : Any
        Learning-rate value from hyperparameter namespace.

    Returns
    -------
    float
        Individual learning rate.
    """
    if isinstance(learning_rate, tuple) and len(learning_rate) == 2:
        return float(learning_rate[1])
    if isinstance(learning_rate, list) and len(learning_rate) == 2:
        return float(learning_rate[1])
    if isinstance(learning_rate, (int, float)):
        return float(learning_rate)
    raise TypeError(
        "Unable to resolve individual learning rate from value "
        f"{learning_rate} (type={type(learning_rate).__name__})."
    )
