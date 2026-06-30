"""Shared ``.pt`` tensor loading utilities with optional deterministic subsampling."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch

DEFAULT_SUBSAMPLE_SEED = 1337


def load_subject_timeseries_pt(
    path: str,
    subsample_size: int | None = None,
    subsample_seed: int = DEFAULT_SUBSAMPLE_SEED,
) -> torch.Tensor:
    """Load training/evaluation tensor as ``(subjects, timesteps, features)``.

    Parameters
    ----------
    path : str
        Path to ``.pt`` file containing a tensor.
    subsample_size : int | None, optional
        Number of rows to sample from axis 0. If ``None``, all rows are used.
    subsample_seed : int, optional
        Random seed for deterministic axis-0 subsampling.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(subjects, timesteps, features)``.
    """
    tensor = _load_pt_tensor(path=path, context="subject_timeseries")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    _ensure_three_dimensional(tensor=tensor, path=path, context="subject_timeseries")
    return _maybe_subsample_axis0(
        tensor=tensor,
        subsample_size=subsample_size,
        subsample_seed=subsample_seed,
        path=path,
        context="subject_timeseries",
    )


def load_signal_samples_pt(
    path: str,
    subsample_size: int | None = None,
    subsample_seed: int = DEFAULT_SUBSAMPLE_SEED,
) -> torch.Tensor:
    """Load benchmark tensor as ``(samples, timesteps, channels)``.

    Parameters
    ----------
    path : str
        Path to ``.pt`` file containing a tensor.
    subsample_size : int | None, optional
        Number of rows to sample from axis 0. If ``None``, all rows are used.
    subsample_seed : int, optional
        Random seed for deterministic axis-0 subsampling.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(samples, timesteps, channels)``.
    """
    tensor = _load_pt_tensor(path=path, context="signal_samples")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    _ensure_three_dimensional(tensor=tensor, path=path, context="signal_samples")
    return _maybe_subsample_axis0(
        tensor=tensor,
        subsample_size=subsample_size,
        subsample_seed=subsample_seed,
        path=path,
        context="signal_samples",
    )


def _load_pt_tensor(path: str, context: str) -> torch.Tensor:
    """Load and validate a tensor from a ``.pt`` file."""
    path_obj = Path(path)
    if path_obj.suffix.lower() != ".pt":
        warnings.warn(
            f"Expected a '.pt' file for {context}, but got '{path}'. Attempting to load anyway.",
            stacklevel=2,
        )
    if not path_obj.exists():
        raise FileNotFoundError(f"{context}: data file '{path}' does not exist.")

    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{context}: expected torch.Tensor in '{path}', but got type '{type(tensor).__name__}'."
        )
    if tensor.ndim < 2:
        raise ValueError(
            f"{context}: expected at least 2 dimensions, but got shape {tuple(tensor.shape)} from '{path}'."
        )
    if tensor.shape[0] == 0:
        raise ValueError(f"{context}: axis-0 is empty in '{path}', cannot build dataset from zero rows.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{context}: tensor in '{path}' contains non-finite values (NaN or inf).")
    return tensor


def _ensure_three_dimensional(tensor: torch.Tensor, path: str, context: str) -> None:
    """Raise if tensor is not exactly three-dimensional after expansion."""
    if tensor.ndim != 3:
        raise ValueError(
            f"{context}: expected 2D or 3D tensor from '{path}', resolved shape is {tuple(tensor.shape)}."
        )


def _maybe_subsample_axis0(
    tensor: torch.Tensor,
    subsample_size: int | None,
    subsample_seed: int,
    path: str,
    context: str,
) -> torch.Tensor:
    """Optionally subsample axis-0 with deterministic randomness."""
    total_rows = int(tensor.shape[0])
    if subsample_size is None:
        return tensor
    if not isinstance(subsample_size, int):
        raise TypeError(
            f"{context}: expected subsample_size to be int or null, but got {type(subsample_size).__name__}."
        )
    if subsample_size <= 0:
        raise ValueError(
            f"{context}: expected subsample_size > 0, but got {subsample_size} for '{path}'."
        )
    if subsample_size > total_rows:
        raise ValueError(
            f"{context}: subsample_size={subsample_size} exceeds available rows ({total_rows}) in '{path}'."
        )
    if not isinstance(subsample_seed, int):
        raise TypeError(
            f"{context}: expected subsample_seed to be int, but got {type(subsample_seed).__name__}."
        )
    if subsample_size == total_rows:
        warnings.warn(
            f"{context}: subsample_size equals total rows ({total_rows}) for '{path}'; subsampling is a no-op.",
            stacklevel=2,
        )
        return tensor

    generator = torch.Generator(device="cpu")
    generator.manual_seed(subsample_seed)
    selected = torch.randperm(total_rows, generator=generator)[:subsample_size]
    return tensor.index_select(0, selected)