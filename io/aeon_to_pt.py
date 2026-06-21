from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aeon.datasets import load_classification


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for aeon-to-PT conversion.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert an aeon classification dataset to the tensor format required by this project."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="EpilepticSeizures",
        help="Aeon dataset name passed to aeon.datasets.load_classification.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "test"],
        default="all",
        help="Data split to load from aeon.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/aeon/epileptic_seizures.pt",
        help="Path to write the converted tensor (.pt).",
    )
    parser.add_argument(
        "--variable_length_policy",
        type=str,
        choices=["error", "truncate", "pad"],
        default="error",
        help="How to handle variable-length cases returned by aeon.",
    )
    parser.add_argument(
        "--pad_value",
        type=float,
        default=0.0,
        help="Padding value used when variable_length_policy='pad'.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Output tensor floating-point dtype.",
    )
    parser.add_argument(
        "--save_auxiliary",
        action="store_true",
        help="Save labels (.npy) and metadata (.json) next to the tensor file.",
    )
    return parser.parse_args()


def load_aeon_classification_dataset(
    dataset_name: str,
    split: str | None,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Load an aeon classification dataset with metadata.

    Parameters
    ----------
    dataset_name : str
        Name of the aeon dataset.
    split : str | None
        Optional split name ("train", "test", or ``None`` for all data).

    Returns
    -------
    tuple[Any, np.ndarray, dict[str, Any]]
        Raw aeon inputs ``X``, labels ``y``, and metadata dictionary.
    """
    try:
        X, y, metadata = load_classification(dataset_name, split=split, return_metadata=True)
    except TypeError:
        # Backward compatibility for aeon variants exposing the older keyword.
        X, y, metadata = load_classification(dataset_name, split=split, meta_data=True)
    return X, y, metadata


def convert_aeon_cases_to_tensor(
    X: Any,
    variable_length_policy: str,
    pad_value: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert aeon case representation to project tensor format.

    The project expects shape ``(num_subjects, timesteps, num_features)``.

    Parameters
    ----------
    X : Any
        Aeon case container.
    variable_length_policy : str
        One of ``{"error", "truncate", "pad"}``.
    pad_value : float
        Value for temporal padding when ``variable_length_policy='pad'``.
    dtype : torch.dtype
        Output tensor dtype.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(S, T, F)`` suitable for ``MultiSubjectDataset``.

    Raises
    ------
    TypeError
        If ``X`` has an unsupported type or malformed case entries.
    ValueError
        If dimensions or variable-length handling constraints are violated.
    """
    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            X = X[:, None, :]
        if X.ndim != 3:
            raise ValueError(
                f"Expected aeon ndarray with ndim 3 after normalization, got shape {X.shape}."
            )
        converted = np.transpose(X, (0, 2, 1))
        return torch.as_tensor(converted, dtype=dtype)

    if isinstance(X, list):
        case_arrays = [_normalize_case_array(case, idx) for idx, case in enumerate(X)]
        lengths = [arr.shape[1] for arr in case_arrays]
        unique_lengths = sorted(set(lengths))

        if len(unique_lengths) > 1:
            if variable_length_policy == "error":
                raise ValueError(
                    "Dataset contains variable-length time series with lengths "
                    f"{unique_lengths}. Choose --variable_length_policy truncate or pad."
                )
            if variable_length_policy == "truncate":
                target_length = min(unique_lengths)
                case_arrays = [arr[:, :target_length] for arr in case_arrays]
            elif variable_length_policy == "pad":
                target_length = max(unique_lengths)
                case_arrays = [
                    _pad_case_array(arr, target_length=target_length, pad_value=pad_value)
                    for arr in case_arrays
                ]
            else:
                raise ValueError(
                    f"Unknown variable_length_policy '{variable_length_policy}'."
                )

        stacked = np.stack(case_arrays, axis=0)
        converted = np.transpose(stacked, (0, 2, 1))
        return torch.as_tensor(converted, dtype=dtype)

    raise TypeError(
        "Unsupported aeon output type for X. Expected numpy.ndarray or list of arrays, "
        f"but got '{type(X).__name__}'."
    )


def _normalize_case_array(case: Any, index: int) -> np.ndarray:
    """Normalize one aeon case to ``(channels, timesteps)`` array format.

    Parameters
    ----------
    case : Any
        Single aeon case.
    index : int
        Case index for detailed error messages.

    Returns
    -------
    np.ndarray
        Case array with shape ``(channels, timesteps)``.

    Raises
    ------
    TypeError
        If case cannot be interpreted as a NumPy array.
    ValueError
        If case has invalid dimensionality.
    """
    if not isinstance(case, np.ndarray):
        raise TypeError(
            f"Expected case {index} to be a numpy.ndarray, got '{type(case).__name__}'."
        )
    if case.ndim == 1:
        return case[None, :]
    if case.ndim != 2:
        raise ValueError(
            f"Expected case {index} with ndim 1 or 2, but got shape {case.shape}."
        )
    return case


def _pad_case_array(arr: np.ndarray, target_length: int, pad_value: float) -> np.ndarray:
    """Pad a single case in time dimension to a target length.

    Parameters
    ----------
    arr : np.ndarray
        Case array with shape ``(channels, timesteps)``.
    target_length : int
        Desired number of timesteps.
    pad_value : float
        Constant value used for right-padding.

    Returns
    -------
    np.ndarray
        Padded array with shape ``(channels, target_length)``.
    """
    pad_len = target_length - arr.shape[1]
    if pad_len <= 0:
        return arr
    return np.pad(arr, pad_width=((0, 0), (0, pad_len)), mode="constant", constant_values=pad_value)


def main() -> None:
    """Convert an aeon dataset and store it as `.pt` for model training/evaluation."""
    args = parse_args()
    split = None if args.split == "all" else args.split
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    X, y, metadata = load_aeon_classification_dataset(args.dataset_name, split=split)
    tensor = convert_aeon_cases_to_tensor(
        X,
        variable_length_policy=args.variable_length_policy,
        pad_value=args.pad_value,
        dtype=dtype,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)

    print(f"Saved tensor to {output_path} with shape {tuple(tensor.shape)}")
    print("Expected format is (num_subjects, timesteps, num_features).")

    if args.save_auxiliary:
        labels_path = output_path.with_name(output_path.stem + "_labels.npy")
        metadata_path = output_path.with_name(output_path.stem + "_metadata.json")
        np.save(labels_path, y)
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        print(f"Saved labels to {labels_path}")
        print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
