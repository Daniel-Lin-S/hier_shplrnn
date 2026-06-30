"""Evaluate hierarchical checkpoints in legacy mode.

Outputs
-------
This entry script writes metrics and feature artifacts under ``{save_path}``.

Output files include:
- ``results.csv``: one column per ``(run_path, subject_index)`` pair and includes:
- ``dstsp``: state-space divergence value.
- ``pse``: power-spectrum error value.
- ``subject_features.csv``: extracted per-subject feature vectors.

Benchmark mode has moved to ``main_eval_feature_extractor.py``.
"""

import argparse
import torch

from data_io.pt_tensor import DEFAULT_SUBSAMPLE_SEED
from eval.legacy_eval_runtime import run_legacy_evaluation

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
        "--subsample_size",
        type=int,
        default=None,
        help="Maximum number of subjects to subsample from the dataset.",
    )
    io_group.add_argument(
        "--subsample_seed",
        type=int,
        default=DEFAULT_SUBSAMPLE_SEED,
        help="Random seed for subject subsampling.",
    )
    io_group.add_argument(
        "--save_path",
        type=str,
        default=None,
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
    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--use_gpu", action="store_true", help="Use GPU when available.")

    return parser.parse_args()


def main() -> None:
    """Entry point for legacy checkpoint evaluation."""
    args = parse_args()
    run_legacy_evaluation(args)

if __name__ == "__main__":
    main()
