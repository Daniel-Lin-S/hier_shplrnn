"""Train hierarchical PLRNN models from tensor time series data.

Outputs
-------
This entry script writes all artifacts under:
``{save_path}/{experiment}/{name}/{run:03d}``.

If ``--num_repetitions > 1``, the ``run`` index corresponds to the
repetition ID (1, 2, ...), and the ``--run`` argument is ignored.
Otherwise, the provided ``--run`` argument is used (default 001).

Created files include:
- ``hypers.txt``: serialized merged runtime/config arguments used for the run.
- ``events.out.tfevents.*``: TensorBoard scalar/text/figure logs.
- ``model_<epoch>.pt``: checkpoint files saved periodically during training.
"""

import argparse
import random

import numpy as np
import torch

from trainers.bptt import BPTT
from config_loader import apply_main_config
from data_io.dataset import MultiSubjectDataset
from data_io.pt_tensor import DEFAULT_SUBSAMPLE_SEED

torch.set_num_threads(1)
torch.set_float32_matmul_precision('high')

# torch.autograd.set_detect_anomaly(True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training script.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Train a hierarchical shPLRNN model.")

    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="Path to the hierarchical YAML configuration file.",
    )

    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data_path",
        type=str,
        default="./data/lorenz63/3params64sub/noisy.pt",
        help="Path to training data (.pt).",
    )
    data_group.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="Path to optional long evaluation data (.pt).",
    )
    data_group.add_argument(
        "--subsample_size",
        type=int,
        default=None,
        help="Maximum number of subjects to subsample from the dataset.",
    )
    data_group.add_argument(
        "--subsample_seed",
        type=int,
        default=DEFAULT_SUBSAMPLE_SEED,
        help="Random seed for subject subsampling.",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--save_path",
        type=str,
        default="./trained_models",
        help="Path to save model checkpoints and logs.",
    )
    output_group.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        help="Experiment folder name.",
    )
    output_group.add_argument("--name", type=str, default="name", help="Model name within an experiment.")
    output_group.add_argument("--run", type=int, default=1, help="Run index.")

    finetune_group = parser.add_argument_group("Finetune")
    finetune_group.add_argument("--finetune", action="store_true", help="Finetune a pretrained model.")
    finetune_group.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a pretrained model directory.",
    )
    finetune_group.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint id to load. Defaults to latest.",
    )

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--use_gpu", action="store_true", help="Enable GPU if available.")
    runtime_group.add_argument("--device_id", type=int, default=0, help="CUDA device id.")
    runtime_group.add_argument(
        "--num_repetitions",
        type=int,
        default=1,
        help="Number of repeated training runs with different random seeds.",
    )
    runtime_group.add_argument(
        "--seed_start",
        type=int,
        default=0,
        help="Starting seed used for repetition 1.",
    )

    return parser.parse_args()


def _set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible repeated runs.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_single_run(args: argparse.Namespace) -> None:
    """Run one training/finetuning execution with current arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Prepared runtime arguments.
    """
    dataset = get_dataset(args)
    training_algorithm = BPTT(args, dataset)
    run_training = training_algorithm.train if not args.finetune else training_algorithm.finetune

    if args.compile:
        run_training = torch.compile(run_training)
    run_training()


def get_dataset(args: argparse.Namespace) -> MultiSubjectDataset:
    """Build the multi-subject dataset for training.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and expanded argument namespace.

    Returns
    -------
    MultiSubjectDataset
        Training dataset object.
    """
    return MultiSubjectDataset(
        args.data_path,
        args.seq_len,
        args.train_set_size,
        args.subjects_per_batch,
        args.num_workers,
        args.device,
        getattr(args, "subsample_size", None),
        getattr(args, "subsample_seed", DEFAULT_SUBSAMPLE_SEED),
    )


def get_device(args: argparse.Namespace) -> argparse.Namespace:
    """Select the execution device based on runtime flags.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed argument namespace.

    Returns
    -------
    argparse.Namespace
        Namespace with a populated ``device`` attribute.
    """
    args.device = "cpu"
    if args.use_gpu:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        args.device = f"{args.device}:{args.device_id}"
    print(f"Using device: {args.device}", flush=True)
    return args


def handle_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply dependent defaults after config and CLI merging.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and expanded argument namespace.

    Returns
    -------
    argparse.Namespace
        Namespace with normalized learning-rate and teacher-forcing values.

    Raises
    ------
    ValueError
        If required optimization or teacher-forcing values are missing.
    """
    if args.learning_rate is None:
        raise ValueError("Missing learning_rate after parsing config and CLI arguments.")
    if args.tf_alpha_start is None:
        raise ValueError("Missing tf_alpha_start after parsing config and CLI arguments.")

    if args.individual_learning_rate is not None:
        args.learning_rate = (args.learning_rate, args.individual_learning_rate)
    else:
        args.learning_rate = (args.learning_rate, args.learning_rate)

    if args.tf_alpha_end is None:
        args.tf_alpha_end = args.tf_alpha_start
    return args


def main() -> None:
    """Run model training or finetuning."""
    args = parse_args()

    # Capture CLI overrides before they are overwritten by config defaults
    cli_subsample_size = args.subsample_size
    cli_subsample_seed = args.subsample_seed

    args = apply_main_config(args)

    # Re-apply CLI overrides if they were provided
    if cli_subsample_size is not None:
        args.subsample_size = cli_subsample_size
    if cli_subsample_seed != DEFAULT_SUBSAMPLE_SEED:
        args.subsample_seed = cli_subsample_seed

    args = get_device(args)
    args = handle_defaults(args)

    if args.num_repetitions <= 0:
        raise ValueError(f"Expected --num_repetitions to be positive, but got {args.num_repetitions}.")

    base_run = int(args.run)
    for repetition_id in range(1, int(args.num_repetitions) + 1):
        repetition_seed = int(args.seed_start + repetition_id - 1)
        repetition_run_id = repetition_id if int(args.num_repetitions) > 1 else base_run

        run_args = argparse.Namespace(**vars(args))
        run_args.seed = repetition_seed
        run_args.run = repetition_run_id

        _set_global_seed(repetition_seed)
        print(
            "Starting training repetition "
            f"{repetition_id}/{args.num_repetitions} with seed={repetition_seed}, run={repetition_run_id}.",
            flush=True,
        )
        _train_single_run(run_args)

if __name__ == '__main__':
    main()