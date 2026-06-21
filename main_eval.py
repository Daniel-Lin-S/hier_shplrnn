"""Evaluate trained hierarchical PLRNN checkpoints and export summary metrics.

Outputs
-------
This entry script writes one CSV file at
``{save_path}/results.csv``.

The CSV stores one column per ``(run_path, subject_index)`` pair and includes:
- ``dstsp``: state-space divergence value.
- ``pse``: power-spectrum error value.
"""

import argparse
import multiprocessing
import os
from typing import cast

import numpy as np
import pandas as pd
import torch

from bptt import load_from_path, read_hypers
from config_loader import apply_main_eval_config
from main import get_device, get_dataset
from model import shallowPLRNN
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
        required=True,
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
        default="./results/experiment",
        help="Path to save result CSV.",
    )

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument("--use_gpu", action="store_true", help="Use GPU when available.")

    return parser.parse_args()


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


def evaluate_model(args_and_path: tuple[argparse.Namespace, str]) -> tuple[str, np.ndarray, np.ndarray]:
    """Evaluate a single model run.

    Parameters
    ----------
    args_and_path : tuple[argparse.Namespace, str]
        Pair of global evaluation arguments and run directory path.

    Returns
    -------
    tuple[str, np.ndarray, np.ndarray]
        Run path, state-space divergence values, and PSE values.
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

    modelargs = get_device(modelargs)
    dataset = get_dataset(modelargs)
    model = shallowPLRNN(modelargs, dataset)
    load_from_path(model, worker_args)
    if modelargs.compile:
        model = cast(shallowPLRNN, torch.compile(model))
    model.eval()
    model.evaluator.compute_expensive(['dstsp', 'pse'])
    dstsp = np.atleast_1d(model.evaluator.get_state_space_divergence().cpu().squeeze().numpy())
    pse = np.atleast_1d(model.evaluator.get_pse().squeeze())
    return model_path, dstsp, pse


def main() -> None:
    """Run multi-process evaluation over discovered model runs."""
    args = parse_args()
    args = apply_main_eval_config(args)

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
    for (p, dstsp, pse) in results_list:
        for s, (kl, hel) in enumerate(zip(dstsp, pse)):
            results[(p, s)] = {'dstsp': kl, 'pse': hel}

    # save results
    df = pd.DataFrame(results)
    # generate folders if necessary
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    df.to_csv(os.path.join(args.save_path, 'results.csv'))
    print("Results saved to ", os.path.join(args.save_path, 'results.csv'), flush=True)

if __name__ == "__main__":
    main()
