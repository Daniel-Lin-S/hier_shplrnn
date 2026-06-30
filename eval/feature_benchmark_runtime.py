"""Feature-benchmark runtime orchestration with repetition support."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from eval.benchmark_results_report import regenerate_benchmark_report
from eval.latent_benchmark import BenchmarkConfig, benchmark_config_from_files, run_latent_benchmark

DETERMINISTIC_EXTRACTOR_TYPES: frozenset[str] = frozenset({
    "pca",
    "bandpower",
    "catch22",
})


def parse_feature_benchmark_args() -> argparse.Namespace:
    """Parse CLI arguments for repeated feature benchmark evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run latent feature extractor benchmark with optional repeated random seeds.",
    )
    parser.add_argument(
        "--benchmark_config",
        type=str,
        required=True,
        help="Path to shared benchmark YAML (datasets, CV, baseline specs).",
    )
    parser.add_argument(
        "--benchmark_model_config",
        type=str,
        required=True,
        help="Path to benchmark model YAML (single `model` or list `models`).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional benchmark output root override.",
    )
    parser.add_argument(
        "--num_seed_repetitions",
        type=int,
        default=10,
        help="Number of random-seed repetitions for stochastic extractors.",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=0,
        help="First repetition seed value.",
    )
    parser.add_argument(
        "--report_summary_filename",
        type=str,
        default="latent_benchmark_summary.csv",
        help="Summary CSV filename used by auto-regenerated benchmark report.",
    )
    parser.add_argument(
        "--report_plots_subdir",
        type=str,
        default="figures",
        help="Subdirectory used by auto-regenerated benchmark report.",
    )
    parser.add_argument(
        "--subsample_size",
        type=int,
        default=None,
        help="Optional override for evaluation-dataset subsample size.",
    )
    parser.add_argument(
        "--subsample_seed",
        type=int,
        default=None,
        help="Optional override for evaluation-dataset subsample seed.",
    )
    return parser.parse_args()


def run_repeated_feature_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    """Run latent benchmark with deterministic/stochastic repetition rules.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments from :func:`parse_feature_benchmark_args`.

    Returns
    -------
    pd.DataFrame
        Concatenated summary rows across all repetitions.
    """
    if args.num_seed_repetitions <= 0:
        raise ValueError(
            "Expected --num_seed_repetitions to be positive, "
            f"but got {args.num_seed_repetitions}."
        )

    base_config = benchmark_config_from_files(
        args.benchmark_config,
        args.benchmark_model_config,
        save_path_override=args.save_path,
        subsample_size_override=args.subsample_size,
        subsample_seed_override=args.subsample_seed,
    )

    print(
        "Running feature benchmark with repetition control: "
        f"dataset_group='{base_config.dataset_group}', num_seed_repetitions={args.num_seed_repetitions}, "
        f"seed_start={args.seed_start}.",
        flush=True,
    )

    all_runs: list[pd.DataFrame] = []
    seed_values = [int(args.seed_start + offset) for offset in range(args.num_seed_repetitions)]
    for repetition_id, seed in enumerate(seed_values, start=1):
        run_config = _build_config_for_repetition(base_config, repetition_id=repetition_id, seed=seed)
        run_config = _filter_repetition_specs(run_config, repetition_id=repetition_id)

        if len(run_config.model_specs) == 0 and len(run_config.baseline_specs) == 0:
            print(
                "Skipping repetition because only deterministic extractors were configured "
                f"and repetition_id={repetition_id} > 1.",
                flush=True,
            )
            continue

        _set_global_seed(seed)

        repetition_df = run_latent_benchmark(run_config)
        repetition_df = repetition_df.copy()
        repetition_df["repetition_id"] = int(repetition_id)
        repetition_df["seed"] = int(seed)

        if not repetition_df.empty:
            all_runs.append(repetition_df)

        _regenerate_group_report(
            run_config=run_config,
            summary_filename=args.report_summary_filename,
            plots_subdir=args.report_plots_subdir,
        )

    if not all_runs:
        raise ValueError("No benchmark rows were produced after applying repetition rules.")

    combined_df = pd.concat(all_runs, axis=0, ignore_index=True)
    combined_path = _group_root(base_config) / "latent_benchmark_repetition_rows.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved repetition rows to {combined_path}", flush=True)

    _regenerate_group_report(
        run_config=base_config,
        summary_filename=args.report_summary_filename,
        plots_subdir=args.report_plots_subdir,
    )
    return combined_df


def _build_config_for_repetition(base: BenchmarkConfig, repetition_id: int, seed: int) -> BenchmarkConfig:
    """Create per-repetition benchmark config.

    Parameters
    ----------
    base : BenchmarkConfig
        Base benchmark config.
    repetition_id : int
        One-based repetition index.
    seed : int
        Random seed for the repetition.

    Returns
    -------
    BenchmarkConfig
        Updated config with repetition-specific random state.
    """
    return BenchmarkConfig(
        dataset_group=base.dataset_group,
        save_path=base.save_path,
        evaluation_datasets=base.evaluation_datasets,
        nested_cv_folds=base.nested_cv_folds,
        ridge_alphas=base.ridge_alphas,
        random_state=seed,
        cache_baselines_once_per_dataset_group=base.cache_baselines_once_per_dataset_group,
        evaluate_baselines=base.evaluate_baselines,
        model_specs=_append_repetition_suffix(base.model_specs, repetition_id, seed),
        baseline_specs=_append_repetition_suffix(base.baseline_specs, repetition_id, seed),
        create_pca_plot=base.create_pca_plot,
        pca_plot_prefix=base.pca_plot_prefix,
        pca_plot_show_arrows=base.pca_plot_show_arrows,
        subsample_size=base.subsample_size,
        subsample_seed=base.subsample_seed,
    )


def _filter_repetition_specs(config: BenchmarkConfig, repetition_id: int) -> BenchmarkConfig:
    """Filter deterministic extractor specs for repeated runs.

    Parameters
    ----------
    config : BenchmarkConfig
        Repetition config before filtering.
    repetition_id : int
        One-based repetition id.

    Returns
    -------
    BenchmarkConfig
        Config with deterministic extractors removed for repetition > 1.
    """
    if repetition_id <= 1:
        return config

    model_specs = [
        spec for spec in config.model_specs if spec.extractor_type not in DETERMINISTIC_EXTRACTOR_TYPES
    ]
    baseline_specs = [
        spec for spec in config.baseline_specs if spec.extractor_type not in DETERMINISTIC_EXTRACTOR_TYPES
    ]
    return BenchmarkConfig(
        dataset_group=config.dataset_group,
        save_path=config.save_path,
        evaluation_datasets=config.evaluation_datasets,
        nested_cv_folds=config.nested_cv_folds,
        ridge_alphas=config.ridge_alphas,
        random_state=config.random_state,
        cache_baselines_once_per_dataset_group=config.cache_baselines_once_per_dataset_group,
        evaluate_baselines=config.evaluate_baselines,
        model_specs=model_specs,
        baseline_specs=baseline_specs,
        create_pca_plot=config.create_pca_plot,
        pca_plot_prefix=config.pca_plot_prefix,
        pca_plot_show_arrows=config.pca_plot_show_arrows,
        subsample_size=config.subsample_size,
        subsample_seed=config.subsample_seed,
    )


def _append_repetition_suffix(specs: Sequence, repetition_id: int, seed: int) -> list:
    """Clone extractor specs with stable repetition suffix in names.

    Parameters
    ----------
    specs : Sequence
        Extractor specs.
    repetition_id : int
        One-based repetition id.
    seed : int
        Random seed.

    Returns
    -------
    list
        Updated specs with unique names.
    """
    updated = []
    for spec in specs:
        updated.append(
            type(spec)(
                name=f"{spec.name}__rep{repetition_id:02d}_seed{seed:04d}",
                extractor_type=spec.extractor_type,
                params=dict(spec.params),
            )
        )
    return updated


def _set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for deterministic repetition runs.

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


def _group_root(config: BenchmarkConfig) -> Path:
    """Resolve benchmark group root path."""
    return Path(config.save_path) / _safe_token(config.dataset_group)


def _safe_token(value: str) -> str:
    """Convert arbitrary string to filesystem-safe token."""
    token = value.strip().replace(" ", "_")
    token = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in token)
    return token if token else "token"


def _regenerate_group_report(
    run_config: BenchmarkConfig,
    summary_filename: str,
    plots_subdir: str,
) -> None:
    """Regenerate benchmark report after each benchmark update."""
    benchmark_root = _group_root(run_config)
    regenerate_benchmark_report(
        benchmark_root=benchmark_root,
        summary_filename=summary_filename,
        plots_subdir=plots_subdir,
    )
