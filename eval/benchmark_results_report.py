"""Collect latent benchmark results and generate faceted uncertainty plots."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY_TEST_METRICS: tuple[str, ...] = (
    "test_accuracy",
    "test_auroc",
    "test_aic",
    "test_bic",
    "test_mutual_information_mean",
    "test_mutual_information_max",
)

COLOR_BY_GROUP: dict[str, str] = {
    "model": "#1f77b4",
    "baseline": "#ff7f0e",
    "unknown": "#7f7f7f",
}

DETERMINISTIC_EXTRACTOR_TYPES: frozenset[str] = frozenset({
    "pca",
    "bandpower",
    "catch22",
})


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark reporting.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Collect model/baseline benchmark outputs and create faceted uncertainty reports.",
    )
    parser.add_argument(
        "--benchmark_root",
        type=str,
        required=True,
        help="Dataset benchmark root containing models and optionally baselines directories.",
    )
    parser.add_argument(
        "--summary_filename",
        type=str,
        default="latent_benchmark_summary.csv",
        help="Output summary CSV filename written under benchmark_root.",
    )
    parser.add_argument(
        "--plots_subdir",
        type=str,
        default="figures",
        help="Subdirectory under benchmark_root where figures are saved.",
    )
    return parser.parse_args()


def collect_benchmark_rows(benchmark_root: Path) -> pd.DataFrame:
    """Collect one-row classification metrics from model and baseline folders.

    Parameters
    ----------
    benchmark_root : Path
        Root directory containing benchmark outputs.

    Returns
    -------
    pd.DataFrame
        Combined summary dataframe.
    """
    rows: list[dict[str, Any]] = []
    rows.extend(_collect_group_rows(benchmark_root, group_folder="models", group_name="model"))
    rows.extend(_collect_group_rows(benchmark_root, group_folder="baselines", group_name="baseline"))

    if not rows:
        raise FileNotFoundError(
            "No classification_metrics.csv files were found under models/ or baselines/. "
            f"benchmark_root='{benchmark_root}'."
        )

    summary_df = pd.DataFrame(rows)
    if "model_name" not in summary_df.columns:
        raise ValueError("Collected metrics are missing required column 'model_name'.")

    summary_df["model_name"] = summary_df["model_name"].astype(str)
    summary_df["extractor_group"] = summary_df["extractor_group"].astype(str)
    if "extractor_type" in summary_df.columns:
        summary_df["extractor_type"] = summary_df["extractor_type"].astype(str)
        missing_mask = summary_df["extractor_type"].str.strip() == ""
        summary_df.loc[missing_mask, "extractor_type"] = summary_df.loc[missing_mask, "model_name"].map(
            _extractor_type_from_name
        )
    else:
        summary_df["extractor_type"] = summary_df["model_name"].map(_extractor_type_from_name)
    summary_df["base_model_name"] = summary_df["model_name"].map(_base_model_name)
    summary_df["repetition_id"] = summary_df["model_name"].map(_repetition_id_from_name)
    summary_df = summary_df.sort_values(["extractor_group", "base_model_name", "model_name"]).reset_index(drop=True)
    return summary_df


def _collect_group_rows(benchmark_root: Path, group_folder: str, group_name: str) -> list[dict[str, Any]]:
    """Collect metric rows for one extractor group."""
    group_path = benchmark_root / group_folder
    if not group_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(group_path.glob("*/classification_metrics.csv")):
        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.shape[0] != 1:
            raise ValueError(
                "Expected each classification_metrics.csv to contain exactly one row, "
                f"but got {metrics_df.shape[0]} rows in '{metrics_path}'."
            )

        row = dict(metrics_df.iloc[0].to_dict())
        if "model_name" not in row or str(row["model_name"]).strip() == "":
            row["model_name"] = metrics_path.parent.name

        row["extractor_group"] = group_name
        row["artifact_dir"] = str(metrics_path.parent.relative_to(benchmark_root))
        rows.append(row)

    return rows


def _base_model_name(model_name: str) -> str:
    """Remove repetition suffix from a model name."""
    token = str(model_name)
    match = re.match(r"^(?P<base>.+)__rep\d+_seed\d+$", token)
    if match is None:
        return token
    return str(match.group("base"))


def _repetition_id_from_name(model_name: str) -> int | None:
    """Extract repetition id from a model name suffix."""
    token = str(model_name)
    match = re.match(r"^.+__rep(?P<rep>\d+)_seed\d+$", token)
    if match is None:
        return None
    return int(match.group("rep"))


def _extractor_type_from_name(model_name: str) -> str:
    """Heuristically infer extractor type from model name."""
    token = _base_model_name(str(model_name)).lower()
    if "bandpower" in token:
        return "bandpower"
    if "catch22" in token:
        return "catch22"
    if "pca" in token:
        return "pca"
    if "hier_shplrnn_finetuned" in token:
        return "hier_shplrnn_finetuned"
    if "hier_shplrnn_checkpoint" in token or "hier_shplrnn_epoch" in token:
        return "hier_shplrnn_checkpoint"
    if "cbramod" in token:
        return "cbramod_pretrained"
    return "unknown"


def aggregate_repetition_statistics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated benchmark rows into mean and uncertainty columns.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Raw benchmark rows.

    Returns
    -------
    pd.DataFrame
        Aggregated rows with ``*_mean`` and ``*_std`` columns.
    """
    missing_cols = {"base_model_name", "extractor_group", "extractor_type"} - set(summary_df.columns)
    if missing_cols:
        raise ValueError(
            "Cannot aggregate benchmark rows because required columns are missing: "
            f"{sorted(missing_cols)}."
        )

    available_metrics = [metric for metric in KEY_TEST_METRICS if metric in summary_df.columns]
    if not available_metrics:
        raise ValueError(
            "None of the key test metrics are available in the summary dataframe. "
            f"Expected any of {KEY_TEST_METRICS}."
        )

    for metric in available_metrics:
        summary_df[metric] = pd.to_numeric(summary_df[metric], errors="coerce")

    grouped = summary_df.groupby(["base_model_name", "extractor_group", "extractor_type"], dropna=False)
    rows: list[dict[str, Any]] = []
    for (base_name, group, extractor_type), frame in grouped:
        row: dict[str, Any] = {
            "base_model_name": str(base_name),
            "extractor_group": str(group),
            "extractor_type": str(extractor_type),
            "num_repetitions": int(frame.shape[0]),
        }
        is_deterministic = str(extractor_type) in DETERMINISTIC_EXTRACTOR_TYPES

        for metric in available_metrics:
            values = frame[metric].to_numpy(dtype=np.float64)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                mean_value = float("nan")
                std_value = float("nan")
            else:
                mean_value = float(np.mean(finite_values))
                if is_deterministic:
                    std_value = 0.0
                elif finite_values.size == 1:
                    std_value = 0.0
                else:
                    std_value = float(np.std(finite_values, ddof=1))

            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value

        rows.append(row)

    aggregated = pd.DataFrame(rows)
    aggregated = aggregated.sort_values(["extractor_group", "base_model_name"]).reset_index(drop=True)
    return aggregated


def save_summary_csv(summary_df: pd.DataFrame, summary_path: Path) -> None:
    """Save benchmark summary CSV."""
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved collected benchmark summary to '{summary_path}'.", flush=True)


def plot_faceted_dot_error_bars(aggregated_df: pd.DataFrame, output_path: Path) -> None:
    """Plot faceted dot-and-error-bar panels for key metrics.

    Parameters
    ----------
    aggregated_df : pd.DataFrame
        Aggregated metric table.
    output_path : Path
        Output figure path.
    """
    metric_names = [metric for metric in KEY_TEST_METRICS if f"{metric}_mean" in aggregated_df.columns]
    if not metric_names:
        raise ValueError("No aggregated key metrics found for faceted plot generation.")

    y_labels = aggregated_df["base_model_name"].astype(str).tolist()
    y_positions = np.arange(len(y_labels), dtype=np.int64)

    num_cols = 2
    num_rows = int(math.ceil(len(metric_names) / float(num_cols)))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(8.5 * num_cols, max(4.8, 0.6 * len(y_labels)) * num_rows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for axis, metric in zip(axes_flat, metric_names):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        means = aggregated_df[mean_col].to_numpy(dtype=np.float64)
        stds = aggregated_df[std_col].to_numpy(dtype=np.float64)
        groups = aggregated_df["extractor_group"].astype(str).tolist()
        extractor_types = aggregated_df["extractor_type"].astype(str).tolist()

        zipped_values = zip(means, stds, groups, extractor_types)
        for index, (mean_value, std_value, group, extractor_type) in enumerate(zipped_values):
            color = COLOR_BY_GROUP.get(group, COLOR_BY_GROUP["unknown"])
            y_coord = y_positions[index]

            if not np.isfinite(mean_value):
                continue

            deterministic = extractor_type in DETERMINISTIC_EXTRACTOR_TYPES or np.isclose(std_value, 0.0)
            axis.scatter(mean_value, y_coord, color=color, s=28, zorder=3)
            if not deterministic and np.isfinite(std_value) and std_value > 0:
                axis.errorbar(
                    x=mean_value,
                    y=y_coord,
                    xerr=std_value,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.6,
                    capsize=2.8,
                    alpha=0.9,
                    zorder=2,
                )

        axis.set_yticks(y_positions)
        axis.set_yticklabels(y_labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25, linestyle="--")
        axis.set_xlabel(metric)
        axis.set_title(metric)

    for axis in axes_flat[len(metric_names):]:
        axis.axis("off")

    fig.suptitle("Latent Benchmark: Mean Score With Uncertainty", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved faceted metric plot to '{output_path}'.", flush=True)


def regenerate_benchmark_report(
    benchmark_root: Path,
    summary_filename: str = "latent_benchmark_summary.csv",
    plots_subdir: str = "figures",
) -> None:
    """Regenerate summary CSV and faceted figure for a benchmark root.

    Parameters
    ----------
    benchmark_root : Path
        Benchmark root containing model and baseline outputs.
    summary_filename : str, optional
        Output summary CSV name.
    plots_subdir : str, optional
        Output plot subdirectory.
    """
    if not benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark root '{benchmark_root}' does not exist.")

    raw_df = collect_benchmark_rows(benchmark_root)
    summary_path = benchmark_root / summary_filename
    save_summary_csv(raw_df, summary_path)

    aggregated_df = aggregate_repetition_statistics(raw_df)
    aggregated_path = benchmark_root / "latent_benchmark_aggregated_summary.csv"
    aggregated_df.to_csv(aggregated_path, index=False)
    print(f"Saved aggregated benchmark summary to '{aggregated_path}'.", flush=True)

    plots_root = benchmark_root / plots_subdir
    plot_faceted_dot_error_bars(aggregated_df, plots_root / "key_test_metrics_facet_dot_error.png")


def main() -> None:
    """Collect benchmark results, regenerate summaries, and save faceted figure."""
    args = parse_args()
    regenerate_benchmark_report(
        benchmark_root=Path(args.benchmark_root),
        summary_filename=args.summary_filename,
        plots_subdir=args.plots_subdir,
    )


if __name__ == "__main__":
    main()
