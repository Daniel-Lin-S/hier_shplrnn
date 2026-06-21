"""Collect latent benchmark results and generate visual comparisons."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

KEY_TEST_METRICS: tuple[str, ...] = (
    "test_accuracy",
    "test_auroc",
    "test_aic",
    "test_bic",
    "test_mutual_information_mean",
    "test_mutual_information_max",
)

LOWER_IS_BETTER: set[str] = {"test_aic", "test_bic"}
COLOR_BY_GROUP: dict[str, str] = {
    "model": "#1f77b4",
    "baseline": "#ff7f0e",
    "unknown": "#7f7f7f",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark reporting.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Collect model/baseline benchmark outputs and create comparison reports.",
    )
    parser.add_argument(
        "--benchmark_root",
        type=str,
        required=True,
        help=(
            "Dataset benchmark root containing 'models' and optionally 'baselines' directories. "
        ),
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
    summary_df = summary_df.sort_values(["extractor_group", "model_name"]).reset_index(drop=True)
    return summary_df


def _collect_group_rows(benchmark_root: Path, group_folder: str, group_name: str) -> list[dict[str, Any]]:
    """Collect metric rows for one extractor group.

    Parameters
    ----------
    benchmark_root : Path
        Root benchmark directory.
    group_folder : str
        Folder name under benchmark_root.
    group_name : str
        Group label stored in the output dataframe.

    Returns
    -------
    list[dict[str, Any]]
        Extracted metric rows.
    """
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


def save_summary_csv(summary_df: pd.DataFrame, summary_path: Path) -> None:
    """Save benchmark summary CSV.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Benchmark summary dataframe.
    summary_path : Path
        Output CSV path.
    """
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved collected benchmark summary to '{summary_path}'.", flush=True)


def plot_key_metric_bars(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot key test metrics as ranked horizontal bar charts.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Benchmark summary dataframe.
    output_path : Path
        Output figure path.
    """
    available_metrics = [metric for metric in KEY_TEST_METRICS if metric in summary_df.columns]
    if not available_metrics:
        raise ValueError(
            "None of the key test metrics are available in the summary dataframe. "
            f"Expected any of {KEY_TEST_METRICS}."
        )

    num_cols = 2
    num_rows = int(math.ceil(len(available_metrics) / float(num_cols)))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(8.0 * num_cols, 5.0 * num_rows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for axis, metric in zip(axes_flat, available_metrics):
        _plot_metric_axis(axis, summary_df, metric)

    for axis in axes_flat[len(available_metrics):]:
        axis.axis("off")

    legend_handles = [
        Patch(color=COLOR_BY_GROUP["model"], label="model"),
        Patch(color=COLOR_BY_GROUP["baseline"], label="baseline"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2)
    fig.suptitle("Latent Benchmark Key Test Metrics", fontsize=16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved key metric bar chart to '{output_path}'.", flush=True)


def _plot_metric_axis(axis: plt.Axes, summary_df: pd.DataFrame, metric: str) -> None:
    """Render one metric ranking panel.

    Parameters
    ----------
    axis : plt.Axes
        Matplotlib axis.
    summary_df : pd.DataFrame
        Benchmark summary dataframe.
    metric : str
        Metric name to plot.
    """
    plot_df = summary_df[["model_name", "extractor_group", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])

    if plot_df.empty:
        axis.text(0.5, 0.5, f"No values for {metric}", ha="center", va="center")
        axis.set_axis_off()
        return

    ascending = metric in LOWER_IS_BETTER
    ranked = plot_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    colors = ranked["extractor_group"].map(COLOR_BY_GROUP).fillna(COLOR_BY_GROUP["unknown"]).tolist()

    axis.barh(ranked["model_name"], ranked[metric], color=colors)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.3, linestyle="--")
    axis.set_xlabel(metric)

    direction = "lower is better" if ascending else "higher is better"
    axis.set_title(f"{metric} ({direction})")


def plot_metric_score_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Create a normalized score heatmap across test metrics.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Benchmark summary dataframe.
    output_path : Path
        Output figure path.
    """
    available_metrics = [metric for metric in KEY_TEST_METRICS if metric in summary_df.columns]
    if not available_metrics:
        raise ValueError(
            "Cannot build heatmap because none of the key metrics are available in the summary dataframe."
        )

    score_df = summary_df[["model_name"] + available_metrics].copy()
    score_df = score_df.drop_duplicates(subset=["model_name"], keep="last").reset_index(drop=True)

    normalized_columns: dict[str, np.ndarray] = {}
    for metric in available_metrics:
        values = pd.to_numeric(score_df[metric], errors="coerce").to_numpy(dtype=np.float64)
        normalized_columns[metric] = _normalize_metric(values, higher_is_better=metric not in LOWER_IS_BETTER)

    norm_df = pd.DataFrame(normalized_columns)
    norm_df.insert(0, "model_name", score_df["model_name"])
    matrix = norm_df[available_metrics].to_numpy(dtype=np.float64)

    fig_width = max(8.0, 1.2 * len(available_metrics))
    fig_height = max(4.5, 0.55 * matrix.shape[0] + 2.5)
    fig, axis = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(available_metrics)))
    axis.set_xticklabels(available_metrics, rotation=35, ha="right")
    axis.set_yticks(np.arange(matrix.shape[0]))
    axis.set_yticklabels(norm_df["model_name"].tolist())
    axis.set_title("Normalized Test-Metric Score Heatmap (1 = best)")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isfinite(value):
                text_color = "white" if value < 0.55 else "black"
                axis.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color=text_color)

    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Normalized score")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved metric-score heatmap to '{output_path}'.", flush=True)


def _normalize_metric(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Normalize metric values into [0, 1], where 1 indicates best value.

    Parameters
    ----------
    values : np.ndarray
        Metric values.
    higher_is_better : bool
        Whether larger values indicate better performance.

    Returns
    -------
    np.ndarray
        Normalized metric values.
    """
    finite_mask = np.isfinite(values)
    normalized = np.full(values.shape, np.nan, dtype=np.float64)

    if not np.any(finite_mask):
        return normalized

    finite_values = values[finite_mask]
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))

    if np.isclose(maximum, minimum):
        normalized[finite_mask] = 0.5
        return normalized

    if higher_is_better:
        normalized[finite_mask] = (finite_values - minimum) / (maximum - minimum)
    else:
        normalized[finite_mask] = (maximum - finite_values) / (maximum - minimum)

    return normalized


def main() -> None:
    """Collect benchmark results, regenerate summary CSV, and save figures."""
    args = parse_args()

    benchmark_root = Path(args.benchmark_root)
    if not benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark root '{benchmark_root}' does not exist.")

    summary_df = collect_benchmark_rows(benchmark_root)

    summary_path = benchmark_root / args.summary_filename
    save_summary_csv(summary_df, summary_path)

    plots_root = benchmark_root / args.plots_subdir
    plot_key_metric_bars(summary_df, plots_root / "key_test_metrics_bars.png")
    plot_metric_score_heatmap(summary_df, plots_root / "key_test_metrics_heatmap.png")


if __name__ == "__main__":
    main()
