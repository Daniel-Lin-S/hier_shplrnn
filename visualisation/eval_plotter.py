"""Plotting utilities for hierarchical model evaluation."""

import importlib
import textwrap
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_subject_feature_pca(
    feature_vectors: np.ndarray,
    labels: np.ndarray,
    run_path: str,
    output_path: str,
) -> None:
    """Plot a label-aware 2D PCA projection of subject feature vectors.

    Parameters
    ----------
    feature_vectors : np.ndarray
        Feature matrix of shape ``(num_subjects, feature_dim)``.
    labels : np.ndarray
        One label per subject.
    run_path : str
        Evaluated run path used for figure title context.
    output_path : str
        File path where the plot is written.

    Raises
    ------
    ValueError
        If feature vectors or labels have incompatible shapes.
    """
    if feature_vectors.ndim != 2:
        raise ValueError(
            "PCA plotting expects a 2D feature matrix with shape "
            f"(num_subjects, feature_dim), but got shape {feature_vectors.shape}."
        )
    if feature_vectors.shape[1] < 2:
        raise ValueError(
            "PCA plotting requires at least two feature dimensions, "
            f"but got {feature_vectors.shape[1]}."
        )
    if labels.shape[0] != feature_vectors.shape[0]:
        raise ValueError(
            "Label count must match number of feature vectors, "
            f"but got {labels.shape[0]} labels for {feature_vectors.shape[0]} vectors."
        )

    # Dynamic import to avoid hard dependency in non-plotting environments
    import matplotlib.pyplot as plt
    try:
        sns_module = importlib.import_module("seaborn")
        sns = cast(Any, sns_module)
    except ImportError:
        sns = None

    labels_str = labels.astype(str)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_vectors)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(features_scaled)
    loadings = pca.components_.T

    unique_labels = sorted(np.unique(labels_str))
    num_unique = len(unique_labels)
    
    # Scale figure and fonts based on number of subjects and unique labels
    base_size = 8
    width = max(12, base_size + num_unique * 0.5)
    height = 8
    
    # Font sizes
    title_fs = 22
    label_fs = 20
    tick_fs = 18
    legend_fs = 18

    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    
    if sns:
        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("colorblind", n_colors=max(num_unique, 1))
    else:
        palette = plt.cm.get_cmap("tab10", max(num_unique, 1)).colors

    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")

    for idx, label in enumerate(unique_labels):
        marker = marker_cycle[idx % len(marker_cycle)]
        color = palette[idx % len(palette)]
        label_mask = labels_str == label
        
        if sns:
            scatter_df = pd.DataFrame(
                {
                    "pc1": scores[label_mask, 0],
                    "pc2": scores[label_mask, 1],
                }
            )
            sns.scatterplot(
                data=scatter_df,
                x="pc1",
                y="pc2",
                marker=marker,
                color=color,
                s=120, # Larger dots
                linewidth=2.0, # Thicker lines (instruction: 2 or 2.5)
                edgecolor="black",
                ax=ax,
                label=f"label={label}",
            )
        else:
            ax.scatter(
                scores[label_mask, 0],
                scores[label_mask, 1],
                marker=marker,
                color=color,
                s=120,
                linewidth=2.0,
                edgecolor="black",
                label=f"label={label}",
            )

    # Principal Component Arrows
    max_score = np.max(np.linalg.norm(scores, axis=1))
    if not np.isfinite(max_score) or max_score <= 0:
        max_score = 1.0
    arrow_scale = 0.7 * max_score
    arrow_vectors = loadings * arrow_scale

    for dim_index, (x_end, y_end) in enumerate(arrow_vectors):
        ax.annotate(
            "",
            xy=(x_end, y_end),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "->", "linewidth": 2.5, "color": "black"},
        )
        ax.text(
            1.12 * x_end,
            1.12 * y_end,
            f"p{dim_index + 1}",
            color="black",
            fontsize=label_fs,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Origin lines
    ax.axhline(0.0, color="black", linewidth=1.5, linestyle="--", alpha=0.3)
    ax.axvline(0.0, color="black", linewidth=1.5, linestyle="--", alpha=0.3)

    # Axis limits - focus strictly on data range to exclude excessive whitespace
    x_values = np.concatenate((scores[:, 0], arrow_vectors[:, 0], np.array([0.0])))
    y_values = np.concatenate((scores[:, 1], arrow_vectors[:, 1], np.array([0.0])))
    x_min, x_max = x_values.min(), x_values.max()
    y_min, y_max = y_values.min(), y_values.max()
    x_margin = 0.12 * (x_max - x_min if x_max > x_min else 1.0)
    y_margin = 0.12 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Clean up frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Labels and Title
    explained = 100.0 * pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained[0]:.2f}% variance)", fontsize=label_fs)
    ax.set_ylabel(f"PC2 ({explained[1]:.2f}% variance)", fontsize=label_fs)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)

    title_text = (
        "2D PCA Projection of Subject Feature Vectors with Principal-Component Arrows\n"
        f"Run: {run_path}"
    )
    # Wrap long titles and keep gap from figure
    wrapped_title = "\n".join(textwrap.wrap(title_text, width=70))
    ax.set_title(wrapped_title, fontsize=title_fs, pad=25)

    # Legend outside, frameoff
    ax.legend(
        title="True Label",
        title_fontsize=label_fs,
        fontsize=legend_fs,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0), # Outside
        frameon=False, # Frame off
        borderaxespad=0.0,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Success: PCA plot saved to {output_path}", flush=True)
