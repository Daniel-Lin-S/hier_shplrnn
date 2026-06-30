"""Run latent feature extractor benchmark with repeated random seeds.

Usage
-----
./.conda/bin/python main_eval_feature_extractor.py \
    --benchmark_config configs/latent_benchmark_epileptic_small.yaml \
    --benchmark_model_config configs/latent_model_hier_shplrnn_scratch.yaml

Outputs
-------
This entrypoint writes grouped benchmark results under:
``results/{dataset_group}``

Key files created in the group directory include:
- ``latent_benchmark_repetition_rows.csv``: raw results for all repetitions.
- ``latent_benchmark_summary.csv``: aggregated metrics (mean/std).
- ``repetition_plots/``: faceted dot and error-bar plots for each metric.

This entrypoint is dedicated to benchmark mode and keeps ``main_eval.py``
focused on legacy checkpoint evaluation.
"""

from __future__ import annotations

from eval.feature_benchmark_runtime import parse_feature_benchmark_args, run_repeated_feature_benchmark


def main() -> None:
    """Run feature benchmark with configured repetitions."""
    args = parse_feature_benchmark_args()
    run_repeated_feature_benchmark(args)


if __name__ == "__main__":
    main()
