"""Base interfaces for latent feature extractors."""

from __future__ import annotations

import abc

import numpy as np


class LatentFeatureExtractor(abc.ABC):
    """Base interface for latent feature extractors."""

    name: str

    def set_runtime_output_dir(self, output_dir: str) -> None:
        """Set benchmark runtime output directory.

        Parameters
        ----------
        output_dir : str
            Directory where the benchmark stores artifacts for this extractor.
        """
        del output_dir

    def set_evaluation_indices(self, source_indices: np.ndarray) -> None:
        """Set source-row indices used to build the evaluation subset.

        Parameters
        ----------
        source_indices : np.ndarray
            Integer indices that map each evaluation sample back to its row in
            the full dataset before subset selection.

        Notes
        -----
        Most extractors derive features directly from the provided ``signals``
        array and can safely ignore this metadata. Extractors that load
        checkpoint-resident features can override this method to preserve
        deterministic row-to-subject alignment.
        """
        del source_indices

    @abc.abstractmethod
    def extract(self, signals: np.ndarray, dataset_name: str | None = None) -> np.ndarray:
        """Convert one evaluation dataset into latent feature vectors.

        Parameters
        ----------
        signals : np.ndarray
            EEG tensor in ``(samples, timesteps, channels)`` format.
        dataset_name : str | None, optional
            Optional evaluation dataset identifier from benchmark config.

        Returns
        -------
        np.ndarray
            Two-dimensional feature matrix.
        """
