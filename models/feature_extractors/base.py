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
