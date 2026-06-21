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
    def fit(self, train_signals: np.ndarray, train_labels: np.ndarray) -> None:
        """Fit extractor state using training split data.

        Parameters
        ----------
        train_signals : np.ndarray
            Training EEG tensor in ``(samples, timesteps, channels)`` format.
        train_labels : np.ndarray
            One-dimensional training label array.
        """

    @abc.abstractmethod
    def transform(self, signals: np.ndarray, split_name: str) -> np.ndarray:
        """Convert a split into latent feature vectors.

        Parameters
        ----------
        signals : np.ndarray
            EEG tensor in ``(samples, timesteps, channels)`` format.
        split_name : str
            Split identifier, typically ``"train"`` or ``"test"``.

        Returns
        -------
        np.ndarray
            Two-dimensional feature matrix.
        """
