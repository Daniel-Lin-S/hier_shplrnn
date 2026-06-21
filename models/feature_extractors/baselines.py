"""Baseline feature extractors for latent benchmarking."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.feature_extractors.base import LatentFeatureExtractor
from models.feature_extractors.utils import BAND_LIMITS_HZ, flatten_signals, validate_feature_matrix


class RawPCAFeatureExtractor(LatentFeatureExtractor):
    """PCA baseline extractor on flattened signal windows."""

    def __init__(self, n_components: int) -> None:
        """Initialize PCA extractor.

        Parameters
        ----------
        n_components : int
            Number of retained PCA components.
        """
        self.n_components = n_components
        self.name = f"baseline_pca_{n_components}"
        self.pipeline: Pipeline | None = None

    def fit(self, train_signals: np.ndarray, train_labels: np.ndarray) -> None:
        """Fit scaler and PCA pipeline.

        Parameters
        ----------
        train_signals : np.ndarray
            Training EEG tensor.
        train_labels : np.ndarray
            Training labels.
        """
        del train_labels

        flat_train = flatten_signals(train_signals)
        if self.n_components > flat_train.shape[1]:
            raise ValueError(
                "PCA component count exceeds flattened feature dimension: "
                f"requested={self.n_components}, available={flat_train.shape[1]}."
            )

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=self.n_components, random_state=0)),
            ]
        )
        self.pipeline.fit(flat_train)

    def transform(self, signals: np.ndarray, split_name: str) -> np.ndarray:
        """Project flattened windows into PCA features.

        Parameters
        ----------
        signals : np.ndarray
            EEG tensor.
        split_name : str
            Split identifier.

        Returns
        -------
        np.ndarray
            PCA feature matrix.
        """
        del split_name

        if self.pipeline is None:
            raise RuntimeError("PCA extractor was not fitted before transform was called.")

        features = self.pipeline.transform(flatten_signals(signals))
        validate_feature_matrix(features, self.name)
        return features


class BandPowerFeatureExtractor(LatentFeatureExtractor):
    """Canonical EEG bandpower baseline extractor."""

    def __init__(self, sample_rate_hz: int) -> None:
        """Initialize bandpower extractor.

        Parameters
        ----------
        sample_rate_hz : int
            Sampling rate of the EEG windows.
        """
        self.name = "baseline_bandpower"
        self.sample_rate_hz = sample_rate_hz

    def fit(self, train_signals: np.ndarray, train_labels: np.ndarray) -> None:
        """No-op for deterministic handcrafted features.

        Parameters
        ----------
        train_signals : np.ndarray
            Training EEG tensor.
        train_labels : np.ndarray
            Training labels.
        """
        del train_signals, train_labels

    def transform(self, signals: np.ndarray, split_name: str) -> np.ndarray:
        """Extract log-bandpower features.

        Parameters
        ----------
        signals : np.ndarray
            EEG tensor.
        split_name : str
            Split identifier.

        Returns
        -------
        np.ndarray
            Bandpower feature matrix.
        """
        del split_name

        num_samples, num_timesteps, num_channels = signals.shape
        frequencies = np.fft.rfftfreq(num_timesteps, d=1.0 / float(self.sample_rate_hz))

        demeaned = signals - signals.mean(axis=1, keepdims=True)
        spectrum = np.abs(np.fft.rfft(demeaned, axis=1)) ** 2
        spectrum = spectrum / float(num_timesteps)

        feature_blocks: list[np.ndarray] = []
        for _, (low_hz, high_hz) in BAND_LIMITS_HZ:
            band_mask = (frequencies >= low_hz) & (frequencies < high_hz)
            if not np.any(band_mask):
                raise ValueError(
                    "The selected EEG band has no FFT bins for the current sample rate and length: "
                    f"band=({low_hz}, {high_hz})Hz, num_timesteps={num_timesteps}, "
                    f"sample_rate_hz={self.sample_rate_hz}."
                )

            band_power = spectrum[:, band_mask, :].mean(axis=1)
            feature_blocks.append(np.log1p(band_power))

        features = np.concatenate(feature_blocks, axis=1)
        expected_shape = (num_samples, num_channels * len(BAND_LIMITS_HZ))
        if features.shape != expected_shape:
            raise ValueError(
                "Bandpower feature shape mismatch: "
                f"expected {expected_shape}, got {features.shape}."
            )

        validate_feature_matrix(features, self.name)
        return features


class Catch22FeatureExtractor(LatentFeatureExtractor):
    """pycatch22 baseline extractor."""

    def __init__(self) -> None:
        """Initialize catch22 extractor."""
        self.name = "baseline_catch22"
        self.pycatch22_module: Any | None = None

    def fit(self, train_signals: np.ndarray, train_labels: np.ndarray) -> None:
        """Import pycatch22 module.

        Parameters
        ----------
        train_signals : np.ndarray
            Training EEG tensor.
        train_labels : np.ndarray
            Training labels.
        """
        del train_signals, train_labels

        try:
            import pycatch22  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pycatch22 is required for catch22 baseline features. "
                "Install it with 'pip install pycatch22'."
            ) from exc

        self.pycatch22_module = pycatch22

    def transform(self, signals: np.ndarray, split_name: str) -> np.ndarray:
        """Extract catch22 features for each sample and channel.

        Parameters
        ----------
        signals : np.ndarray
            EEG tensor.
        split_name : str
            Split identifier.

        Returns
        -------
        np.ndarray
            Catch22 feature matrix.
        """
        del split_name

        if self.pycatch22_module is None:
            raise RuntimeError("Catch22 extractor was not fitted before transform was called.")

        num_samples, _, num_channels = signals.shape
        features = np.zeros((num_samples, num_channels * 22), dtype=np.float64)

        for sample_index in range(num_samples):
            channel_features: list[np.ndarray] = []
            for channel_index in range(num_channels):
                series = signals[sample_index, :, channel_index].astype(np.float64)
                result = self.pycatch22_module.catch22_all(series.tolist())
                values = np.asarray(result["values"], dtype=np.float64)

                if values.shape[0] != 22:
                    raise ValueError(
                        "pycatch22 returned an unexpected feature count: "
                        f"expected 22, got {values.shape[0]} for sample {sample_index}, "
                        f"channel {channel_index}."
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        "pycatch22 returned non-finite values for "
                        f"sample {sample_index}, channel {channel_index}."
                    )

                channel_features.append(values)

            features[sample_index] = np.concatenate(channel_features)

        validate_feature_matrix(features, self.name)
        return features
