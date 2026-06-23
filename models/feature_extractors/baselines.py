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

    def extract(self, signals: np.ndarray, dataset_name: str | None = None) -> np.ndarray:
        """Extract PCA features from one evaluation dataset.

        Parameters
        ----------
        signals : np.ndarray
            Evaluation EEG tensor.
        dataset_name : str | None, optional
            Optional dataset identifier.

        Returns
        -------
        np.ndarray
            PCA feature matrix.
        """
        del dataset_name

        flattened = flatten_signals(signals)
        if self.n_components > flattened.shape[1]:
            raise ValueError(
                "PCA component count exceeds flattened feature dimension: "
                f"requested={self.n_components}, available={flattened.shape[1]}."
            )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=self.n_components, random_state=0)),
            ]
        )
        features = pipeline.fit_transform(flattened)
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

    def extract(self, signals: np.ndarray, dataset_name: str | None = None) -> np.ndarray:
        """Extract log-bandpower features for one evaluation dataset.

        Parameters
        ----------
        signals : np.ndarray
            EEG tensor.
        dataset_name : str | None, optional
            Optional dataset identifier.

        Returns
        -------
        np.ndarray
            Bandpower feature matrix.
        """
        del dataset_name

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

    def _load_pycatch22(self) -> Any:
        """Load pycatch22 dependency once.

        Returns
        -------
        Any
            Imported ``pycatch22`` module.
        """
        if self.pycatch22_module is not None:
            return self.pycatch22_module

        try:
            import pycatch22  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pycatch22 is required for catch22 baseline features. "
                "Install it with 'pip install pycatch22'."
            ) from exc

        self.pycatch22_module = pycatch22
        return self.pycatch22_module

    def extract(self, signals: np.ndarray, dataset_name: str | None = None) -> np.ndarray:
        """Extract catch22 features for each sample and channel.

        Parameters
        ----------
        signals : np.ndarray
            EEG tensor.
        dataset_name : str | None, optional
            Optional dataset identifier.

        Returns
        -------
        np.ndarray
            Catch22 feature matrix.
        """
        del dataset_name

        pycatch22 = self._load_pycatch22()

        num_samples, _, num_channels = signals.shape
        features = np.zeros((num_samples, num_channels * 22), dtype=np.float64)

        for sample_index in range(num_samples):
            channel_features: list[np.ndarray] = []
            for channel_index in range(num_channels):
                series = signals[sample_index, :, channel_index].astype(np.float64)
                result = pycatch22.catch22_all(series.tolist())
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
