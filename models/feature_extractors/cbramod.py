"""CBraMod latent feature extractor."""

from __future__ import annotations

import os

import numpy as np
import torch

from models.cbramod import CBraMod
from models.feature_extractors.base import LatentFeatureExtractor
from models.feature_extractors.utils import prepare_cbramod_input, validate_feature_matrix


class CBraModFeatureExtractor(LatentFeatureExtractor):
    """Pooled CBraMod latent extractor for window-level EEG features."""

    def __init__(
        self,
        weights_path: str,
        input_rate_hz: int,
        target_rate_hz: int,
        patch_size: int,
        device: str = "cpu",
    ) -> None:
        """Initialize a CBraMod extractor.

        Parameters
        ----------
        weights_path : str
            Path to pre-trained CBraMod weights.
        input_rate_hz : int
            Sampling rate of input EEG windows.
        target_rate_hz : int
            Target sampling rate expected by CBraMod.
        patch_size : int
            Patch size used by CBraMod.
        device : str, optional
            Torch device string, by default ``"cpu"``.
        """
        self.name = "cbramod_pretrained"
        self.weights_path = weights_path
        self.input_rate_hz = input_rate_hz
        self.target_rate_hz = target_rate_hz
        self.patch_size = patch_size
        self.device = device
        self.model: CBraMod | None = None

    def fit(self, train_signals: np.ndarray, train_labels: np.ndarray) -> None:
        """Load pre-trained CBraMod weights.

        Parameters
        ----------
        train_signals : np.ndarray
            Training EEG tensor.
        train_labels : np.ndarray
            Training labels.
        """
        del train_signals, train_labels

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"CBraMod weights file '{self.weights_path}' does not exist. "
                "Please provide a valid pre-trained weight path."
            )
        self.model = CBraMod.from_version("default")
        self.model.load_pretrained_weights(self.weights_path, device=self.device, proj=False)
        self.model.eval()

    def transform(self, signals: np.ndarray, split_name: str) -> np.ndarray:
        """Extract pooled CBraMod features.

        Parameters
        ----------
        signals : np.ndarray
            EEG tensor in ``(samples, timesteps, channels)`` format.
        split_name : str
            Split identifier.

        Returns
        -------
        np.ndarray
            Feature matrix in ``(samples, d_model)`` format.
        """
        del split_name

        if self.model is None:
            raise RuntimeError("CBraMod extractor was not fitted before transform was called.")

        patched_input = prepare_cbramod_input(
            signals,
            input_rate_hz=self.input_rate_hz,
            target_rate_hz=self.target_rate_hz,
            patch_size=self.patch_size,
        )
        patched_input = patched_input.to(self.device)

        with torch.no_grad():
            features = self.model(patched_input, proj=False)

        pooled = features.mean(dim=(1, 2)).detach().cpu().numpy().astype(np.float64)
        validate_feature_matrix(pooled, self.name)
        return pooled
