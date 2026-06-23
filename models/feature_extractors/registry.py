"""Registry utilities for latent feature extractors."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from models.feature_extractors.base import LatentFeatureExtractor
from models.feature_extractors.baselines import (
    BandPowerFeatureExtractor,
    Catch22FeatureExtractor,
    RawPCAFeatureExtractor,
)
from models.feature_extractors.cbramod import CBraModFeatureExtractor
from models.feature_extractors.hier_shplrnn import (
    HierShPLRNNFinetunedPVectorExtractor,
    HierShPLRNNFromScratchPVectorExtractor,
)

FEATURE_EXTRACTOR_BUILDERS: dict[str, Callable[..., LatentFeatureExtractor]] = {}


def register_feature_extractor(name: str, builder: Callable[..., LatentFeatureExtractor]) -> None:
    """Register a feature extractor builder.

    Parameters
    ----------
    name : str
        Registry key.
    builder : Callable[..., LatentFeatureExtractor]
        Builder callable that returns a configured extractor.
    """
    if name in FEATURE_EXTRACTOR_BUILDERS:
        raise ValueError(f"A feature extractor named '{name}' is already registered.")
    FEATURE_EXTRACTOR_BUILDERS[name] = builder


def register_default_feature_extractors() -> None:
    """Register default extractors once."""
    if FEATURE_EXTRACTOR_BUILDERS:
        return

    register_feature_extractor("cbramod_pretrained", CBraModFeatureExtractor)
    register_feature_extractor("hier_shplrnn_finetuned", HierShPLRNNFinetunedPVectorExtractor)
    register_feature_extractor("hier_shplrnn_finetuned_pvector", HierShPLRNNFinetunedPVectorExtractor)
    register_feature_extractor("hier_shplrnn_scratch", HierShPLRNNFromScratchPVectorExtractor)
    register_feature_extractor("hier_shplrnn_train_from_scratch", HierShPLRNNFromScratchPVectorExtractor)
    register_feature_extractor("pca", RawPCAFeatureExtractor)
    register_feature_extractor("bandpower", BandPowerFeatureExtractor)
    register_feature_extractor("catch22", Catch22FeatureExtractor)


def create_feature_extractor(extractor_type: str, params: Mapping[str, Any]) -> LatentFeatureExtractor:
    """Create an extractor instance from registry.

    Parameters
    ----------
    extractor_type : str
        Registered extractor key.
    params : Mapping[str, Any]
        Keyword arguments passed to the extractor constructor.

    Returns
    -------
    LatentFeatureExtractor
        Configured extractor instance.
    """
    register_default_feature_extractors()

    if extractor_type not in FEATURE_EXTRACTOR_BUILDERS:
        raise KeyError(
            f"Unknown extractor type '{extractor_type}'. "
            f"Available options: {sorted(FEATURE_EXTRACTOR_BUILDERS.keys())}."
        )

    builder = FEATURE_EXTRACTOR_BUILDERS[extractor_type]
    try:
        return builder(**dict(params))
    except TypeError as exc:
        raise TypeError(
            f"Failed to build extractor '{extractor_type}' with params={dict(params)}."
        ) from exc
