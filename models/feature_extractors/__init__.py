"""Latent feature extractor package."""

from models.feature_extractors.base import LatentFeatureExtractor
from models.feature_extractors.baselines import (
    BandPowerFeatureExtractor,
    Catch22FeatureExtractor,
    RawPCAFeatureExtractor,
)
from models.feature_extractors.cbramod import CBraModFeatureExtractor
from models.feature_extractors.hier_shplrnn import (
    HierShPLRNNCheckpointPVectorExtractor,
    HierShPLRNNFinetunedPVectorExtractor,
)
from models.feature_extractors.registry import (
    create_feature_extractor,
    register_default_feature_extractors,
    register_feature_extractor,
)

__all__ = [
    "BandPowerFeatureExtractor",
    "CBraModFeatureExtractor",
    "Catch22FeatureExtractor",
    "HierShPLRNNCheckpointPVectorExtractor",
    "HierShPLRNNFinetunedPVectorExtractor",
    "LatentFeatureExtractor",
    "RawPCAFeatureExtractor",
    "create_feature_extractor",
    "register_default_feature_extractors",
    "register_feature_extractor",
]
