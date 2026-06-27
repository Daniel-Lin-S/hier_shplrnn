"""BPTT-based hier-shPLRNN feature extractors."""

from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from data_io.dataset import MultiSubjectDataset
from main import get_device as training_get_device
from models.feature_extractors.base import LatentFeatureExtractor
from models.feature_extractors.utils import (
    cast_tensor,
    clean_state_dict_keys,
    resolve_checkpoint_path,
    validate_feature_matrix,
)
from models.hier_shplrnn import shallowPLRNN
from trainers.bptt import BPTT, read_hypers

DEFAULT_COHORT_SUBJECT_THRESHOLD = 80
DEFAULT_BATCH_SIZE_MULTIPLIER = 4


def _resolve_model_dir(model_path: str) -> str:
    """Resolve model run directory from a model path.

    Parameters
    ----------
    model_path : str
        Path to a model run directory or checkpoint file.

    Returns
    -------
    str
        Directory containing ``hypers.txt`` and model checkpoints.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

    model_dir = path.parent if path.is_file() else path
    hypers_path = model_dir / "hypers.txt"
    if not hypers_path.exists():
        raise FileNotFoundError(
            "Could not find 'hypers.txt' required for BPTT initialization. "
            f"Resolved model_dir='{model_dir}', model_path='{model_path}'."
        )
    return str(model_dir)


def _normalize_learning_rate(value: Any) -> tuple[float, float]:
    """Normalize learning-rate values into ``(shared_lr, individual_lr)``.

    Parameters
    ----------
    value : Any
        Learning-rate value from runtime arguments.

    Returns
    -------
    tuple[float, float]
        Shared and individual learning rates.
    """
    if isinstance(value, tuple) and len(value) == 2:
        shared_lr = float(value[0])
        individual_lr = float(value[1])
    elif isinstance(value, list) and len(value) == 2:
        shared_lr = float(value[0])
        individual_lr = float(value[1])
    elif isinstance(value, (int, float)):
        shared_lr = float(value)
        individual_lr = float(value)
    else:
        raise TypeError(
            "Expected learning_rate to be scalar or length-2 sequence, "
            f"but got value={value!r} of type {type(value).__name__}."
        )

    if shared_lr <= 0 or individual_lr <= 0:
        raise ValueError(
            "Learning rates must be positive, "
            f"but got shared_lr={shared_lr}, individual_lr={individual_lr}."
        )
    return shared_lr, individual_lr


def _dataset_token(dataset_name: str | None) -> str:
    """Convert optional dataset names to stable artifact tokens."""
    if dataset_name is None or dataset_name.strip() == "":
        return "evaluation"
    return dataset_name.strip().replace(" ", "_")


class _HierShPLRNNBPTTBaseExtractor(LatentFeatureExtractor):
    """Shared BPTT execution helpers for hier-shPLRNN extractors."""

    def __init__(
        self,
        name: str,
        expected_dim: int,
        use_gpu: bool = False,
        device_id: int = 0,
        cohort_subject_threshold: int = DEFAULT_COHORT_SUBJECT_THRESHOLD,
        batch_size_multiplier: int = DEFAULT_BATCH_SIZE_MULTIPLIER,
    ) -> None:
        """Initialize shared extractor state.

        Parameters
        ----------
        name : str
            Extractor identifier.
        expected_dim : int
            Expected p-vector dimensionality.
        use_gpu : bool, optional
            Whether to request GPU execution, by default False.
        device_id : int, optional
            CUDA device id if ``use_gpu`` is enabled, by default 0.
        cohort_subject_threshold : int, optional
            Subject-count threshold used for adaptive batching. Small cohorts
            train with all subjects per iteration, while larger cohorts sample
            exactly this many subjects per iteration. By default
            ``DEFAULT_COHORT_SUBJECT_THRESHOLD``.
        batch_size_multiplier : int, optional
            Number of sequence samples per selected subject, by default
            ``DEFAULT_BATCH_SIZE_MULTIPLIER``.
        """
        if expected_dim <= 0:
            raise ValueError(f"Expected expected_dim > 0, but got {expected_dim}.")
        if cohort_subject_threshold <= 0:
            raise ValueError(
                "Expected cohort_subject_threshold > 0, "
                f"but got {cohort_subject_threshold}."
            )
        if batch_size_multiplier <= 0:
            raise ValueError(f"Expected batch_size_multiplier > 0, but got {batch_size_multiplier}.")

        self.name = name
        self.expected_dim = expected_dim
        self.use_gpu = use_gpu
        self.device_id = device_id
        self.cohort_subject_threshold = cohort_subject_threshold
        self.batch_size_multiplier = batch_size_multiplier
        self._runtime_output_dir: Path | None = None

    def set_runtime_output_dir(self, output_dir: str) -> None:
        """Set runtime output directory for this benchmark run.

        Parameters
        ----------
        output_dir : str
            Output directory passed by benchmark orchestrator.
        """
        self._runtime_output_dir = Path(output_dir)

    def _resolve_save_root(self) -> Path:
        """Resolve root folder where saver artifacts are written.

        Returns
        -------
        Path
            Root artifact directory.
        """
        if self._runtime_output_dir is not None:
            save_root = self._runtime_output_dir
        else:
            save_root = Path("./results/epileptic_latent_benchmark/models") / self.name
        save_root.mkdir(parents=True, exist_ok=True)
        return save_root

    def _set_common_save_args(self, args: Namespace, mode: str, dataset_name: str | None) -> Namespace:
        """Apply saver output layout shared by both extractors.

        Parameters
        ----------
        args : Namespace
            Runtime argument namespace.
        mode : str
            Mode token used in saver path naming.
        dataset_name : str | None
            Optional dataset identifier.

        Returns
        -------
        Namespace
            Updated namespace.
        """
        token = _dataset_token(dataset_name)
        args.save_path = str(self._resolve_save_root())
        args.experiment = "training_progress"
        args.name = f"{self.name}_{mode}_{token}"
        args.run = 1
        return args

    def _build_dataset(self, args: Namespace, expected_subjects: int, dataset_name: str | None) -> MultiSubjectDataset:
        """Build multi-subject dataset and validate subject count.

        Parameters
        ----------
        args : Namespace
            Runtime argument namespace.
        expected_subjects : int
            Expected number of subjects from input signals.
        dataset_name : str | None
            Optional dataset identifier.

        Returns
        -------
        MultiSubjectDataset
            Constructed dataset instance.
        """
        dataset = MultiSubjectDataset(
            args.data_path,
            args.seq_len,
            args.train_set_size,
            args.subjects_per_batch,
            args.num_workers,
            args.device,
        )

        if dataset.num_subjects != expected_subjects:
            raise ValueError(
                "Dataset subject count mismatch during extractor optimization. "
                f"extractor='{self.name}', dataset='{dataset_name}', expected={expected_subjects}, "
                f"actual={dataset.num_subjects}."
            )
        return dataset

    def _finalize_training_artifacts(
        self,
        trainer: BPTT,
        num_epochs: int,
        run_expensive_evaluation: bool = False,
    ) -> None:
        """Flush saver artifacts similarly to regular training runs.

        Parameters
        ----------
        trainer : BPTT
            Trained BPTT instance.
        num_epochs : int
            Number of epochs that were run.
        run_expensive_evaluation : bool, optional
            Whether to run expensive evaluation before closing the writer,
            by default False.
        """
        trained_model = cast(shallowPLRNN, trainer.model)
        saver = cast(Any, trained_model.saver)
        try:
            if run_expensive_evaluation:
                saver.save_expensive(num_epochs)
            else:
                saver.save_cheap(num_epochs)
        finally:
            saver.writer.close()

    def _validate_p_vectors(self, features: np.ndarray, signals: np.ndarray, dataset_name: str | None) -> None:
        """Validate output p-vectors against expectations.

        Parameters
        ----------
        features : np.ndarray
            Extracted p-vector matrix.
        signals : np.ndarray
            Input EEG tensor.
        dataset_name : str | None
            Optional dataset identifier.
        """
        if features.ndim != 2:
            raise ValueError(
                "Expected extracted p-vectors to be a 2D matrix, "
                f"but got shape {features.shape} for extractor='{self.name}', dataset='{dataset_name}'."
            )
        if features.shape[0] != signals.shape[0]:
            raise ValueError(
                "P-vector count does not match evaluation samples. "
                f"extractor='{self.name}', dataset='{dataset_name}', features={features.shape[0]}, "
                f"samples={signals.shape[0]}."
            )
        if features.shape[1] != self.expected_dim:
            raise ValueError(
                "Unexpected p-vector dimensionality. "
                f"extractor='{self.name}', dataset='{dataset_name}', expected_dim={self.expected_dim}, "
                f"actual_dim={features.shape[1]}."
            )

        validate_feature_matrix(features, f"{self.name}_{_dataset_token(dataset_name)}")

    def _snapshot_shared_parameters(self, trainer: BPTT) -> list[torch.Tensor]:
        """Snapshot shared-group parameters before finetuning.

        Parameters
        ----------
        trainer : BPTT
            Trainer instance.

        Returns
        -------
        list[torch.Tensor]
            Detached shared-parameter snapshots.
        """
        shared_parameters, _ = trainer.model.hierarchisation_scheme.grouped_parameters()
        return [parameter.detach().cpu().clone() for parameter in shared_parameters]

    def _assert_shared_parameters_frozen(
        self,
        trainer: BPTT,
        snapshots: list[torch.Tensor],
    ) -> None:
        """Assert finetuning did not modify shared-group parameters.

        Parameters
        ----------
        trainer : BPTT
            Trainer instance after finetuning.
        snapshots : list[torch.Tensor]
            Shared-parameter snapshots captured before finetuning.
        """
        shared_parameters, _ = trainer.model.hierarchisation_scheme.grouped_parameters()
        if len(shared_parameters) != len(snapshots):
            raise RuntimeError(
                "Shared parameter count changed during finetuning. "
                f"before={len(snapshots)}, after={len(shared_parameters)}."
            )

        for index, (parameter, snapshot) in enumerate(zip(shared_parameters, snapshots), start=1):
            current = parameter.detach().cpu()
            if not torch.equal(current, snapshot):
                raise RuntimeError(
                    "Detected updates in shared-group parameters during finetuning. "
                    f"extractor='{self.name}', parameter_index={index}."
                )

    def _configure_adaptive_batching(
        self,
        args: Namespace,
        num_subjects: int,
        mode: str,
    ) -> Namespace:
        """Set batching policy for small and large cohorts.

        Parameters
        ----------
        args : Namespace
            Runtime argument namespace.
        num_subjects : int
            Number of subjects in current stage.
        mode : str
            Human-readable mode token used in logs.

        Returns
        -------
        Namespace
            Namespace with updated batching fields.
        """
        if num_subjects <= 0:
            raise ValueError(f"Expected num_subjects > 0 for adaptive batching, but got {num_subjects}.")

        if num_subjects <= self.cohort_subject_threshold:
            args.subjects_per_batch = int(num_subjects)
            args.batch_size = int(self.batch_size_multiplier * num_subjects)
            args.reshuffle_subjects_each_iteration = False
            sampling_strategy = "all_subjects_each_iteration"
        else:
            sampled_subjects = int(self.cohort_subject_threshold)
            args.subjects_per_batch = int(sampled_subjects)
            args.batch_size = int(self.batch_size_multiplier * sampled_subjects)
            args.reshuffle_subjects_each_iteration = True
            sampling_strategy = "random_subject_subset_per_iteration"

        if args.batch_size <= 0:
            raise ValueError(
                "Adaptive batching produced non-positive batch size. "
                f"mode='{mode}', num_subjects={num_subjects}, batch_size={args.batch_size}."
            )

        print(
            "Applied adaptive batching policy: "
            f"mode='{mode}', num_subjects={num_subjects}, batch_size={args.batch_size}, "
            f"subjects_per_batch={args.subjects_per_batch}, strategy='{sampling_strategy}'.",
            flush=True,
        )
        return args

    def _resolve_run_dir(self, args: Namespace) -> Path:
        """Resolve saver run directory for runtime arguments.

        Parameters
        ----------
        args : Namespace
            Runtime argument namespace.

        Returns
        -------
        Path
            Resolved run directory.
        """
        if getattr(args, "save_path", None) is None:
            raise ValueError("Expected args.save_path to be set before resolving run directory.")
        if getattr(args, "experiment", None) is None:
            raise ValueError("Expected args.experiment to be set before resolving run directory.")
        if getattr(args, "name", None) is None:
            raise ValueError("Expected args.name to be set before resolving run directory.")
        if getattr(args, "run", None) is None:
            raise ValueError("Expected args.run to be set before resolving run directory.")

        run = int(args.run)
        return Path(str(args.save_path)) / str(args.experiment) / str(args.name) / f"{run:03d}"


class HierShPLRNNFinetunedPVectorExtractor(_HierShPLRNNBPTTBaseExtractor):
    """Finetune individual parameters from a pretrained model and return p-vectors."""

    def __init__(
        self,
        model_path: str,
        expected_dim: int = 6,
        finetune_epochs: int = 20,
        finetune_batches_per_epoch: int = 8,
        finetune_num_workers: int = 0,
        use_gpu: bool = False,
        device_id: int = 0,
        cohort_subject_threshold: int = DEFAULT_COHORT_SUBJECT_THRESHOLD,
        batch_size_multiplier: int = DEFAULT_BATCH_SIZE_MULTIPLIER,
    ) -> None:
        """Initialise the finetuned p-vector extractor.

        Parameters
        ----------
        model_path : str
            Path to pretrained model directory or checkpoint.
        expected_dim : int, optional
            Expected p-vector dimensionality, by default 6.
        finetune_epochs : int, optional
            Number of finetuning epochs, by default 20.
        finetune_batches_per_epoch : int, optional
            Number of batches per epoch, by default 8.
        finetune_num_workers : int, optional
            DataLoader worker count, by default 0.
        use_gpu : bool, optional
            Whether to request GPU execution, by default False.
        device_id : int, optional
            CUDA device id if ``use_gpu`` is enabled, by default 0.
        cohort_subject_threshold : int, optional
            Subject-count threshold used by adaptive batching. Small cohorts
            use all subjects per iteration, while large cohorts sample this
            many subjects per iteration. By default 80.
        batch_size_multiplier : int, optional
            Sequences per selected subject in one batch, by default 4.
        """
        super().__init__(
            name="hier_shplrnn_finetuned",
            expected_dim=expected_dim,
            use_gpu=use_gpu,
            device_id=device_id,
            cohort_subject_threshold=cohort_subject_threshold,
            batch_size_multiplier=batch_size_multiplier,
        )
        self.model_path = _resolve_model_dir(model_path)
        self.finetune_epochs = finetune_epochs
        self.finetune_batches_per_epoch = finetune_batches_per_epoch
        self.finetune_num_workers = finetune_num_workers
        self._feature_cache: dict[str, np.ndarray] = {}

    def extract(self, signals: np.ndarray, dataset_name: str | None = None) -> np.ndarray:
        """Run BPTT finetuning and return learned p-vectors.

        Parameters
        ----------
        signals : np.ndarray
            Evaluation EEG tensor.
        dataset_name : str | None, optional
            Optional dataset identifier.

        Returns
        -------
        np.ndarray
            Learned p-vector matrix.
        """
        cache_key = _dataset_token(dataset_name)
        if cache_key not in self._feature_cache:
            self._feature_cache[cache_key] = self._extract_finetuned_vectors(signals, dataset_name)

        features = self._feature_cache[cache_key]
        self._validate_p_vectors(features, signals, dataset_name)
        return np.copy(features)

    def _extract_finetuned_vectors(self, signals: np.ndarray, dataset_name: str | None) -> np.ndarray:
        """Optimize individual parameters with BPTT finetuning.

        Parameters
        ----------
        signals : np.ndarray
            Evaluation EEG tensor.
        dataset_name : str | None
            Optional dataset identifier.

        Returns
        -------
        np.ndarray
            Learned p-vector matrix.
        """
        if signals.shape[0] <= 0:
            raise ValueError("Expected at least one subject for finetuning, but received an empty tensor.")

        with tempfile.TemporaryDirectory(prefix="hier_shplrnn_finetune_") as temp_dir:
            data_path = Path(temp_dir) / f"{self.name}_{_dataset_token(dataset_name)}.pt"
            torch.save(torch.as_tensor(signals, dtype=torch.float32), data_path)

            args = self._build_finetune_args(
                data_path=str(data_path),
                num_subjects=signals.shape[0],
                dataset_name=dataset_name,
            )
            dataset = self._build_dataset(args, signals.shape[0], dataset_name)

            trainer = BPTT(args, dataset)
            shared_snapshots = self._snapshot_shared_parameters(trainer)

            run_training = trainer.finetune
            if args.compile:
                run_training = torch.compile(run_training)
            run_training()

            self._assert_shared_parameters_frozen(trainer, shared_snapshots)
            self._finalize_training_artifacts(trainer, args.num_epochs)

            trained_model = cast(shallowPLRNN, trainer.model)
            features = cast_tensor(cast(torch.Tensor, trained_model.p_vector))

        return features

    def _build_finetune_args(self, data_path: str, num_subjects: int, dataset_name: str | None) -> Namespace:
        """Build runtime arguments for BPTT finetuning.

        Parameters
        ----------
        data_path : str
            Temporary ``.pt`` path containing evaluation signals.
        num_subjects : int
            Number of subjects in evaluation tensor.
        dataset_name : str | None
            Optional dataset identifier.

        Returns
        -------
        Namespace
            Prepared finetuning arguments.
        """
        if self.finetune_epochs <= 0:
            raise ValueError(f"Expected finetune_epochs > 0, but got {self.finetune_epochs}.")
        if self.finetune_batches_per_epoch <= 0:
            raise ValueError(
                f"Expected finetune_batches_per_epoch > 0, but got {self.finetune_batches_per_epoch}."
            )
        if self.finetune_num_workers < 0:
            raise ValueError(f"Expected finetune_num_workers >= 0, but got {self.finetune_num_workers}.")

        args = read_hypers(Namespace(model_path=self.model_path))
        args.model_path = self.model_path
        args.finetune = True
        args.checkpoint = None

        args.data_path = data_path
        args.eval_data_path = data_path
        args.num_epochs = int(self.finetune_epochs)
        args.batches_per_epoch = int(self.finetune_batches_per_epoch)
        args.num_workers = int(self.finetune_num_workers)
        args = self._configure_adaptive_batching(args, num_subjects, mode="finetune")

        args.learning_rate = _normalize_learning_rate(getattr(args, "learning_rate", None))
        args.individual_learning_rate = args.learning_rate[1]

        if getattr(args, "tf_alpha_start", None) is None:
            args.tf_alpha_start = 0.1
        if getattr(args, "tf_alpha_end", None) is None:
            args.tf_alpha_end = args.tf_alpha_start
        if getattr(args, "metrics", None) is None:
            args.metrics = ["kl", "pse", "mse"]
        if getattr(args, "plots", None) is None:
            args.plots = []

        args.use_gpu = self.use_gpu
        args.device_id = self.device_id
        args = training_get_device(args)
        args.compile = False

        args = self._set_common_save_args(args, mode="finetune", dataset_name=dataset_name)

        print(
            "Running hier-shPLRNN finetune extraction with parameters: "
            f"dataset='{dataset_name}', num_subjects={num_subjects}, num_epochs={args.num_epochs}, "
            f"batch_size={args.batch_size}, subjects_per_batch={args.subjects_per_batch}, "
            f"reshuffle_subjects_each_iteration={args.reshuffle_subjects_each_iteration}, "
            f"batches_per_epoch={args.batches_per_epoch}, save_root='{args.save_path}'.",
            flush=True,
        )
        return args


class HierShPLRNNCheckpointPVectorExtractor(_HierShPLRNNBPTTBaseExtractor):
    """Load hier-shPLRNN p-vectors directly from a trained checkpoint."""

    def __init__(
        self,
        model_path: str,
        expected_dim: int = 6,
        checkpoint: int | None = None,
    ) -> None:
        """Initialize checkpoint-based p-vector extractor.

        Parameters
        ----------
        model_path : str
            Path to a model run directory or a single ``.pt`` checkpoint.
        expected_dim : int, optional
            Expected p-vector dimensionality, by default 6.
        checkpoint : int | None, optional
            Explicit checkpoint epoch id when ``model_path`` is a directory.
            If ``None``, the latest checkpoint in the directory is used.
        """
        super().__init__(
            name="hier_shplrnn_checkpoint",
            expected_dim=expected_dim,
            use_gpu=False,
            device_id=0,
            cohort_subject_threshold=DEFAULT_COHORT_SUBJECT_THRESHOLD,
            batch_size_multiplier=DEFAULT_BATCH_SIZE_MULTIPLIER,
        )
        self.model_path = model_path
        self.checkpoint = checkpoint
        self._feature_cache: dict[str, np.ndarray] = {}
        self._evaluation_indices: np.ndarray | None = None

    def set_evaluation_indices(self, source_indices: np.ndarray) -> None:
        """Set deterministic source indices for checkpoint feature alignment.

        Parameters
        ----------
        source_indices : np.ndarray
            One-dimensional non-negative integer indices selecting the rows to
            keep from the checkpoint's full ``p_vector`` table.
        """
        indices = np.asarray(source_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(
                "Expected source_indices to be one-dimensional, "
                f"but got shape {indices.shape}."
            )
        if indices.size == 0:
            raise ValueError("Expected source_indices to be non-empty for checkpoint extraction.")
        if np.any(indices < 0):
            raise ValueError(
                "Expected source_indices to contain non-negative values, "
                f"but got values={indices.tolist()}."
            )
        unique_count = np.unique(indices).size
        if unique_count != indices.size:
            raise ValueError(
                "Expected source_indices to contain unique subject rows, "
                f"but got {indices.size - unique_count} duplicate entries."
            )

        self._evaluation_indices = indices.copy()

    def extract(self, signals: np.ndarray, dataset_name: str | None = None) -> np.ndarray:
        """Load p-vectors from checkpoint and validate against evaluation batch.

        Parameters
        ----------
        signals : np.ndarray
            Evaluation EEG tensor.
        dataset_name : str | None, optional
            Optional dataset identifier used for cache keys.

        Returns
        -------
        np.ndarray
            Subject feature matrix loaded from checkpoint.
        """
        cache_key = self._cache_key(dataset_name)
        if cache_key not in self._feature_cache:
            self._feature_cache[cache_key] = self._load_checkpoint_vectors()

        features = self._feature_cache[cache_key]
        self._validate_p_vectors(features, signals, dataset_name)
        return np.copy(features)

    def _cache_key(self, dataset_name: str | None) -> str:
        """Build cache key including deterministic index selection."""
        token = _dataset_token(dataset_name)
        if self._evaluation_indices is None:
            return f"{token}::all"
        index_token = ",".join(str(int(index)) for index in self._evaluation_indices.tolist())
        return f"{token}::{index_token}"

    def _resolve_checkpoint_file(self) -> str:
        """Resolve checkpoint file path from extractor settings.

        Returns
        -------
        str
            Checkpoint file path.
        """
        if self.checkpoint is None:
            return resolve_checkpoint_path(self.model_path)

        model_dir = Path(self.model_path)
        if model_dir.is_file():
            raise ValueError(
                "Explicit 'checkpoint' cannot be combined with a direct checkpoint file path. "
                f"Received model_path='{self.model_path}', checkpoint={self.checkpoint}."
            )

        checkpoint_path = model_dir / f"model_{int(self.checkpoint)}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Could not find requested checkpoint file for extractor. "
                f"Expected '{checkpoint_path}'."
            )
        return str(checkpoint_path)

    def _load_checkpoint_vectors(self) -> np.ndarray:
        """Load p-vectors from a model checkpoint.

        Returns
        -------
        np.ndarray
            Feature matrix with shape ``(num_subjects, expected_dim)``.
        """
        checkpoint_file = self._resolve_checkpoint_file()
        raw_state_dict = torch.load(checkpoint_file, map_location="cpu")
        if not isinstance(raw_state_dict, dict):
            raise TypeError(
                "Expected checkpoint to contain a state_dict mapping, "
                f"but got type {type(raw_state_dict).__name__} from '{checkpoint_file}'."
            )

        state_dict = clean_state_dict_keys(raw_state_dict)
        if "p_vector" not in state_dict:
            raise KeyError(
                "Checkpoint does not contain 'p_vector'. "
                f"Cannot extract subject features from '{checkpoint_file}'."
            )

        p_vector = state_dict["p_vector"]
        if not isinstance(p_vector, torch.Tensor):
            raise TypeError(
                "Expected checkpoint key 'p_vector' to be a torch.Tensor, "
                f"but got type {type(p_vector).__name__} from '{checkpoint_file}'."
            )

        feature_vectors = cast_tensor(p_vector)
        if feature_vectors.ndim != 2:
            raise ValueError(
                "Expected checkpoint p-vectors to be a 2D matrix, "
                f"but got shape {feature_vectors.shape} from '{checkpoint_file}'."
            )
        if feature_vectors.shape[1] != self.expected_dim:
            raise ValueError(
                "Unexpected p-vector dimensionality loaded from checkpoint. "
                f"expected_dim={self.expected_dim}, actual_dim={feature_vectors.shape[1]}, "
                f"checkpoint='{checkpoint_file}'."
            )

        if self._evaluation_indices is None:
            return feature_vectors

        max_index = int(self._evaluation_indices.max())
        if max_index >= feature_vectors.shape[0]:
            raise ValueError(
                "Evaluation subset references subject rows that are outside the checkpoint range. "
                f"max_index={max_index}, checkpoint_rows={feature_vectors.shape[0]}, "
                f"checkpoint='{checkpoint_file}'."
            )

        return feature_vectors[self._evaluation_indices]
