"""BPTT-based hier-shPLRNN feature extractors."""

from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from config_loader import apply_main_config
from data_io.dataset import MultiSubjectDataset
from main import get_device as training_get_device
from main import handle_defaults as training_handle_defaults
from models.feature_extractors.base import LatentFeatureExtractor
from models.feature_extractors.utils import cast_tensor, validate_feature_matrix
from models.hier_shplrnn import shallowPLRNN
from trainers.bptt import BPTT, read_hypers


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
        """
        self.name = name
        self.expected_dim = expected_dim
        self.use_gpu = use_gpu
        self.device_id = device_id
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

    def _finalize_training_artifacts(self, trainer: BPTT, num_epochs: int) -> None:
        """Flush saver artifacts similarly to regular training runs.

        Parameters
        ----------
        trainer : BPTT
            Trained BPTT instance.
        num_epochs : int
            Number of epochs that were run.
        """
        trained_model = cast(shallowPLRNN, trainer.model)
        saver = cast(Any, trained_model.saver)
        saver.save_expensive(num_epochs)
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
        """
        super().__init__(
            name="hier_shplrnn_finetuned",
            expected_dim=expected_dim,
            use_gpu=use_gpu,
            device_id=device_id,
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

        # Keep the finetune batching policy requested for benchmark adaptation.
        args.subjects_per_batch = int(num_subjects)
        args.batch_size = int(4 * num_subjects)

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
            f"batches_per_epoch={args.batches_per_epoch}, save_root='{args.save_path}'.",
            flush=True,
        )
        return args


class HierShPLRNNFromScratchPVectorExtractor(_HierShPLRNNBPTTBaseExtractor):
    """Train hier-shPLRNN from scratch and return learned p-vectors."""

    def __init__(
        self,
        config_path: str,
        expected_dim: int = 6,
        use_gpu: bool = False,
        device_id: int = 0,
    ) -> None:
        """Initialize from-scratch p-vector extractor.

        Parameters
        ----------
        config_path : str
            Training YAML path used by ``main.py``.
        expected_dim : int, optional
            Expected p-vector dimensionality, by default 6.
        use_gpu : bool, optional
            Whether to request GPU execution, by default False.
        device_id : int, optional
            CUDA device id if ``use_gpu`` is enabled, by default 0.
        """
        super().__init__(
            name="hier_shplrnn_scratch",
            expected_dim=expected_dim,
            use_gpu=use_gpu,
            device_id=device_id,
        )
        self.config_path = config_path
        self._feature_cache: dict[str, np.ndarray] = {}

    def extract(self, signals: np.ndarray, dataset_name: str | None = None) -> np.ndarray:
        """Train from scratch and return learned p-vectors.

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
            self._feature_cache[cache_key] = self._extract_scratch_vectors(signals, dataset_name)

        features = self._feature_cache[cache_key]
        self._validate_p_vectors(features, signals, dataset_name)
        return np.copy(features)

    def _extract_scratch_vectors(self, signals: np.ndarray, dataset_name: str | None) -> np.ndarray:
        """Run main.py-equivalent scratch training on evaluation signals.

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
            raise ValueError("Expected at least one subject for scratch training, but received an empty tensor.")

        with tempfile.TemporaryDirectory(prefix="hier_shplrnn_scratch_") as temp_dir:
            data_path = Path(temp_dir) / f"{self.name}_{_dataset_token(dataset_name)}.pt"
            torch.save(torch.as_tensor(signals, dtype=torch.float32), data_path)

            args = self._build_scratch_args(data_path=str(data_path), dataset_name=dataset_name)
            dataset = self._build_dataset(args, signals.shape[0], dataset_name)

            trainer = BPTT(args, dataset)

            run_training = trainer.train
            if args.compile:
                run_training = torch.compile(run_training)
            run_training()

            self._finalize_training_artifacts(trainer, args.num_epochs)

            trained_model = cast(shallowPLRNN, trainer.model)
            features = cast_tensor(cast(torch.Tensor, trained_model.p_vector))

        return features

    def _build_scratch_args(self, data_path: str, dataset_name: str | None) -> Namespace:
        """Build runtime arguments exactly through ``main.py`` config flow.

        Parameters
        ----------
        data_path : str
            Temporary ``.pt`` path containing evaluation signals.
        dataset_name : str | None
            Optional dataset identifier.

        Returns
        -------
        Namespace
            Prepared scratch-training arguments.
        """
        args = Namespace(
            config=self.config_path,
            data_path=data_path,
            eval_data_path=data_path,
            save_path=None,
            experiment=None,
            name=None,
            run=None,
            finetune=False,
            model_path=None,
            checkpoint=None,
            use_gpu=self.use_gpu,
            device_id=self.device_id,
        )

        # Match main.py argument preparation exactly.
        args = apply_main_config(args)
        args = training_get_device(args)
        args = training_handle_defaults(args)

        args.data_path = data_path
        args.eval_data_path = data_path
        args.finetune = False
        args.model_path = None
        args.checkpoint = None

        if args.num_epochs <= 0:
            raise ValueError(
                f"Expected num_epochs > 0 from config '{self.config_path}', but got {args.num_epochs}."
            )
        if args.batch_size <= 0:
            raise ValueError(
                f"Expected batch_size > 0 from config '{self.config_path}', but got {args.batch_size}."
            )

        args = self._set_common_save_args(args, mode="scratch", dataset_name=dataset_name)

        print(
            "Running hier-shPLRNN scratch extraction with main.py flow: "
            f"config='{self.config_path}', dataset='{dataset_name}', num_epochs={args.num_epochs}, "
            f"batch_size={args.batch_size}, subjects_per_batch={args.subjects_per_batch}, "
            f"batches_per_epoch={args.batches_per_epoch}, save_root='{args.save_path}'.",
            flush=True,
        )
        return args
