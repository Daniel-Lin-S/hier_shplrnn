"""Hierarchical shPLRNN feature extractor."""

from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from data_io.dataset import MultiSubjectDataset
from models.feature_extractors.base import LatentFeatureExtractor
from models.feature_extractors.utils import (
    cast_tensor,
    clean_state_dict_keys,
    resolve_checkpoint_path,
    resolve_individual_learning_rate,
    validate_feature_matrix,
)
from models.hier_shplrnn import nll_loss, shallowPLRNN
from trainers.bptt import load_from_path, read_hypers


class HierShPLRNNFinetunedPVectorExtractor(LatentFeatureExtractor):
    """Extractor that uses shPLRNN individual parameter vectors (p-vectors).

    The training split uses stored p-vectors from the checkpoint when available.
    The test split is obtained by finetuning individual vectors on the new test
    samples, as required by the benchmark protocol.
    """

    def __init__(
        self,
        model_path: str,
        expected_dim: int = 6,
        finetune_epochs: int = 20,
        finetune_batches_per_epoch: int = 8,
        finetune_num_workers: int = 0,
    ) -> None:
        """Initialize the hierarchical p-vector extractor.

        Parameters
        ----------
        model_path : str
            Path to a trained shPLRNN run directory or checkpoint file.
        expected_dim : int, optional
            Expected p-vector dimensionality, by default 6.
        finetune_epochs : int, optional
            Number of finetuning epochs for new split adaptation, by default 20.
        finetune_batches_per_epoch : int, optional
            Number of batches per finetuning epoch, by default 8.
        finetune_num_workers : int, optional
            DataLoader worker count for finetuning, by default 0.
        """
        self.name = "hier_shplrnn_finetuned"
        self.model_path = model_path
        self.expected_dim = expected_dim
        self.finetune_epochs = finetune_epochs
        self.finetune_batches_per_epoch = finetune_batches_per_epoch
        self.finetune_num_workers = finetune_num_workers

        self._train_signals_cache: np.ndarray | None = None
        self._train_features_cache: np.ndarray | None = None
        self._runtime_output_dir: Path | None = None

    def set_runtime_output_dir(self, output_dir: str) -> None:
        """Set benchmark runtime output directory.

        Parameters
        ----------
        output_dir : str
            Extractor artifact directory for the active benchmark run.
        """
        self._runtime_output_dir = Path(output_dir)

    def fit(self, train_signals: np.ndarray, train_labels: np.ndarray) -> None:
        """Load or derive training split p-vectors.

        Parameters
        ----------
        train_signals : np.ndarray
            Training EEG tensor.
        train_labels : np.ndarray
            Training labels.
        """
        del train_labels

        checkpoint_path = resolve_checkpoint_path(self.model_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cleaned_state = clean_state_dict_keys(state_dict)

        if "p_vector" in cleaned_state:
            train_features = cast_tensor(cleaned_state["p_vector"])
            if train_features.ndim != 2:
                raise ValueError(
                    "Expected checkpoint 'p_vector' to be 2D, "
                    f"but got shape {train_features.shape} in '{checkpoint_path}'."
                )

            if train_features.shape[0] == train_signals.shape[0]:
                self._train_features_cache = train_features
            else:
                print(
                    "Checkpoint p-vector subject count does not match benchmark train split. "
                    "Deriving training p-vectors via finetuning instead.",
                    flush=True,
                )
                self._train_features_cache = self._finetune_subject_vectors(
                    train_signals,
                    split_name="train",
                )
        else:
            print(
                "Checkpoint has no p_vector entry. Deriving training p-vectors via finetuning.",
                flush=True,
            )
            self._train_features_cache = self._finetune_subject_vectors(
                train_signals,
                split_name="train",
            )

        if self._train_features_cache.shape[1] != self.expected_dim:
            raise ValueError(
                "Unexpected p-vector dimensionality for train split: "
                f"expected {self.expected_dim}, got {self._train_features_cache.shape[1]}."
            )

        self._train_signals_cache = np.copy(train_signals)
        validate_feature_matrix(self._train_features_cache, f"{self.name}_train")

    def transform(self, signals: np.ndarray, split_name: str) -> np.ndarray:
        """Return p-vectors for the requested split.

        Parameters
        ----------
        signals : np.ndarray
            EEG tensor for the requested split.
        split_name : str
            Split identifier.

        Returns
        -------
        np.ndarray
            P-vector feature matrix with dimension ``expected_dim``.
        """
        if split_name == "train":
            if self._train_features_cache is None or self._train_signals_cache is None:
                raise RuntimeError("Train features are not available. Call fit before transform.")

            if signals.shape != self._train_signals_cache.shape:
                raise ValueError(
                    "Train split shape changed between fit and transform: "
                    f"fit_shape={self._train_signals_cache.shape}, transform_shape={signals.shape}."
                )
            return np.copy(self._train_features_cache)

        if split_name != "test":
            raise ValueError(f"Unsupported split_name '{split_name}'. Expected 'train' or 'test'.")

        test_features = self._finetune_subject_vectors(signals, split_name="test")
        if test_features.shape[1] != self.expected_dim:
            raise ValueError(
                "Unexpected p-vector dimensionality for test split: "
                f"expected {self.expected_dim}, got {test_features.shape[1]}."
            )
        validate_feature_matrix(test_features, f"{self.name}_test")
        return test_features

    def _finetune_subject_vectors(self, signals: np.ndarray, split_name: str) -> np.ndarray:
        """Finetune individual vectors on a split and return p-vectors.

        Parameters
        ----------
        signals : np.ndarray
            Split EEG tensor.
        split_name : str
            Split identifier for logging.

        Returns
        -------
        np.ndarray
            Finetuned p-vector matrix.
        """
        if split_name not in {"train", "test"}:
            raise ValueError(
                f"split_name must be either 'train' or 'test', but got '{split_name}'."
            )

        monitor_root = self._resolve_monitor_root()
        with tempfile.TemporaryDirectory(prefix="hier_finetune_") as tmp_dir:
            temp_data_path = Path(tmp_dir) / "finetune_split.pt"
            torch.save(torch.as_tensor(signals, dtype=torch.float32), temp_data_path)

            model_args = self._build_finetune_args(
                data_path=str(temp_data_path),
                num_subjects=signals.shape[0],
            )
            dataset = MultiSubjectDataset(
                model_args.data_path,
                model_args.seq_len,
                model_args.train_set_size,
                model_args.subjects_per_batch,
                model_args.num_workers,
                model_args.device,
            )
            if dataset.num_subjects != signals.shape[0]:
                raise ValueError(
                    "Finetune dataset subject count mismatch: "
                    f"dataset.num_subjects={dataset.num_subjects}, "
                    f"signals.shape[0]={signals.shape[0]}, split='{split_name}'."
                )

            model = shallowPLRNN(model_args, dataset)
            load_args = Namespace(model_path=self.model_path, finetune=True)
            load_from_path(model, load_args)

            history_df = self._run_finetune_loop(
                model=model,
                dataset=dataset,
                args=model_args,
                split_name=split_name,
                monitor_root=monitor_root,
            )
            model.saver.writer.close()

            p_vector = cast_tensor(model.p_vector)

        history_path = monitor_root / f"finetune_history_{split_name}.csv"
        history_df.to_csv(history_path, index=False)
        print(
            f"Saved hier finetune history for split '{split_name}' to '{history_path}'.",
            flush=True,
        )

        if p_vector.shape[0] != signals.shape[0]:
            raise ValueError(
                "Finetuned p-vector count does not match input sample count: "
                f"features={p_vector.shape[0]}, samples={signals.shape[0]}."
            )
        return p_vector

    def _build_finetune_args(self, data_path: str, num_subjects: int) -> Namespace:
        """Build finetuning argument namespace from checkpoint hypers.

        Parameters
        ----------
        data_path : str
            Path to temporary split tensor.
        num_subjects : int
            Number of subjects in the active finetune split.

        Returns
        -------
        Namespace
            Configured finetuning namespace.
        """
        args = read_hypers(Namespace(model_path=self.model_path))
        args.model_path = self.model_path
        args.finetune = True
        args.use_gpu = False
        args.device_id = 0
        args.device = "cpu"
        args.compile = False

        args.data_path = data_path
        args.eval_data_path = data_path

        args.experiment = "latent_benchmark"
        args.name = "hier_finetune"
        args.run = 1
        args.save_path = str(Path(data_path).parent)

        args.num_epochs = self.finetune_epochs
        args.batches_per_epoch = self.finetune_batches_per_epoch
        args.num_workers = self.finetune_num_workers

        if num_subjects <= 0:
            raise ValueError(f"Expected num_subjects > 0 for finetuning, but got {num_subjects}.")

        args.subjects_per_batch = int(num_subjects)
        args.batch_size = int(4 * num_subjects)
        return args

    def _run_finetune_loop(
        self,
        model: shallowPLRNN,
        dataset: MultiSubjectDataset,
        args: Namespace,
        split_name: str,
        monitor_root: Path,
    ) -> pd.DataFrame:
        """Finetune only individual parameters on the provided split.

        Parameters
        ----------
        model : shallowPLRNN
            Instantiated model.
        dataset : MultiSubjectDataset
            Dataset bound to the split to adapt on.
        args : Namespace
            Finetuning arguments.
        split_name : str
            Split identifier for logging.
        monitor_root : Path
            Root directory where finetune progress artifacts are saved.

        Returns
        -------
        pd.DataFrame
            Epoch-level finetune loss history.
        """
        shared, individual = model.hierarchisation_scheme.grouped_parameters()
        shared_snapshots = [parameter.detach().cpu().clone() for parameter in shared]
        for shared_param in shared:
            shared_param.requires_grad_(False)

        individual_lr = resolve_individual_learning_rate(args.learning_rate)
        optimizer = Adam(individual, lr=individual_lr)

        monitor_root.mkdir(parents=True, exist_ok=True)
        log_dir = monitor_root / f"{split_name}_finetune"
        writer = SummaryWriter(log_dir=str(log_dir))

        history_rows: list[dict[str, float]] = []
        global_step = 0

        model.train()
        for epoch in range(args.num_epochs):
            model.hierarchisation_scheme.step = epoch + 1
            dataloader = dataset.get_dataloader(args.batch_size, args.batches_per_epoch)

            epoch_rnn_loss = 0.0
            epoch_hier_loss = 0.0
            epoch_loss = 0.0
            batch_count = 0
            for data, target, subject in dataloader:
                data = data.to(args.device)
                target = target.to(args.device)
                subject = subject.to(args.device)

                prediction = model(data, subject)
                optimizer.zero_grad()

                rnn_loss = nll_loss(prediction, target, model.noise_cov[subject])
                hier_loss = torch.tensor(0.0, device=args.device)
                if args.lam > 0:
                    hier_loss = args.lam * model.hierarchisation_scheme.loss()

                total_loss = rnn_loss + hier_loss
                total_loss.backward()
                optimizer.step()

                batch_count += 1
                batch_rnn_loss = float(rnn_loss.item())
                batch_hier_loss = float(hier_loss.item())
                batch_total_loss = float(total_loss.item())

                epoch_rnn_loss += batch_rnn_loss
                epoch_hier_loss += batch_hier_loss
                epoch_loss += batch_total_loss

                writer.add_scalar(f"{split_name}/batch_rnn_loss", batch_rnn_loss, global_step)
                writer.add_scalar(f"{split_name}/batch_hier_loss", batch_hier_loss, global_step)
                writer.add_scalar(f"{split_name}/batch_total_loss", batch_total_loss, global_step)
                global_step += 1

            model.tf_alpha *= model.tf_gamma

            if batch_count == 0:
                raise RuntimeError(
                    "Hier finetune loop produced zero batches; cannot compute optimization progress. "
                    f"batch_size={args.batch_size}, batches_per_epoch={args.batches_per_epoch}, "
                    f"dataset_size={len(dataset)}, split='{split_name}'."
                )

            mean_rnn_loss = epoch_rnn_loss / batch_count
            mean_hier_loss = epoch_hier_loss / batch_count
            mean_total_loss = epoch_loss / batch_count
            history_rows.append(
                {
                    "epoch": float(epoch + 1),
                    "mean_rnn_loss": mean_rnn_loss,
                    "mean_hier_loss": mean_hier_loss,
                    "mean_total_loss": mean_total_loss,
                }
            )

            writer.add_scalar(f"{split_name}/epoch_rnn_loss", mean_rnn_loss, epoch + 1)
            writer.add_scalar(f"{split_name}/epoch_hier_loss", mean_hier_loss, epoch + 1)
            writer.add_scalar(f"{split_name}/epoch_total_loss", mean_total_loss, epoch + 1)

        writer.flush()
        writer.close()

        self._assert_shared_parameters_frozen(shared, shared_snapshots)
        print(
            f"Saved hier finetune TensorBoard logs for split '{split_name}' under '{log_dir}'.",
            flush=True,
        )
        return pd.DataFrame(history_rows)

    def _resolve_monitor_root(self) -> Path:
        """Resolve persistent monitor root for finetune logging artifacts.

        Returns
        -------
        Path
            Directory where finetune logs and histories are written.
        """
        if self._runtime_output_dir is not None:
            monitor_root = self._runtime_output_dir
        else:
            monitor_root = Path("./results/epileptic_latent_benchmark/models/hier_shplrnn_finetuned")

        monitor_root.mkdir(parents=True, exist_ok=True)
        return monitor_root

    def _assert_shared_parameters_frozen(
        self,
        shared_parameters: list[torch.nn.Parameter],
        shared_snapshots: list[torch.Tensor],
    ) -> None:
        """Assert that frozen shared parameters remained unchanged during finetuning.

        Parameters
        ----------
        shared_parameters : list[torch.nn.Parameter]
            Shared parameters after finetuning.
        shared_snapshots : list[torch.Tensor]
            Pre-finetune copies of shared parameters.
        """
        for index, (parameter, snapshot) in enumerate(zip(shared_parameters, shared_snapshots), start=1):
            current = parameter.detach().cpu()
            if not torch.equal(current, snapshot):
                raise RuntimeError(
                    "Detected updates in frozen shared parameter group during finetuning: "
                    f"parameter_index={index}."
                )
