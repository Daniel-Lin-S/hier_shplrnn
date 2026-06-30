import torch
from torch.utils.tensorboard.writer import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from argparse import Namespace
from typing import Any, Mapping


CHEAP_EVALS: frozenset[str] = frozenset({
    'mse',
    'dstsp',
    'pse',
})


class Saver:
    """
    Save checkpoints and TensorBoard artifacts during training.
    """

    def __init__(self, model: Any, args: Namespace, dataset: Any) -> None:
        """Initialise saver state and TensorBoard writer.

        Parameters
        ----------
        model : Any
            Trained model instance. Expected to expose ``state_dict()``,
            ``eval()``, ``evaluator``, and ``plotter`` interfaces.
        args : Namespace
            Runtime namespace. Accepted attributes:
            - ``save_path`` (type: str, meaning: output root path,
                default: ``'./trained_models'`` from ``main.py --save_path``)
            - ``experiment`` (type: str, meaning: experiment directory name,
                default: ``'experiment'`` from ``main.py --experiment``)
            - ``name`` (type: str, meaning: model directory name inside experiment,
                default: ``'name'`` from ``main.py --name``)
            - ``run`` (type: int, meaning: run folder index used as zero-padded
                directory name,
                default: ``1`` from ``main.py --run``)
            - ``metrics`` (type: list[str], meaning: enabled metric callbacks,
                default: ``['kl', 'pse']`` from ``configs/default.yaml``)
            - ``plots`` (type: list[str], meaning: enabled plot callbacks,
                default: ``['pow', 'hier']`` from ``configs/default.yaml``)
            - ``cheap_eval_interval`` (type: int, meaning: epoch cadence for
                cheap evaluations,
                default: ``100`` from ``evaluation.intervals.cheap`` in
                ``configs/default.yaml``)
            - ``expensive_eval_interval`` (type: int, meaning: epoch cadence for
                expensive evaluations,
                default: ``500`` from ``evaluation.intervals.expensive`` in
                ``configs/default.yaml``)
            - ``checkpoint_interval`` (type: int | None, meaning: epoch cadence
                for model checkpoint saving,
                default: ``100`` from ``training.optimization.checkpoint_interval``
                in ``configs/default.yaml``)
        dataset : Any
            Dataset object with ``num_subjects`` attribute.

        Raises
        ------
        ValueError
            If configured evaluation intervals are invalid.
        """
        self.model = model
        self.path = os.path.join(
            args.save_path, args.experiment, args.name, str(args.run).zfill(3))
        self.dataset = dataset
        self.num_subjects = dataset.num_subjects
        selected = get_attr_names(args)
        self.cheap, self.expensive = self._split_eval_groups(selected)

        # intervals for periodic saving and evaluation
        self.cheap_interval = args.cheap_eval_interval
        self.expensive_interval = args.expensive_eval_interval
        self.checkpoint_interval = args.checkpoint_interval
        self._validate_intervals()

        self.writer = SummaryWriter(self.path, purge_step=0)

        self.save_args(args)

    @staticmethod
    def _split_eval_groups(selected: list[str]) -> tuple[list[str], list[str]]:
        """Split selected evaluation names into cheap and expensive groups.

        Parameters
        ----------
        selected : list[str]
            Saver callback names requested by runtime args.

        Returns
        -------
        tuple[list[str], list[str]]
            Pair ``(cheap, expensive)`` where:
            - ``cheap`` includes only callbacks in the predefined cheap set.
            - ``expensive`` includes every selected callback not in ``cheap``.

        Notes
        -----
        This split controls cadence in :meth:`save_periodic`: expensive
        callbacks are preferred when both intervals trigger at the same epoch.
        """
        cheap = [name for name in CHEAP_EVALS if name in selected]
        expensive = [name for name in selected if name not in cheap]

        return cheap, expensive

    def _validate_intervals(self) -> None:
        """Validate checkpoint/evaluation interval configuration.

        Raises
        ------
        ValueError
            If cheap or expensive interval is missing or non-positive, or if
            checkpoint interval is provided but non-positive.
        """
        if self.cheap_interval is None or self.cheap_interval <= 0:
            raise ValueError(
                "Expected 'cheap_eval_interval' to be a positive integer, "
                f"but got {self.cheap_interval}."
            )
        if self.expensive_interval is None or self.expensive_interval <= 0:
            raise ValueError(
                "Expected 'expensive_eval_interval' to be a positive integer, "
                f"but got {self.expensive_interval}."
            )
        if self.checkpoint_interval is not None and self.checkpoint_interval <= 0:
            raise ValueError(
                "Expected 'checkpoint_interval' to be positive when provided, "
                f"but got {self.checkpoint_interval}."
            )
    
    def save_args(self, args: Namespace) -> None:
        """Save resolved runtime arguments to TensorBoard and ``hypers.txt``.

        Parameters
        ----------
        args : Namespace
            Runtime argument namespace to serialize.
        """
        # write to tensorboard
        args_dict = vars(args)
        args_text = ""
        for key, value in args_dict.items():
            args_text += f'{key}: {value}\n'
        self.writer.add_text('hypers', args_text)
        with open(os.path.join(self.path, 'hypers.txt'), 'w') as f:
            f.write(args_text)
    
    @torch.compiler.disable
    def save_cheap(self, epoch: int) -> None:
        """Saves everything that is cheap to compute. So that it
        can be done frequently."""
        if len(self.cheap) == 0:
            return

        self.model.eval()
        eval_start = time.time()
        self.model.evaluator.compute_cheap(self.cheap)
        eval_elapsed = time.time() - eval_start
        print(f"--- Cheap metric computation took {eval_elapsed:.2f}s ---", flush=True)

        if 'trajectory' in self.cheap:
            self.save_trajectory(epoch)

        for name in self.cheap:
            if name == 'trajectory':
                continue
            getattr(self, f'save_{name}')(epoch)
    
    @torch.compiler.disable
    def save_expensive(self, epoch: int) -> None:
        """Computes both expensive and cheap stuff. So as to only
        be called every now and then."""
        if len(self.expensive) == 0:
            return

        self.model.eval()
        eval_start = time.time()
        self.model.evaluator.compute_expensive(self.expensive)
        eval_elapsed = time.time() - eval_start
        print(f"--- Expensive metric computation took {eval_elapsed:.2f}s ---", flush=True)

        if 'trajectory' in self.expensive:
            self.save_trajectory(epoch)

        for name in self.expensive:
            if name == 'trajectory':
                continue
            getattr(self, f'save_{name}')(epoch)
    
    @torch.compiler.disable
    def save_mse(self, epoch: int) -> None:
        self.writer.add_scalar('mse/5_step', self.model.evaluator.get_n_step_mse(5), epoch)
        self.writer.add_scalar('mse/10_step', self.model.evaluator.get_n_step_mse(10), epoch)
        self.writer.add_scalar('mse/15_step', self.model.evaluator.get_n_step_mse(15), epoch)
    
    @torch.compiler.disable
    def save_scyfi(self, epoch: int) -> None:
        if self.model.dx > 3:
            return
        fig = self.model.plotter.plot_fixed_points()
        if fig is not None:
            self.writer.add_figure(f'fixed_points', fig, epoch)
            plt.close(fig)
    
    @torch.compiler.disable
    def save_loss(self, epoch: int, losses: Mapping[str, float]) -> None:
        for key, val in losses.items():
            self.writer.add_scalar(f'loss/{key}', val, epoch)
    
    @torch.compiler.disable
    def save_periodic(self, epoch: int, is_final: bool = False) -> None:
        """Saves model and computes metrics/plots periodically.
        Args:
            epoch: current epoch
            is_final: whether this is the final epoch
        """
        # 1. Checkpoint saving
        if is_final:
            self.save_model(epoch)
        elif self.checkpoint_interval is not None and epoch % self.checkpoint_interval == 0:
            self.save_model(epoch)

        # 2. Evaluation
        if epoch % self.expensive_interval == 0:
            self.save_expensive(epoch)
        elif epoch % self.cheap_interval == 0:
            self.save_cheap(epoch)

    @torch.compiler.disable
    def save_pse(self, epoch: int) -> None:
        pses = self.model.evaluator.get_pse()
        for i, pse in enumerate(pses):
            self.writer.add_scalar(f"PSE/{i}", pse, epoch)
        self.writer.add_scalar('mean_metrics/PSE', np.mean(pses), epoch)
        self.writer.add_scalar('median_metrics/PSE', np.median(pses), epoch)
    
    @torch.compiler.disable
    def save_dstsp(self, epoch: int) -> None:
        Ds = self.model.evaluator.get_state_space_divergence()
        for i, d in enumerate(Ds):
            self.writer.add_scalar(f"D_stsp/{i}", d, epoch)
        self.writer.add_scalar('mean_metrics/D_stsp', torch.mean(Ds), epoch)
        self.writer.add_scalar('median_metrics/D_stsp', torch.median(Ds), epoch)
    
    @torch.compiler.disable
    def save_hierarchisation_plots(self, epoch: int) -> None:
        """Saves the plots given from the hierarchisation scheme."""
        for fig, name in self.model.plotter.plot_hierarchisation_stuff():
            self.writer.add_figure(name, fig, epoch)
            plt.close(fig)
    
    @torch.compiler.disable
    def save_power_spectrum(self, epoch: int) -> None:
        """Saves the power spectrum of the test trajectory
        and the generated trajectory of same length.
        Args:
            epoch: epoch at which the power spectrum is saved
        """
        fig = self.model.plotter.plot_power_spectrum()
        self.writer.add_figure(f'power_spectrum', fig, epoch)
        plt.close(fig)
    
    @torch.compiler.disable
    def save_trajectory(self, epoch: int) -> None:
        """Saves a test trajectory and a generated trajectory
        of same length.
        Args:
            epoch: epoch at which the trajectory is saved
            ground_truth: ground truth trajectory
            subject: subject to save the trajectory for
        """
        fig = self.model.plotter.plot_trajectory()
        self.writer.add_figure(f'trajectory', fig, epoch)
        plt.close(fig)

    @torch.compiler.disable
    def save_3D_trajectory(self, epoch: int) -> None:
        """Saves a test trajectory and a generated trajectory
        of same length.
        Args:
            epoch: epoch at which the trajectory is saved
        """
        if self.model.dx > 3:
            return
        fig = self.model.plotter.plot_3D_trajectory()
        if fig is not None:
            self.writer.add_figure(f'3D_trajectory', fig, epoch)
            plt.close(fig)

    @torch.compiler.disable
    def save_hovmoller(self, epoch: int) -> None:
        """Saves the hovmoller diagram of the test trajectory
        and the generated trajectory of same length.
        Args:
            epoch: epoch at which the hovmoller diagram is saved
        """
        fig = self.model.plotter.plot_hovmoller()
        self.writer.add_figure(f'hovmöller', fig, epoch)
        plt.close(fig)
    
    @torch.compiler.disable
    def save_trajectory_train(self, epoch: int) -> None:
        """Copy of save_3D_trajectory but uses a train istance.
        Args:
            epoch: epoch at which the trajectory is saved
        """
        fig = self.model.plotter.plot_trajectory_train()
        self.writer.add_figure(f'trajectory_train', fig, epoch)
        plt.close(fig)
    
    @torch.compiler.disable
    def save_model(self, epoch: int) -> None:
        """Saves the model.
        Args:
            epoch: epoch at which the model is saved
        """
        torch.save(self.model.state_dict(), os.path.join(self.path, f'model_{epoch}.pt'))


def get_attr_names(args: Namespace) -> list[str]:
    """Resolve saver callback names from configured metrics and plots.

    Parameters
    ----------
    args : Namespace
        Runtime configuration namespace.
        Accepted attributes:
        - ``metrics`` (type: list[str], meaning: enabled evaluation metrics,
            default: ``['kl', 'pse']`` from ``configs/default.yaml``)
        - ``plots`` (type: list[str], meaning: enabled evaluation plots,
            default: ``['pow', 'hier']`` from ``configs/default.yaml``)

    Returns
    -------
    list[str]
        Saver callback names (for example ``'dstsp'``, ``'power_spectrum'``)
        used to call ``save_<name>`` methods.
    """
    names = []
    if 'kl' in args.metrics:
        names.append('dstsp')
    if 'pse' in args.metrics:
        names.append('pse')
    if 'mse' in args.metrics:
        names.append('mse')
    if 'scyfi' in args.metrics:
        names.append('scyfi')
    if 'hovmoller' in args.plots:
        names.append('hovmoller')
    if '3D' in args.plots:
        names.append('3D_trajectory')
    if 'pow' in args.plots:
        names.append('power_spectrum')
    if 'hier' in args.plots:
        names.append('hierarchisation_plots')
    if 'trajectory' in args.plots:
        names.append('trajectory')
    return names
