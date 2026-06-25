import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
import os
import time


def get_attr_names(args):
    """Returns the names of the function of the saver, plotter and evaluator classes
    that need to be executed."""
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


class Saver:
    """Class for saving the results of the model."""
    def __init__(self, model, args, dataset):
        """Initializes the saver.
        Args:
            model: model to save the results for
            args: args from the command line
            dataset: dataset to save the results for
        """
        self.model = model
        self.path = os.path.join(args.save_path, args.experiment, args.name, str(args.run).zfill(3))
        self.dataset = dataset
        self.num_subjects = dataset.num_subjects
        selected = get_attr_names(args)
        self.cheap, self.expensive = self._split_eval_groups(selected)

        # intervals for periodic saving and evaluation
        self.cheap_interval = args.cheap_eval_interval
        self.expensive_interval = args.expensive_eval_interval
        self.checkpoint_interval = args.checkpoint_interval

        # initialise tensorboard writer
        self.writer = SummaryWriter(self.path, purge_step=0)
        # save args
        self.save_args(args)

    @staticmethod
    def _split_eval_groups(selected: list[str]) -> tuple[list[str], list[str]]:
        """Split selected evaluation names into cheap and expensive groups.

        Parameters
        ----------
        selected : list[str]
            Metrics and plots requested by runtime args.

        Returns
        -------
        tuple[list[str], list[str]]
            Cheap and expensive evaluation names.
        """
        cheap_candidates = ['mse']
        cheap = [name for name in cheap_candidates if name in selected]
        expensive = [name for name in selected if name not in cheap]
        return cheap, expensive
    
    def save_args(self, args):
        """Saves the args.
        Args:
            args: args to save
        """
        # write to tensorboard
        dict = vars(args)
        str = ""
        for key, value in dict.items():
            str += f'{key}: {value}\n'
        self.writer.add_text('hypers', str)
        with open(os.path.join(self.path, 'hypers.txt'), 'w') as f:
            f.write(str)
    
    @torch.compiler.disable()
    def save_cheap(self, epoch):
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
    
    @torch.compiler.disable()
    def save_expensive(self, epoch):
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
    
    @torch.compiler.disable()
    def save_mse(self, epoch):
        self.writer.add_scalar('mse/5_step', self.model.evaluator.get_n_step_mse(5), epoch)
        self.writer.add_scalar('mse/10_step', self.model.evaluator.get_n_step_mse(10), epoch)
        self.writer.add_scalar('mse/15_step', self.model.evaluator.get_n_step_mse(15), epoch)
    
    @torch.compiler.disable()
    def save_scyfi(self, epoch):
        if self.model.dx > 3:
            return
        fig = self.model.plotter.plot_fixed_points()
        if fig is not None:
            self.writer.add_figure(f'fixed_points', fig, epoch)
            plt.close(fig)
    
    @torch.compiler.disable()
    def save_loss(self, epoch, losses):
        for key, val in losses.items():
            self.writer.add_scalar(f'loss/{key}', val, epoch)
    
    @torch.compiler.disable()
    def save_periodic(self, epoch, is_final=False):
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

    @torch.compiler.disable()
    def save_pse(self, epoch):
        pses = self.model.evaluator.get_pse()
        for i, pse in enumerate(pses):
            self.writer.add_scalar(f"PSE/{i}", pse, epoch)
        self.writer.add_scalar('mean_metrics/PSE', np.mean(pses), epoch)
        self.writer.add_scalar('median_metrics/PSE', np.median(pses), epoch)
    
    @torch.compiler.disable()
    def save_dstsp(self, epoch):
        Ds = self.model.evaluator.get_state_space_divergence()
        for i, d in enumerate(Ds):
            self.writer.add_scalar(f"D_stsp/{i}", d, epoch)
        self.writer.add_scalar('mean_metrics/D_stsp', torch.mean(Ds), epoch)
        self.writer.add_scalar('median_metrics/D_stsp', torch.median(Ds), epoch)
    
    @torch.compiler.disable()
    def save_hierarchisation_plots(self, epoch):
        """Saves the plots given from the hierarchisation scheme."""
        for fig, name in self.model.plotter.plot_hierarchisation_stuff():
            self.writer.add_figure(name, fig, epoch)
            plt.close(fig)
    
    @torch.compiler.disable()
    def save_power_spectrum(self, epoch):
        """Saves the power spectrum of the test trajectory
        and the generated trajectory of same length.
        Args:
            epoch: epoch at which the power spectrum is saved
        """
        fig = self.model.plotter.plot_power_spectrum()
        self.writer.add_figure(f'power_spectrum', fig, epoch)
        plt.close(fig)
    
    @torch.compiler.disable()
    def save_trajectory(self, epoch):
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

    @torch.compiler.disable()
    def save_3D_trajectory(self, epoch):
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

    @torch.compiler.disable()
    def save_hovmoller(self, epoch):
        """Saves the hovmoller diagram of the test trajectory
        and the generated trajectory of same length.
        Args:
            epoch: epoch at which the hovmoller diagram is saved
        """
        fig = self.model.plotter.plot_hovmoller()
        self.writer.add_figure(f'hovmöller', fig, epoch)
        plt.close(fig)
    
    @torch.compiler.disable()
    def save_trajectory_train(self, epoch):
        """Copy of save_3D_trajectory but uses a train istance.
        Args:
            epoch: epoch at which the trajectory is saved
        """
        fig = self.model.plotter.plot_trajectory_train()
        self.writer.add_figure(f'trajectory_train', fig, epoch)
        plt.close(fig)
    
    @torch.compiler.disable()
    def save_model(self, epoch):
        """Saves the model.
        Args:
            epoch: epoch at which the model is saved
        """
        torch.save(self.model.state_dict(), os.path.join(self.path, f'model_{epoch}.pt'))