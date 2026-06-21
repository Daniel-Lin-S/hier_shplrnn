# Learning Interpretable Hierarchical Dynamical Systems Models From Time Series Data [[ICLR 2025]](https://openreview.net/forum?id=Vp2OAxMs2s)

## Requirements

We include the `requirements.txt` file to clone our python environment. Simply run
```
pip install -r requirements.txt
```

The training and evaluation entry points now read hierarchical YAML configuration files, so
`PyYAML` is required (already included in `requirements.txt`).

## Data Format (`.pt`)

Both `main.py` (training) and `main_eval.py` (evaluation) expect `.pt` files that contain a
single `torch.Tensor` with one of these shapes:

1. `(num_subjects, timesteps, num_features)`
2. `(timesteps, num_features)` for a single-subject case (automatically expanded to 3D)

### Training data (`--data_path`)

1. Tensor must contain floating-point observations.
2. The last dimension must match `model.dimensions.obs_size` from the YAML config.
3. The time axis must be long enough for `training.sequence.seq_len` and
     `training.sequence.train_set_size`.

### Evaluation data (`--eval_data_path`)

1. Same shape contract as training data.
2. **Structural Requirement**: The number of subjects ($S$) and number of features ($F$) must strictly match the training set. The model assumes a 1-to-1 mapping between the $i$-th subject in training and the $i$-th subject in evaluation.
3. A longer trajectory is recommended for state-space divergence (`kl`) metrics.

### Minimal example

```python
import torch

tensor = torch.randn(64, 2000, 3)  # (subjects, timesteps, features)
torch.save(tensor, "./data/my_dataset/noisy.pt")
```

## Single-Source Configuration

`main.py` and `main_eval.py` now use YAML as the **single source of model/training/evaluation parameters**.

Parser arguments are only for run-specific I/O/runtime settings (for example `data_path`,
`save_path`, `eval_data_path`, `model_path`, `use_gpu`).

## Configuration Reference

All keys below are defined in `configs/default.yaml`.

### `model`

- `model.dimensions.obs_size` (`int`): Observation feature dimension `dx` (number of features in the input).
- `model.dimensions.latent_size` (`int | null`): Latent dimension `dz`; `null` defaults to `obs_size`.
- `model.dimensions.hidden_size` (`int`): Hidden state dimension `dh` of the shallow PLRNN, determining the capacity of the non-linear transition.
- `model.dimensions.forcing_size` (`int | null`): Number of latent dimensions that are teacher-forced during training; `null` means all `dz` dimensions.
- `model.observation.obs_model` (`str`): The type of mapping from latent space to observation space:
  - `identity`: Uses a fixed identity matrix (or rectangular identity if `dz != dx`); latent dimensions correspond directly to observations.
  - `linear`: Uses a learnable weight matrix to project latent states to observations.
- `model.observation.clipped` (`bool`): If `true`, uses the clipped ReLU variant of the PLRNN state update for potentially more stable long-term dynamics.
- `model.observation.learn_noise_cov` (`bool`): If `true`, the diagonal noise covariance matrix $\Sigma$ is learned alongside other model parameters.
- `model.hierarchy.scheme` (`str`): The strategy for parameterizing differences between subjects:
  - `projection`: Subject-specific parameters are derived from a low-dimensional individual vector projected through shared matrices.
  - `baseline`: Subject-specific parameters are learned independently for each subject without a shared hierarchical structure.
- `model.hierarchy.num_individual_params` (`int`): Dimension of the individual parameter vector used when the `projection` scheme is active.
- `model.hierarchy.lambda_regularization` (`float`): Regularization weight $\lambda$ for hierarchical losses (e.g., penalties on the individual parameter vectors).

### `training`

- `training.sequence.seq_len` (`int`): Length of trajectory segments sampled for each BPTT training step.
- `training.sequence.train_set_size` (`int`): Total number of timepoints from the beginning of each subject's data reserved for training.
- `training.optimization.num_epochs` (`int`): Total number of passes over the training dataset.
- `training.optimization.batch_size` (`int`): Number of sequences processed in parallel per gradient update.
- `training.optimization.batches_per_epoch` (`int | null`): Number of gradient steps per epoch; `null` defaults to `dataset_size / batch_size`.
- `training.optimization.subjects_per_batch` (`int | null`): Number of unique subjects sampled in each batch; `null` uses all available subjects.
- `training.optimization.num_workers` (`int | null`): Number of parallel threads for data loading; `null` enables an automated performance search.
- `training.optimization.weight_decay` (`float`): Strength of L2 regularization applied specifically to shared (non-subject-specific) parameters.
- `training.optimization.clip_grad_norm` (`float`): Maximum allowed norm for gradients; values above 0 trigger clipping to prevent exploding gradients.
- `training.optimization.learning_rate.shared` (`float`): Step size for optimizing shared model parameters.
- `training.optimization.learning_rate.individual` (`float | null`): Step size for subject-specific parameters; `null` reuses the `shared` learning rate.
- `training.teacher_forcing.alpha_start` (`float`): Initial strength of teacher forcing (ground-truth injection) at the start of training.
- `training.teacher_forcing.alpha_end` (`float | null`): Final strength of teacher forcing; if not `null`, $\alpha$ is linearly decayed from `alpha_start`. If not provided, alpha is constant during training.

### `evaluation`

- `evaluation.metrics.enabled` (`list[str]`): List of quantitative measures to calculate:
  - `kl`: State-space divergence. Measures the similarity between ground-truth and generated attractors using histogram binning or GMM.
  - `pse`: Power Spectrum Error. Measures the Hellinger distance between power spectral densities.
  - `mse`: Mean Squared Error. Calculates n-step predictive accuracy of the model.
  - `scyfi`: Fixed Point analysis. Extracts and analyzes the stability of dynamical fixed points (requires `dz <= 3`).
- `evaluation.metrics.kl_bins` (`int`): Number of bins per dimension for the `kl` metric; `0` switches to the GMM-based divergence.
- `evaluation.metrics.pse_smooth` (`int`): Standard deviation for Gaussian smoothing of the power spectrum before computing `pse`.
- `evaluation.plots.enabled` (`list[str]`): List of visualizations to generate:
  - `pow`: Comparison of power spectra between simulated and real data.
  - `hier`: Visualization of hierarchisation parameters (e.g., the individual vectors $p_i$).
  - `3D`: 3D rendering of generated trajectories (requires `dx <= 3`).
  - `hovmoller`: Space-time heatmap (Hovmöller diagram) of the trajectory.

### `runtime`

- `runtime.compile` (`bool`): Enable `torch.compile` for model/training callable.

## TensorBoard Visualisation

Training progress and model diagnostics are logged to TensorBoard. Panels are organized by metric category:

### Scalar Panels
- **`loss/`**:
  - `rnn`: Negative log-likelihood of the observations given the latent states and noise covariance.
  - `hier`: Regularization loss from the hierarchisation scheme (e.g., penalties on subject-specific parameters).
- **`mean_metrics/` & `median_metrics/`**: Aggregate dynamical performance across all subjects.
  - `PSE`: Average/Median Power Spectrum Error. 0 is perfect reconstruction.
  - `D_stsp`: Average/Median State-Space Divergence (KL). Lower is better.
- **`PSE/` & `D_stsp/`**: Subject-specific metric values (indexed 0 to S-1). Useful for identifying outliers.
- **`tf_alpha`**: Current value of the teacher-forcing interpolation coefficient.
- **`lr/`**: Learning rate values for shared and individual parameter groups.

### Image & Figure Panels
- **`trajectory`**: Overlay of generated vs. ground-truth observation trajectories for each subject.
- **`power_spectrum`**: Comparison of the frequency content of generated vs. real data.
- **`hovmöller`**: Heatmap visualizing the temporal evolution of all features simultaneously.
- **`3D_trajectory`**: 3D phase-space plots (only for `obs_size <= 3`).
- **`fixed_points`**: Stability analysis of the system's dynamical equilibria (only for `latent_size <= 3`).
- **`noise_covariance`**: Heatmap showing the learned noise level per subject/feature.

## Usage

### Training

```bash
python main.py \
    --config ./configs/default.yaml \
    --data_path ./data/lorenz63/3params64sub/noisy.pt \
    --eval_data_path ./data/lorenz63/3params64sub/full.pt \
    --save_path ./trained_models \
    --experiment lorenz63 \
    --name projection \
    --run 1
```

Running multiple trainings, potentially in parallel, can still be done via:

```bash
python ubermain.py
```

### Evaluation

```bash
python main_eval.py \
    --config ./configs/default.yaml \
    --model_path ./trained_models/lorenz63/projection \
    --save_path ./results/lorenz63
```

## Dataset

To use a dataset from the Time Series Classification benchmark, please use `data_io/aeon_to_pt.py` to convert 

```bash
python data_io/aeon_to_pt.py \
    --dataset_name EpilepticSeizures \
    --split all \
    --output_path ./data/epileptic/seizures_all.pt \
    --save_auxiliary
```

## Citation

If you find the repository and/or paper helpful for your own research, please cite our work.
```
@inproceedings{
    brenner2025learning,
    title={Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data},
    author={Manuel Brenner and Elias Weber and Georgia Koppe and Daniel Durstewitz},
    booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
    year={2025},
    url={https://openreview.net/forum?id=Vp2OAxMs2s}
}
```

## Acknowledgements

This work was funded by the European Union’s Horizon 2020 programme under grant agreement 945263 (IMMERSE), by the German Ministry for Education \& Research (BMBF) within the FEDORA (01EQ2403F) consortium, by the Federal Ministry of Science, Education, and Culture (MWK) of the state of Baden-Württemberg within the AI Health Innovation Cluster Initiative and living lab (grant number 31-7547.223-7/3/2), by the German Research Foundation (DFG) within the collaborative research center TRR-265 (project A06  \& B08) and by the Hector-II foundation.