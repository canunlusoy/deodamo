import logging
from typing import Callable, Union, Type, ClassVar, Iterable
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from utils.iox import ProgramData
from datamodels.variables import Variable
from mdev.utilities import activation_fcns_by_key, optimizers_by_key, data_container_row_index, NoiseGenerator, MDevDataset, get_fc_network
from mdev.generative.base import ConditionalGenerativeModel, Validator


logger = logging.getLogger(__name__)


class CRNVP(ConditionalGenerativeModel):
    """Conditional RealNVP (Real-Valued Non-Volume-Preserving Model)"""

    MODEL_TYPE_ABBREVIATION = 'CRNVP'
    MODEL_TYPE_NAME_LONG = 'Conditional RealNVP'

    KEY_LOSS_MAP = 'MAP'
    """Key for maximum a priori loss"""

    @dataclass
    class ModelHyperparameters(ConditionalGenerativeModel.ModelHyperparameters):

        n_generate_features: int
        """Number of features to generate"""

        n_latent_features: int
        """Number of features in the latent space"""

        n_condition_features: int
        """Number of condition features"""

        n_coupling_layers: int
        """Number of coupling layers"""

        hidden_layers_neurons: tuple[int]
        """Number of neurons in hidden layers"""

        hidden_layers_activation_fcns_keys: tuple[str, ...]
        """Keys of activation functions to come after each hidden layer"""

        n_features_upper_deck: int
        """Number of features to be split to the upper deck in coupling layers"""

        double_st: bool
        """Double s & t networks"""

        sigma_theta: float
        """Standard deviation of model parameters"""

        batch_norm: bool
        """Switch if batch normalization is to be applied"""

        shuffle_patterns: tuple[tuple, ...] | None
        """Shuffle patterns for each coupling layer."""

        _data_type_str: ClassVar[str] = 'crnvp_model_hyperparameters'
        _data_type_key: ClassVar[int] = 550

        _save_fields: ClassVar[list[str]] = ConditionalGenerativeModel.ModelHyperparameters._save_fields + [
            'n_latent_features', 'n_coupling_layers', 'hidden_layers_neurons', 'hidden_layers_activation_fcns_keys', 'n_features_upper_deck', 'double_st',
            'sigma_theta', 'batch_norm', 'shuffle_patterns'
        ]

        def __post_init__(self):
            if self.n_generate_features != self.n_latent_features:
                message = 'In RNVP models, latent and generate spaces must have the same number of dimensions!'
                raise ValueError(message)

            if not len(self.hidden_layers_activation_fcns_keys) != len(self.hidden_layers_neurons) - 1:
                message = f'Expecting {len(self.hidden_layers_neurons) - 1} activation functions.'
                raise ValueError(message)

            if not all(key in activation_fcns_by_key for key in self.hidden_layers_activation_fcns_keys):
                message = f'Unrecognized activation function key.'
                raise KeyError(message)

            if self.shuffle_patterns is None:
                logger.info('Generating new shuffle patterns.')
                self._create_shuffle_patterns()
            else:
                logger.warning('Using provided shuffle patterns.')

        def _create_shuffle_patterns(self):
            pattern_length = self.n_generate_features
            assert self.n_generate_features == self.n_latent_features

            shuffle_patterns = []
            for layer_index in range(self.n_coupling_layers):
                pattern = torch.randperm(pattern_length).tolist()
                # In this simple dataclass, store things as primitives!
                shuffle_patterns.append(tuple(pattern))

            self.shuffle_patterns = tuple(shuffle_patterns)

    @dataclass
    class ModelRecord(ConditionalGenerativeModel.ModelRecord):
        # Extending from base record data type – additional fields follow:
        network_state: dict
        optimizer_state: dict

    def __init__(self,
                 instance_id: str | int,
                 model_recipe: 'CRNVP.ModelRecipe',
                 training_controls: 'CRNVP.TrainingControls',
                 device: torch.device
                 ):

        super().__init__(instance_id, model_recipe, training_controls, device)

        # Re-setting these here to fix type hinting errors - reinforce that these are the classes of CRNVP
        self.model_specification: CRNVP.ModelSpecification = self.recipe.specification
        self.hps: CRNVP.ModelHyperparameters = self.recipe.hyperparameters

        self.ngen = NoiseGenerator('normal', n_dimensions=self.hps.n_latent_features, device=self.device)

        logger.info(f'Model initialization complete. Model directory is "{self.training_controls.autosave_dir}"')

    @property
    def training_data_groups(self) -> Iterable[list[Variable]]:
        """Returns lists of ``Variable`` instances. In each batch, training data is expected to be provided by
        ``DataLoader`` instances in groups, each of which contain the variables in these lists, at the same order."""
        assert isinstance(self.model_specification, CRNVP.ModelSpecification)
        return ([data_container_row_index],
                self.model_specification.generate_space.variables,
                self.model_specification.condition_space.variables)

    def get_record(self) -> ModelRecord:
        return self.ModelRecord(
            instance_id=self.instance_id,
            recipe=self.recipe,
            training_controls=self.training_controls,
            training_status=self.get_training_status(),
            network_state = self.network.state_dict(),
            optimizer_state=self.optimizer.state_dict()
        )

    @classmethod
    def load(cls, fp: str | Path, device: torch.device) -> 'CRNVP':

        model = super().load(fp, device)
        assert isinstance(model, CRNVP)

        record: CRNVP.ModelRecord = torch.load(fp)

        model.network.load_state_dict(record.network_state)
        model.optimizer.load_state_dict(record.optimizer_state)

        return model

    def _create_model(self):
        self.network = self.Network(self.device, self.hps)
        self.network.to(self.device)

    def _init_optimizer(self):
        self.optimizer = optimizers_by_key[self.training_settings.optimizer_key](
            self.network.parameters(),
            **self.training_settings.optimizer_settings
        )

    def activate_train_mode(self):
        self.network.train()

    def _train_batch(self, batch_index: int, batch_data: torch.Tensor):
        """Process one (mini)batch of training data and update the network using loss based on the batch."""

        if CRNVP.KEY_LOSS_MAP not in self._current_epoch_losses_sums:
            self._current_epoch_losses_sums[CRNVP.KEY_LOSS_MAP] = 0.0

        batch_sample_indices, batch_gvars, batch_conds = batch_data

        self.network.zero_grad()
        loss_map = self.loss_maximum_a_priori(batch_gvars, batch_conds)
        self._current_epoch_losses_sums[CRNVP.KEY_LOSS_MAP] += loss_map.item()

        loss_map.backward()
        self.optimizer.step()

    def loss_maximum_a_priori(self,
                              batch_fts: Tensor,
                              batch_conds: Tensor) -> Tensor:

        N = batch_fts.shape[0]
        model_input = torch.hstack((batch_fts, batch_conds))

        x_out, sum_log_det = self.network(model_input)
        self.last_batch_generated_data = x_out

        # Term A: log likelihood
        term_A = (1 / N) * (0.5 * x_out.norm(2, dim=1, keepdim=True).pow(2) - sum_log_det).sum(0)

        # Term B: log of prior on model parameters (log p(theta)) - the "augmented" part of the objective
        theta_sq = self.get_model_param_square_sum()
        term_B = theta_sq / (2 * self.hps.sigma_theta)  # TODO: sigma_theta may need to be squared

        loss = term_A + term_B

        pitch_mse = 1 * torch.pow(batch_fts[:, 5] - x_out[:, 5], 2).mean(0)  # Pitch penalty
        loss += pitch_mse

        return loss

    def get_model_param_square_sum(self) -> Tensor:
        """Returns the sum of the squares of all model parameters."""

        sum = torch.tensor(0, device=self.device, dtype=torch.float32)
        for param_group in self.network.parameters():
            if param_group.requires_grad:
                sum += param_group.data.pow(2).sum()

        return sum

    @property
    def n_parameters(self) -> int:
        return self.network.n_parameters

    def generate(self, n_samples: int, condition: torch.Tensor = None) -> torch.Tensor:
        if condition is None:
            raise Exception()

        if condition.dim() != 1:
            message = 'Provide a 1-dimensional condition vector.'
            raise ValueError(message)

        self.network.eval()

        noise = self.ngen.get_noise(n_samples)
        condition = condition.repeat(n_samples, 1)

        reverse_inputs = torch.hstack((noise, condition))

        with torch.no_grad():
            x_out = self.network.reverse(reverse_inputs)

        if self.hps.double_st:  # TODO: VERIFY THIS!
            # IT APPEARS ONLY IN DOUBLE_ST MODE WE GET CONDS BACK
            x_out = x_out[:, :-self.hps.n_condition_features]
        return x_out

    class Network(nn.Module):

        def __init__(self,
                     device: torch.device,
                     hyperparameters: 'CRNVP.ModelHyperparameters'):
            """Conditional RealNVP model network, consisting of coupling layers, and optionally,
            including batch norm layers."""

            super().__init__()
            self.hps = hyperparameters

            self.device = device
            self.hps = hyperparameters

            self.coupling_layers = nn.ModuleList()
            self._create_coupling_layers()

        def _create_coupling_layers(self):

            if self.hps.double_st:
                cl_type = self.CouplingLayerDoubleNetworks
            else:
                cl_type = self.CouplingLayer

            for layer_index, gvar_shuffle_pattern in zip(range(self.hps.n_coupling_layers), self.hps.shuffle_patterns):
                layer = cl_type(
                    device=self.device,
                    shuffle_pattern=gvar_shuffle_pattern,
                    hyperparameters=self.hps
                )
                layer.to(self.device)
                self.coupling_layers.append(layer)

                if self.hps.batch_norm:
                    bn_layer = self.BatchNormLayer(
                        n_features=len(gvar_shuffle_pattern) + self.hps.n_condition_features,
                        affine=False,
                        device=self.device
                    )
                    bn_layer.to(self.device)
                    self.coupling_layers.append(bn_layer)

        def forward(self, x: Tensor):
            """Pass a vector of generation variables through the network in forward direction towards noise."""
            x_j = x
            sum_log_det = torch.zeros((x.shape[0], 1), device=self.device)

            for lindex, layer_j in enumerate(self.coupling_layers):
                x_j, log_det_j = layer_j(x_j)
                sum_log_det += log_det_j

            return x_j, sum_log_det

        def reverse(self, z: Tensor) -> Tensor:
            """Pass a noise vector through the network in reverse direction towards generation variables."""
            z_j = z
            for layer_minus_j in reversed(self.coupling_layers):
                assert isinstance(layer_minus_j, (self.CouplingLayer, self.BatchNormLayer))
                z_j = layer_minus_j.reverse(z_j)

            x = z_j
            return x

        class CouplingLayer(nn.Module):

            def __init__(self,
                         device: torch.device,
                         shuffle_pattern: Tensor,
                         hyperparameters: 'CRNVP.ModelHyperparameters'):
                """
                Single s-t network coupling layer module.
                :param device: Device to compute on.
                :param shuffle_pattern: Shuffle pattern of generation variables / noise variables (not the conditions!)
                :param hyperparameters: Hyperparameters
                """

                super().__init__()
                self.device = device
                self.hps = hyperparameters

                self.shuffle_pattern = shuffle_pattern

                # Determine which generation variables go to upper & lower decks of the coupling layer
                self.indices_gvars_u1 = self.shuffle_pattern[:self.hps.n_features_upper_deck]
                self.indices_gvars_u2 = self.shuffle_pattern[self.hps.n_features_upper_deck:]

                n_gvars = len(self.shuffle_pattern)  # number of generation variables
                self.indices_conds = torch.arange(n_gvars, n_gvars + self.hps.n_condition_features, device=self.device)

                self._create_network()

            def _create_network(self):
                # u1 is upper deck, u1 goes through the s-t networks
                # u2 is lower deck

                self.indices_fts_u1 = torch.hstack((self.indices_gvars_u1, self.indices_conds))
                self.indices_fts_u2 = self.indices_gvars_u2

                self.n_dims_u1 = len(self.indices_fts_u1)
                self.n_dims_u2 = len(self.indices_fts_u2)

                # Starting layer should have u1 # of neurons, must output u2 # of neurons
                self.layer_neuron_counts = (
                    [self.n_dims_u1]
                    + list(self.hps.hidden_layers_neurons)
                    + [self.n_dims_u2]
                )

                self.s = get_fc_network(self.layer_neuron_counts, self.hps.hidden_layers_activation_fcns_keys)
                self.t = get_fc_network(self.layer_neuron_counts, self.hps.hidden_layers_activation_fcns_keys)

            def forward(self, x: Tensor):
                """
                Forward pass on the coupling layer.

                :param x: 2D Tensor - one minibatch of data - each row is one sample, columns are features,
                          may have condition features at the end

                :returns: **y** 2D Tensor - same shape as x - output of coupling layer
                 **log_det**: logarithm of the determinant of the Jacobian of the layer for use in the loss function.
                """

                # Split! u1 goes to upper deck, u2 goes to lower.
                u1 = x[:, self.indices_fts_u1].squeeze()
                u2 = x[:, self.indices_fts_u2].squeeze()

                v1 = u1
                s, t = self.s(u1), self.t(u1)
                v2 = (u2 * torch.exp(s) + t)

                # Merge! In the new y (output) vector
                y = torch.zeros_like(x, device=self.device)
                y[:, self.indices_fts_u1] = v1
                y[:, self.indices_fts_u2] = v2

                assert s.dim() == 2
                # Log-determinant because the actual determinant is ``exp(s.sum())``
                log_det = s.sum(1, keepdim=True)  # sum each row

                # Note that in the single storey configuration, the conditioning inputs move into the outputs unchanged.
                # Condition features are included in u1. u1 directly becomes v1, unchanged. v1 features get placed into
                # the y vector at the same indices as in i_fts_u1. Condition features are the last n_cond_fts features.
                return y, log_det

            def reverse(self, z: Tensor) -> Tensor:
                v1 = z[:, self.indices_fts_u1].squeeze()
                v2 = z[:, self.indices_fts_u2].squeeze()

                u1 = v1
                s_rev, t_rev = self.s(v1), self.t(v1)
                u2 = (v2 - t_rev) * (-1 * s_rev).exp()

                x = torch.zeros_like(z, device=self.device)
                x[:, self.indices_fts_u1] = u1
                x[:, self.indices_fts_u2] = u2

                return x

        class CouplingLayerDoubleNetworks(CouplingLayer):
            """Coupling layer model with double s-t networks, i.e. both the lower and upper deck vectors pass through their
            respective s & t networks. Has higher complexity compared to a single s-t network coupling layer."""

            def _create_network(self):
                # u1 is upper deck
                # u2 is lower deck

                self.indices_fts_u1 = self.indices_gvars_u1
                self.indices_fts_u2 = self.indices_gvars_u2

                self.st1_layer_neuron_counts = (
                        [len(self.indices_fts_u2) + self.hps.n_condition_features]
                        + list(self.hps.hidden_layers_neurons)
                        + [len(self.indices_fts_u1)]
                )

                self.st2_layer_neuron_counts = (
                        [len(self.indices_fts_u1) + self.hps.n_condition_features]
                        + list(self.hps.hidden_layers_neurons)
                        + [len(self.indices_fts_u2)]
                )

                self.s1 = get_fc_network(self.st1_layer_neuron_counts, self.hps.hidden_layers_activation_fcns_keys)
                self.t1 = get_fc_network(self.st1_layer_neuron_counts, self.hps.hidden_layers_activation_fcns_keys)

                self.s2 = get_fc_network(self.st2_layer_neuron_counts, self.hps.hidden_layers_activation_fcns_keys)
                self.t2 = get_fc_network(self.st2_layer_neuron_counts, self.hps.hidden_layers_activation_fcns_keys)

            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:

                # Split! u1 goes to upper deck, u2 goes to lower.
                u1 = x[:, self.indices_fts_u1].squeeze()
                u2 = x[:, self.indices_fts_u2].squeeze()

                conds = x[:, self.indices_conds]

                u2_and_conds = torch.hstack((u2, conds))
                s1, t1 = self.s1(u2_and_conds), self.t1(u2_and_conds)
                v1 = (u1 * torch.exp(s1)) + t1

                v1_and_conds = torch.hstack((v1, conds))
                s2, t2 = self.s2(v1_and_conds), self.t2(v1_and_conds)
                v2 = (u2 * torch.exp(s2)) + t2

                # Merge and put things back together: conditions added unchanged
                y = torch.zeros_like(x, device=self.device)
                y[:, self.indices_fts_u1] = v1
                y[:, self.indices_fts_u2] = v2
                y[:, self.indices_conds] = conds

                # See arXiv:1907.02392v3 for explicit statement of this, I also verified it by hand derivation
                log_det = s1.sum(1, keepdim=True) + s2.sum(1, keepdim=True)

                return y, log_det

            def reverse(self, z: Tensor) -> Tensor:
                v1 = z[:, self.indices_fts_u1].squeeze()
                v2 = z[:, self.indices_fts_u2].squeeze()

                conds = z[:, self.indices_conds]

                v1_and_conds = torch.hstack((v1, conds))
                s2_rev, t2_rev = self.s2(v1_and_conds), self.t2(v1_and_conds)
                u2 = (v2 - t2_rev) * torch.exp(-1 * s2_rev)

                u2_and_conds = torch.hstack((u2, conds))
                s1_rev, t1_rev = self.s1(u2_and_conds), self.t1(u2_and_conds)
                u1 = (v1 - t1_rev) * torch.exp(-1 * s1_rev)

                x = torch.zeros_like(z, device=self.device)
                x[:, self.indices_fts_u1] = u1
                x[:, self.indices_fts_u2] = u2
                x[:, self.indices_conds] = conds

                return x

        class BatchNormLayer(nn.modules.batchnorm._BatchNorm):

            def __init__(self, n_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = False,
                         track_running_stats: bool = True, device=None, dtype=None) -> None:
                """
                Borrowed heavily from base class ``nn.modules.batchnorm._BatchNorm``. Customized to enable reverse passes.

                :param n_features: number of features or channels C of the input
                :param eps: a value added to the denominator for numerical stability. Default: 1e-5
                :param momentum: the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
                :param affine: a boolean value that when set to True, this module has learnable affine parameters. Default: False
                :param track_running_stats: a boolean value that when set to True, this module tracks the running mean and variance, and when set to False, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None. When these buffers are None, this module always uses batch statistics. in both training and eval modes. Default: True
                :param device: Device.
                :param dtype: Data type.
                """

                # Untouched from _BatchNorm
                factory_kwargs = {'device': device, 'dtype': dtype}
                super().__init__(n_features, eps, momentum, affine, track_running_stats, **factory_kwargs)

            def _check_input_dim(self, input):
                """Copied from ``nn.modules.batchnorm.BatchNorm1d`` """
                if input.dim() != 2 and input.dim() != 3:
                    raise ValueError("expected 2D or 3D input (got {}D input)".format(input.dim()))

            def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
                """Modified from PyTorch original to **return the Jacobian as well.**"""

                self._check_input_dim(input)

                # exponential_average_factor is set to self.momentum
                # (when it is available) only so that it gets updated
                # in ONNX graph when this node is exported to ONNX.
                if self.momentum is None:
                    exponential_average_factor = 0.0
                else:
                    exponential_average_factor = self.momentum

                if self.training and self.track_running_stats:
                    # if statement only here to tell the jit to skip emitting this when it is None
                    if self.num_batches_tracked is not None:  # type: ignore[has-type]
                        self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                        if self.momentum is None:  # use cumulative moving average
                            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                        else:  # use exponential moving average
                            exponential_average_factor = self.momentum

                # Decide whether the mini-batch stats should be used for normalization rather than the buffers.
                # Mini-batch stats are used in training mode, and in eval mode when buffers are None.
                if self.training:
                    bn_training = True
                else:
                    bn_training = (self.running_mean is None) and (self.running_var is None)

                # Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
                # passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
                # used for normalization (i.e. in eval mode when buffers are not None).

                # Modified
                normalized = F.batch_norm(input,
                                          # If buffers are not to be tracked, ensure that they won't be updated
                                          self.running_mean if not self.training or self.track_running_stats else None,
                                          self.running_var if not self.training or self.track_running_stats else None,
                                          self.weight,
                                          self.bias,
                                          bn_training,
                                          exponential_average_factor,
                                          self.eps)

                current_var = self.running_var.view((1, self.num_features)).expand_as(input)  # make N rows of same thing
                denominator = (current_var + self.eps)

                log_det = -0.5 * torch.log(denominator).sum(1, keepdim=True)

                return normalized, log_det

            def reverse(self, z: Tensor) -> Tensor:
                """New method for RealNVP. Reverse pass."""

                current_var = self.running_var.view((1, self.num_features)).expand_as(z)
                current_mean = self.running_mean.view((1, self.num_features)).expand_as(z)
                denominator = (current_var + self.eps)
                x = z * denominator.sqrt() + current_mean

                return x


CRNVP.ModelRecipe._used_classes += [CRNVP.ModelHyperparameters]


class CRNVPMAPValidator(Validator):

    validation_loss_type = CRNVP.KEY_LOSS_MAP

    def __init__(self,
                 model: CRNVP,
                 validation_dataset: MDevDataset):
        super().__init__(model)

        self.validation_dataset = validation_dataset
        self.model.set_mdev_dataset_groups(self.validation_dataset)

    def calculate_validation_loss(self, epoch_report: CRNVP.EpochReport):

        with torch.no_grad():

            assert isinstance(self.model, CRNVP)
            indices, generate_vars, conds = self.validation_dataset.get_all_rows_in_groups()
            val_loss = self.model.loss_maximum_a_priori(generate_vars, conds).item()

        val_losses = {CRNVP.KEY_LOSS_MAP: val_loss}
        self._add_to_validation_losses(epoch_report.epoch_index, val_losses[self.validation_loss_type])


if __name__ == '__main__':

    print(5)


