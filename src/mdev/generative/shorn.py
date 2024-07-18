import logging
from typing import Callable, Union, Type, ClassVar, Iterable, Any, Literal
from pathlib import Path
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from src.mdev.utilities import activation_fcns_by_key, optimizers_by_key, data_container_row_index, NoiseGenerator, MDevDataset, get_fc_network
from src.mdev.generative.base import GenerativeModel, ConditionalGenerativeModel, Validator
from src.mdev.logger import LOG_LEVEL_NAME_VALIDATION, LOG_LEVEL_VALIDATION
from src.mdev.archspec import NetworkSpecification, Linear, ActivationFunction


logger = logging.getLogger(__name__)


# Based on work of Jean Feydy
# https://github.com/jeanfeydy/global-divergences/blob/master/common/sinkhorn_balanced.py

# Based on adaptation on Jean Feydy's work for CEBGAN by Qiuyi Chen et al.
# https://github.com/wchen459/CEBGAN_JMD_2021


def LSE(V_ij: Tensor):
    """(Stabilized) log-sum-exp reduction from Feydy, 2019
    :arg V_ij: 2D tensor"""

    # Pick maximum in each row (along direction 1)
    # Pick the values with [0] - ``max`` returns (max, max_indices)
    # Form into a column vector with whatever # of rows (-1) and 1 column
    V_i = torch.max(V_ij, 1)[0].view(-1, 1)

    # Sum each row (along direction 1)
    # Form into a column vector with whatever # of rows (-1) and 1 column
    return V_i + (V_ij - V_i).exp().sum(1).log().view(-1, 1)


def get_sinkhorn_operators(
    epsilon: float,
    x_i: Tensor,
    y_j: Tensor,
    cost_func: Callable
) -> tuple[Callable, Callable]:

    C_e = cost_func(x_i, y_j) / epsilon

    S_x = lambda f_i: -1 * LSE(f_i.view(1, -1) - C_e.T)
    S_y = lambda f_j: -1 * LSE(f_j.view(1, -1) - C_e)

    return S_x, S_y


def sinkhorn_algorithm(
    alpha_i: Tensor,
    x_i: Tensor,
    beta_j: Tensor,
    y_j: Tensor,
    cost_fcn: Callable,
    epsilon: float = 0.1,
    n_iters: int = 100,
    tolerance: float = 1e-3,
    assume_convergence: bool = False
) -> tuple[Tensor, Tensor]:

    log_alpha_i, log_beta_j = alpha_i.log(), beta_j.log()
    B_i, A_j = torch.zeros_like(alpha_i), torch.zeros_like(beta_j)  # zero matrix of same shape

    with torch.set_grad_enabled(not assume_convergence):
        # If we assume convergence, we can calculate the optimal f and g Lagrange multipliers
        # using the "rule", ignoring that they are a function

        for i in range(n_iters - 1):
            B_i_last = B_i

            C_e = cost_fcn(x_i, y_j) / epsilon
            A_j = -1 * LSE((B_i + log_alpha_i).view(1, -1) - C_e.T)
            B_i = -1 * LSE((A_j + log_beta_j).view(1, -1) - C_e)

            # Stopping criterion: L1 norm of updates
            error = epsilon * (B_i - B_i_last).abs().mean()
            if error.item() < tolerance:
                break

    if not assume_convergence:
        C_e = cost_fcn(x_i, y_j) / epsilon
        A_j = -1 * LSE((B_i + log_alpha_i).view(1, -1) - C_e.T)
        B_i = -1 * LSE((A_j + log_beta_j ).view(1, -1) - C_e)

    else:
        # Assume that we have converged, and can thus use the "exact" and cheap gradient's formula

        C_e = cost_fcn(x_i.detach(), y_j) / epsilon
        A_j = -1 * LSE((B_i + log_alpha_i).detach().view(1, -1) - C_e.T)

        C_e = cost_fcn(x_i, y_j.detach()) / epsilon
        B_i = -1 * LSE((A_j + log_beta_j).detach().view(1, -1) - C_e)

    a_y, b_x = epsilon * A_j.view(-1), epsilon * B_i.view(-1)

    return a_y, b_x


def sinkhorn_symmetric_algorithm(
    alpha_i: Tensor,
    x_i: Tensor,
    cost_fcn: Callable,
    epsilon: float = 0.1,
    n_iters: int = 100,
    tolerance: float = 1e-3,
    assume_convergence: bool = False
) -> Tensor:
    """In the specific case of the (symmetric) corrective terms ``OT_ε(α,α)`` and ``OT_ε(β,β)``,
    we can do better than the regular Sinkhorn algorithm."""

    log_alpha_i = alpha_i.log()
    A_i = torch.zeros_like(alpha_i)  # zeros, same shape as alpha_i

    C_e = cost_fcn(x_i, x_i) / epsilon

    with torch.set_grad_enabled(not assume_convergence):

        for i in range(n_iters - 1):

            A_i_prev = A_i

            A_i = 0.5 * (A_i + -1 * LSE((A_i + log_alpha_i).view(1, -1) - C_e.T))

            # Stopping criterion: L1 norm of the updates
            mean_delta = epsilon * (A_i - A_i_prev).abs().mean()
            if mean_delta.item() < tolerance:
                break

    if not assume_convergence:
        W_i = A_i + log_alpha_i

        a_x = epsilon * -1 * LSE(W_i.view(1, -1) - C_e.T).view(-1)
        return a_x

    else:
        W_i = (A_i + log_alpha_i).detach()
        # detach tensor from the current computational graph
        # don't need it to be traced for the gradient computation

        C_e = cost_fcn(x_i.detach(), x_i) / epsilon
        a_x = epsilon * -1 * LSE(W_i.view(1, -1) - C_e.T).view(-1)
        return a_x


def sinkhorn_divergence(
    alpha: Tensor,
    x: Tensor,
    beta: Tensor,
    y: Tensor,
    epsilon: float,
    cost_fcn: Callable,
    n_iters_asymmetric: int,
    n_iters_symmetric: int,
    tolerance: float,
    assume_convergence: bool
) -> Tensor:
    """
    Calculates the Sinkhorn divergence S_ε(α, β)
    See paper "Global divergences between measures from Hausdorff distance to Optimal Transport"

    :param alpha: Measure weigths, α ∈ R^N+, non-negative vector of shape [N], that sums up to 1
    :param x: x ∈ (R^D)^N, real-valued tensor of shape [N, D]
    :param beta: Measure weights, β ∈ R^M+, non-negative vector of shape [M], that sums up to 1
    :param y: y ∈ (R^D)^M, real-valued tensor of shape [M, D]
    :return: Sinkhorn divergence / loss
    """

    # Cross-correlation
    a_y, b_x = sinkhorn_algorithm(
        alpha_i=alpha, x_i=x,
        beta_j=beta, y_j=y,
        cost_fcn=cost_fcn,
        epsilon=epsilon,
        n_iters=n_iters_asymmetric,
        tolerance=tolerance,
        assume_convergence=assume_convergence
    )

    # Autocorrelation
    a_x = sinkhorn_symmetric_algorithm(
        alpha, x, cost_fcn,
        epsilon=epsilon, n_iters=n_iters_symmetric,
        tolerance=tolerance, assume_convergence=assume_convergence
    )

    b_y = sinkhorn_symmetric_algorithm(
        beta, y, cost_fcn,
        epsilon=epsilon, n_iters=n_iters_symmetric,
        tolerance=tolerance, assume_convergence=assume_convergence
    )

    # Sinkhorn divergence / loss
    # Page 10 in paper "Global divergences between measures from Hausdorff distance to Optimal Transport"
    # cross-correlation dual vectors f and g -> b_x and a_y
    #   autocorrelation dual vectors p and q -> a_x and b_y

    return (b_x - a_x) @ alpha + (a_y - b_y) @ beta


def distance_L1(a: Tensor, b: Tensor):
    """1-norm distance between each pair of the two collections of row vectors."""
    return torch.cdist(a, b, p=1)


def distance_L2(a: Tensor, b: Tensor):
    return torch.cdist(a, b, p=2)


class SHORN(GenerativeModel):

    MODEL_TYPE_ABBREVIATION = 'SHORN'
    MODEL_TYPE_NAME_LONG = 'Sinkhorn Generator'
    
    KEY_LOSS_GENERATOR_SINKHORN = 'SINKHORN'

    KEY_LATENT_DIST_STANDARD_NORMAL = 'standard_normal'

    cost_fcns_by_key = {'L1': distance_L1, 'L2': distance_L2}

    @dataclass
    class ModelHyperparameters(GenerativeModel.ModelHyperparameters):

        n_generate_features: int
        """Number of features to generate"""

        n_latent_features: int
        """Number of features in the latent space"""

        generator_spec: NetworkSpecification
        """Architecture specification for the generator network"""

        latent_distribution: str
        """Prior distribution for latent features"""

        cost_function: Literal['L1', 'L2']
        """Cost function for use in Sinkhorn divergence calculation"""

        epsilon: float
        """Epsilon used in Sinkhorn divergence calculation"""

        _data_type_str: ClassVar[str] = 'shorn_generator_model_hyperparameters'
        _data_type_key: ClassVar[int] = 550

        _save_fields: ClassVar[list[str]] = GenerativeModel.ModelHyperparameters._save_fields + [
            'n_generate_features', 'n_latent_features', 'generator_spec', 'latent_distribution',
            'cost_function', 'epsilon'
        ]

        _used_classes = GenerativeModel.ModelHyperparameters._used_classes + [NetworkSpecification]

        def __post_init__(self):
            # Check network outer layer feature counts
            errors = []

            generator_first_layer = self.generator_spec.layers[0]
            if isinstance(generator_first_layer, Linear) and generator_first_layer.in_features != self.n_latent_features:
                errors.append(f'Generator network first layer must have {self.n_latent_features} features as per ``n_latent_features``.')

            generator_last_layer = self.generator_spec.layers[-1]
            if isinstance(generator_last_layer, ActivationFunction):
                generator_last_layer = self.generator_spec.layers[-2]  # try the one before
            if isinstance(generator_last_layer, Linear) and generator_last_layer.out_features != self.n_generate_features:
                errors.append(f'Generator network last layer must have {self.n_generate_features} features as per ``n_generate_features``.')

            if errors:
                errors_str = "\n".join(errors)
                message = f'The following errors were found with the generator network specification:\n{errors_str}'
                raise ValueError(message)
            else:
                logger.info('Network outer layer feature counts are consistent with other model hyperparameters.')

    @dataclass
    class ModelRecord(GenerativeModel.ModelRecord):
        # Extending from base record data type – additional fields follow:
        g_network_state: dict  # generator network state
        optimizer_state: dict

    def __init__(self,
                 instance_id: str | int,
                 model_recipe: 'SHORN.ModelRecipe',
                 training_controls: 'SHORN.TrainingControls',
                 device: torch.device
                 ):

        super().__init__(instance_id, model_recipe, training_controls, device)

        # Re-setting these here to fix type hinting errors - reinforce that these are the classes of CRNVP
        self.model_specification: SHORN.ModelSpecification = self.recipe.specification
        self.hps: SHORN.ModelHyperparameters = self.recipe.hyperparameters
        self.training_settings: SHORN.TrainingSettings = self.recipe.training_settings

        self.noise_generator = NoiseGenerator(
            NoiseGenerator.KEY_NOISE_TYPE_NORMAL,
            n_dimensions=self.hps.n_latent_features,
            device=self.device
        )

    def _create_model(self):
        self.generator = self.hps.generator_spec.get_network(device=self.device)

    def get_record(self) -> 'SHORN.ModelRecord':
        return self.ModelRecord(
            instance_id=self.instance_id,
            recipe=self.recipe,
            training_controls=self.training_controls,
            training_status=self.get_training_status(),
            g_network_state=self.generator.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
        )

    @classmethod
    def load(cls, fp: str | Path, device: torch.device) -> 'SHORN':

        model = super().load(fp, device)
        assert isinstance(model, SHORN)

        record: SHORN.ModelRecord = torch.load(fp)

        model.generator.load_state_dict(record.g_network_state)
        model.optimizer.load_state_dict(record.optimizer_state)

        return model

    def _init_optimizer(self):
        self.optimizer = optimizers_by_key[self.training_settings.optimizer_key](
            self.generator.parameters(),
            **self.training_settings.optimizer_settings
        )

    def activate_train_mode(self):
        self.generator.train()

    def _train_batch(self, batch_index: int, batch_data: tuple[torch.Tensor, ...]):

        if SHORN.KEY_LOSS_GENERATOR_SINKHORN not in self._current_epoch_losses_sums:
            self._current_epoch_losses_sums.update({SHORN.KEY_LOSS_GENERATOR_SINKHORN: 0.0})

        self.optimizer.zero_grad()

        batch_sample_indices, batch_gvars = batch_data

        loss_generator = self.loss_generator(batch_gvars)

        self._current_epoch_losses_sums[SHORN.KEY_LOSS_GENERATOR_SINKHORN] += loss_generator.item()

        loss_generator.backward()
        self.optimizer.step()

    def loss_generator(self, batch_fts: torch.Tensor) -> torch.Tensor:

        g_out = self._get_generator_outputs(n_samples=batch_fts.shape[0])

        loss_sinkhorn = self.get_sinkhorn_div(
            real_fts=batch_fts,
            fake_fts=g_out
        )

        return loss_sinkhorn

    def _get_generator_outputs(self, n_samples: int) -> torch.Tensor:
        latents = self.get_latent_samples(n_samples)
        g_out = self.generator(latents)
        return g_out

    def get_sinkhorn_div(self,
                         real_fts: Tensor,
                         fake_fts: Tensor,
                         ) -> Tensor:
        """Returns ``S_λ(p_r, p_g)``"""

        a = torch.ones(len(real_fts), 1, device=real_fts.device) / len(real_fts)
        b = torch.ones(len(fake_fts), 1, device=fake_fts.device) / len(fake_fts)

        div = sinkhorn_divergence(
            a, real_fts,
            b, fake_fts,
            epsilon=self.hps.epsilon,
            assume_convergence=True,
            cost_fcn=self.cost_fcns_by_key[self.hps.cost_function],
            n_iters_asymmetric=1000,
            n_iters_symmetric=1000,
            tolerance=1e-3
        )
        return div

    def generate(self, n_samples: int) -> torch.Tensor:

        self.generator.eval()

        with torch.no_grad():
            g_out = self._get_generator_outputs(n_samples)

        return g_out

    def get_latent_samples(self, n_samples: int) -> torch.Tensor:
        match self.hps.latent_distribution:

            case SHORN.KEY_LATENT_DIST_STANDARD_NORMAL:
                latents = torch.randn((n_samples, self.hps.n_latent_features), device=self.device)

            case _:
                raise NotImplementedError()

        return latents


SHORN.ModelRecipe._used_classes += [SHORN.ModelHyperparameters]


class CSHORN(SHORN, ConditionalGenerativeModel):

    MODEL_TYPE_ABBREVIATION = 'CSHORN'
    MODEL_TYPE_NAME_LONG = 'Conditional Sinkhorn Generator'

    @dataclass
    class ModelHyperparameters(SHORN.ModelHyperparameters):

        n_condition_features: int
        """Number of condition features"""

        _data_type_str: ClassVar[str] = 'cshorn_model_hyperparameters'
        _data_type_key: ClassVar[int] = 550

        _save_fields: ClassVar[list[str]] = SHORN.ModelHyperparameters._save_fields + [
            'n_condition_features'
        ]

        def __post_init__(self):
            # Check network outer layer feature counts
            errors = []

            generator_first_layer = self.generator_spec.layers[0]
            if isinstance(generator_first_layer, Linear) and generator_first_layer.in_features != self.n_latent_features + self.n_condition_features:
                errors.append(f'Generator network first layer must have {self.n_latent_features + self.n_condition_features} features '
                              f'(n_latent_features + n_condition_features)')

            generator_last_layer = self.generator_spec.layers[-1]
            if isinstance(generator_last_layer, ActivationFunction):
                generator_last_layer = self.generator_spec.layers[-2]  # try the one before
            if isinstance(generator_last_layer, Linear) and generator_last_layer.out_features != self.n_generate_features:
                errors.append(f'Generator network last layer must have {self.n_generate_features} features as per ``n_generate_features``.')

            if errors:
                errors_str = "\n".join(errors)
                message = f'The following errors were found with the generator network specification:\n{errors_str}'
                raise ValueError(message)
            else:
                logger.info('Network outer layer feature counts are consistent with other model hyperparameters.')


    def _train_batch(self, batch_index: int, batch_data: tuple[torch.Tensor, ...]):
        if SHORN.KEY_LOSS_GENERATOR_SINKHORN not in self._current_epoch_losses_sums:
            self._current_epoch_losses_sums.update({SHORN.KEY_LOSS_GENERATOR_SINKHORN: 0.0})

        self.optimizer.zero_grad()

        batch_sample_indices, batch_fts, batch_conds = batch_data

        loss_generator = self.loss_generator(batch_fts, batch_conds)

        self._current_epoch_losses_sums[SHORN.KEY_LOSS_GENERATOR_SINKHORN] += loss_generator.item()

        loss_generator.backward()
        self.optimizer.step()

    def loss_generator(self, batch_fts: torch.Tensor, batch_conds: torch.Tensor = None) -> torch.Tensor:
        if batch_conds is None:
            message = 'Conditional generative model expects conditions to be provided for running the generator'
            raise Exception(message)

        g_out = self._get_generator_outputs(batch_conds)

        loss_sinkhorn = self.get_sinkhorn_div(
            real_fts=batch_fts,
            fake_fts=g_out,
            real_conds=batch_conds,
            fake_conds=batch_conds
        )

        return loss_sinkhorn

    def _get_generator_outputs(self, conds: torch.Tensor) -> torch.Tensor:
        latents = self.get_latent_samples(n_samples=conds.shape[0])
        latents_conds = torch.concat((latents, conds), dim=1).to(self.device)

        g_out = self.generator(latents_conds)
        return g_out

    def get_sinkhorn_div(self,
                         real_fts: Tensor,
                         fake_fts: Tensor,
                         real_conds: Tensor = None,
                         fake_conds: Tensor = None
                         ) -> Tensor:
        """Returns ``S_λ(p_r, p_g)``"""

        a = torch.ones(len(real_fts), 1, device=real_fts.device) / len(real_fts)
        b = torch.ones(len(fake_fts), 1, device=fake_fts.device) / len(fake_fts)

        div = sinkhorn_divergence(
            a, torch.hstack((real_fts, real_conds)),
            b, torch.hstack((fake_fts, fake_conds)),
            epsilon=self.hps.epsilon,
            assume_convergence=True,
            cost_fcn=self.cost_fcns_by_key[self.hps.cost_function],
            n_iters_asymmetric=1000,
            n_iters_symmetric=1000,
            tolerance=1e-3
        )
        return div

    def generate(self, n_samples: int, condition: torch.Tensor = None) -> torch.Tensor:
        if condition is None:
            message = 'Conditional generative model expects conditions to be provided for running the generator'
            raise Exception(message)

        if condition.dim() != 1:
            message = 'Provide a 1-dimensional condition vector.'
            raise ValueError(message)

        self.generator.eval()

        condition = condition.repeat(n_samples, 1)

        with torch.no_grad():
            g_out = self._get_generator_outputs(conds=condition)

        return g_out


CSHORN.ModelRecipe._used_classes += [CSHORN.ModelHyperparameters]


class SHORNValidatorWrapper:

    class _SHORNValidator(Validator):

        def __init__(self,
                     model: SHORN,
                     validation_dataset: MDevDataset,
                     validation_loss_type: str,
                     loss_method: Callable):

            super().__init__(model)

            self.validation_loss_type = validation_loss_type
            self.loss_method = loss_method

            self.validation_dataset = validation_dataset
            self.model.set_mdev_dataset_groups(self.validation_dataset)

        def calculate_validation_loss(self, epoch_report: SHORN.EpochReport):
            with torch.no_grad():
                assert isinstance(self.model, SHORN)
                indices, generate_vars = self.validation_dataset.get_all_rows_in_groups()
                val_loss = self.loss_method(generate_vars).item()

            val_losses = {self.validation_loss_type: val_loss}
            self._add_to_validation_losses(epoch_report.epoch_index, val_losses[self.validation_loss_type])
            logger.log(LOG_LEVEL_VALIDATION, f'Epoch {epoch_report.epoch_index}: {self.validation_loss_type}={val_loss}')

    class _CSHORNValidator(_SHORNValidator):
        """Overrides regular ``SHORNValidator`` only in validation loss calculation as it also sends the condition
        data to the loss calculator."""

        def calculate_validation_loss(self, epoch_report: CSHORN.EpochReport):
            with torch.no_grad():
                assert isinstance(self.model, CSHORN)
                indices, generate_vars, conds = self.validation_dataset.get_all_rows_in_groups()
                val_loss = self.loss_method(generate_vars, conds).item()  # DIFFERENT FOR CSHORN

            val_losses = {self.validation_loss_type: val_loss}
            self._add_to_validation_losses(epoch_report.epoch_index, val_losses[self.validation_loss_type])
            logger.log(LOG_LEVEL_VALIDATION, f'Epoch {epoch_report.epoch_index}: {self.validation_loss_type}={val_loss}')

    def __init__(self,
                 model: SHORN | CSHORN,
                 validation_dataset: MDevDataset):
        """Container for generator and discriminator validators.
        Can work with both unconditional & conditional SHORNs."""

        self.model = model
        self.validation_dataset = validation_dataset

        if isinstance(self.model, CSHORN):
            validator_class = self._CSHORNValidator
        else:
            assert isinstance(self.model, SHORN)
            validator_class = self._SHORNValidator

        # Generator validation loss calculator
        self.g_validator = validator_class(self.model,
                                           self.validation_dataset,
                                           validation_loss_type=self.model.KEY_LOSS_GENERATOR_SINKHORN,
                                           loss_method=self.model.loss_generator)

    def calculate_validation_loss(self, epoch_report: CSHORN.EpochReport):
        self.g_validator.calculate_validation_loss(epoch_report)

    @property
    def validator(self) -> Validator:
        return self.g_validator
