import logging
from typing import Callable, Union, Type, ClassVar, Iterable, Any, Literal
from pathlib import Path
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from utils.iox import ProgramData
from datamodels.variables import Variable
from mdev.archspec import *
from mdev.utilities import activation_fcns_by_key, optimizers_by_key, data_container_row_index, NoiseGenerator, MDevDataset, get_fc_network
from mdev.generative.base import GenerativeModel, ConditionalGenerativeModel, Validator
from mdev.logger import LOG_LEVEL_NAME_VALIDATION, LOG_LEVEL_VALIDATION

logger = logging.getLogger(__name__)


@dataclass
class _BaseGANModelHyperparameters(GenerativeModel.ModelHyperparameters):
    """Container for GAN hyperparameters - baseclass, further customized for different GAN variants"""

    n_generate_features: int
    """Number of features to generate"""

    n_latent_features: int
    """Number of features in the latent space"""

    generator_spec: NetworkSpecification
    """Specification of the generator network"""

    discriminator_spec: NetworkSpecification
    """Specification of the discriminator"""

    reduction: str
    """Loss reduction method across samples in a batch"""

    epsilon: float
    """Epsilon for stability"""

    clamp_discriminator_outputs: bool

    _data_type_str: ClassVar[str] = 'base_gan_model_hyperparameters'
    _data_type_key: ClassVar[int] = 550

    _save_fields: ClassVar[list[str]] = GenerativeModel.ModelHyperparameters._save_fields + [
        'n_generate_features', 'n_latent_features', 'generator_spec', 'discriminator_spec', 'reduction', 'epsilon',
        'clamp_discriminator_outputs'
    ]

    _used_classes = GenerativeModel.ModelHyperparameters._used_classes + [NetworkSpecification]

    KEY_REDUCE_MEAN: ClassVar[str] = 'mean'
    KEY_REDUCE_SUM: ClassVar[str] = 'sum'
    reduction_types: ClassVar[list[str]] = [KEY_REDUCE_MEAN, KEY_REDUCE_SUM]

    def __post_init__(self):
        if self.reduction not in self.reduction_types:
            message = f'Invalid reduction type.'
            raise ValueError(message)


@dataclass
class _BaseCGANModelHyperparameters(_BaseGANModelHyperparameters):
    """Container for hyperparameters of a **conditional** GAN"""

    n_condition_features: int
    """Number of condition features"""

    _data_type_str: ClassVar[str] = 'base_cgan_model_hyperparameters'
    _data_type_key: ClassVar[int] = 550

    _save_fields: ClassVar[list[str]] = _BaseGANModelHyperparameters._save_fields + [
        'n_condition_features'
    ]


class GAN(GenerativeModel):
    """Vanilla GAN (Generative Adversarial Network)"""

    MODEL_TYPE_ABBREVIATION = 'GAN'
    MODEL_TYPE_NAME_LONG = 'Generative Adversarial Network'

    KEY_LOSS_GEN = 'GEN'  # generator loss
    KEY_LOSS_DISC = 'DISC'  # discriminator loss
    KEY_LOSS_ADV = 'ADV'
    """Key for adversarial loss"""

    KEY_FORMULATION_NON_SATURATING = 'non_saturating'
    """Discriminator loss same as "classic" formulation, generator loss modified to prevent saturation."""

    KEY_FORMULATION_CLASSIC = 'classic'
    """Equation 1 of the GAN paper. Generator loss saturates easily! Not very usable."""

    KEY_FORMULATION_WASSERSTEIN = 'wasserstein'
    """Wasserstein loss formulation for WGAN"""

    KEY_FORMULATION_WASSERSTEIN_GP = 'wasserstein_gp'
    """Wasserstein loss formulation for WGAN with gradient penalty"""

    @dataclass
    class ModelHyperparameters(_BaseGANModelHyperparameters):
        """Container for hyperparameters of a vanilla GAN"""

        g_loss_formulation: Literal['non_saturating', 'classic']

        d_loss_formulation: Literal['non_saturating', 'classic']

        _data_type_str: ClassVar[str] = 'gan_model_hyperparameters'
        _data_type_key: ClassVar[int] = 550

        _save_fields: ClassVar[list[str]] = _BaseGANModelHyperparameters._save_fields + [
            'g_loss_formulation', 'd_loss_formulation'
        ]

        def __post_init__(self):
            errors = []

            if not isinstance(self.discriminator_spec.layers[-1], Sigmoid):
                errors.append('Using BCE loss formulation for DISC, but DISC last hidden layer activation function is not sigmoid.')

            g_linear_layers = [layer for layer in self.generator_spec.layers if isinstance(layer, Linear)]
            d_linear_layers = [layer for layer in self.discriminator_spec.layers if isinstance(layer, Linear)]

            # Check generator
            # Unconditional GAN - initial layer should take n_latent_features number of features
            # Final layer should output number of features to generate
            g_initial_layer = g_linear_layers[0]
            g_final_layer = g_linear_layers[-1]

            if g_initial_layer.in_features == -1:
                g_initial_layer.in_features = self.n_latent_features
            elif g_initial_layer.in_features != self.n_latent_features:
                message = ('First linear layer in the generator network should take in number of features equal to the number '
                           'of latent dimensions!')
                errors.append(message)

            if g_final_layer.out_features == -1:
                g_final_layer.out_features = self.n_generate_features
            elif g_final_layer.out_features != self.n_generate_features:
                message = ('Last linear layer in the generator network must output number of features equal to the number '
                           'of features to generate!')
                errors.append(message)

            # Check discriminator
            # Unconditional GAN - initial layer should take number of features generated
            # Final layer should output a single feature
            d_initial_layer = d_linear_layers[0]
            d_final_layer = d_linear_layers[-1]

            if d_initial_layer.in_features == -1:
                d_initial_layer.in_features = self.n_generate_features
            elif d_initial_layer.in_features != self.n_generate_features:
                message = ('First linear layer of the discriminator network should take in number of features equal to the '
                           'number of features generated!')
                errors.append(message)

            if d_final_layer.out_features == -1:
                d_final_layer.out_features = 1
            elif d_final_layer.out_features != 1:
                message = 'Final layer of the discriminator network must output 1 feature!'
                errors.append(message)

            if errors:
                newline = '\n'
                raise ValueError(f'Following errors were found with the hyperparameters:{newline.join(errors)}')
            else:
                logger.info(f'Hyperparameter checks passed.')

    @dataclass
    class TrainingSettings(ProgramData):
        """Container for training settings of a vanilla GAN"""

        batch_size: int

        dataloader_drop_last: bool
        """Drop last incomplete batch in the dataloader if the batch size is less than specified."""
        dataloader_shuffle: bool

        g_optimizer_key: str
        """String key identifying an optimizer type"""
        g_optimizer_settings: dict[str, Any]
        """Arguments of the optimizer instance, will be expanded inside the optimizer instance constructor."""

        d_optimizer_key: str
        """String key identifying an optimizer type"""
        d_optimizer_settings: dict[str, Any]
        """Arguments of the optimizer instance, will be expanded inside the optimizer instance constructor."""

        n_g_step_per_batch: int
        """Set to a value greater than 1 to make multiple training steps for the generator in each batch."""
        n_d_step_per_batch: int
        """Set to a value greater than 1 to make multiple training steps for the discriminator in each batch."""

        torch_manual_seed: int | float | None

        _data_type_str: ClassVar[str] = 'gan_training_settings'
        _data_type_key: ClassVar[int] = 550

        _save_fields: ClassVar[list[str]] = [
            'batch_size', 'dataloader_drop_last', 'dataloader_shuffle',
            'g_optimizer_key', 'g_optimizer_settings', 'd_optimizer_key', 'd_optimizer_settings',
            'n_g_step_per_batch', 'n_d_step_per_batch', 'torch_manual_seed'
        ]

    @dataclass
    class ModelRecord(ConditionalGenerativeModel.ModelRecord):
        """Record of a vanilla GAN model"""

        # Extending from base record data type – additional fields follow:
        g_network_state: dict
        d_network_state: dict

        g_optimizer_state: dict
        d_optimizer_state: dict

    def __init__(self,
                 instance_id: str | int,
                 model_recipe: 'GAN.ModelRecipe',
                 training_controls: 'GAN.TrainingControls',
                 device: torch.device,
                 ):

        super().__init__(instance_id, model_recipe, training_controls, device)

        self.model_specification: GAN.ModelSpecification = self.recipe.specification
        self.hps: GAN.ModelHyperparameters = self.recipe.hyperparameters
        self.training_settings: CGAN.TrainingSettings = self.recipe.training_settings

        self.noise_generator = NoiseGenerator(
            NoiseGenerator.KEY_NOISE_TYPE_NORMAL,
            n_dimensions=self.hps.n_latent_features,
            device=self.device
        )

    def check_and_complete_network_specs(self):
        """Checks if the first and last linear layers take in and put out the correct number of features.
        Will not work for networks using layers other than Linear."""

        g_linear_layers = [layer for layer in self.hps.generator_spec.layers if isinstance(layer, Linear)]
        d_linear_layers = [layer for layer in self.hps.discriminator_spec.layers if isinstance(layer, Linear)]

        # Check generator
        # Unconditional GAN - initial layer should take n_latent_features number of features
        # Final layer should output number of features to generate
        g_initial_layer = g_linear_layers[0]
        g_final_layer = g_linear_layers[-1]

        if g_initial_layer.in_features == -1:
            g_initial_layer.in_features = self.hps.n_latent_features
        elif g_initial_layer.in_features != self.hps.n_latent_features:
            message = ('First linear layer in the generator network should take in number of features equal to the number '
                       'of latent dimensions!')
            raise ValueError(message)

        if g_final_layer.out_features == -1:
            g_final_layer.out_features = self.hps.n_generate_features
        elif g_final_layer.out_features != self.hps.n_generate_features:
            message = ('Last linear layer in the generator network must output number of features equal to the number '
                       'of features to generate!')
            raise ValueError(message)

        # Check discriminator
        # Unconditional GAN - initial layer should take number of features generated
        # Final layer should output a single feature
        d_initial_layer = d_linear_layers[0]
        d_final_layer = d_linear_layers[-1]

        if d_initial_layer.in_features == -1:
            d_initial_layer.in_features = self.hps.n_generate_features
        elif d_initial_layer.in_features != self.hps.n_generate_features:
            message = ('First linear layer of the discriminator network should take in number of features equal to the '
                       'number of features generated!')
            raise ValueError(message)

        if d_final_layer.out_features == -1:
            d_final_layer.out_features = 1
        elif d_final_layer.out_features != 1:
            message = 'Final layer of the discriminator network must output 1 feature!'
            raise ValueError(message)

    def _create_model(self):

        self.check_and_complete_network_specs()

        # Create generator
        self.generator = self.hps.generator_spec.get_network(device=self.device)

        # g_layer_neuron_counts = [
        #     self.hps.n_latent_features,
        #     *self.hps.g_hidden_layers_neurons,
        #     self.hps.n_generate_features
        # ]

        # Create discriminator
        self.discriminator = self.hps.discriminator_spec.get_network(device=self.device)
        # d_layer_neuron_counts = [
        #     self.hps.n_generate_features,
        #     *self.hps.d_hidden_layers_neurons,
        #     1  # binary output
        # ]

        self.discriminator.insert(0, nn.Dropout())

    def get_record(self) -> 'GAN.ModelRecord':
        return self.ModelRecord(
            instance_id=self.instance_id,
            recipe=self.recipe,
            training_controls=self.training_controls,
            training_status=self.get_training_status(),
            g_network_state=self.generator.state_dict(),
            d_network_state=self.discriminator.state_dict(),
            g_optimizer_state=self.optimizer_g.state_dict(),
            d_optimizer_state=self.optimizer_d.state_dict()
        )

    @classmethod
    def load(cls, fp: str | Path, device: torch.device) -> 'GAN':

        model = super().load(fp, device)
        assert isinstance(model, GAN)

        record: GAN.ModelRecord = torch.load(fp)

        model.generator.load_state_dict(record.g_network_state)
        model.discriminator.load_state_dict(record.d_network_state)

        model.optimizer_g.load_state_dict(record.g_optimizer_state)
        model.optimizer_d.load_state_dict(record.d_optimizer_state)

        return model

    def _init_optimizer(self):
        self.optimizer_g = optimizers_by_key[self.training_settings.g_optimizer_key](
            self.generator.parameters(),
            **self.training_settings.g_optimizer_settings
        )
        self.optimizer_d = optimizers_by_key[self.training_settings.d_optimizer_key](
            self.discriminator.parameters(),
            **self.training_settings.d_optimizer_settings
        )

    def generate(self, n_samples: int) -> torch.Tensor:
        self.generator.eval()
        with torch.no_grad():
            g_out = self._get_generator_outputs(n_samples)
        return g_out

    def activate_train_mode(self):
        self.generator.train()
        self.discriminator.train()

    def _train_batch(self, batch_index: int, batch_data: tuple[torch.Tensor, ...]):
        """Process one (mini)batch of training data and update the network using loss based on the batch."""

        if GAN.KEY_LOSS_ADV not in self._current_epoch_losses_sums:  # TODO ERROR HERE LIKELY RESETTING SUM EVERY ITER
            self._current_epoch_losses_sums.update({GAN.KEY_LOSS_GEN: 0.0, GAN.KEY_LOSS_DISC: 0.0})

        batch_sample_indices, batch_gvars = batch_data

        for i_disc_step in range(self.training_settings.n_d_step_per_batch):
            self.optimizer_d.zero_grad()
            loss_d = self.loss_discriminator(batch_gvars)
            self._current_epoch_losses_sums[GAN.KEY_LOSS_DISC] += loss_d.item()
            loss_d.backward()
            self.optimizer_d.step()

            self._post_discriminator_step(batch_index, batch_data)

        for i_gen_step in range(self.training_settings.n_g_step_per_batch):
            self.optimizer_g.zero_grad()
            loss_g = self.loss_generator(batch_gvars)
            self._current_epoch_losses_sums[GAN.KEY_LOSS_GEN] += loss_g.item()
            loss_g.backward()
            self.optimizer_g.step()

    def loss_generator(self, batch_fts: torch.Tensor) -> torch.Tensor:

        g_out = self._get_generator_outputs(n_samples=batch_fts.shape[0])

        # Discriminator output for fake data (data from generator)
        d_out_fake = self._get_discriminator_outputs(g_out)

        loss_g = self._loss_generator(d_out_fake)
        return self._reduce_loss(loss_g)

    def _loss_generator(self, d_out_fake: torch.Tensor) -> torch.Tensor:
        """Generalized for use by both **unconditional** and **conditional** model"""

        match self.hps.g_loss_formulation:

            case self.KEY_FORMULATION_CLASSIC:
                # Directly from Equation 1 of the original GAN paper
                loss_g = torch.log(1.0 - d_out_fake)

            case self.KEY_FORMULATION_NON_SATURATING:
                # See original GAN paper - classic generator loss saturates.
                # This non-saturating alternative is much more workable.
                loss_g = -1 * torch.log(d_out_fake)

                # # Not using ``binary_cross_entropy_with_logits`` - automatically applies sigmoid
                # # Using ``binary_cross_entropy`` - does not apply sigmoid. Last layer should have sigmoid.
                # # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/13
                #
                # fake_loss = F.binary_cross_entropy(
                #     d_out_fake,
                #     torch.ones_like(d_out_fake, device=self.device, requires_grad=False)
                # )
                # # fake_loss is the difference between discriminator outputs and ONES
                # # if the two are similar, BCE between them will be low, which is what we
                # # want to get towards in training, by minimizing loss_g = fake_loss.
                #
                # loss_g = fake_loss

            case self.KEY_FORMULATION_WASSERSTEIN | self.KEY_FORMULATION_WASSERSTEIN_GP:
                loss_g = -1 * d_out_fake

            case _:
                message = 'Generator loss formulation not recognized.'
                raise KeyError(message)

        return loss_g

    def loss_discriminator(self, batch_fts: torch.Tensor) -> torch.Tensor:

        data_fake = self._get_generator_outputs(n_samples=batch_fts.shape[0])
        d_out_fake = self._get_discriminator_outputs(data_fake)

        data_real = batch_fts
        d_out_real = self._get_discriminator_outputs(data_real)

        loss_d = self._loss_discriminator(d_out_real=d_out_real, d_out_fake=d_out_fake)
        return self._reduce_loss(loss_d)

    def _loss_discriminator(self, d_out_real: torch.Tensor, d_out_fake: torch.Tensor) -> torch.Tensor:
        """Generalized for use by both **unconditional** and **conditional** model"""

        match self.hps.d_loss_formulation:

            case self.KEY_FORMULATION_CLASSIC:
                loss_d = -1 * (torch.log(d_out_real) + torch.log(1.0 - d_out_fake))

            case self.KEY_FORMULATION_NON_SATURATING:
                # Same as classic formulation.
                loss_d = -1 * (torch.log(d_out_real) + torch.log(1.0 - d_out_fake))
                # # Equivalent to:
                # real_loss = F.binary_cross_entropy(
                #     d_out_real,
                #     torch.ones_like(d_out_real, device=self.device, requires_grad=False)
                # )
                # # real_loss is the difference between discriminator outputs for real data and ONES
                # # discriminator should ideally label all real data as REAL = 1
                # # Want to minimize real_loss in training, i.e. discriminator should
                # # label all real data as ONES, and difference of its labels to ONES should decrease
                #
                # fake_loss = F.binary_cross_entropy(
                #     d_out_fake,
                #     torch.zeros_like(d_out_fake, device=self.device, requires_grad=False)
                # )
                # # fake_loss is the difference between discriminator outputs for fake data and ZEROS
                # # discriminator should ideally label all fake data as FAKE = 0
                # # Want to minimize fake_loss in training, i.e. discriminator should
                # # label all fake data as ZEROS, and the difference of its labels to ZEROS should decrease
                #
                # loss_d = real_loss + fake_loss

            case self.KEY_FORMULATION_WASSERSTEIN:
                loss_d = -1 * d_out_real + d_out_fake

            case _:
                message = 'Discriminator loss formulation not recognized.'
                raise KeyError(message)

        return loss_d

    def _post_discriminator_step(self, batch_index: int, batch_data: tuple[torch.Tensor, ...]):
        pass

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.hps.reduction == self.hps.KEY_REDUCE_SUM:
            return loss.sum()
        elif self.hps.reduction == self.hps.KEY_REDUCE_MEAN:
            return loss.mean()

    def _get_generator_outputs(self, n_samples: int):
        latents = self.noise_generator.get_noise(n_samples)
        g_out = self.generator.forward(latents)
        return g_out

    def _get_discriminator_outputs(self, data: torch.Tensor):
        d_out = self.discriminator.forward(data)

        clamped = self._clamp_discriminator_outputs(d_out)
        return clamped

    def _clamp_discriminator_outputs(self, d_out: torch.Tensor):

        if self.hps.clamp_discriminator_outputs:
            clamped = torch.clamp(
                d_out,
                min=self.hps.epsilon,
                max=1.0 - self.hps.epsilon
            )
        else:
            # Not clamped
            clamped = d_out

        return clamped

    class Unit(nn.Module):

        def __init__(self,
                     device: torch.device,
                     layer_neuron_counts: list[int],
                     activation_fcns: tuple[str, ...]):
            """Container for a network. Independent networks, e.g. discriminator and the generator, can each be modeled
            as a different "``Unit``"."""

            super().__init__()
            self.layer_neuron_counts = layer_neuron_counts
            self.activation_fcns = activation_fcns

            self.device = device
            self._create_network()

        def _create_network(self):
            self.network = get_fc_network(self.layer_neuron_counts, self.activation_fcns)

        def forward(self, _input: torch.Tensor) -> torch.Tensor:
            return self.network(_input)


GAN.ModelRecipe._used_classes += [GAN.ModelHyperparameters, GAN.TrainingSettings]


class CGAN(GAN, ConditionalGenerativeModel):

    MODEL_TYPE_ABBREVIATION = 'CGAN'
    MODEL_TYPE_NAME_LONG = 'Conditional Generative Adversarial Network'

    KEY_LOSS_GEN = 'GEN'
    KEY_LOSS_DISC = 'DISC'
    KEY_LOSS_ADV = 'ADV'
    """Key for adversarial loss"""

    @dataclass
    class ModelHyperparameters(GAN.ModelHyperparameters, _BaseCGANModelHyperparameters):

        _data_type_str: ClassVar[str] = 'cgan_model_hyperparameters'
        _data_type_key: ClassVar[int] = 550

        _save_fields = sorted(list(set(GAN.ModelHyperparameters._save_fields + _BaseCGANModelHyperparameters._save_fields)))

        def __post_init__(self):
            errors = []

            if not isinstance(self.discriminator_spec.layers[-1], Sigmoid):
                errors.append('Using BCE loss formulation for DISC, but DISC last hidden layer activation function is not sigmoid.')

            g_linear_layers = [layer for layer in self.generator_spec.layers if isinstance(layer, Linear)]
            d_linear_layers = [layer for layer in self.discriminator_spec.layers if isinstance(layer, Linear)]

            # Check generator
            # CONDITIONAL GAN - initial layer should take n_latent_features + n_cond number of features
            # Final layer should output number of features to generate
            g_initial_layer = g_linear_layers[0]
            g_final_layer = g_linear_layers[-1]

            g_initial_layer_n_fts = self.n_latent_features + self.n_condition_features
            if g_initial_layer.in_features == -1:
                g_initial_layer.in_features = g_initial_layer_n_fts
            elif g_initial_layer.in_features != g_initial_layer_n_fts:
                message = ('First linear layer in the generator network should take in number of features equal to the number '
                           f'of latent dimensions and the number of conditions! (={g_initial_layer_n_fts})')
                errors.append(message)

            if g_final_layer.out_features == -1:
                g_final_layer.out_features = self.n_generate_features
            elif g_final_layer.out_features != self.n_generate_features:
                message = ('Last linear layer in the generator network must output number of features equal to the number '
                           f'of features to generate! (={self.n_generate_features})')
                errors.append(message)

            # Check discriminator
            # CONDITIONAL GAN - initial layer should take number of features generated + number of conditions
            # Final layer should output a single feature
            d_initial_layer = d_linear_layers[0]
            d_final_layer = d_linear_layers[-1]

            d_initial_layer_n_fts = self.n_generate_features + self.n_condition_features
            if d_initial_layer.in_features == -1:
                d_initial_layer.in_features = d_initial_layer_n_fts
            elif d_initial_layer.in_features != d_initial_layer_n_fts:
                message = ('First linear layer of the discriminator network should take in number of features equal to the '
                           f'number of features generated and the number of conditions! (={d_initial_layer_n_fts})')
                errors.append(message)

            if d_final_layer.out_features == -1:
                d_final_layer.out_features = 1
            elif d_final_layer.out_features != 1:
                message = 'Final layer of the discriminator network must output 1 feature!'
                errors.append(message)

            if errors:
                newline = '\n'
                raise ValueError(f'Following errors were found with the hyperparameters:{newline.join(errors)}')
            else:
                logger.info(f'Hyperparameter checks passed.')

    def __init__(self,
                 instance_id: str | int,
                 model_recipe: 'CGAN.ModelRecipe',
                 training_controls: 'CGAN.TrainingControls',
                 device: torch.device,
                 ):

        super().__init__(instance_id, model_recipe, training_controls, device)

        # Re-setting these here to fix type hinting errors - reinforce that these are the classes of CRNVP
        self.model_specification: CGAN.ModelSpecification = self.recipe.specification
        self.hps: CGAN.ModelHyperparameters = self.recipe.hyperparameters
        self.training_settings: CGAN.TrainingSettings = self.recipe.training_settings

    def check_and_complete_network_specs(self):
        """Checks if the first and last linear layers take in and put out the correct number of features.
        Will not work for networks using layers other than Linear."""

        g_linear_layers = [layer for layer in self.hps.generator_spec.layers if isinstance(layer, Linear)]
        d_linear_layers = [layer for layer in self.hps.discriminator_spec.layers if isinstance(layer, Linear)]

        # Check generator
        # CONDITIONAL GAN - initial layer should take n_latent_features + n_cond number of features
        # Final layer should output number of features to generate
        g_initial_layer = g_linear_layers[0]
        g_final_layer = g_linear_layers[-1]

        g_initial_layer_n_fts = self.hps.n_latent_features + self.hps.n_condition_features
        if g_initial_layer.in_features == -1:
            g_initial_layer.in_features = g_initial_layer_n_fts
        elif g_initial_layer.in_features != g_initial_layer_n_fts:
            message = ('First linear layer in the generator network should take in number of features equal to the number '
                       f'of latent dimensions and the number of conditions! (={g_initial_layer_n_fts})')
            raise ValueError(message)

        if g_final_layer.out_features == -1:
            g_final_layer.out_features = self.hps.n_generate_features
        elif g_final_layer.out_features != self.hps.n_generate_features:
            message = ('Last linear layer in the generator network must output number of features equal to the number '
                       f'of features to generate! (={self.hps.n_generate_features})')
            raise ValueError(message)

        # Check discriminator
        # CONDITIONAL GAN - initial layer should take number of features generated + number of conditions
        # Final layer should output a single feature
        d_initial_layer = d_linear_layers[0]
        d_final_layer = d_linear_layers[-1]

        d_initial_layer_n_fts = self.hps.n_generate_features + self.hps.n_condition_features
        if d_initial_layer.in_features == -1:
            d_initial_layer.in_features = d_initial_layer_n_fts
        elif d_initial_layer.in_features != d_initial_layer_n_fts:
            message = ('First linear layer of the discriminator network should take in number of features equal to the '
                       f'number of features generated and the number of conditions! (={d_initial_layer_n_fts})')
            raise ValueError(message)

        if d_final_layer.out_features == -1:
            d_final_layer.out_features = 1
        elif d_final_layer.out_features != 1:
            message = 'Final layer of the discriminator network must output 1 feature!'
            raise ValueError(message)

    def _create_model(self):

        self.check_and_complete_network_specs()

        # Create generator
        # g_layer_neuron_counts = [
        #     self.hps.n_latent_features + self.hps.n_condition_features,
        #     *self.hps.g_hidden_layers_neurons,
        #     self.hps.n_generate_features
        # ]

        self.generator = self.hps.generator_spec.get_network(device=self.device)

        # Create discriminator
        # d_layer_neuron_counts = [
        #     self.hps.n_generate_features + self.hps.n_condition_features,
        #     *self.hps.d_hidden_layers_neurons,
        #     1  # binary output
        # ]

        self.discriminator = self.hps.discriminator_spec.get_network(device=self.device)

        self.discriminator.insert(0, nn.Dropout())

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

    def _train_batch(self, batch_index: int, batch_data: tuple[torch.Tensor, ...]):
        """Process one (mini)batch of training data and update the network using loss based on the batch."""

        if CGAN.KEY_LOSS_ADV not in self._current_epoch_losses_sums:
            self._current_epoch_losses_sums.update({CGAN.KEY_LOSS_GEN: 0.0, CGAN.KEY_LOSS_DISC: 0.0})

        batch_sample_indices, batch_gvars, batch_conds = batch_data

        for i_disc_step in range(self.training_settings.n_d_step_per_batch):
            self.optimizer_d.zero_grad()
            loss_d = self.loss_discriminator(batch_gvars, batch_conds)
            self._current_epoch_losses_sums[CGAN.KEY_LOSS_DISC] += loss_d.item()
            loss_d.backward(retain_graph=False)
            self.optimizer_d.step()

            self._post_discriminator_step(batch_index, batch_data)

        for i_gen_step in range(self.training_settings.n_g_step_per_batch):
            self.optimizer_g.zero_grad()
            loss_g = self.loss_generator(batch_gvars, batch_conds)
            self._current_epoch_losses_sums[CGAN.KEY_LOSS_GEN] += loss_g.item()
            loss_g.backward(retain_graph=False)
            self.optimizer_g.step()

    def loss_generator(self, batch_fts: torch.Tensor, batch_conds: torch.Tensor = None) -> torch.Tensor:
        if batch_conds is None:
            raise Exception()

        g_out = self._get_generator_outputs(batch_conds)

        # Discriminator output for fake data (data from generator)
        d_out_fake = self._get_discriminator_outputs(g_out, batch_conds)

        loss_g = self._loss_generator(d_out_fake)
        return self._reduce_loss(loss_g)

    def loss_discriminator(self, batch_fts: torch.Tensor, batch_conds: torch.Tensor = None) -> torch.Tensor:
        if batch_conds is None:
            raise Exception()

        data_fake = self._get_generator_outputs(batch_conds)
        d_out_fake = self._get_discriminator_outputs(data_fake, batch_conds)

        data_real = batch_fts
        d_out_real = self._get_discriminator_outputs(data_real, batch_conds)

        loss_d = self._loss_discriminator(d_out_real=d_out_real, d_out_fake=d_out_fake)
        return self._reduce_loss(loss_d)

    def _get_generator_outputs(self, conds: torch.Tensor):
        latents = self.noise_generator.get_noise(n_samples=conds.shape[0])
        latents_conds = torch.concat((latents, conds), dim=1).to(self.device)

        g_out = self.generator.forward(latents_conds)
        return g_out

    def _get_discriminator_outputs(self, data: torch.Tensor, conds: torch.Tensor = None):
        if conds is None:
            raise Exception()

        data_conds = torch.concat((data, conds), dim=1).to(self.device)
        d_out = self.discriminator.forward(data_conds)

        clamped = self._clamp_discriminator_outputs(d_out)

        return clamped


CGAN.ModelRecipe._used_classes += [CGAN.ModelHyperparameters, CGAN.TrainingSettings]


class WGAN(GAN):
    """Wasserstein GAN. Uses a different loss formulation for the generator and the discriminator compared
    to the vanilla GAN."""

    MODEL_TYPE_ABBREVIATION = 'WGAN'
    MODEL_TYPE_NAME_LONG = 'Wasserstein Generative Adversarial Network'

    @dataclass
    class ModelHyperparameters(_BaseGANModelHyperparameters):

        d_weight_clip_bounds: tuple[float, float]

        _data_type_str: ClassVar[str] = 'wgan_model_hyperparameters'
        _data_type_key: ClassVar[int] = 550

        _save_fields: ClassVar[list[str]] = _BaseGANModelHyperparameters._save_fields + [
            'd_weight_clip_bounds'
        ]

        # Default, fixed override of loss formulation fields.
        d_loss_formulation: ClassVar[str] = GAN.KEY_FORMULATION_WASSERSTEIN
        g_loss_formulation: ClassVar[str] = GAN.KEY_FORMULATION_WASSERSTEIN

        def __post_init__(self):
            if not self.d_weight_clip_bounds[0] < 0 or self.d_weight_clip_bounds[1] > 1:
                logger.warning('WGAN discriminator weight clipping bounds: Lower limit is not negative, '
                               'upper limit is not positive.')

    @dataclass
    class ModelRecipe(GAN.ModelRecipe):

        hyperparameters: 'WGAN.ModelHyperparameters'

        _data_type_str: ClassVar[str] = 'wgan_model_recipe'
        _data_type_key: ClassVar[int] = 550

    ModelRecipe._used_classes += [ModelHyperparameters]

    def _post_discriminator_step(self, batch_index: int, batch_data: tuple[torch.Tensor, ...]):
        # Wasserstein GAN discriminator weight clipping
        for d_param in self.discriminator.parameters():
            assert isinstance(self.hps, WGAN.ModelHyperparameters)
            d_param.data.clamp_(*self.hps.d_weight_clip_bounds)


class CWGAN(WGAN, CGAN):

    MODEL_TYPE_ABBREVIATION = 'CWGAN'
    MODEL_TYPE_NAME_LONG = 'Conditional Wasserstein Generative Adversarial Network'

    @dataclass
    class ModelHyperparameters(WGAN.ModelHyperparameters, _BaseCGANModelHyperparameters):

        _data_type_str: ClassVar[str] = 'cwgan_model_hyperparameters'
        _data_type_key: ClassVar[int] = 550

        _save_fields: ClassVar[list[str]] = _BaseCGANModelHyperparameters._save_fields + WGAN.ModelHyperparameters._save_fields

    @dataclass
    class ModelRecipe(GAN.ModelRecipe):

        hyperparameters: 'CWGAN.ModelHyperparameters'

        _data_type_str: ClassVar[str] = 'cwgan_model_recipe'
        _data_type_key: ClassVar[int] = 550

    ModelRecipe._used_classes += [ModelHyperparameters]


class WGANGP(GAN):

    MODEL_TYPE_ABBREVIATION = 'WGANGP'
    MODEL_TYPE_NAME_LONG = 'Wasserstein Generative Adversarial Network with Gradient Penalty'

    @dataclass
    class ModelHyperparameters(_BaseGANModelHyperparameters):

        gp_multiplier: float

        _data_type_str: ClassVar[str] = 'wgangp_model_hyperparameters'
        _data_type_key: ClassVar[int] = 550

        _save_fields: ClassVar[list[str]] = _BaseGANModelHyperparameters._save_fields + [
            'gp_multiplier'
        ]

        # Default, fixed override of loss formulation fields.
        d_loss_formulation: ClassVar[str] = GAN.KEY_FORMULATION_WASSERSTEIN_GP
        g_loss_formulation: ClassVar[str] = GAN.KEY_FORMULATION_WASSERSTEIN_GP

    @dataclass
    class ModelRecipe(GAN.ModelRecipe):

        hyperparameters: 'WGANGP.ModelHyperparameters'

    ModelRecipe._used_classes += [ModelHyperparameters]

    def loss_discriminator(self, batch_fts: torch.Tensor) -> torch.Tensor:

        data_fake = self._get_generator_outputs(n_samples=batch_fts.shape[0])
        d_out_fake = self._get_discriminator_outputs(data_fake)

        data_real = batch_fts
        d_out_real = self._get_discriminator_outputs(data_real)

        # Random interpolation weights between real and fake data
        alpha = torch.randn(size=(batch_fts.shape[0], 1), device=self.device)

        # Artificial data, interpolation between real and fake data
        data_interp = (alpha * data_real + (1.0 - alpha) * data_fake).requires_grad_(True)
        d_out_interp = self._get_discriminator_outputs(data_interp)

        gradients = torch.autograd.grad(
            outputs=d_out_interp,
            inputs=data_interp,
            grad_outputs=torch.ones_like(d_out_interp, device=self.device, requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        gradients = gradients[0]
        gradients = gradients.view(gradients.shape[0], -1)
        grad_penalty = torch.mean((gradients.norm(2, dim=1) - 1.0) ** 2)

        # Wasserstein loss formulation, with added gradient penalty
        assert isinstance(self.hps, WGANGP.ModelHyperparameters)
        loss_d = -1 * d_out_real + d_out_fake + self.hps.gp_multiplier * grad_penalty
        return self._reduce_loss(loss_d)


class CWGANGP(WGANGP, CGAN):

    MODEL_TYPE_ABBREVIATION = 'CWGANGP'
    MODEL_TYPE_NAME_LONG = 'Conditional Wasserstein Generative Adversarial Network with Gradient Penalty'

    @dataclass
    class ModelHyperparameters(WGANGP.ModelHyperparameters, _BaseCGANModelHyperparameters):

        _data_type_str: ClassVar[str] = 'cwgangp_model_hyperparameters'
        _data_type_key: ClassVar[int] = 550

        _save_fields: ClassVar[list[str]] = _BaseCGANModelHyperparameters._save_fields + WGANGP.ModelHyperparameters._save_fields

    @dataclass
    class ModelRecipe(GAN.ModelRecipe):

        hyperparameters: 'CWGANGP.ModelHyperparameters'

        _data_type_str: ClassVar[str] = 'cwgangp_model_recipe'
        _data_type_key: ClassVar[int] = 550

    ModelRecipe._used_classes += [ModelHyperparameters]

    def loss_discriminator(self, batch_fts: torch.Tensor, batch_conds: torch.Tensor = None) -> torch.Tensor:

        data_fake = self._get_generator_outputs(batch_conds)
        d_out_fake = self._get_discriminator_outputs(data_fake, batch_conds)

        data_real = batch_fts
        d_out_real = self._get_discriminator_outputs(data_real, batch_conds)

        # Random interpolation weights between real and fake data
        alpha = torch.randn(size=(batch_fts.shape[0], 1), device=self.device)

        # Artificial data, interpolation between real and fake data
        data_interp = (alpha * data_real + (1.0 - alpha) * data_fake).requires_grad_(True)
        d_out_interp = self._get_discriminator_outputs(data_interp, batch_conds)

        gradients = torch.autograd.grad(
            outputs=d_out_interp,
            inputs=data_interp,
            grad_outputs=torch.ones_like(d_out_interp, device=self.device, requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        gradients = gradients[0]
        gradients = gradients.view(gradients.shape[0], -1)
        grad_penalty = torch.mean((gradients.norm(2, dim=1) - 1.0) ** 2)

        # Wasserstein loss formulation, with added gradient penalty
        assert isinstance(self.hps, CWGANGP.ModelHyperparameters)
        loss_d = -1 * d_out_real + d_out_fake + self.hps.gp_multiplier * grad_penalty
        return self._reduce_loss(loss_d)


class GANValidatorBundle:

    class GANValidator(Validator):

        def __init__(self,
                     model: GAN,
                     validation_dataset: MDevDataset,
                     validation_loss_type: str,
                     loss_method: Callable):
            super().__init__(model)

            self.validation_loss_type = validation_loss_type
            self.loss_method = loss_method

            self.validation_dataset = validation_dataset
            self.model.set_mdev_dataset_groups(self.validation_dataset)

        def calculate_validation_loss(self, epoch_report: GAN.EpochReport):

            with torch.no_grad():

                assert isinstance(self.model, GAN)
                indices, generate_vars = self.validation_dataset.get_all_rows_in_groups()
                val_loss = self.loss_method(generate_vars).item()

            val_losses = {self.validation_loss_type: val_loss}
            self._add_to_validation_losses(epoch_report.epoch_index, val_losses[self.validation_loss_type])
            logger.log(LOG_LEVEL_VALIDATION, f'Epoch {epoch_report.epoch_index}: {self.validation_loss_type}={val_loss}')
            
    class CGANValidator(GANValidator):
        """Overrides regular ``GANValidator`` only in validation loss calculation as it also sends the condition
        data to the loss calculator."""

        def calculate_validation_loss(self, epoch_report: CGAN.EpochReport):
            with torch.no_grad():

                assert isinstance(self.model, CGAN)
                indices, generate_vars, conds = self.validation_dataset.get_all_rows_in_groups()
                val_loss = self.loss_method(generate_vars, conds).item()  # DIFFERENT FOR CGAN

            val_losses = {self.validation_loss_type: val_loss}
            self._add_to_validation_losses(epoch_report.epoch_index, val_losses[self.validation_loss_type])
            logger.log(LOG_LEVEL_VALIDATION, f'Epoch {epoch_report.epoch_index}: {self.validation_loss_type}={val_loss}')

    def __init__(self,
                 model: GAN | CGAN,
                 validation_dataset: MDevDataset):
        """Container for generator and discriminator validators.
        Can work with both unconditional & conditional GANs."""

        self.model = model
        self.validation_dataset = validation_dataset

        if isinstance(self.model, CGAN):
            validator_class = self.CGANValidator
        else:
            assert isinstance(self.model, GAN)
            validator_class = self.GANValidator

        # Generator validation loss calculator
        self.g_validator = validator_class(self.model,
                                           self.validation_dataset,
                                           validation_loss_type=self.model.KEY_LOSS_GEN,
                                           loss_method=self.model.loss_generator)

        # Discriminator validation loss calculator
        self.d_validator = validator_class(self.model,
                                           self.validation_dataset,
                                           validation_loss_type=self.model.KEY_LOSS_DISC,
                                           loss_method=self.model.loss_discriminator)

    @property
    def validators(self) -> list[Validator]:
        return [self.g_validator, self.d_validator]
