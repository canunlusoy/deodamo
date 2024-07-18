import logging
from typing import Callable, Union, Type, ClassVar, Iterable, Any, Literal
from pathlib import Path
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from src.utils.iox import ProgramData
from src.datamodels.variables import Variable
from src.mdev.utilities import activation_fcns_by_key, optimizers_by_key, data_container_row_index, NoiseGenerator, MDevDataset, get_fc_network
from src.mdev.generative.base import GenerativeModel, ConditionalGenerativeModel, Validator
from src.mdev.logger import LOG_LEVEL_NAME_VALIDATION, LOG_LEVEL_VALIDATION
from src.mdev.archspec import NetworkSpecification, Linear, ActivationFunction


logger = logging.getLogger(__name__)


class VAE(GenerativeModel):

    MODEL_TYPE_ABBREVIATION = 'VAE'
    MODEL_TYPE_NAME_LONG = 'Variational Autoencoder'

    KEY_LATENT_DIST_STANDARD_NORMAL = 'standard_normal'

    KEY_LOSS_NEG_ELBO = 'neg_elbo'
    KEY_LOSS_RECONSTRUCTION = 'reconstruction'
    KEY_LOSS_KL = 'kl_divergence'

    KEY_RECONSTRUCTION_MSE = 'mse'

    @dataclass
    class ModelHyperparameters(GenerativeModel.ModelHyperparameters):

        n_generate_features: int
        """Number of features to generate"""

        n_latent_features: int
        """Number of features in the latent space"""

        encoder_spec: NetworkSpecification
        """Architecture specification for the encoder"""

        decoder_spec: NetworkSpecification
        """Architecture specification for the decoder"""

        latent_distribution: str
        """Prior distribution for latent features"""

        loss_reconstruction_formulation: str
        """Reconstruction loss formulation"""

        loss_reconstruction_multiplier: float | int
        """Multiplier for weighing the reconstruction loss in ELBO"""

        _data_type_str: ClassVar[str] = 'vae_model_hyperparameters'
        _data_type_key: ClassVar[int] = 550

        _save_fields: ClassVar[list[str]] = GenerativeModel.ModelHyperparameters._save_fields + [
            'n_generate_features', 'n_latent_features', 'encoder_spec', 'decoder_spec', 'latent_distribution',
            'loss_reconstruction_formulation', 'loss_reconstruction_multiplier'
        ]

        _used_classes = GenerativeModel.ModelHyperparameters._used_classes + [NetworkSpecification]

        def __post_init__(self):
            # Check network outer layer feature counts
            errors = []

            encoder_first_layer = self.encoder_spec.layers[0]
            if isinstance(encoder_first_layer, Linear) and encoder_first_layer.in_features != self.n_generate_features:
                errors.append(f'Encoder first layer must have {self.n_generate_features} features as per ``n_generate_features``.')

            encoder_last_layer = self.encoder_spec.layers[-1]
            if isinstance(encoder_last_layer, ActivationFunction):
                encoder_last_layer = self.encoder_spec.layers[-2]  # try the one before
            if isinstance(encoder_last_layer, Linear) and encoder_last_layer.out_features / 2 != self.n_latent_features:
                errors.append(f'Encoder last layer must have 2 x {self.n_latent_features} features as per ``n_latent_features``.')

            decoder_first_layer = self.decoder_spec.layers[0]
            if isinstance(decoder_first_layer, Linear) and decoder_first_layer.in_features != self.n_latent_features:
                errors.append(f'Decoder first layer must have {self.n_latent_features} features as per ``n_latent_features``.')

            decoder_last_layer = self.decoder_spec.layers[-1]
            if isinstance(decoder_last_layer, ActivationFunction):
                decoder_last_layer = self.decoder_spec.layers[-2]  # try the one before
            if isinstance(decoder_last_layer, Linear) and decoder_last_layer.out_features != self.n_generate_features:
                errors.append(f'Decoder last layer must have {self.n_generate_features} features as per ``n_generate_features``.')

            if errors:
                errors_str = "\n".join(errors)
                message = f'The following errors were found with the encoder and/or decoder network specification:\n{errors_str}'
                raise ValueError(message)
            else:
                logger.info('Network outer layer feature counts are consistent with other model hyperparameters.')

    @dataclass
    class ModelRecord(GenerativeModel.ModelRecord):
        # Extending from base record data type – additional fields follow:
        bundle_network_state: dict
        optimizer_state: dict

    def __init__(self,
                 instance_id: str | int,
                 model_recipe: 'VAE.ModelRecipe',
                 training_controls: 'VAE.TrainingControls',
                 device: torch.device
                 ):

        super().__init__(instance_id, model_recipe, training_controls, device)


        # Re-setting these here to fix type hinting errors - reinforce that these are the classes of CRNVP
        self.model_specification: VAE.ModelSpecification = self.recipe.specification
        self.hps: VAE.ModelHyperparameters = self.recipe.hyperparameters
        self.training_settings: VAE.TrainingSettings = self.recipe.training_settings

    def _create_model(self):
        self.bundle = VAE._Bundle(self.device, self.hps.encoder_spec, self.hps.decoder_spec, self)
        self.encoder = self.bundle.encoder  # alias
        self.decoder = self.bundle.decoder  # alias

    def get_record(self) -> 'VAE.ModelRecord':
        return self.ModelRecord(
            instance_id=self.instance_id,
            recipe=self.recipe,
            training_controls=self.training_controls,
            training_status=self.get_training_status(),
            bundle_network_state=self.bundle.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
        )

    @classmethod
    def load(cls, fp: str | Path, device: torch.device) -> 'VAE':

        model = super().load(fp, device)
        assert isinstance(model, VAE)

        record: VAE.ModelRecord = torch.load(fp)

        model.bundle.load_state_dict(record.bundle_network_state)
        model.optimizer.load_state_dict(record.optimizer_state)

        return model

    def _init_optimizer(self):
        self.optimizer = optimizers_by_key[self.training_settings.optimizer_key](
            self.bundle.parameters(),
            **self.training_settings.optimizer_settings
        )

    def activate_train_mode(self):
        self.bundle.train()

    def _train_batch(self, batch_index: int, batch_data: tuple[torch.Tensor, ...]):

        self.activate_train_mode()

        if VAE.KEY_LOSS_NEG_ELBO not in self._current_epoch_losses_sums:
            self._current_epoch_losses_sums.update({VAE.KEY_LOSS_NEG_ELBO: 0.0, VAE.KEY_LOSS_RECONSTRUCTION: 0.0, VAE.KEY_LOSS_KL: 0.0})

        self.optimizer.zero_grad()

        batch_sample_indices, batch_gvars = batch_data

        loss_reconstruction, loss_kl, loss_neg_elbo = self.loss_neg_elbo(batch_gvars)

        self._current_epoch_losses_sums[VAE.KEY_LOSS_RECONSTRUCTION] += loss_reconstruction.item()
        self._current_epoch_losses_sums[VAE.KEY_LOSS_KL] += loss_kl.item()
        self._current_epoch_losses_sums[VAE.KEY_LOSS_NEG_ELBO] += loss_neg_elbo.item()

        loss_neg_elbo.backward()
        self.optimizer.step()

    def loss_neg_elbo(self, batch_fts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the negative ELBO and returns it alongside its constituents.

        :returns: loss_reconstruction, loss_kl, loss_neg_elbo
        """
        x = batch_fts
        mean, log_variance = self.encoder.encode(x)

        # Sampling and re-parameterization trick
        z = self.encoder.sample(mean, log_variance)

        # Decoding
        x_reconstruction = self.decoder.decode(z)

        # Calculate ELBO
        re_multipler = self.hps.loss_reconstruction_multiplier  # alias

        loss_reconstruction = self._loss_reconstruction(x, x_reconstruction)
        loss_kl = self._loss_kl(mean, log_variance)

        loss_neg_elbo = (re_multipler * loss_reconstruction + loss_kl).mean(dim=0)

        return loss_reconstruction, loss_kl, loss_neg_elbo

    def _loss_reconstruction(self, x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:

        match self.hps.loss_reconstruction_formulation:

            case self.KEY_RECONSTRUCTION_MSE:
                loss_re = torch.pow(x - x_reconstructed, 2).mean()

            case _:
                raise NotImplementedError()

        return loss_re

    def _loss_kl(self, mean: torch.Tensor, log_variance: torch.Tensor):
        """Returns the Kullback-Leibler divergence"""

        # Equation 10 - appendix B in Auto-Encoding Variational Bayes - https://arxiv.org/pdf/1312.6114.pdf
        loss_kl = torch.mean(-0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=1), dim=0)

        return loss_kl

    def generate(self, n_samples: int) -> torch.Tensor:

        self.bundle.eval()

        with torch.no_grad():
            z = self.get_latent_samples(n_samples)
            x_reconstructed = self.decoder.decode(z)

        return x_reconstructed

    def get_latent_samples(self, n_samples: int) -> torch.Tensor:
        match self.hps.latent_distribution:

            case VAE.KEY_LATENT_DIST_STANDARD_NORMAL:
                latents = torch.randn((n_samples, self.hps.n_latent_features), device=self.device)

            case _:
                raise NotImplementedError()

        return latents

    class _Bundle(nn.Module):

        def __init__(self,
                     device: torch.device,
                     encoder_spec: NetworkSpecification,
                     decoder_spec: NetworkSpecification,
                     vae: 'VAE',
                     ):

            super().__init__()

            self.device = device

            self.encoder = VAE.Encoder(self.device, encoder_spec, vae, vae.hps)
            self.decoder = VAE.Decoder(self.device, decoder_spec)

    class Unit(nn.Module):

        def __init__(self,
                     device: torch.device,
                     network_specification: NetworkSpecification
                     ):

            super().__init__()
            self.device = device
            self.netspec = network_specification

            self._create_network()
            self.network.to(self.device)

        def _create_network(self):
            self.network = self.netspec.get_network()

        def forward(self, _input: torch.Tensor) -> torch.Tensor:
            return self.network(_input)

    class Decoder(Unit):

        def __init__(self,
                     device: torch.device,
                     network_specification: NetworkSpecification,
                     ):

            super().__init__(device, network_specification)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            x_reconstructed = self.network(z)
            return x_reconstructed

    class Encoder(Unit):

        def __init__(self,
                     device: torch.device,
                     network_specification: NetworkSpecification,
                     vae: 'VAE',
                     hps: 'VAE.ModelHyperparameters'
                     ):

            super().__init__(device, network_specification)
            self.vae = vae
            self.hps = hps

        def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Returns mean and log_variance pairs corresponding to inputs X

            :return: mean tensor, log_variance tensor"""

            h_e = self.network(x)

            # Split a tensor into the specified number of chunks.
            # Each chunk is a view of the input tensor.
            # Split output into 2, half is mu, other half is log variance
            mean, log_variance = torch.chunk(h_e, 2, dim=1)

            return mean, log_variance

        def sample(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
            """
            Returns z samples corresponding to x's.

            Cannot backpropagate through random normal sampling. Use re-parameterization trick:
            μ and σ come from network, stochastic samples are given as input to network -
            remove sampling step from main pipeline.
            """

            standard_deviation = torch.exp(0.5 * log_variance)

            eps = self.vae.get_latent_samples(n_samples=mean.shape[0])

            # Map standard normal to distribution given by mu and var
            # z = (x-mu)/sigma
            # x = z * sigma + mu
            mapped = mean + standard_deviation * eps
            return mapped

        def log_prob(self, z: torch.Tensor, mean: torch.Tensor, log_variance: torch.Tensor):

            match self.hps.latent_distribution:

                case VAE.KEY_LATENT_DIST_STANDARD_NORMAL:
                    # Returns logarithm of the probability density function value for a normal distribution parameterized by
                    # ``mean`` and ``log_variance`` for samples in ``z``. Assumes a **diagonal covariance matrix**.
                    # See for example implementation:
                    # https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal.log_prob

                    D = z.shape[1]
                    # The multiplication with ``D`` confuses me - PyTorch implementation doesn't seem to have it
                    # Does it add just some constant factor to all so doesn't matter?
                    log_prob = (
                            - 0.5 * D * torch.log(Tensor(2. * torch.pi))
                            - 0.5 * log_variance
                            - 0.5 * torch.exp(-log_variance) * (z - mean) ** 2
                    )

                case _:
                    raise NotImplementedError()

            return log_prob


VAE.ModelRecipe._used_classes += [VAE.ModelHyperparameters]


class CVAE(VAE, ConditionalGenerativeModel):

    MODEL_TYPE_ABBREVIATION = 'CVAE'
    MODEL_TYPE_NAME_LONG = 'Conditional Variational Autoencoder'

    @dataclass
    class ModelHyperparameters(VAE.ModelHyperparameters):

        n_condition_features: int
        """Number of condition features"""

        _data_type_str: ClassVar[str] = 'cvae_model_hyperparameters'
        _data_type_key: ClassVar[int] = 550

        _save_fields: ClassVar[list[str]] = VAE.ModelHyperparameters._save_fields + [
            'n_condition_features'
        ]

        def __post_init__(self):
            # Check network outer layer feature counts
            errors = []

            encoder_first_layer = self.encoder_spec.layers[0]
            if isinstance(encoder_first_layer, Linear) and encoder_first_layer.in_features != self.n_generate_features + self.n_condition_features:
                errors.append(f'Encoder first layer must have {self.n_generate_features + self.n_condition_features} features as per '
                              f'``n_generate_features + n_condition_features``.')

            encoder_last_layer = self.encoder_spec.layers[-1]
            if isinstance(encoder_last_layer, ActivationFunction):
                encoder_last_layer = self.encoder_spec.layers[-2]  # try the one before
            if isinstance(encoder_last_layer, Linear) and encoder_last_layer.out_features / 2 != self.n_latent_features:
                errors.append(f'Encoder last layer must have 2 x {self.n_latent_features} features as per ``n_latent_features``.')

            decoder_first_layer = self.decoder_spec.layers[0]
            if isinstance(decoder_first_layer, Linear) and decoder_first_layer.in_features != self.n_latent_features + self.n_condition_features:
                errors.append(f'Decoder first layer must have {self.n_latent_features + self.n_condition_features} features as per '
                              f'``n_latent_features + n_condition_features``.')

            decoder_last_layer = self.decoder_spec.layers[-1]
            if isinstance(decoder_last_layer, ActivationFunction):
                decoder_last_layer = self.decoder_spec.layers[-2]  # try the one before
            if isinstance(decoder_last_layer, Linear) and decoder_last_layer.out_features != self.n_generate_features:
                errors.append(f'Decoder last layer must have {self.n_generate_features} features as per ``n_generate_features``.')

            if errors:
                errors_str = "\n".join(errors)
                message = f'The following errors were found with the encoder and/or decoder network specification:\n{errors_str}'
                raise ValueError(message)
            else:
                logger.info('Network outer layer feature counts are consistent with other model hyperparameters.')

    def _train_batch(self, batch_index: int, batch_data: tuple[torch.Tensor, ...]):
        self.activate_train_mode()

        if VAE.KEY_LOSS_NEG_ELBO not in self._current_epoch_losses_sums:
            self._current_epoch_losses_sums.update({VAE.KEY_LOSS_NEG_ELBO: 0.0, VAE.KEY_LOSS_RECONSTRUCTION: 0.0, VAE.KEY_LOSS_KL: 0.0})

        self.optimizer.zero_grad()

        batch_sample_indices, batch_fts, batch_conds = batch_data

        loss_reconstruction, loss_kl, loss_neg_elbo = self.loss_neg_elbo(batch_fts, batch_conds)

        self._current_epoch_losses_sums[VAE.KEY_LOSS_RECONSTRUCTION] += loss_reconstruction.item()
        self._current_epoch_losses_sums[VAE.KEY_LOSS_KL] += loss_kl.item()
        self._current_epoch_losses_sums[VAE.KEY_LOSS_NEG_ELBO] += loss_neg_elbo.item()

        loss_neg_elbo.backward()
        self.optimizer.step()

    def loss_neg_elbo(self, batch_fts: torch.Tensor, batch_conds: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the negative ELBO and returns it alongside its constituents.

        :returns: loss_reconstruction, loss_kl, loss_neg_elbo
        """
        if batch_conds is None:
            message = 'Conditional generative model expects conditions to be provided for running the generator'
            raise Exception(message)

        x = batch_fts
        x_c = torch.concat((x, batch_conds), dim=1).to(self.device)
        mean, log_variance = self.encoder.encode(x_c)

        # Sampling and re-parameterization trick
        z = self.encoder.sample(mean, log_variance)

        # Decoding
        z_c = torch.concat((z, batch_conds), dim=1).to(self.device)
        x_reconstruction = self.decoder.decode(z_c)

        # Calculate ELBO
        re_multipler = self.hps.loss_reconstruction_multiplier  # alias

        loss_reconstruction = self._loss_reconstruction(x, x_reconstruction)
        loss_kl = self._loss_kl(mean, log_variance)

        loss_neg_elbo = (re_multipler * loss_reconstruction + loss_kl).mean(dim=0)

        return loss_reconstruction, loss_kl, loss_neg_elbo

    def generate(self, n_samples: int, condition: torch.Tensor = None) -> torch.Tensor:
        if condition is None:
            message = 'Conditional generative model expects conditions to be provided for running the generator'
            raise Exception(message)

        if condition.dim() != 1:
            message = 'Provide a 1-dimensional condition vector.'
            raise ValueError(message)

        self.bundle.eval()

        with torch.no_grad():
            z = self.get_latent_samples(n_samples)
            z_c = torch.concat((z, condition), dim=1).to(self.device)
            x_reconstructed = self.decoder.decode(z_c)

        return x_reconstructed

CVAE.ModelRecipe._used_classes += [CVAE.ModelHyperparameters]



class VAEValidator(Validator):

    validation_loss_type = VAE.KEY_LOSS_NEG_ELBO

    def __init__(self,
                 model: VAE,
                 validation_dataset: MDevDataset):
        super().__init__(model)

        self.validation_dataset = validation_dataset
        self.model.set_mdev_dataset_groups(self.validation_dataset)

    def calculate_validation_loss(self, epoch_report: VAE.EpochReport):

        with torch.no_grad():

            if isinstance(self.model, VAE) and not isinstance(self.model, CVAE):
                indices, generate_vars = self.validation_dataset.get_all_rows_in_groups()
                loss_reconstruction, loss_kl, loss_neg_elbo = self.model.loss_neg_elbo(generate_vars)

            elif isinstance(self.model, CVAE):
                indices, generate_vars, conds = self.validation_dataset.get_all_rows_in_groups()
                loss_reconstruction, loss_kl, loss_neg_elbo = self.model.loss_neg_elbo(generate_vars, conds)

        self._add_to_validation_losses(epoch_report.epoch_index, loss_neg_elbo.item())

