from pathlib import Path

import torch

from src.pman.datamodels.problems import ConditionalGenerativeModelingProblem
from src.datamodels.datasets import Dataset
from src.mdev.utilities import MDevDataset
from src.mdev.generative.gan import CGAN
from src.mdev.archspec import *


wd = Path(__file__).parent
ds_dir = wd.parent.parent.parent / 'data'


if __name__ == '__main__':

    RECIPE_ID = 'mc62'

    # Problem
    fp_problem = [fp for fp in list(ds_dir.glob('problem_*_cond.json'))][0]
    problem = ConditionalGenerativeModelingProblem.from_file(fp_problem)

    ds = Dataset.from_dir(ds_dir)
    ds_mean_std = ds.get_cols_mean_std()
    dev_ds_dummy = MDevDataset(ds, device=torch.device('cpu'), cols_means_stds=ds_mean_std)

    # Recipe
    n_latent_features = 2

    recipe = CGAN.ModelRecipe(
        id=RECIPE_ID,
        specification=CGAN.ModelSpecification(
            generate_space=dev_ds_dummy.get_space_of_corresponding_normalized_vars(problem.generate_space),
            condition_space=dev_ds_dummy.get_space_of_corresponding_normalized_vars(problem.condition_space)
        ),
        hyperparameters=CGAN.ModelHyperparameters(
            n_generate_features=problem.generate_space.n_dims,
            n_latent_features=n_latent_features,
            n_condition_features=problem.condition_space.n_dims,
            generator_spec=NetworkSpecification(layers=[
                Linear(n_latent_features + problem.condition_space.n_dims, 1024),
                BatchNorm1D(kwargs={'num_features': 1024}),
                ReLU(),
                Linear(1024, 1024),
                BatchNorm1D(kwargs={'num_features': 1024}),
                ReLU(),
                Linear(1024, 1024),
                BatchNorm1D(kwargs={'num_features': 1024}),
                ReLU(),
                Linear(1024, 1024),
                BatchNorm1D(kwargs={'num_features': 1024}),
                ReLU(),
                Linear(1024, problem.generate_space.n_dims),
            ]),
            discriminator_spec=NetworkSpecification(layers=[
                Linear(problem.generate_space.n_dims + problem.condition_space.n_dims, 1024),
                ReLU(),
                Linear(1024, 1024),
                ReLU(),
                Linear(1024, 1),
                Sigmoid()
            ]),
            g_loss_formulation='non_saturating',
            d_loss_formulation='non_saturating',
            reduction='mean',
            epsilon=1e-5,
            clamp_discriminator_outputs=True
        ),
        training_settings=CGAN.TrainingSettings(
            batch_size=400,
            dataloader_drop_last=True,
            dataloader_shuffle=True,
            g_optimizer_key='adam',
            g_optimizer_settings={
                'lr': 3e-4,
                'betas': (0.5, 0.99),
                'eps': 1e-8
            },
            d_optimizer_key='adam',
            d_optimizer_settings={
                'lr': 3e-4,
                'betas': (0.5, 0.99),
                'eps': 1e-8
            },
            n_g_step_per_batch=1,
            n_d_step_per_batch=1,
            torch_manual_seed=None
        )
    )

    recipe.write(wd / f'recipe_{RECIPE_ID}.json')
