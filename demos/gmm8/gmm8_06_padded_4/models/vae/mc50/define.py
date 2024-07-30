from pathlib import Path

from src.pman.datamodels.problems import ConditionalGenerativeModelingProblem
from src.datamodels.datasets import Dataset
from src.mdev.utilities import MDevDataset
from src.mdev.generative.vae import CVAE
from src.mdev.archspec import *


wd = Path(__file__).parent
ds_dir = wd.parent.parent.parent / 'data'


if __name__ == '__main__':

    RECIPE_ID = 'mc50'

    # Problem
    fp_problem = [fp for fp in list(ds_dir.glob('problem_*_cond*.json'))][0]
    problem = ConditionalGenerativeModelingProblem.from_file(fp_problem)

    ds = Dataset.from_dir(ds_dir)
    ds_mean_std = ds.get_cols_mean_std()
    dev_ds_dummy = MDevDataset(ds, device=torch.device('cpu'), cols_means_stds=ds_mean_std)

    # Recipe
    n_latent_features = 893
    n_condition_features = 1

    recipe = CVAE.ModelRecipe(
        id=RECIPE_ID,
        specification=CVAE.ModelSpecification(
            generate_space=dev_ds_dummy.get_space_of_corresponding_normalized_vars(problem.generate_space),
            condition_space=dev_ds_dummy.get_space_of_corresponding_normalized_vars(problem.condition_space)
        ),
        hyperparameters=CVAE.ModelHyperparameters(
            n_generate_features=problem.generate_space.n_dims,
            n_latent_features=n_latent_features,
            n_condition_features=problem.condition_space.n_dims,
            encoder_spec=NetworkSpecification(layers=[
                Linear(problem.generate_space.n_dims + n_condition_features, 16),
                ReLU(),
                BatchNorm1D(kwargs={'num_features': 16}),
                Linear(16, 77),
                ReLU(),
                BatchNorm1D(kwargs={'num_features': 77}),
                Linear(77, n_latent_features * 2)
            ]),
            decoder_spec=NetworkSpecification(layers=[
                Linear(n_latent_features + n_condition_features, 16),
                ReLU(),
                BatchNorm1D(kwargs={'num_features': 16}),
                Linear(16, 77),
                ReLU(),
                BatchNorm1D(kwargs={'num_features': 77}),
                Linear(77, problem.generate_space.n_dims)
            ]),
            loss_reconstruction_formulation=CVAE.KEY_RECONSTRUCTION_MSE,
            loss_reconstruction_multiplier=1,
            latent_distribution=CVAE.KEY_LATENT_DIST_STANDARD_NORMAL
        ),
        training_settings=CVAE.TrainingSettings(
            batch_size=1600,
            dataloader_drop_last=True,
            dataloader_shuffle=True,
            optimizer_key='adam',
            optimizer_settings={
                'lr': 0.0015576719018882922,
                'betas': (0.5, 0.99),
                'eps': 1e-8
            },
            torch_manual_seed=None
        )
    )

    recipe.write(wd / f'recipe_{RECIPE_ID}.json')
