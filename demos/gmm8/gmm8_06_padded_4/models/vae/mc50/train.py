from pathlib import Path

import torch

from datamodels.datasets import Dataset

from mdev.generative.base import TrainingProgressPlotter
from mdev.generative.vae import CVAE, VAEValidator
from mdev.utilities import MDevDataset, split_into_train_test

from demos.gmm8.plotter import GMM8GenerationPlotter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_dir = Path(__file__).parent
ds_dir = model_dir.parent.parent.parent / 'data'


n_epochs = 100


if __name__ == '__main__':

    RECIPE_ID = 'mc50'
    INSTANCE_ID = 'i01'

    # Data
    ds = Dataset.from_dir(ds_dir)
    ds_mean_std = ds.get_cols_mean_std()

    ds_train, ds_validate = split_into_train_test(
        ds,
        test_size=0.2,
        split_random_state=None
    )

    dev_ds_train = MDevDataset(
        ds_train, device=device, cols_means_stds=ds_mean_std,
    )

    dev_ds_validate = MDevDataset(
        ds_validate, device=device, cols_means_stds=ds_mean_std,
    )

    # Recipe
    fp_recipe = model_dir / f'recipe_{RECIPE_ID}.json'
    recipe = CVAE.ModelRecipe.from_file(fp_recipe)

    # Model
    instance_dir = model_dir / INSTANCE_ID
    instance_dir.mkdir(exist_ok=True)

    model = CVAE(
        instance_id=INSTANCE_ID,
        model_recipe=recipe,
        training_controls=CVAE.TrainingControls(
            autosave_period=100,
            autosave_dir=str(instance_dir)
        ),
        device=device
    )

    validator = VAEValidator(model, dev_ds_validate)
    loss_plotter = TrainingProgressPlotter(model, validators=[validator])
    gen_plotter = GMM8GenerationPlotter(model, fixed_limits=((-20, 20), (-20, 20)))

    # Model event bindings
    event_10_epochs = model.create_event_for_nth_epoch(10)
    event_2_epochs = model.create_event_for_nth_epoch(2)
    event_50_epochs = model.create_event_for_nth_epoch(50)
    event_1_epoch = model.create_event_for_nth_epoch(1)

    model.event_handler.subscribe(event_10_epochs, validator.calculate_validation_loss)
    model.event_handler.subscribe(event_10_epochs, loss_plotter.generate_progress_plot)
    model.event_handler.subscribe(event_1_epoch, gen_plotter.plot)

    model.train(dev_ds_train, epochs=n_epochs)
    model.save_record()
