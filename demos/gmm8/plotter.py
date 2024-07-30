import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.mdev.generative.base import GenerativeModel, ConditionalGenerativeModel, GenerativeModelOutputAdapter, ConditionalGenerativeModelOutputAdapter

from demos.gmm8.problem import cond_problem, x, y


logger = logging.getLogger(__name__)


class GMM8GenerationPlotter:

    base_problem = cond_problem
    colormap = mpl.cm.hsv

    def __init__(self,
                 model: GenerativeModel | ConditionalGenerativeModel,
                 n_samples_per_condition: int = 100,
                 fixed_limits: tuple = None
                 ):

        self.model = model
        self.n_samples_per_condition = n_samples_per_condition
        self.fixed_limits = fixed_limits

        if isinstance(self.model, ConditionalGenerativeModel):
            self.adapter = ConditionalGenerativeModelOutputAdapter(self.model, self.base_problem)
        else:
            self.adapter = GenerativeModelOutputAdapter(self.model, self.base_problem)

    def plot(self, epoch_report: ConditionalGenerativeModel.EpochReport):

        fig, ax = plt.subplots()
        ax.set_aspect(1)

        all_xs, all_ys, all_cs = [], [], []

        if isinstance(self.model, ConditionalGenerativeModel):
            for condition in range(8):
                assert isinstance(self.adapter, ConditionalGenerativeModelOutputAdapter)
                generation_outputs = self.adapter.generate_useful(self.n_samples_per_condition, (float(condition),))

                all_xs += [gen_out[x] for gen_out in generation_outputs]
                all_ys += [gen_out[y] for gen_out in generation_outputs]
                all_cs += [condition for gen_out in generation_outputs]

        else:
            assert isinstance(self.adapter, GenerativeModelOutputAdapter)
            generation_outputs = self.adapter.generate_useful(self.n_samples_per_condition * 8)

            all_xs += [gen_out[x] for gen_out in generation_outputs]
            all_ys += [gen_out[y] for gen_out in generation_outputs]
            all_cs += [0 for gen_out in generation_outputs]

        ax.scatter(all_xs, all_ys, c=all_cs, s=1, cmap=self.colormap)

        if self.fixed_limits:
            ax.set_xlim(*self.fixed_limits[0])
            ax.set_ylim(*self.fixed_limits[1])

        fn = f'{self.model.name}_genplot_e{epoch_report.epoch_index}.png'
        plt.savefig(Path(self.model.training_controls.autosave_dir) / fn)
        plt.close(fig)
        logger.info('Sample generated data plotted.')
