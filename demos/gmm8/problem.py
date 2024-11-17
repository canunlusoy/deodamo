import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from datamodels.assets import Asset
from datamodels.parameterizations import Parameterization
from datamodels.spaces import Space
from datamodels.variables import Variable
from datamodels.datasets import Dataset, DatasetStandard, DatasetSpecification

from pman.datamodels.problems import ConditionalGenerativeModelingProblem, GenerativeModelingProblem


x = Variable('x')
y = Variable('y')
c = Variable('c')
plane = Asset('gmm8_plane')


points = Parameterization(
    'gmm8_points',
    parameterized_asset=plane,
    parameters=[x, y]
)


cond_problem = ConditionalGenerativeModelingProblem(
    id='gmm8_cond',
    name='8-Mode Gaussian Mixture Conditional Generation',
    generate_space=Space(points.parameters),
    condition_space=Space([c])
)


uncond_problem = GenerativeModelingProblem(
    id='gmm8',
    name='8-Mode Gaussian Mixture Generation',
    generate_space=Space(points.parameters)
)


class GMM8DataGenerator:

    def __init__(self,
                 arrangement_radius: float = 8.0,
                 n_samples_mode: int = 400
                 ):
        self._radius_arrangement = arrangement_radius
        self._n_samples_mode = n_samples_mode

        self._modes_centers = {}
        self._modes_pts = {}

        self._all_data = None
        self._col_standard = None

        self._generate()

    @property
    def arrangement_radius(self) -> float:
        return self._radius_arrangement

    @property
    def mode_centers(self) -> dict[int, tuple[float, float]]:
        return self._modes_centers

    @property
    def modes_points(self) -> dict[int, np.ndarray]:
        return self._modes_pts

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def standard(self) -> DatasetStandard:
        return self._col_standard

    def _generate(self):
        center_angular_positions = np.arange(0, 2 * np.pi, np.pi / 4)

        modes_rows = []
        for mode_index, center_angular_position in enumerate(center_angular_positions):
            center_x = self._radius_arrangement * np.cos(center_angular_position)
            center_y = self._radius_arrangement * np.sin(center_angular_position)
            self._modes_centers[mode_index] = (center_x, center_y)

            samples = np.random.randn(self._n_samples_mode, 2)
            samples[:, 0] += center_x
            samples[:, 1] += center_y

            self._modes_pts[mode_index] = samples

            mode_index_col = np.ones((self._n_samples_mode, 1)) * mode_index
            mode_rows = np.concatenate((mode_index_col, samples), axis=1)
            modes_rows.append(mode_rows)

        self._data = np.concatenate(modes_rows, axis=0)
        col_spec = DatasetSpecification(columns=[c, x, y])
        self._col_standard = DatasetStandard(col_spec, columns_standards=[points for var in col_spec.columns])


def plot_data(data: np.ndarray):
    fig, ax = plt.subplots()
    ax.set_aspect(1)

    ax.scatter(data[:, 1], data[:, 2], c=data[:, 0], s=1, cmap=mpl.cm.hsv)
    plt.show()
