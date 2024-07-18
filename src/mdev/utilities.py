from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, ClassVar, Callable, Union, Type, Literal
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split as _sklearn_train_test_split

from src.utils.iox import ProgramData
from src.datamodels.assets import Asset
from src.datamodels.datasets import DataContainer, Dataset as DManDataset, DatasetStandard, DatasetSpecification
from src.datamodels.variables import Variable
from src.datamodels.spaces import Space
from src.datamodels.parameterizations import Parameterization
from src.pman.datamodels.problems import GenerativeModelingProblem


activation_fcns_by_key: dict[str, Union[nn.Module, None]] = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'none': None
}

optimizers_by_key: dict[str, Type[Optimizer]] = {
    'adam': Adam
}


_ml_dataset = Asset('ml_dataset')
data_container_row_index = Variable('data_container_row_index')
_mdev_support_vars = Parameterization('mdev_support_vars', parameterized_asset=_ml_dataset, parameters=[data_container_row_index])


@dataclass(frozen=True)
class NormalizedVariable(Variable):
    """Represents the normalized version of a variable. Contains reference to the standard variable
    and the mean and standard deviation used to normalize its values."""

    base_variable: Variable
    normalize_mean: float
    normalize_std: float

    _data_type_str: ClassVar[str] = 'variable:normalized'
    _data_type_key: ClassVar[int] = 550

    _save_fields: ClassVar[list[str]] = Variable._save_fields + ['base_variable', 'normalize_mean', 'normalize_std']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [Variable]


# If MDevDataset is being used, this is likely needed
# A hacky way of doing this...
Space._used_classes.append(NormalizedVariable)


def get_padded_dataset(data_container: DataContainer,
                       n_dimensions_padding: int,
                       pad_noise_scale: float = 1e-4
                       ) -> tuple[DataContainer, list[Variable]]:

    original = data_container

    vars_padding = [Variable(f'padding_{i}') for i in range(n_dimensions_padding)]
    asset_dataset = Asset(f'dataset_{original.id}')

    parameterization_padding = Parameterization(
        name=f'padding_{n_dimensions_padding}',
        parameterized_asset=asset_dataset,
        parameters=vars_padding
    )

    new_standard = DatasetStandard(
        specification=DatasetSpecification(original.standard.specification.columns + vars_padding),
        columns_standards=original.standard.columns_standards + [parameterization_padding for pad_var in vars_padding]
    )

    padding = np.random.rand(original.data.shape[0], n_dimensions_padding) * pad_noise_scale
    new_data = np.concatenate((original.data, padding), axis=1)

    new_container = type(original)(
        id=f'{original.id}_padded_{n_dimensions_padding}',
        data=new_data,
        standard=new_standard,
    )

    return new_container, vars_padding


def split_into_train_test(data_container: DataContainer,
                          test_size: int | float,
                          split_random_state: int | float | None,
                          shuffle: bool = True
                          ) -> tuple[DManDataset, DManDataset]:

    data_container_row_indices = np.arange(data_container.data.shape[0]).reshape((-1, 1))
    indexed_data = np.concatenate((data_container_row_indices, data_container.data), axis=1)

    new_standard = DatasetStandard(
        specification=DatasetSpecification([data_container_row_index] + data_container.standard.specification.columns),
        columns_standards=[_mdev_support_vars] + data_container.standard.columns_standards
    )

    indexed_data_train, indexed_data_test = _sklearn_train_test_split(indexed_data,
                                                                      test_size=test_size,
                                                                      random_state=split_random_state,
                                                                      shuffle=shuffle)

    indexed_dataset_train = DManDataset(
        id=f'{data_container.id}_indexed_train',
        standard=new_standard,
        data=indexed_data_train
    )

    indexed_dataset_test = DManDataset(
        id=f'{data_container.id}_indexed_test',
        standard=new_standard,
        data=indexed_data_test
    )

    return indexed_dataset_train, indexed_dataset_test


class MDevDataset(torch.utils.data.Dataset):

    KEY_MEAN = 'mean'
    KEY_STD = 'std'

    def __init__(self,
                 data_container: DataContainer,
                 device: torch.device,
                 normalize: bool = True,
                 cols_means_stds: dict[Variable, dict[str, float]] = None,
                 cols_to_not_normalize: Iterable[Variable] = (data_container_row_index,)):
        """
        Custom PyTorch Dataset.

        Compared to ``src.datamodels.datasets.Dataset``, this meant for use in machine
        learning model development, whereas the former is meant for general purposes.

        :param data_container: Data container instance, e.g. ``Dataset`` or ``Datapack`` instance
        :param device: Device to store the data on.
        :param normalize: Switch to normalize the data.
        :param cols_means_stds: Optional, provide a dictionary, indexed by variables, each mapped to a sub-dictionary with keys
            "mean" and "std" for the mean and standard deviation value for the column of the variable.
            This argument may be used when mean & standard deviation of the overall dataset is to be used in making an ``MDevDataset``
            out of only a portion of the overall dataset (e.g. training split)
        :param cols_to_not_normalize: If provided, columns of these variables **will not** be normalized.
        """

        self.data_container = data_container

        self.device = device

        self._columns: list[Variable | NormalizedVariable, ...] = []
        """Internal container for variables represented by columns in the data array"""

        self._groups = []
        self.groups_in_col_indices = None

        self.normalize = normalize

        if cols_means_stds is not None:
            if not all(var in cols_means_stds for var in [var for var in data_container.standard.specification.columns if var not in cols_to_not_normalize]):
                message = (f'Mean and standard deviation values are provided manually only for some variables. If they are to be provided manually, they should be '
                           f'provided for all variables making up columns of the data container except for those columns which are not to be normalized '
                           f'(specified in ``cols_to_not_normalize``.')
                raise KeyError(message)

        self.cols_means_stds = cols_means_stds if cols_means_stds else {}
        self.cols_to_not_normalize = cols_to_not_normalize if cols_to_not_normalize else []
        self._prepare_data_for_mdev()

    def _set_groups_in_col_indices(self):

        # Variables represented by the columns of the data
        col_variables = self.columns

        # Determine the indices of columns to be provided in each group
        groups_in_col_indices = []
        for group_in_variables in self._groups:

            for variable in group_in_variables:
                if variable not in col_variables:
                    message = f'Variable "{variable.name}" requested to be released in a group is not a column of the dataset.'
                    raise KeyError(message)

            groups_in_col_indices.append([col_variables.index(var) for var in group_in_variables])

        self.groups_in_col_indices = tuple(groups_in_col_indices)

        # Determine the mean and std values of columns in each group
        cols_means_grouped, cols_stds_grouped = [], []
        for group_in_variables in self._groups:

            group_means, group_stds = [], []
            for group_var in group_in_variables:

                if group_var in self.cols_to_not_normalize:
                    group_means.append(None)
                    group_stds.append(None)
                    continue

                assert isinstance(group_var, NormalizedVariable)
                group_means.append(group_var.normalize_mean)
                group_stds.append(group_var.normalize_std)

            cols_means_grouped.append(tuple(group_means))
            cols_stds_grouped.append(tuple(group_stds))

        self.cols_means_grouped = tuple(cols_means_grouped)
        self.cols_stds_grouped = tuple(cols_stds_grouped)

    def _prepare_data_for_mdev(self):
        """Prepares data for use in machine learning models. Namely,

        - Normalizes the columns (unless a column is explicitly set not to be normalized)
        - Prepares the internal PyTorch Tensor for storing the data
        """

        data = self.data_container.data

        for col_index, var in enumerate(self.data_container.standard.specification.columns):

            if self.normalize and var not in self.cols_to_not_normalize:

                if var in self.cols_means_stds:
                    mean, std = self.cols_means_stds[var]['mean'], self.cols_means_stds[var]['std']
                else:
                    # Mean and std for column/variable not provided manually by user, calculate here
                    mean, std = data[:, col_index].mean(axis=0), data[:, col_index].std(axis=0)

                data[:, col_index] = (data[:, col_index] - mean) / std

                self._columns.append(
                    NormalizedVariable(
                        f'normalized_{var.name}',
                        base_variable=var,
                        normalize_mean=mean,
                        normalize_std=std
                    )
                )
            else:
                self._columns.append(var)

        self.data = torch.tensor(data, dtype=torch.float, device=self.device)

    @property
    def columns(self) -> tuple[Variable | NormalizedVariable, ...]:
        # Hiding it so it is not settable easily from outside
        return tuple(self._columns)

    def __getitem__(self, item) -> tuple[torch.Tensor, ...]:
        """Returns data at row ``item`` **in _groups** as per ``MDevDataset._groups``."""
        return tuple(self.data[item, group_cols] for group_cols in self.groups_in_col_indices)

    def __len__(self) -> int:
        """Number of samples / rows in the dataset"""
        return self.data.shape[0]

    def get_all_rows_in_groups(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.data[:, group_cols] for group_cols in self.groups_in_col_indices)

    def set_groups(self, groups: Iterable[list[Variable]]):
        self._groups = groups
        self._set_groups_in_col_indices()

    def get_space_of_corresponding_normalized_vars(self, space: Space) -> Space:
        """
        Returns a new ``Space`` instance made up of ``NormalizedVariable`` instances whose ``base_variable``s are
        the variables making up the provided ``space``.

        Note that the provided space must be made up of only non-normalized variables, i.e. just regular ``Variable``s.

        For each variable in the provided space, a corresponding ``NormalizedVariable`` must exist in this ``MDevDataset`` instance.
        """
        if not self.normalize:
            message = 'Dataset is not normalized, and does not contain normalized variables.'
            raise Exception(message)

        normalized_vars = []

        for var in space.variables:
            assert not isinstance(var, NormalizedVariable)

            corresponding_normalized_var_found = False
            for ds_col_var in self._columns:

                if isinstance(ds_col_var, NormalizedVariable) and ds_col_var.base_variable == var:
                    normalized_vars.append(ds_col_var)
                    corresponding_normalized_var_found = True
                    break

            if corresponding_normalized_var_found:
                continue
            else:
                message = f'No normalized variable corresponding to the space variable "{var.name}" is included in the dataset.'
                raise KeyError(message)

        return Space(normalized_vars)



class NoiseGenerator:

    KEY_NOISE_TYPE_NORMAL = 'normal'
    KEY_NOISE_TYPE_UNIFORM = 'uniform'

    noise_types_generators = {KEY_NOISE_TYPE_NORMAL: torch.randn,
                              KEY_NOISE_TYPE_UNIFORM: torch.rand}

    def __init__(self,
                 noise_type: Literal['normal', 'uniform'],
                 n_dimensions: int,
                 device: torch.device
                 ):

        self.noise_type = noise_type
        self.n_dimensions = n_dimensions

        self.device = device

    def get_noise(self, n_samples: int = 1):

        noise_fcn = self.noise_types_generators[self.noise_type]
        noises = noise_fcn((n_samples, self.n_dimensions), device=self.device)

        return noises


def get_fc_network(layer_neuron_counts: list[int], activation_fcns: tuple[str | None, ...]) -> nn.Sequential:
    """
    Returns a fully-connected network to the provided specifications.
    :param layer_neuron_counts: Number of neurons in each layer
    :param activation_fcns: List of activation function modules, to be appended after every layer as of 2nd layer.
    :return: An ``nn.Sequential`` instance of ``nn.Linear`` and activation function module instances
    """
    layers = []
    for layer_index in range(len(layer_neuron_counts) - 1):
        layers.append(
            nn.Linear(in_features=layer_neuron_counts[layer_index],
                      out_features=layer_neuron_counts[layer_index + 1])
        )

        activation_fcn_type_key = activation_fcns[layer_index]
        if activation_fcn_type_key is None:
            continue

        activation_fcn_type = activation_fcns_by_key[activation_fcn_type_key]
        if activation_fcn_type is None:
            continue
        layers.append(activation_fcn_type())

    network = nn.Sequential(*layers)
    return network

