from glob import glob
from typing import ClassVar, Type, Iterable
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import train_test_split

from datamodels.analyses import AnalysisStandard
from datamodels.parameterizations import Parameterization

from datamodels.variables import Variable
from utils.iox import ProgramData


@dataclass
class DatasetSpecification(ProgramData):
    """An ordered list of variables represented by each column in the dataset."""

    columns: list[Variable]

    _data_type_str: ClassVar[str] = 'specification:dataset'
    _data_type_key: ClassVar[int] = 9100

    _save_fields: ClassVar[list[str]] = ['columns']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [Variable]

    def __eq__(self, other) -> bool:
        """
        Returns ``True`` if both specifications have columns of the same variables, in the same order.
        """
        if not isinstance(other, DatasetSpecification):
            return False
        return self.columns == other.columns


@dataclass
class DatasetStandard(ProgramData):
    """Includes the dataset specification (ordered list of variables represented by each column) and
    the dataset standard (standard, e.g. analysis standard, parameterization, to which the variables belong)"""

    specification: DatasetSpecification
    columns_standards: list[AnalysisStandard | Parameterization]

    _data_type_str: ClassVar[str] = 'standard:dataset'
    _data_type_key: ClassVar[int] = 9200

    _save_fields: ClassVar[list[str]] = ['specification', 'columns_standards']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [DatasetSpecification, AnalysisStandard, Parameterization]

    def __post_init__(self):
        if not isinstance(self.specification, DatasetSpecification):
            return

        if len(self.specification.columns) != len(self.columns_standards):
            message = ('Length of column standard listing does not match the number of columns. Each column must have a '
                       'standard declared.')
            raise ValueError(message)

    def __eq__(self, other) -> bool:
        """
        Returns ``True`` if:

        - Both standards' ``DatasetSpecification`` are the same, i.e. columns of same variables, in same order
        - Both standards' columns' data come from the same standard, e.g. same ``AnalysisStandard`` or same ``Parameterization``
        """
        if not isinstance(other, DatasetStandard):
            return False
        return self.specification == other.specification and self.columns_standards == other.columns_standards


class DataContainer:

    def __init__(self,
                 id: str,
                 standard: DatasetStandard,
                 data: np.ndarray,
                 sample_metadata: dict[int, dict] = None):
        """
        Base class for any tabular data container. In addition to the data itself, contains metadata about
        the data container.

        :param id: Unique ID for the data container
        :param standard: ``DatasetStandard`` instance explaining the data container
        :param data: A NumPy array of the data (obviously the data is expected to be tabular)
        :param sample_metadata: Dictionary of metadata for each sample in the data container. Keys are row indices, values are
            user-formatted dictionaries containing the sample metadata.
        """

        self.id = id
        self.standard = standard

        self.data = data

        self.sample_metadata = sample_metadata if sample_metadata is not None else {}
        """Metadata for rows (i.e. each sample), indexed by row index."""

    def get_cols_mean_std(self) -> dict[Variable, dict[str, float]]:
        """Returns a dictionary with keys of ``Variable`` instances corresponding to each column of the data container,
        and values of dictionaries. Value dictionaries have keys ``mean`` and ``std``, containing the mean and
        standard deviation of the values of the column of the variable."""
        cols_means_stds = {}
        for col_index, var in enumerate(self.standard.specification.columns):
            cols_means_stds[var] = {'mean': self.data[:, col_index].mean(axis=0),
                                    'std': self.data[:, col_index].std(axis=0)}

        return cols_means_stds

    def save_as_dir(self, directory: str | Path, exist_ok: bool = False):
        """Saves the data container as a directory. The provided directory will belong to this data container and no other
        data should be created in it.

        :param exist_ok: If ``False``, if the provided directory exists, will raise an error."""
        if Path(directory).exists() and not exist_ok:
            message = f'Directory to save already exists. Please provide a clean directory to save the data container.'
            raise OSError(message)
        Path(directory).mkdir(exist_ok=exist_ok)

    @staticmethod
    def from_dir(directory: str | Path):
        """Loads data container from directory."""
        pass


class Dataset(DataContainer):

    @dataclass
    class Card(ProgramData):
        """Container of metadata about the dataset."""

        id: str
        standard: DatasetStandard
        n_samples: int = None
        sample_metadata: dict[int, dict] = None

        # Private class attributes
        _data_type_str: ClassVar[str] = 'dataset_card'
        _data_type_key: ClassVar[int] = 10000

        _save_fields: ClassVar[list[str]] = ['id', 'standard', 'n_samples', 'sample_metadata']
        _used_classes: ClassVar[list[Type['ProgramData']]] = [DatasetStandard]

    FN_PREFIX = 'dataset'
    FN_SUFFIX_CARD = 'card'
    FN_SUFFIX_DATA = 'data'

    def __init__(self,
                 id: str,
                 standard: DatasetStandard,
                 data: np.ndarray,
                 sample_metadata: dict[int, dict] = None):

        super().__init__(id, standard, data, sample_metadata)

        if isinstance(self.standard.specification, DatasetSpecification) and not self.data.shape[1] == len(self.standard.specification.columns):
            raise ValueError(f'Provided data array has more columns than in specification.')

    def get_card(self) -> Card:
        """Returns ``Dataset`` metadata in ``Card`` format."""
        return Dataset.Card(self.id,
                            self.standard,
                            n_samples=self.data.shape[0],
                            sample_metadata=self.sample_metadata)

    def save_as_dir(self, directory: str | Path, exist_ok: bool = False):
        super().save_as_dir(directory, exist_ok)

        if not isinstance(directory, Path):
            directory = Path(directory)

        # Save dataset card
        self.get_card().write(directory / f'{self.FN_PREFIX}_{self.id}_{self.FN_SUFFIX_CARD}.json')

        # Save data itself
        np.save(directory / f'{self.FN_PREFIX}_{self.id}_{self.FN_SUFFIX_DATA}.npy', self.data)

    @staticmethod
    def from_dir(directory: str | Path):
        if not isinstance(directory, Path):
            directory = Path(directory)

        # Find card file
        matches_card = glob(str(directory / f'{Dataset.FN_PREFIX}_*_{Dataset.FN_SUFFIX_CARD}.json'))
        if len(matches_card) > 1:
            message = 'Multiple files found for a dataset card in directory.'
            raise ValueError(message)

        card = Dataset.Card.from_file(matches_card[0])
        _id = card.id

        # Find data file
        matches_data = glob(str(directory / f'{Dataset.FN_PREFIX}_{_id}_{Dataset.FN_SUFFIX_DATA}.npy'))
        if len(matches_data) > 1:
            message = 'Multiple files found for data in directory.'
            raise ValueError(message)

        data = np.load(matches_data[0])

        return Dataset(card.id, card.standard, data, card.sample_metadata)

    def n_samples(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        return f'{self.id}: {self.data.shape[0]} x {self.data.shape[1]}'


@dataclass
class CalibrationProtocol(ProgramData):

    source_specification: DatasetSpecification
    target_specification: DatasetSpecification
    calibrators: dict[Variable]

    _data_type_str: ClassVar[str] = 'protocol:calibration'
    _data_type_key: ClassVar[int] = 840

    _save_fields: ClassVar[list[str]] = ['source_specification', 'target_specification', 'calibrators']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [Variable, DatasetSpecification]


class Datapack(DataContainer):

    _data_type_str: ClassVar[str] = 'datapack'
    _data_type_key: ClassVar[int] = 11000

    FN_PREFIX = 'datapack'
    FN_SUFFIX_CARD = 'card'
    FN_SUFFIX_DATA = 'data'

    @dataclass
    class Card(ProgramData):

        id: str
        sources: Iterable | None
        standard: DatasetStandard
        sample_metadata: dict[int, dict] = None

        # Private class attributes
        _data_type_str: ClassVar[str] = 'datapack_card'
        _data_type_key: ClassVar[int] = 10000

        _save_fields: ClassVar[list[str]] = ['id', 'sources', 'standard', 'sample_metadata']
        _used_classes: ClassVar[list[Type['ProgramData']]] = [DatasetStandard]

    def __init__(self,
                 id: str,
                 standard: DatasetStandard,
                 data: np.ndarray,
                 sample_metadata: dict = None,
                 sources: Iterable = None):
        """
        Combines multiple ``Dataset`` instances.

        :param id: Short, unique identifier for datapack instance
        :param specification: Specification for the data columns
        :param column_standards: Standards for the column variables
        """

        super().__init__(id, standard, data, sample_metadata)

        self.sources = sources if sources is not None else []

    def get_card(self) -> Card:
        return Datapack.Card(self.id, self.sources, self.standard, self.sample_metadata)

    def save_as_dir(self, directory: str | Path, exist_ok: bool = False):
        super().save_as_dir(directory, exist_ok)

        if not isinstance(directory, Path):
            directory = Path(directory)

        # Save dataset card
        self.get_card().write(directory / f'{self.FN_PREFIX}_{self.id}_{self.FN_SUFFIX_CARD}.json')

        # Save data itself
        np.save(directory / f'{self.FN_PREFIX}_{self.id}_{self.FN_SUFFIX_DATA}.npy', self.data)

    @staticmethod
    def from_dir(directory: str | Path):
        if not isinstance(directory, Path):
            directory = Path(directory)

        # Find card file
        matches_card = glob(str(directory / f'{Datapack.FN_PREFIX}_*_{Datapack.FN_SUFFIX_CARD}.json'))
        if len(matches_card) > 1:
            message = 'Multiple files found for a dataset card in directory.'
            raise ValueError(message)

        card = Datapack.Card.from_file(matches_card[0])
        _id = card.id

        # Find data file
        matches_data = glob(str(directory / f'{Datapack.FN_PREFIX}_{_id}_{Datapack.FN_SUFFIX_DATA}.npy'))
        if len(matches_data) > 1:
            message = 'Multiple files found for data in directory.'
            raise ValueError(message)

        data = np.load(matches_data[0])

        return Datapack(card.id, card.standard, data, card.sample_metadata)

    @staticmethod
    def from_datasets(datapack_id: str,
                      datapack_standard: DatasetStandard,
                      datasets: Iterable[Dataset],
                      calibration_protocols: dict[DatasetStandard, CalibrationProtocol] = None) -> 'Datapack':
        """
        Constructs a ``Datapack`` from provided ``Dataset`` instances.

        :param datapack_id: ID for the new ``Datapack`` being constructed.
        :param datapack_standard: ``DatasetStandard`` instance, including ordered list of variables represented by the
        columns (``DatasetSpecification``) and the standards of the variables.
        :param datasets: ``Dataset`` instances to merge in forming the ``Datapack``.
        :param calibration_protocols: Dictionary mapping different ``DatasetStandard`` instances to the ``CalibrationProtocol``
        mapping them to the datapack standard.
        :return: New datapack, merging the data of the individual datasets.
        """

        if calibration_protocols is None:
            calibration_protocols = {}

        calibrated_datasets = []
        for dataset in datasets:

            if dataset.standard != datapack_standard:
                if dataset.standard not in calibration_protocols:
                    message = (f'A calibration protocol is needed between the datapack standard and the standard of '
                               f'dataset "{dataset.id}".')
                    raise KeyError(message)

                # TODO - calibrate dataset to datapack standard
                message = 'Dataset calibration has not been implemented.'
                raise NotImplementedError(message)

            else:
                calibrated_datasets.append(dataset)

        datas = [dataset.data for dataset in calibrated_datasets]
        all_data = np.concatenate(datas, axis=0)  # concatenate vertically, along rows

        merged_samples_metadata = {}
        start_row = 0
        for dataset in calibrated_datasets:
            dataset_n_rows = dataset.data.shape[0]

            for dataset_row_index in range(dataset_n_rows):

                dataset_row_metadata = dataset.sample_metadata.get(dataset_row_index, {})

                datapack_row_index = start_row + dataset_row_index
                merged_samples_metadata[f'{datapack_row_index}'] = {'source_dataset_id': dataset.id,
                                                                    **dataset_row_metadata}

            end_row = start_row + dataset_n_rows - 1
            start_row = end_row + 1

        return Datapack(id=datapack_id,
                        standard=datapack_standard,
                        data=all_data,
                        sample_metadata=merged_samples_metadata,
                        sources=[ds.get_card() for ds in calibrated_datasets])
