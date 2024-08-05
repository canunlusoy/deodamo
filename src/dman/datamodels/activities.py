from typing import ClassVar, Type
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from datamodels.spaces import Space, AbstractSpace, SamplingSpace, DesignSpace
from datamodels.mappers import S2SMapper
from datamodels.variables import SamplingVariable
from datamodels.datasets import Dataset, DatasetStandard

from utils.iox import ProgramData


@dataclass
class DesignSamplingActivityDefinition(ProgramData):
    """
    Data container for the definition of a design sampling activity.
    Also assists in conducting the sampling.
    """

    id: str
    """Unique ID for the sampling activity"""

    n_samples: int
    """Number of design samples to generate"""

    starting_space: AbstractSpace | SamplingSpace | DesignSpace
    """
    Space from which we will start generating samples. Typically an ``AbstractSpace``.
    Since ``AbstractSpace`` contains a reference to a ``SamplingSpace``, which in turn contains a reference to a
    ``DesignSpace``, we can figure out the full chain of spaces leading to the final design space from an initial one.
    """

    directory: str | Path
    """Data of the activity will be saved to its directory."""

    description: str

    _data_type_str: ClassVar[str] = 'definition:activity:sampling'
    _data_type_key: ClassVar[int] = 51

    _save_fields: ClassVar[list[str]] = [
        'id', 'n_samples', 'starting_space', 'directory', 'description'
    ]
    _used_classes: ClassVar[list[Type['ProgramData']]] = [AbstractSpace, SamplingSpace, DesignSpace]

    def __post_init__(self):
        self.directory = Path(self.directory)

    def generate_samples(self):

        if isinstance(self.starting_space, AbstractSpace):
            as_ = self.starting_space
            asps_ds = as_.generate_dataset(n_points=self.n_samples, id=f'{self.id}_asps')
            asps_ds.save_as_dir(directory=(self.directory / '0_asps'), exist_ok=False)

            ss = as_.associated_sampling_space
            ssps_ds = as_.map_asps_to_ssps(asps_ds, ssp_ds_id=f'{self.id}_ssps')
            ssps_ds.save_as_dir(directory=(self.directory / '1_ssps'), exist_ok=False)

            ds = ss.associated_design_space
            dsps_ds = ss.map_ssps_to_dsps(ssps_ds, dsp_ds_id=f'{self.id}_dsps')
            dsps_ds.save_as_dir(directory=(self.directory / '2_dsps'), exist_ok=False)



        else:
            raise NotImplementedError()

class AbstractSpacePointDataset(Dataset):

    def __init__(self,
                 id: str,
                 standard: DatasetStandard,
                 data: np.ndarray,
                 sample_metadata: dict[int, dict] = None):

        super().__init__(id, standard, data, sample_metadata)
