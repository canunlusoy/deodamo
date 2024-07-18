from typing import ClassVar, Type
from dataclasses import dataclass

from datamodels.spaces import SamplingSpace
from datamodels.mappers import S2SMapper
from datamodels.variables import SamplingVariable

from utils.iox import ProgramData


@dataclass
class SamplingActivityDefinition(ProgramData):

    id: str

    sampling_space: SamplingSpace
    n_samples: int

    description: str

    abstract_to_sampling_space_mappers: dict[SamplingVariable, S2SMapper] = None

    _data_type_str: ClassVar[str] = 'definition:activity:sampling'
    _data_type_key: ClassVar[int] = 51

    _save_fields: ClassVar[list[str]] = ['id', 'sampling_space', 'n_samples', 'description', 'abstract_to_sampling_space_mappers']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [SamplingSpace, SamplingVariable, S2SMapper]

    def __post_init__(self):

        # If abstract_to_sampling_space_mappers are provided, check if there is one for all svars
        if self.abstract_to_sampling_space_mappers is not None:
            if not all(svar in self.abstract_to_sampling_space_mappers for svar in self.sampling_space.sampling_variables):
                message = f'All sampling variables must have a function defined for the mapping from abstract spaces to the sampling space.'
                raise ValueError(message)



