from copy import deepcopy
from typing import Callable, ClassVar, Type
from dataclasses import dataclass, field

from src.datamodels.assets import Asset
from src.datamodels.mappers import S2SMapper, CustomFunctionMapper
from src.datamodels.points import DesignSpacePoint, SamplingSpacePoint
from src.datamodels.analyses import PerformanceMetric
from src.datamodels.variables import Variable, DesignVariable, SamplingVariable

from utils.iox import ProgramData
from utils.programming import is_lambda_function


@dataclass(frozen=True)
class Space(ProgramData):

    variables: list[Variable]

    _data_type_str: ClassVar[str] = 'definition:space'
    _data_type_key: ClassVar[int] = 550

    _save_fields: ClassVar[list[str]] = ['variables']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [Variable]

    @property
    def n_dims(self) -> int:
        return len(self.variables)

    def get_variable_by_name(self, name: str) -> Variable:
        for variable in self.variables:
            if variable.name == name:
                return variable


@dataclass(frozen=True)
class AbstractSpace(Space):

    _data_type_str: ClassVar[str] = 'definition:space:abstract'
    _data_type_key: ClassVar[int] = 551

    _save_fields: ClassVar[list[str]] = Space._save_fields
    _used_classes: ClassVar[list[Type['ProgramData']]] = Space._used_classes


@dataclass(frozen=True)
class DesignSpace(Space):

    variables: list[DesignVariable]

    _data_type_str: ClassVar[str] = 'definition:space:design'
    _data_type_key: ClassVar[int] = 555

    _save_fields: ClassVar[list[str]] = ['variables']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [DesignVariable]

    def __post_init__(self):
        dvar_names = [dvar.name for dvar in self.variables]
        unique_dvar_names = set(dvar_names)
        if len(dvar_names) != len(unique_dvar_names):
            repeated_names = [dvar_name for dvar_name in unique_dvar_names if dvar_names.count(dvar_name) != 1]
            error = f'The following names are used by more than one design variable: {", ".join(repeated_names)}'
            raise ValueError(error)


@dataclass(frozen=True)
class SamplingSpace(Space):

    variables: list[SamplingVariable]

    associated_design_space: DesignSpace
    mappers_to_design_vars: dict[DesignVariable, S2SMapper | Callable]
    """A sampling space is necessarily linked to a design space. The activity is a design sampling activity, and
    the eventual outputs are designs, belonging to a design space. Therefore, the sampling space points should
    be mappable to design space points."""

    _data_type_str: ClassVar[str] = 'definition:space:sampling'
    _data_type_key: ClassVar[int] = 553

    _save_fields: ClassVar[list[str]] = ['variables', 'associated_design_space', 'mappers_to_design_vars']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [SamplingVariable, DesignVariable]

    def __post_init__(self):
        """Validate inputs."""

        dvars_missing_mappers = [dvar for dvar in self.associated_design_space.variables if dvar not in self.mappers_to_design_vars]
        if dvars_missing_mappers:
            error = f'No mappers are provided for the following design variables: {", ".join([dvar.name for dvar in dvars_missing_mappers])}'
            raise KeyError(error)

        # dvars_with_improper_mappers = [dvar for dvar, mapper in self.mappers_to_design_vars.items() if not isinstance(mapper, Mapper)]
        # if dvars_with_improper_mappers:
        #     error = (f'Mappers provided for the following design variables are not of the proper type: '
        #              f'{", ".join([dvar.name for dvar in dvars_with_improper_mappers])}')
        #     raise KeyError(error)

        for dvar, svar_to_dvar_mapper in deepcopy(self.mappers_to_design_vars).items():
            if is_lambda_function(svar_to_dvar_mapper):
                mapper = CustomFunctionMapper.from_lambda(svar_to_dvar_mapper, self.variables)
                self.mappers_to_design_vars[dvar] = mapper


        svar_names = [svar.name for svar in self.variables]
        unique_svar_names = set(svar_names)
        if len(svar_names) != len(unique_svar_names):
            repeated_names = [svar_name for svar_name in unique_svar_names if svar_names.count(svar_name) != 1]
            error = f'The following names are used by more than one sampling variable: {", ".join(repeated_names)}'
            raise ValueError(error)

    def map_ssp_to_dsp(self, ssp: SamplingSpacePoint) -> DesignSpacePoint:
        dsp = DesignSpacePoint()
        for dvar in self.associated_design_space.variables:
            dvar_mapper = self.mappers_to_design_vars[dvar]
            dvar_value = dvar_mapper.map(ssp)
            dsp[dvar] = dvar_value
        return dsp


@dataclass(frozen=True)
class PerformanceSpace(Space):

    variables: list[PerformanceMetric]

    associated_assets: list[Asset]  # performance metrics of designs of this space make up this performance space

    _data_type_str: ClassVar[str] = 'definition:space:performance'
    _data_type_key: ClassVar[int] = 560

    _save_fields: ClassVar[list[str]] = ['variables']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [PerformanceMetric, Asset]

