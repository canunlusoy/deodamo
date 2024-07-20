from copy import deepcopy
from typing import Callable, ClassVar, Type
from dataclasses import dataclass, field

from scipy.stats.qmc import LatinHypercube

from src.datamodels.assets import Asset
from src.datamodels.mappers import S2SMapper, SympyExpression
from src.datamodels.points import DesignSpacePoint, SamplingSpacePoint, AbstractSpacePoint
from src.datamodels.analyses import PerformanceMetric
from src.datamodels.variables import (
    Variable, ContinuousVariable, DesignVariable, SamplingVariable, ContinuousSamplingVariable, ContinuousDesignVariable
)

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
    _used_classes: ClassVar[list[Type['ProgramData']]] = [
        SamplingVariable, ContinuousSamplingVariable,
        DesignVariable, ContinuousDesignVariable, DesignSpace,
        SympyExpression
    ]

    def __post_init__(self):
        """Validate inputs."""

        # Check if all svar names are unique
        svar_names = [svar.name for svar in self.variables]
        unique_svar_names = set(svar_names)
        if len(svar_names) != len(unique_svar_names):
            repeated_names = [svar_name for svar_name in unique_svar_names if svar_names.count(svar_name) != 1]
            error = f'The following names are used by more than one sampling variable: {", ".join(repeated_names)}'
            raise ValueError(error)

        # Check if there are mappers for all design variables of the associated design space
        dvars_missing_mappers = [dvar for dvar in self.associated_design_space.variables if dvar not in self.mappers_to_design_vars]
        if dvars_missing_mappers:
            error = f'No mappers are provided for the following design variables: {", ".join([dvar.name for dvar in dvars_missing_mappers])}'
            raise KeyError(error)

        # Check if mappers to design space variables use sampling space dimensions
        for dvar, mapper in self.mappers_to_design_vars.items():
            if isinstance(mapper, SympyExpression):
                dependent_svar_names = mapper.variables
                if not all(svar_name in [var.name for var in self.variables] for svar_name in dependent_svar_names):
                    message = (f'The mapper for design space variable "{dvar.name}" uses variables that are not sampling space dimensions. '
                               f'\nMapper uses variables: {", ".join(mapper.variables)}'
                               f'\nSampling space variables are: {", ".join([var.name for var in self.variables])}')
                    raise KeyError(message)

    def map_ssp_to_dsp(self, ssp: SamplingSpacePoint) -> DesignSpacePoint:
        dsp = DesignSpacePoint()
        for dvar in self.associated_design_space.variables:
            dvar_mapper = self.mappers_to_design_vars[dvar]
            dvar_value = dvar_mapper.map(ssp)
            dsp[dvar.name] = dvar_value
        return dsp

    def _get_raw_data(self):
        dct = {key: getattr(self, key) for key in self._save_fields}
        dct['mappers_to_design_vars'] = {dvar.name: mapper for dvar, mapper in dct['mappers_to_design_vars'].items()}
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> 'SamplingSpace':
        field_data = {}
        for key, val in dct.items():
            if key in cls._save_fields:
                field_data[key] = val

        dvars_by_name = {var.name: var for var in field_data['associated_design_space'].variables}

        field_data['mappers_to_design_vars'] = {dvars_by_name[dvar_name]: mapper for dvar_name, mapper in field_data['mappers_to_design_vars'].items()}
        return cls(**field_data)


@dataclass(frozen=True)
class AbstractSpace(Space):

    associated_sampling_space: SamplingSpace
    mappers_to_sampling_vars: dict[SamplingVariable, S2SMapper]

    _data_type_str: ClassVar[str] = 'definition:space:abstract'
    _data_type_key: ClassVar[int] = 551

    _save_fields: ClassVar[list[str]] = Space._save_fields + ['associated_sampling_space' + 'mappers_to_sampling_vars']
    _used_classes: ClassVar[list[Type['ProgramData']]] = Space._used_classes + [SamplingVariable, S2SMapper]

    def __post_init__(self):
        """Validate inputs."""

        # Check if there are mappers for all sampling variables of the associated sampling space
        svars_missing_mappers = [svar for svar in self.associated_sampling_space.variables if svar not in self.mappers_to_sampling_vars]
        if svars_missing_mappers:
            error = f'No mappers are provided for the following sampling variables: {", ".join([svar.name for svar in svars_missing_mappers])}'
            raise KeyError(error)

        # Check if mappers to sampling space variables use sampling space dimensions
        for dvar, mapper in self.mappers_to_design_vars.items():
            if isinstance(mapper, SympyExpression):
                dependent_svar_names = mapper.variables
                if not all(svar_name in [var.name for var in self.variables] for svar_name in dependent_svar_names):
                    message = (f'The mapper for design space variable "{dvar.name}" uses variables that are not sampling space dimensions. '
                               f'\nMapper uses variables: {", ".join(mapper.variables)}'
                               f'\nSampling space variables are: {", ".join([var.name for var in self.variables])}')
                    raise KeyError(message)

    @property
    def n_dims(self) -> int:
        return self.associated_sampling_space.n_dims

    def map_asp_to_ssp(self, asp: AbstractSpacePoint) -> SamplingSpacePoint:
        ssp = SamplingSpacePoint()
        for svar in self.associated_sampling_space.variables:
            svar_mapper = self.mappers_to_sampling_vars[svar]
            svar_value = svar_mapper.map(ssp)
            ssp[svar.name] = svar_value
        return ssp

    def _get_raw_data(self):
        dct = {key: getattr(self, key) for key in self._save_fields}
        dct['mappers_to_design_vars'] = {dvar.name: mapper for dvar, mapper in dct['mappers_to_design_vars'].items()}
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> 'SamplingSpace':
        field_data = {}
        for key, val in dct.items():
            if key in cls._save_fields:
                field_data[key] = val

        dvars_by_name = {var.name: var for var in field_data['associated_design_space'].variables}

        field_data['mappers_to_design_vars'] = {dvars_by_name[dvar_name]: mapper for dvar_name, mapper in field_data['mappers_to_design_vars'].items()}
        return cls(**field_data)

    def generate_points(self, n_points: int) -> list[AbstractSpacePoint]:
        lh = LatinHypercube(d=self.n_dims)
        raw_points = lh.random(n=n_points)
        points = [
            AbstractSpacePoint(
                {col: raw_points[row, col] for row, col in range(raw_points.shape[0]), range(raw_points.shape[1])}
            )
        ]
        return points


@dataclass(frozen=True)
class PerformanceSpace(Space):

    variables: list[PerformanceMetric]

    associated_assets: list[Asset]  # performance metrics of designs of this space make up this performance space

    _data_type_str: ClassVar[str] = 'definition:space:performance'
    _data_type_key: ClassVar[int] = 560

    _save_fields: ClassVar[list[str]] = ['variables']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [PerformanceMetric, Asset]
