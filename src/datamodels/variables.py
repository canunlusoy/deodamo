from typing import ClassVar
from dataclasses import dataclass

from utils.iox import ProgramData


@dataclass(frozen=True)
class Variable(ProgramData):

    name: str

    _data_type_str: ClassVar[str] = 'definition:variable'
    _data_type_key: ClassVar[int] = 400

    _save_fields: ClassVar[list[str]] = ['name']

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return self.name == other.name


@dataclass(frozen=True)
class ContinuousVariable(Variable):

    _data_type_str: ClassVar[str] = 'definition:variable:continuous'
    _data_type_key: ClassVar[int] = 401


@dataclass(frozen=True)
class Metric(Variable):

    _data_type_str: ClassVar[str] = 'definition:variable:metric'
    _data_type_key: ClassVar[int] = 421


@dataclass(frozen=True)
class DesignVariable(Variable):

    _data_type_str: ClassVar[str] = 'definition:variable:design'
    _data_type_key: ClassVar[int] = 410

    def __repr__(self):
        return f'{self.name}'


@dataclass(frozen=True)
class ContinuousDesignVariable(DesignVariable, ContinuousVariable):

    typical_range: tuple[float | int, float | int] = None

    _data_type_str: ClassVar[str] = 'definition:variable:design:continuous'
    _data_type_key: ClassVar[int] = 411

    _save_fields: ClassVar[list[str]] = DesignVariable._save_fields + ['typical_range']

    def __repr__(self):
        return f'{self.name}'


@dataclass(frozen=True)
class SamplingVariable(Variable):

    _data_type_str: ClassVar[str] = 'definition:variable:sampling'
    _data_type_key: ClassVar[int] = 480

    @staticmethod
    def from_design_variable(dvar: DesignVariable, new_name: str = None) -> 'SamplingVariable':
        name = new_name if new_name is not None else dvar.name
        return SamplingVariable(name)

    def __repr__(self):
        return f'{self.name}'


@dataclass(frozen=True)
class ContinuousSamplingVariable(SamplingVariable, ContinuousVariable):

    typical_range: tuple[float | int, float | int] = None

    _data_type_str: ClassVar[str] = 'definition:variable:sampling:continuous'
    _data_type_key: ClassVar[int] = 481

    _save_fields: ClassVar[list[str]] = SamplingVariable._save_fields + ['typical_range']

    @staticmethod
    def from_cts_design_variable(cts_dvar: ContinuousDesignVariable, new_name: str) -> 'ContinuousSamplingVariable':
        if new_name == cts_dvar.name:
            message = 'Sampling variable name cannot be identical to the source design variable.'
            raise ValueError(message)
        return ContinuousSamplingVariable(new_name, typical_range=cts_dvar.typical_range)
