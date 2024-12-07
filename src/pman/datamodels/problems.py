from typing import ClassVar, Type
from dataclasses import dataclass, field

from datamodels.datasets import DatasetSpecification
from datamodels.spaces import Space, DesignSpace, PerformanceSpace
from datamodels.variables import Variable
from datamodels.parameterizations import DesignParameterization
from datamodels.assets import DesignAsset

from utils.iox import ProgramData


@dataclass
class Problem(ProgramData):

    id: str
    name: str

    related_problems: list['Problem'] = field(default_factory=list)

    _data_type_str: ClassVar[str] = 'definition:problem'
    _data_type_key: ClassVar[int] = 8000

    _save_fields: ClassVar[list[str]] = ['id', 'name', 'related_problems']
    _used_classes: ClassVar[list[Type['ProgramData']]] = []


@dataclass(kw_only=True)  # https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
class OptimizationProblem(Problem):

    _data_type_str: ClassVar[str] = 'definition:problem:optimization'
    _data_type_key: ClassVar[int] = 8005

    _save_fields: ClassVar[list[str]] = []
    _used_classes: ClassVar[list[Type['ProgramData']]] = []


@dataclass(kw_only=True)
class DesignProblem(Problem):

    assets_to_design: list[DesignAsset]
    asset_parameterizations: dict[DesignAsset, DesignParameterization]

    _data_type_str: ClassVar[str] = 'definition:problem:design'
    _data_type_key: ClassVar[int] = 8010

    _save_fields: ClassVar[list[str]] = Problem._save_fields + ['assets_to_design', 'asset_parameterizations']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [DesignAsset, DesignParameterization]

    def __post_init__(self):
        if not all(asset_type in self.asset_parameterizations for asset_type in self.assets_to_design):
            msg = (f'A parameterization of record must be provided for all assets to be designed '
                   f'as part of the design problem must have')
            raise KeyError(msg)

    def _get_raw_data(self):
        dct = {key: getattr(self, key) for key in self._save_fields}
        dct['asset_parameterizations'] = {asset.name: parameterization for asset, parameterization in dct['asset_parameterizations'].items()}
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> 'DesignProblem':
        field_data = {}
        for key, val in dct.items():
            if key in cls._save_fields:
                field_data[key] = val

        assets_by_name = {asset.name: asset for asset in field_data['assets_to_design']}
        field_data['asset_parameterizations'] = {assets_by_name[asset_name]: parameterization for asset_name, parameterization in field_data['asset_parameterizations'].items()}
        return cls(**field_data)


@dataclass(kw_only=True)
class DesignOptimizationProblem(DesignProblem, OptimizationProblem):

    _data_type_str: ClassVar[str] = 'definition:problem:designOptimization'
    _data_type_key: ClassVar[int] = 8015


@dataclass(kw_only=True)  # https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
class GenerativeModelingProblem(Problem):

    generate_space: Space

    _data_type_str: ClassVar[str] = 'definition:problem:generativeModeling'
    _data_type_key: ClassVar[int] = 8070

    _save_fields: ClassVar[list[str]] = Problem._save_fields + ['generate_space']
    _used_classes: ClassVar[list[Type['ProgramData']]] = Problem._used_classes + [Space]


@dataclass(kw_only=True)
class ConditionalGenerativeModelingProblem(GenerativeModelingProblem):

    condition_space: Space

    _data_type_str: ClassVar[str] = 'definition:problem:generativeModeling:conditional'
    _data_type_key: ClassVar[int] = 8071

    _save_fields: ClassVar[list[str]] = GenerativeModelingProblem._save_fields + ['condition_space']

    def get_model_input_space(self, n_padding: int = None) -> Space:
        input_space_params = self.generate_space

        padding_vars = []
        if n_padding is not None:
            padding_vars = [Variable(f'padding_{i}') for i in range(n_padding)]

        return Space(input_space_params.variables + padding_vars)


@dataclass(kw_only=True)
class GenerativeDesignProblem(DesignProblem, GenerativeModelingProblem):

    _data_type_str: ClassVar[str] = 'definition:problem:generativeDesign'
    _data_type_key: ClassVar[int] = 8075

    _save_fields: ClassVar[list[str]] = GenerativeModelingProblem._save_fields + DesignProblem._save_fields
    _used_classes: ClassVar[list[Type['ProgramData']]] = list(set(GenerativeModelingProblem._used_classes + DesignProblem._used_classes + [DesignSpace]))

    @property
    def vars_generated(self) -> list[Variable]:
        return self.generate_space.variables


@dataclass(kw_only=True)
class ConditionalGenerativeDesignProblem(ConditionalGenerativeModelingProblem, DesignProblem):

    generate_space: Space
    condition_space: Space

    _data_type_str: ClassVar[str] = 'definition:problem:generativeDesign:conditional'
    _data_type_key: ClassVar[int] = 8076

    _save_fields: ClassVar[list[str]] = GenerativeDesignProblem._save_fields + ['condition_space']
    _used_classes: ClassVar[list[Type['ProgramData']]] = GenerativeDesignProblem._used_classes + [Space, PerformanceSpace]

    def get_training_dataset_spec(self) -> 'DatasetSpecification':
        return DatasetSpecification(columns=self.generate_space.variables + self.condition_space.variables)


if __name__ == '__main__':

    pass