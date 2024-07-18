from typing import ClassVar, Type
from dataclasses import dataclass, field

from src.datamodels.datasets import DatasetSpecification
from src.datamodels.spaces import Space, DesignSpace
from src.datamodels.variables import Variable
from src.datamodels.parameterizations import DesignParameterization
from src.datamodels.assets import DesignAsset

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

    assets_to_design: list[Type[DesignAsset]]
    asset_parameterizations: dict[Type[DesignAsset], Type[DesignParameterization]]

    _data_type_str: ClassVar[str] = 'definition:problem:design'
    _data_type_key: ClassVar[int] = 8010

    _save_fields: ClassVar[list[str]] = Problem._save_fields + ['assets_to_design', 'asset_parameterizations']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [DesignAsset]

    def __post_init__(self):
        if not all(asset_type in self.asset_parameterizations for asset_type in self.assets_to_design):
            msg = (f'A parameterization of record must be provided for all assets to be designed '
                   f'as part of the design problem must have')
            raise KeyError(msg)


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
class GenerativeDesignProblem(GenerativeModelingProblem, DesignProblem):

    _data_type_str: ClassVar[str] = 'definition:problem:generativeDesign'
    _data_type_key: ClassVar[int] = 8075

    _save_fields: ClassVar[list[str]] = GenerativeModelingProblem._save_fields + ['design_space']
    _used_classes: ClassVar[list[Type['ProgramData']]] = GenerativeModelingProblem._used_classes + [DesignSpace]

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
    _used_classes: ClassVar[list[Type['ProgramData']]] = GenerativeDesignProblem._used_classes + [Space]

    def get_training_dataset_spec(self) -> 'DatasetSpecification':
        return DatasetSpecification(columns=self.generate_space.variables + self.condition_space.variables)


if __name__ == '__main__':

    pass