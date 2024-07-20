from typing import ClassVar, Type
from dataclasses import dataclass

from src.datamodels.assets import Asset, DesignAsset
from src.datamodels.spaces import DesignSpace
from src.datamodels.variables import Variable, DesignVariable

from utils.iox import ProgramData


@dataclass
class Parameterization(ProgramData):

    name: str
    parameterized_asset: Asset
    parameters: list[Variable]

    _data_type_str: ClassVar[str] = 'definition:parameterization'
    _data_type_key: ClassVar[int] = 500

    _save_fields: ClassVar[list[str]] = ['name', 'parameterized_asset', 'parameters']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [Asset, Variable]


@dataclass
class DesignParameterization(Parameterization):

    name: str

    parameterized_asset: DesignAsset
    # Direction of referencing should be from parameterization to design asset
    # as multiple parameterizations for the same asset may emerge over time.

    parameters: list[DesignVariable]

    _data_type_str: ClassVar[str] = 'definition:parameterization:design'
    _data_type_key: ClassVar[int] = 501

    _used_classes: ClassVar[list[Type['ProgramData']]] = [DesignAsset, DesignVariable]

    def get_design_space_from_parameters(self) -> DesignSpace:
        return DesignSpace(variables=self.parameters)

    def __repr__(self) -> str:
        return f'{self.name} ({len(self.parameters)} parameters describing "{self.parameterized_asset.name}")'

