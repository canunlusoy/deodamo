from typing import ClassVar, Type
from dataclasses import dataclass

from utils.iox import ProgramData


@dataclass(frozen=True)
class Asset(ProgramData):

    name: str
    parent: 'Asset' = None

    _data_type_str: ClassVar[str] = 'definition:asset'
    _data_type_key: ClassVar[int] = 1000

    _save_fields: ClassVar[list[str]] = ['name', 'parent']
    _used_classes: ClassVar[list[Type['ProgramData']]] = []


@dataclass(frozen=True)
class DesignAsset(Asset):

    _data_type_str: ClassVar[str] = 'definition:asset:design'
    _data_type_key: ClassVar[int] = 1100

    _save_fields: ClassVar[list[str]] = Asset._save_fields
    _used_classes: ClassVar[list[Type['ProgramData']]] = []

    def __repr__(self):
        to_return = f'{self.name}'
        if self.parent is not None and isinstance(self.parent, Asset):
            to_return += f' ∈ {self.parent.name}'
        return to_return

