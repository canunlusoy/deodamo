from copy import copy
from typing import ClassVar, Type
from dataclasses import dataclass, field, asdict, make_dataclass
import torch
import torch.nn as nn

from src.utils.iox import ProgramData


class Layers:

    LINEAR = 'linear'

    BATCH_NORM_1D = 'batch_norm_1d'

    DROPOUT = 'dropout'

    keys_callables = {
        LINEAR: nn.Linear,
        BATCH_NORM_1D: nn.BatchNorm1d,
        DROPOUT: nn.Dropout
    }


class ActivationFunctions:

    RELU = 'relu'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'

    keys_callables = {
        RELU: nn.ReLU,
        TANH: nn.Tanh,
        SIGMOID: nn.Sigmoid
    }


_all_keys_callables: dict[str, nn.Module] = {
    **Layers.keys_callables,
    **ActivationFunctions.keys_callables
}


@dataclass
class NetworkSpecification(ProgramData):

    layers: list['LayerSpecification'] = field(default_factory=list)

    _data_type_str: ClassVar[str] = 'network_specification'
    _data_type_key: ClassVar[int] = 900

    _save_fields: ClassVar[list[str]] = ['layers']
    _used_classes: ClassVar[list[Type['ProgramData']]] = []

    def _get_raw_data(self) -> dict:
        raw_spec = [layer.to_dict() for layer in self.layers]
        return {'layers': raw_spec}

    @classmethod
    def from_dict(cls, dct: dict) -> 'ProgramData':
        layers_dcts = dct['layers']
        layers = [get_layer_spec_from_dct(layer_dct) for layer_dct in layers_dcts]
        return NetworkSpecification(layers=layers)

    def get_network(self, device: torch.device = None) -> nn.Sequential:
        modules = []

        for layer_index, layer in enumerate(self.layers):
            if isinstance(layer, ModuleLinkedSpecification):
                # Can directly get the module from it
                modules.append(layer.get_module())
            else:
                raise NotImplementedError()

        network = nn.Sequential(*modules)
        if device is not None:
            network.to(device)
        return network


@dataclass
class LayerSpecification:

    layer_type_key: ClassVar[str] = 'Layer'
    key_type: ClassVar[str] = '__type__'

    def to_dict(self) -> dict:
        return {self.key_type: self.layer_type_key, **asdict(self)}

    @classmethod
    def from_dict(cls, dct: dict) -> 'LayerSpecification':

        if cls.key_type not in dct:
            message = 'Cannot identify type of layer'
            raise KeyError(message)

        if dct[cls.key_type] != cls.layer_type_key:
            message = f'Provided dictionary is not for the specification of layer type "{cls.layer_type_key}"'
            raise ValueError(message)

        dct.pop(cls.key_type)
        return cls(**dct)

    def get_module(self) -> nn.Module:
        pass


@dataclass
class ModuleLinkedSpecification:
    """Marks layer specifications that directly correspond to an ``nn.Module``, i.e. constructing the
    module does not require information from another object."""
    associated_module: ClassVar[Type[nn.Module]] = nn.Module

    def get_module(self) -> nn.Module:
        pass


@dataclass
class _KWOnlyLayerSpecification(LayerSpecification, ModuleLinkedSpecification):

    kwargs: dict = field(default_factory=dict)

    def get_module(self) -> nn.Module:
        return self.associated_module(**self.kwargs)


class ActivationFunction:
    """Simply a designation for activation function modules."""
    pass


@dataclass
class ReLU(ActivationFunction, _KWOnlyLayerSpecification):
    layer_type_key: ClassVar[str] = 'ReLU'
    associated_module: ClassVar[Type[nn.Module]] = nn.ReLU


@dataclass
class LeakyReLU(ActivationFunction, _KWOnlyLayerSpecification):
    layer_type_key: ClassVar[str] = 'Leaky_ReLU'
    associated_module: ClassVar[Type[nn.Module]] = nn.LeakyReLU


@dataclass
class Tanh(ActivationFunction, _KWOnlyLayerSpecification):
    layer_type_key: ClassVar[str] = 'Tanh'
    associated_module: ClassVar[Type[nn.Module]] = nn.Tanh


@dataclass
class Sigmoid(ActivationFunction, _KWOnlyLayerSpecification):
    layer_type_key: ClassVar[str] = 'Sigmoid'
    associated_module: ClassVar[Type[nn.Module]] = nn.Sigmoid


@dataclass
class Dropout(_KWOnlyLayerSpecification):
    layer_type_key: ClassVar[str] = 'Dropout'
    associated_module: ClassVar[Type[nn.Module]] = nn.Dropout


@dataclass
class BatchNorm1D(_KWOnlyLayerSpecification):
    layer_type_key: ClassVar[str] = 'BatchNorm1D'
    associated_module: ClassVar[Type[nn.Module]] = nn.BatchNorm1d


@dataclass
class Linear(LayerSpecification, ModuleLinkedSpecification):
    layer_type_key: ClassVar[str] = 'Linear'
    associated_module: ClassVar[Type[nn.Module]] = nn.Linear

    in_features: int
    out_features: int

    kwargs: dict = field(default_factory=dict)

    def get_module(self) -> nn.Module:
        return nn.Linear(in_features=self.in_features,
                         out_features=self.out_features,
                         **self.kwargs)


layer_types_by_key: dict[str, Type[LayerSpecification]] = {
    type_.layer_type_key: type_ for type_ in [
        ReLU, LeakyReLU, Tanh, Sigmoid, Dropout, BatchNorm1D,
        Linear
    ]
}


def get_layer_spec_from_dct(dct: dict) -> LayerSpecification:

    if LayerSpecification.key_type not in dct:
        message = 'Cannot identify type of layer'
        raise KeyError(message)

    layer_type_key = dct[LayerSpecification.key_type]
    layer_type = layer_types_by_key[layer_type_key]
    return layer_type.from_dict(dct)


if __name__ == '__main__':

    layers = [
        ReLU(),
        BatchNorm1D(),
        Tanh()
    ]
