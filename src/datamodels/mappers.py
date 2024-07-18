from inspect import signature, getsource
from pathlib import Path
from typing import Any, Callable

from src.datamodels.points import Point
from src.datamodels.variables import SamplingVariable

from utils.programming import is_lambda_function


class S2SMapper:

    def map(self, point: Point) -> Any:
        return None

    def to_dict(self) -> dict:
        pass


class Equate(S2SMapper):

    type_string = 'Equate'

    def __init__(self, source_var: SamplingVariable):
        self.source_var = source_var

    def map(self, point: Point) -> Any:
        return point[self.source_var]

    def __repr__(self):
        return f'={self.source_var.name}'

    def to_dict(self) -> dict:
        return {'__type__': self.type_string,
                'source_var': self.source_var}


class FunctionReference(S2SMapper):

    def __init__(self, module_path: Path, fcn_name: str):
        pass



class CustomFunctionMapper(S2SMapper):

    def __init__(self, function_str: str):
        self.function_str = function_str
        self.function: Callable = eval(function_str)
        self.input_arg_names = list(signature(self.function).parameters.keys())

    def map(self, point: Point) -> Any:
        input_arg_values = [point.get_var_value_by_name(arg_name) for arg_name in self.input_arg_names]
        return self.function(*input_arg_values)

    @staticmethod
    def from_lambda(lambda_fcn: Callable, available_sampling_vars: list[SamplingVariable]):

        available_svar_names = [svar.name for svar in available_sampling_vars]

        # Check signature
        sig = signature(lambda_fcn)
        arg_names = list(sig.parameters.keys())

        if not all(arg_name in available_svar_names for arg_name in arg_names):
            invalid_arg_names = [arg_name for arg_name in arg_names if arg_name not in available_svar_names]
            message = f'Following arguments to the lambda function do not match any sampling variables: {", ".join(invalid_arg_names)}'
            raise ValueError(message)

        # Get source
        fcn_str = getsource(lambda_fcn).split('lambda')[1]  # split at keyword ``lambda``, take right of the keyword
        fcn_str = fcn_str.strip().replace('\n', ' ').strip(' \n ,')  # trim
        fcn_str = f'lambda {fcn_str}'
        try:
            fcn_obj = eval(fcn_str)
        except SyntaxError as error:
            message = f'Error while processing lambda function:\n{error}'
            raise ValueError(message)

        if not is_lambda_function(fcn_obj):
            message = f'Provided function definition could not be resolved into a lambda function: "{fcn_str}"'
            raise ValueError(message)

        return CustomFunctionMapper(function_str = fcn_str)

    def __repr__(self):
        return f'{self.function_str}'


class LookupMapper(S2SMapper):

    def __init__(self, function: Callable, lookup_table_fps: dict[str, str | Path]):
        self.function = function
        self.lookup_table_fps = lookup_table_fps
