from pathlib import Path
from typing import Any, Callable, ClassVar, Type
from dataclasses import dataclass

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from scipy.interpolate import RegularGridInterpolator

from src.utils.iox import ProgramData
from src.datamodels.points import Point, AbstractSpacePoint
from src.datamodels.variables import SamplingVariable

from utils.programming import is_lambda_function


class S2SMapper:

    def map(self, point: Point, dimension_index: int, *args, **kwargs) -> Any:
        return None

    def to_dict(self) -> dict:
        pass


@dataclass
class SympyExpressionMapper(ProgramData, S2SMapper):

    raw_expression: str

    _data_type_str: ClassVar[str] = 'sympy_expression_mapper'
    _data_type_key: ClassVar[int] = '8000'

    _save_fields: ClassVar[list[str]] = ['raw_expression']
    _used_classes: ClassVar[list[Type['ProgramData']]] = []

    def __post_init__(self):
        self.expression = parse_expr(self.raw_expression)

    @property
    def variables(self) -> list[str]:
        return list([symbol.name for symbol in self.expression.free_symbols])

    def evaluate(self, variable_values: dict[str, float]):
        if not all(variable in variable_values for variable in self.variables):
            raise KeyError('Provide values for all variables to evaluate the expression. '
                           f'Values for some variables are missing. Required variables: {[", ".join(self.variables)]}')

        result = self.expression.subs(variable_values)
        return result

    def map(self, point: Point) -> Any:
        mapped = self.evaluate(variable_values={variable: value for variable, value in point.items() if variable in self.variables})
        return mapped


@dataclass
class FixedRangeLinearMapper(ProgramData, S2SMapper):

    fixed_range: tuple[float, float]

    _data_type_str: ClassVar[str] = 'fixed_range_linear_mapper'
    _data_type_key: ClassVar[int] = '8005'

    _save_fields: ClassVar[list[str]] = ['fixed_range']
    _used_classes: ClassVar[list[Type['ProgramData']]] = []

    def __post_init__(self):
        if len(self.fixed_range) != 2:
            message = f'Fixed range must be a tuple of 2 elements, lower and upper limits.'
            raise ValueError(message)

    def map(self,
            point: AbstractSpacePoint,
            dimension_index: int,
            *args, **kwargs
            ) -> float:

        if dimension_index is None:
            message = f'Index of the dimension in the abstract space point must be provided!'
            raise ValueError(message)

        raw_value = point[str(dimension_index)]

        lower, upper = self.fixed_range
        mapped_value = lower + (upper - lower) * raw_value

        return mapped_value

    def from_dict(cls, dct: dict) -> 'ProgramData':
        field_data = {}
        for key, val in dct.items():
            if key in cls._save_fields:
                field_data[key] = val

        # Forcibly type cast to tuple - if data was read from JSON, it would have been read as list as JSON has no tuple type
        field_data['fixed_range'] = tuple(field_data['fixed_range'])

        return cls(**field_data)


@dataclass
class BivariateTabulatedRangeLinearMapper(ProgramData, S2SMapper):
    """
    Maps value of a variable from an abstract space (where value is between 0 and 1) to a sampling space (where value is a real number).
    The mapping is linear.

    Lower and upper bounds of the variable value in the target space is given by bivariate tables.
    (One table for lower bound, another table for upper bound)

    "Bivariate table" here means that the rows and columns of the table correspond to values of other sampling space variables.
    """

    range_lower_table_csv_fp: str | Path
    """Filepath to the table listing lower bounds of the variable value"""

    range_upper_table_csv_fp: str | Path
    """Filepath to the table listing upper bounds of the variable value"""

    table_rows_var: SamplingVariable
    """Rows of the tables correspond to values of this sampling variable"""

    table_cols_var: SamplingVariable
    """Columns of the tables correspond to values of this sampling variable"""

    _data_type_str: ClassVar[str] = 'tabulated_range_linear_mapper'
    _data_type_key: ClassVar[int] = '8007'

    _save_fields: ClassVar[list[str]] = ['table_csv_fp', 'dependee_sampling_vars']
    _used_classes: ClassVar[list[Type['ProgramData']]] = []

    def __post_init__(self):
        self.range_lower_table_csv_fp = Path(self.range_lower_table_csv_fp)
        self.range_upper_table_csv_fp = Path(self.range_upper_table_csv_fp)

        range_lower_table_raw = np.genfromtxt(self.range_lower_table_csv_fp, delimiter=',')
        range_upper_table_raw = np.genfromtxt(self.range_upper_table_csv_fp, delimiter=',')
        
        self.range_lower_table_row_var_vals = range_lower_table_raw[0, 1:]
        self.range_lower_table_col_var_vals = range_lower_table_raw[1:, 0]
        self.range_lower_table = range_lower_table_raw[1:, 1:]

        self.range_upper_table_row_var_vals = range_upper_table_raw[0, 1:]
        self.range_upper_table_col_var_vals = range_upper_table_raw[1:, 0]
        self.range_upper_table = range_upper_table_raw[1:, 1:]
        
        self.range_lower_interpolator = RegularGridInterpolator(
            (self.range_lower_table_row_var_vals, self.range_lower_table_col_var_vals),
            self.range_lower_table,
            method='linear',
            bounds_error=True
        )

        self.range_upper_interpolator = RegularGridInterpolator(
            (self.range_upper_table_row_var_vals, self.range_upper_table_col_var_vals),
            self.range_upper_table,
            method='linear',
            bounds_error=True
        )

    def map(self,
            point: AbstractSpacePoint,
            dimension_index: int,
            dependee_sampling_var_values: dict[SamplingVariable, float] = None,
            *args, **kwargs
            ) -> Any:

        if dimension_index is None:
            message = f'Index of the dimension in the abstract space point must be provided!'
            raise ValueError(message)

        if not (self.table_rows_var in dependee_sampling_var_values and self.table_cols_var in dependee_sampling_var_values):
            message = f'Values of sampling variables representing table row and columns must be provided! Some are missing.'
            raise KeyError(message)

        raw_value = point[str(dimension_index)]

        row_var_val = dependee_sampling_var_values[self.table_rows_var]
        col_var_val = dependee_sampling_var_values[self.table_cols_var]

        range_lower_bound = self.range_lower_interpolator((row_var_val, col_var_val))
        range_upper_bound = self.range_upper_interpolator((row_var_val, col_var_val))

        mapped_value = range_lower_bound + (range_upper_bound - range_lower_bound) * raw_value

        return mapped_value

    def _get_raw_data(self):
        dct = {key: getattr(self, key) for key in self._save_fields}
        for key in dct:
            if 'fp' in key:
                # Cast Path types as string for raw data (cannot save Path objects to JSON)
                dct[key] = str(dct[key])

    def from_dict(cls, dct: dict) -> 'ProgramData':
        field_data = {}
        for key, val in dct.items():
            if key in cls._save_fields:
                field_data[key] = val

        for key in field_data:
            if 'fp' in key:
                field_data[key] = Path(field_data[key])

        return cls(**field_data)
