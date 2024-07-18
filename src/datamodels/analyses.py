from typing import ClassVar, Type, Any
from dataclasses import dataclass, field

from src.datamodels.assets import Asset
from src.datamodels.variables import Metric

from utils.iox import ProgramData


@dataclass(frozen=True)
class PerformanceMetric(Metric):
    """Data type to represent a performance metric, independent of any particular analysis tool or standard."""

    name: str

    _data_type_str: ClassVar[str] = 'reference:metric:performance'
    _data_type_key: ClassVar[int] = 750

    _save_fields: ClassVar[list[str]] = ['name']


@dataclass(frozen=True)
class ContinuousPerformanceMetric(PerformanceMetric):
    """Data type to represent a performance metric with continuous values."""

    typical_range: tuple[float | int, float | int] = None

    _data_type_str: ClassVar[str] = 'reference:metric:performance:continuous'
    _data_type_key: ClassVar[int] = 751

    _save_fields: ClassVar[list[str]] = PerformanceMetric._save_fields + ['typical_range']


@dataclass
class AnalysisSetting(ProgramData):
    """Data type to represent an analysis setting and its value.
    Could be used in relation to a particular analysis tool or standard, or in a more general
    sense, that can apply to any analysis standard."""

    name: str
    value: Any = None

    _data_type_str: ClassVar[str] = 'setting:analysis'
    _data_type_key: ClassVar[int] = 97

    _save_fields: ClassVar[list[str]] = ['name', 'value']
    _used_classes: ClassVar[list[Type['ProgramData']]] = []

    def __repr__(self) -> str:
        to_return = f'{self.name}'
        if self.value is not None:
            to_return += f':{self.value}'
        return to_return


@dataclass
class AnalysisStandard(ProgramData):

    name: str

    analyzed_assets: list[Asset] = field(default_factory=list)
    """Assets being analysed."""

    output_metrics: list['PerformanceMetric'] = field(default_factory=list)

    settings: list[AnalysisSetting] = field(default_factory=list)

    _data_type_str: ClassVar[str] = 'reference:standard:analysis'
    _data_type_key: ClassVar[int] = 721

    _save_fields: ClassVar[list[str]] = ['name', 'analyzed_assets', 'output_metrics', 'settings']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [Asset, PerformanceMetric, AnalysisSetting]

    def get_own_performance_metric(self, performance_metric: PerformanceMetric) -> 'AnalysisStandardPerformanceMetric':
        """Returns an ``AnalysisStandardPerformanceMetric`` instance, representing the performance metric
        calculated by this particular analysis standard."""

        if performance_metric not in self.output_metrics:
            raise KeyError(f'Provided performance metric "{performance_metric.name}" is not an output of '
                           f'this analysis standard.')

        return AnalysisStandardPerformanceMetric(analysis_standard=self, performance_metric=performance_metric)

    def __repr__(self) -> str:
        to_return = f'{self.name}'
        if self.analyzed_assets:
            to_return += f' (analyzing {", ".join([asset.name for asset in self.analyzed_assets])})'

        return to_return


@dataclass
class AnalysisStandardPerformanceMetric(ProgramData):
    """Data type to represent performance metric given by a particular analysis standard.
    This is in contrast to just regular ``PerformanceMetric``, which represents a performance metric in
    general, independent of an analysis standard.

    This data type is essentially a container of a reference to an analysis standard and a performance metric.

    Different analysis standards may compute the same metric, but obtain different values due to a variety of
    reasons. As a result, we may need to distinguish between the same metric computed with different standards.
    """

    analysis_standard: AnalysisStandard
    performance_metric: PerformanceMetric

    _data_type_str: ClassVar[str] = 'multiref:metric:performance:standard:analysis'
    _data_type_key: ClassVar[int] = 721750

    _save_fields: ClassVar[list[str]] = ['analysis_standard', 'performance_metric']
    _used_classes: ClassVar[list[Type['ProgramData']]] = [AnalysisStandard, PerformanceMetric]
