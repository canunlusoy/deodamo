from src.datamodels.assets import DesignAsset
from src.datamodels.analyses import PerformanceMetric, AnalysisStandard
from src.datamodels.parameterizations import DesignParameterization

from src.pman.datamodels.campaigns import ConditionalGenerativeDesignCampaign

class Briefcase:

    pass







class DesignLibrary:

    def __init__(self):

        self.assets: list[DesignAsset] = []
        self.parameterizations: list[DesignParameterization] = []



class AnalysisLibrary:

    def __init__(self):

        self.performance_metrics: list[PerformanceMetric] = []
        self.analysis_standards: list[AnalysisStandard] = []


class ProjectLibrary:

    def __init__(self):

        self.campaigns: list[]


