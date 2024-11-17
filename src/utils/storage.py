from datamodels.assets import DesignAsset
from datamodels.analyses import PerformanceMetric, AnalysisStandard
from datamodels.parameterizations import DesignParameterization

from pman.datamodels.campaigns import ConditionalGenerativeDesignCampaign

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

        self.campaigns: list


