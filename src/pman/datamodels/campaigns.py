from dataclasses import dataclass, field

from utils.iox import ProgramData

from src.datamodels.assets import DesignAsset
from src.datamodels.parameterizations import DesignParameterization
from src.pman.datamodels.problems import Problem, ConditionalGenerativeDesignProblem


@dataclass
class Campaign(ProgramData):

    id: str
    name: str

    problems: list[Problem]
    """List of conditional generative design problems solved by this campaign.
    More than one problem of this type may be solved as part of one campaign."""

    # Private fields for data type definition
    _data_type_key = 8800
    _data_type_str = 'definition:campaign'

    _save_fields = ['id', 'problems']
    _used_classes = [Problem]


@dataclass
class ConditionalGenerativeDesignCampaign(Campaign):

    problems: list[ConditionalGenerativeDesignProblem]

    design_record_parameterization: dict[DesignAsset, DesignParameterization]
    """Outputs of the campaign must be designs that can be reported and stored for the long term,
    in terms of a "design record parameterization"."""

    # Private fields for data type definition
    _data_type_key = 8900
    _data_type_str = 'definition:campaign:generativeDesign:conditional'

    _save_fields = Campaign._save_fields + ['design_record_parameterization']
    _used_classes = [ConditionalGenerativeDesignProblem, DesignParameterization]
