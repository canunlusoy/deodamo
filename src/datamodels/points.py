from collections import UserDict

from src.datamodels.variables import SamplingVariable


class Point(UserDict):

    def get_var_value_by_name(self, var_name: str):
        for svar, svar_value in self.items():
            assert isinstance(svar, SamplingVariable)
            if svar.name == var_name:
                return svar_value


class DesignSpacePoint(Point):
    pass


class SamplingSpacePoint(Point):
    pass


class AbstractSpacePoint(Point):
    pass
