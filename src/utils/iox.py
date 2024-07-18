import json
from os import PathLike
from typing import ClassVar, Type, Union
from getpass import getuser
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


KEY_PROGRAM_DATA = '__program_data__'
KEY_DATA_TYPE_KEY = 'data_type_key'
KEY_DATA_TYPE_STR = 'data_type_str'
keys_typeinfo = [KEY_DATA_TYPE_KEY, KEY_DATA_TYPE_STR]
KEY_TYPEINFO = 'typeinfo'
KEY_METADATA = 'metadata'
JSON_INDENT_LEVEL = 4


class ProgramData:

    _data_type_str: ClassVar[str] = None
    _data_type_key: ClassVar[int] = None

    _save_fields: ClassVar[list[str]] = []
    _used_classes: ClassVar[list[Type['ProgramData']]] = []

    def __init__(self):
        self.metadata: dict[str] = None

    def get_raw_data(self) -> dict:
        return {KEY_TYPEINFO: {KEY_PROGRAM_DATA: True,
                               KEY_DATA_TYPE_KEY: self._data_type_key,
                               KEY_DATA_TYPE_STR: self._data_type_str},
                **self._get_raw_data()}

    def _get_raw_data(self):
        return {key: getattr(self, key) for key in self._save_fields}

    def write(self, filepath: Path) -> None:
        to_write = {**self.get_raw_data(),
                    KEY_METADATA: self.get_metadata()}

        with open(filepath, 'w') as file:
            json.dump(to_write, file,
                      indent=JSON_INDENT_LEVEL,
                      default=lambda obj: obj.get_raw_data())

    @classmethod
    def from_file(cls, filepath: str | Path) -> 'ProgramData':
        with open(filepath) as file:
            dct = json.load(file,
                            object_hook=lambda dct: cls._decode_if_possible_or_return(dct))

        if isinstance(dct, dict):
            return cls.from_dict(dct)

        return dct  # at this point, it may be fully deserialized, a proper object

    @classmethod
    def from_dict(cls, dct: dict) -> 'ProgramData':
        field_data = {}
        for key, val in dct.items():
            if key in cls._save_fields:
                field_data[key] = val
        return cls(**field_data)

    @classmethod
    def get_used_classes_keys(cls) -> dict[int, Type['ProgramData']]:

        used_classes = set()

        def recurse(_cls):
            for used_cls in _cls._used_classes:
                recurse(used_cls)

            used_classes.add(_cls)

        recurse(cls)


        return {cls._data_type_key: cls,  # add own type to options
                # if class uses its own type in its data, it is difficult to reference to its own type
                # in cls.used_classes, so add reference here.
                **{_cls._data_type_str: _cls for _cls in used_classes}}



        # TODO NESTED used_classes

    @classmethod
    def _decode_if_possible_or_return(cls, dct: dict) -> Union[dict, 'ProgramData']:
        try:
            return cls._decode_raw_program_data(raw_data=dct)
        except (KeyError, ValueError) as error:
            return dct
        return dct

    @classmethod
    def _decode_raw_program_data(cls, raw_data: dict):
        if KEY_TYPEINFO not in raw_data and any(key not in raw_data[KEY_TYPEINFO] for key in keys_typeinfo):
            raise KeyError()
        typeinfo = raw_data[KEY_TYPEINFO]
        if typeinfo[KEY_PROGRAM_DATA] != True:
            raise ValueError()
        if KEY_DATA_TYPE_KEY not in typeinfo:
            raise KeyError()

        data_type_str = typeinfo[KEY_DATA_TYPE_STR]
        data_type_cls = cls.get_used_classes_keys()[data_type_str]
        decoded = data_type_cls.from_dict(raw_data)
        return decoded

    def get_metadata(self) -> dict:
        return Session().get_user_time()


@dataclass
class UserProfile(ProgramData):

    _data_type_str: ClassVar[str] = 'pdata:userProfile'
    _data_type_key: ClassVar[int] = 12

    name: str
    id: str = None
    org_code: str = None
    email_address: str = None

    _save_fields: ClassVar[list[str]] = ['name', 'id', 'org_code', 'email_address']

    @staticmethod
    def get_from_device() -> 'UserProfile':
        return UserProfile(name=getuser())

    def get_metadata(self) -> dict:
        return {'time': self.get_timestamp()}

    def get_timestamp(self) -> str:
        return datetime.now().astimezone().strftime(TIMESTAMP_FORMAT)


TIMESTAMP_FORMAT = '%Y/%m/%d %H:%M:%S %z'

KEY_USER = 'user'
KEY_TIME = 'time'


class Session:

    def __init__(self):
        self._user = UserProfile.get_from_device()

    def set_user(self, user_profile: UserProfile):
        self._user = user_profile

    def get_user(self) -> UserProfile:
        return self._user

    @staticmethod
    def get_timestamp() -> str:
        return datetime.now().astimezone().strftime(TIMESTAMP_FORMAT)

    def get_user_time(self) -> dict:
        return {KEY_USER: self.get_user(),
                KEY_TIME: self.get_timestamp()}


