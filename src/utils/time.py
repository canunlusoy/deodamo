
from datetime import datetime

TIMESTAMP_FORMAT = '%Y/%m/%d %H:%M:%S %z'


def get_timestamp() -> str:
    return datetime.now().astimezone().strftime(TIMESTAMP_FORMAT)
