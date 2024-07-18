import json
import logging
import logging.config
from pathlib import Path

import colorlog

LOG_LEVEL_NAME_TRAINING = 'TRAIN'
LOG_LEVEL_TRAINING = 21

LOG_LEVEL_NAME_VALIDATION = 'VAL'
LOG_LEVEL_VALIDATION = 22


FN_LOG_CONFIG = 'mdev_log_config.json'

DEFAULT_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            '()': 'colorlog.ColoredFormatter',
            'format': '{name}: {asctime} {log_color}[{levelname}] {message}',
            'style': '{',
            'datefmt': '%H:%M:%S',
            'log_colors': {
                **colorlog.default_log_colors,
                **{LOG_LEVEL_NAME_TRAINING: 'blue',
                   LOG_LEVEL_NAME_VALIDATION: 'purple'}
            }
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {},
    'root': {'level': 'INFO', 'handlers': ['console']}
}



def set_logging_configuration():

    fp_log_config = Path(__file__).parent / FN_LOG_CONFIG

    if not fp_log_config.is_file():
        with open(fp_log_config, 'w') as file:
            json.dump(DEFAULT_LOGGING_CONFIG, file, indent = 4)
        log_config = DEFAULT_LOGGING_CONFIG
    else:
        with open(fp_log_config) as file:
            log_config = json.load(file)

    logging.config.dictConfig(log_config)

    logging.addLevelName(LOG_LEVEL_TRAINING, LOG_LEVEL_NAME_TRAINING)
    logging.addLevelName(LOG_LEVEL_VALIDATION, LOG_LEVEL_NAME_VALIDATION)
