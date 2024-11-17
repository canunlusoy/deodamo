import logging
from mdev.logger import set_logging_configuration

set_logging_configuration()


logger = logging.getLogger(__name__)
logger.debug('Starting logging.')
