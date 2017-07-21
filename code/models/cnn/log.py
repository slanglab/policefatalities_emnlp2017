import logging
import uuid


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup(gpuno):
    guid = str(uuid.uuid4())[0:8]
    # create a file handler
    handler = logging.FileHandler(str(gpuno) + 'log.log')
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(guid + '- %(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

