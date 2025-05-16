import logging


def config_logger(level=0, file=None):
    """
    Configure the logging of a program
    Log is written in stdio, alternatively also in a file

    :param level: If level is 0 only errors are logged, else all is logged
    :param file: Log is written in a file,
    :return:
    """
    # Logging configuration
    logger = logging.getLogger("log")
    if level == 0:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    return logger
