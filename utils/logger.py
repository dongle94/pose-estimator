import logging
import os


logger = None


def init_logger(name="", filename=None, loglevel='INFO'):
    # if not exist log dir, mkdir
    logdir = os.path.dirname(filename)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # https://docs.python.org/ko/3.8/library/logging.html#logrecord-attributes
    log_format = "[%(asctime)s]-[%(levelname)s]-[%(name)s]-[%(module)s](%(process)d): %(message)s"
    date_format = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(filename=filename,
                        filemode='a',
                        format=log_format,
                        datefmt=date_format,
                        level=loglevel)
    logging.info("logging Start")

    global logger
    logger = logging


def get_logger(name=None):
    global logger
    if logger is not None:
        return logger