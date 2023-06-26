import logging
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
import os


logger = None


def init_logger(cfg=None, name="empty_logger", filename="", loglevel="debug"):
    # LOG FORMATTING
    # https://docs.python.org/ko/3.8/library/logging.html#logrecord-attributes
    log_format = "[%(asctime)s]-[%(levelname)s]-[%(name)s]-[%(module)s](%(process)d): %(message)s"
    date_format = '%Y-%m-%d %H:%M:%S'

    if cfg is not None:
        # LOGGER NAME
        name = cfg.LOGGER_NAME if cfg.LOGGER_NAME else "logger"
        _logger = logging.getLogger(name)

        # LOG LEVEL
        log_level = cfg.LOG_LEVEL if cfg.LOG_LEVEL else "DEBUG"
        _logger.setLevel(log_level)

        # LOG CONSOLE
        if cfg.CONSOLE_LOG is True:
            _handler = StreamHandler()
            _handler.setLevel(log_level)

            # logger formatting
            formatter = logging.Formatter(log_format)
            _handler.setFormatter(formatter)
            _logger.addHandler(_handler)

        # LOG FILE
        if cfg.FILE_LOG is True:
            filename = os.path.join(cfg.LOG_FILE_DIR, cfg.LOGGER_NAME + '.log')
            logdir = os.path.dirname(filename)
            if not os.path.exists(logdir):
                os.makedirs(logdir)

            when = cfg.LOG_FILE_ROTATE_TIME if hasattr(cfg, "LOG_FILE_ROTATE_TIME") else "D"
            interval = cfg.LOG_FILE_ROTATE_INTERVAL if hasattr(cfg, "LOG_FILE_ROTATE_INTERVAL") else 1
            _handler = TimedRotatingFileHandler(
                filename=filename,
                when=when,
                interval=interval,
                backupCount=cfg.LOG_FILE_COUNTER,
                encoding='utf8'
            )
            _handler.setLevel(log_level)

            # logger formatting
            formatter = logging.Formatter(log_format)
            _handler.setFormatter(formatter)
            _logger.addHandler(_handler)

    else:       # cfg is None
        _logger = logging.getLogger(name)
        _logger.setLevel(loglevel)

        # CONSOLE LOGGER
        _handler = StreamHandler()
        _handler.setLevel(loglevel)
        formatter = logging.Formatter(log_format)
        _handler.setFormatter(formatter)
        _logger.addHandler(_handler)

        # FILE LOGGER
        filename = os.path.join('./log', name + '.log')
        logdir = os.path.dirname(filename)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        _handler = TimedRotatingFileHandler(
            filename=filename,
            when="D",
            interval=1,
            backupCount=10,
            encoding='utf8'
        )
        _handler.setLevel(loglevel)
        formatter = logging.Formatter(log_format)
        _handler.setFormatter(formatter)
        _logger.addHandler(_handler)

    _logger.info("Start Main logger")
    global logger
    logger = _logger


def get_logger(name=None):
    if name is None:
        global logger
        return logger
    else:
        return logging.getLogger(name)
