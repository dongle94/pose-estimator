import logging
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
import os


logger = None


def init_logger(cfg=None, name="default", filename="", loglevel="debug"):
    # LOG FORMATTING
    # https://docs.python.org/ko/3.8/library/logging.html#logrecord-attributes
    log_format = "[%(asctime)s]-[%(levelname)s]-[%(name)s]-[%(module)s](%(process)d): %(message)s"
    date_format = '%Y-%m-%d %H:%M:%S'

    if cfg is not None:
        # LOGGER NAME
        name = cfg.logger_name if cfg.logger_name else name
        _logger = logging.getLogger(name)

        # LOG LEVEL
        log_level = cfg.log_level.upper() if cfg.log_level else loglevel.upper()
        _logger.setLevel(log_level)

        # LOG CONSOLE
        if cfg.console_log is True:
            _handler = StreamHandler()
            _handler.setLevel(log_level)

            # logger formatting
            formatter = logging.Formatter(log_format)
            _handler.setFormatter(formatter)
            _logger.addHandler(_handler)

        # LOG FILE
        if cfg.file_log is True:
            filename = os.path.join(cfg.file_log_dir, cfg.logger_name + '.log')
            logdir = os.path.dirname(filename)
            if not os.path.exists(logdir):
                os.makedirs(logdir)

            when = cfg.file_log_rotate_time if hasattr(cfg, "file_log_rotate_time") else "D"
            interval = cfg.file_log_rotate_interval if hasattr(cfg, "file_log_rotate_interval") else 1
            _handler = TimedRotatingFileHandler(
                filename=filename,
                when=when,
                interval=interval,
                backupCount=cfg.file_log_counter,
                encoding='utf8'
            )
            _handler.setLevel(log_level)

            # logger formatting
            formatter = logging.Formatter(log_format)
            _handler.setFormatter(formatter)
            _logger.addHandler(_handler)

    else:       # cfg is None
        loglevel = loglevel.upper()
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
