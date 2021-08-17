import logging
from termcolor import colored
import torch.distributed as dist

logger_initialized = {}

VERBOSE_LOG_FORMAT = ('%(asctime)s | %(levelname)s | pid-%(process)d | '
                      '%(filename)s:<%(funcName)s>:%(lineno)d | %(message)s')
BRIEF_LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'

LEVEL_DICT = dict(
    NOTSET=logging.NOTSET,  # 0
    DEBUG=logging.DEBUG,  # 10
    INFO=logging.INFO,  # 20
    WARNING=logging.WARNING,  # 30
    ERROR=logging.ERROR,  # 40
    CRITICAL=logging.CRITICAL,  # 50
)


class _ColorfulFormatter(logging.Formatter):
    """
    detectron2: detectron2/utils/logger.py
    """

    def __init__(self, *args, **kwargs):
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = logging.Formatter.formatMessage(self, record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.INFO:
            prefix = colored("INFO", "green", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


VERBOSE_LEVELS = [logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL]


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = _ColorfulFormatter(colored("[%(asctime)s %(filename)s]: ", "green") + "%(message)s",
                                   datefmt="%m/%d %H:%M:%S")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Parameters
    ----------
    msg : str
        The message to be logged.
    logger : {logging.Logger, str}, optional
        The logger to be used.
        Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
    level : int
        Logging level. Only available when `logger` is a Logger object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = get_logger(name='lightradet', log_file=log_file, log_level=log_level)

    return logger
