import logging
import os
import sys

import oneflow as flow

logger_initialized = []


def setup_logger(output=None, name="pplanedet"):
    """
    Initialize the passl logger and set its verbosity level to "INFO".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S")
    # stdout logging: master only
    local_rank = flow.env.get_rank()
    if local_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if local_rank > 0:
            filename = filename + ".rank{}".format(local_rank)

        # PathManager.mkdirs(os.path.dirname(filename))
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # fh = logging.StreamHandler(_cached_log_stream(filename)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
    logger_initialized.append(name)
    return logger


def get_logger(name='passl'):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    return setup_logger(name=name)