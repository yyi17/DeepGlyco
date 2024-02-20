__all__ = ["get_logger"]

import logging
import os
import sys
from typing import Optional


def get_logger(
    name: Optional[str] = None, console: bool = True, file: Optional[str] = None
):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    formater = logging.Formatter(
        "%(asctime)s %(filename)s: [%(levelname)s] %(message)s"
    )
    if console and not any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    ):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formater)
        logger.addHandler(console_handler)
    if file and not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(file)
        for h in logger.handlers
    ):
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formater)
        logger.addHandler(file_handler)
    return logger
