#!/usr/bin/env python

__status__ = "Development"

import logging
import logging.config
import os

from monty.serialization import loadfn

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def initialize_logging(filename=None, level=None):
    config_dict = loadfn(os.path.join(MODULE_DIR, "logging.yaml"))
    if filename:
        config_dict["handlers"]["file_handler"].update({"filename": filename})
    if level:
        config_dict["handlers"]["file_handler"].update({"level": level})

    logging.config.dictConfig(config_dict)
