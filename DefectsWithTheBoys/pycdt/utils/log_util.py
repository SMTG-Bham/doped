#!/usr/bin/env python

__author__ = "Bharat Medasani"
__copyright__ = "Copyright 2014, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Bharat Medasani"
__email__ = "mbkumar@gmail.com"
__status__ = "Development"
__date__ = "Jul 24, 2016"

import os
import logging
import logging.config

from monty.serialization import loadfn

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def initialize_logging(filename=None, level=None):
    config_dict = loadfn(os.path.join(MODULE_DIR, 'logging.yaml'))
    if filename:
        config_dict['handlers']['file_handler'].update({'filename': filename})
    if level:
        config_dict['handlers']['file_handler'].update({'level': level})

    logging.config.dictConfig(config_dict)

