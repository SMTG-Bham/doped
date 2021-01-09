# coding: utf-8

from __future__ import division

__author__ = "Nils E. R. Zimmermann, Danny Broberg"
__copyright__ = "Copyright 2014, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Nils E. R. Zimmermann"
__email__ = "nils.e.r.zimmermann@gmail.com"
__status__ = "Development"
__date__ = "October 9, 2017"

import os
import logging
import unittest
import logging.config

from doped.pycdt.utils.log_util import initialize_logging

class InitializeLoggingTest(unittest.TestCase):
    def test_initialize_logging(self):
        self.assertEqual(initialize_logging(), None)
        self.assertEqual(initialize_logging(filename='this_tmp_filename.txt'), None)
        os.remove('this_tmp_filename.txt')
        self.assertEqual(initialize_logging(level='WARNING'), None)
        self.assertEqual(initialize_logging(level='DEBUG'), None)
        self.assertEqual(initialize_logging(level='NOTSET'), None)


import unittest
if __name__ == '__main__':
    unittest.main()
