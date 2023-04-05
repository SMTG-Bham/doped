# coding: utf-8

from __future__ import division

__status__ = "Development"

import logging
import logging.config
import os
import unittest

from doped.pycdt.utils.log_util import initialize_logging


class InitializeLoggingTest(unittest.TestCase):
    def test_initialize_logging(self):
        self.assertEqual(initialize_logging(), None)
        self.assertEqual(initialize_logging(filename="this_tmp_filename.txt"), None)
        os.remove("this_tmp_filename.txt")
        self.assertEqual(initialize_logging(level="WARNING"), None)
        self.assertEqual(initialize_logging(level="DEBUG"), None)
        self.assertEqual(initialize_logging(level="NOTSET"), None)


import unittest

if __name__ == "__main__":
    unittest.main()
