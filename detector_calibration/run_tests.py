"""Run producer checks; independent adversarial review reports separately."""
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parent
loader = unittest.TestLoader()
suite = unittest.TestSuite()
for directory in ('tests', 'countermodels', 'experiment'):
    suite.addTests(loader.discover(str(root/directory), pattern='test_*.py',
                                  top_level_dir=str(root/directory)))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
