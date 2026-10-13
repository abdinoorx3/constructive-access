"""Run producer tests once; independent checks have their own result-producing run."""
from pathlib import Path
import sys
import unittest

root=Path(__file__).resolve().parent
suite=unittest.TestSuite()
loader=unittest.TestLoader()
for path in sorted((root/'tests').glob('test_*.py')):
    if path.name!='test_independent.py':
        suite.addTests(loader.discover(str(root/'tests'),pattern=path.name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
