"""Run independent checks, then emit their deterministic evidence summary."""
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
script=root/'tests/test_independent.py'
status=subprocess.run([sys.executable,str(script)]).returncode
if status:raise SystemExit(status)
(root/'results').mkdir(exist_ok=True)
raise SystemExit(subprocess.run([sys.executable,str(script),'--report']).returncode)
