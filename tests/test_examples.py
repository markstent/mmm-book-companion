import subprocess
import sys
from pathlib import Path

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("ex_*.py"))


import pytest


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script):
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
