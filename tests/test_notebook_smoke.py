import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_fixes_smoke_test_passes():
    """Runs scripts/smoke_test_notebook_fixes.py, which checks the actual current
    source of the fixed notebook functions (anomaly-injection baseline, clean-reference
    IsolationForest scaling, run_fold's unknown-model_type error) against synthetic data.
    Wired into pytest so a regression is caught by the standard `pytest tests/` command
    instead of requiring someone to remember to run the script by hand.
    """
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_notebook_fixes.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "kaggle/notebooks/*.ipynb notebook-fix smoke test failed.\n\n"
        f"{result.stdout}\n{result.stderr}"
    )
