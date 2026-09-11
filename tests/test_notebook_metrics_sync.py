import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_metrics_match_src():
    """kaggle/notebooks/03 and 04 embed their own copy of mape/wape/mase (they run in an
    isolated Kaggle environment and can't import src.utils.metrics). This runs the same
    check scripts/sync_notebook_metrics.py uses to catch drift, so a change to the metric
    functions that only gets made in one place fails here instead of shipping silently.
    """
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_notebook_metrics.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "kaggle notebook copies of mape/wape/mase have drifted from src/utils/metrics.py. "
        "Run `python scripts/sync_notebook_metrics.py` to fix, then commit the notebook "
        f"changes.\n\n{result.stdout}\n{result.stderr}"
    )
