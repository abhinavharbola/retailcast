"""
Keeps the mape/wape/mase copies inside the Kaggle notebooks byte-identical to
src/utils/metrics.py, which is the canonical source.

Why this exists: kaggle/notebooks/03_statistical_models.ipynb and
kaggle/notebooks/04_ml_models.ipynb run in an isolated Kaggle environment and can't
`import src.utils.metrics`, so they each carry their own copy of these three functions.
Copy-pasted code drifts silently - this script is what prevents that, by treating
src/utils/metrics.py as ground truth and mechanically overwriting the notebook copies
with it, delimited by "# BEGIN SYNCED METRICS: <name>" / "# END SYNCED METRICS: <name>"
markers that exist in all three files.

Usage:
    python scripts/sync_notebook_metrics.py            # rewrite the notebooks in place
    python scripts/sync_notebook_metrics.py --check     # exit 1 if anything would change,
                                                         # print a diff, change nothing

tests/test_notebook_metrics_sync.py runs this in --check mode, so drift fails the test
suite instead of silently shipping.
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_OF_TRUTH = ROOT / "src" / "utils" / "metrics.py"
NOTEBOOK_TARGETS = [
    ROOT / "kaggle" / "notebooks" / "03_statistical_models.ipynb",
    ROOT / "kaggle" / "notebooks" / "04_ml_models.ipynb",
]

MARKER_RE = re.compile(
    r"# BEGIN SYNCED METRICS: (\w+)\n.*?# END SYNCED METRICS: \1\n",
    re.DOTALL,
)


def extract_blocks(text):
    """Returns {name: full_block_text_including_markers} for a file's content."""
    return {m.group(1): m.group(0) for m in MARKER_RE.finditer(text)}


def sync_file(target_path, canonical_blocks, check_only):
    original = target_path.read_text()
    target_blocks = extract_blocks(original)

    missing = set(canonical_blocks) - set(target_blocks)
    if missing:
        raise SystemExit(
            f"{target_path}: missing SYNCED METRICS marker(s) for {sorted(missing)}. "
            "Add matching '# BEGIN/END SYNCED METRICS: <name>' markers before syncing."
        )

    updated = original
    changed_names = []
    for name, target_block in target_blocks.items():
        canonical_block = canonical_blocks[name]
        if canonical_block != target_block:
            changed_names.append(name)
            updated = updated.replace(target_block, canonical_block)

    if not changed_names:
        return False

    if check_only:
        diff = "\n".join(difflib.unified_diff(
            original.splitlines(), updated.splitlines(),
            fromfile=str(target_path), tofile=str(target_path) + " (synced)",
            lineterm="",
        ))
        print(f"DRIFT in {target_path} for: {', '.join(changed_names)}\n{diff}\n")
    else:
        target_path.write_text(updated)
        print(f"Synced {target_path}: {', '.join(changed_names)}")

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                         help="Report drift and exit 1 instead of writing changes.")
    args = parser.parse_args()

    canonical_blocks = extract_blocks(SOURCE_OF_TRUTH.read_text())
    if not canonical_blocks:
        raise SystemExit(f"No '# BEGIN/END SYNCED METRICS' markers found in {SOURCE_OF_TRUTH}.")

    any_drift = False
    for target in NOTEBOOK_TARGETS:
        if sync_file(target, canonical_blocks, check_only=args.check):
            any_drift = True

    if args.check:
        if any_drift:
            print("Notebook metrics are out of sync with src/utils/metrics.py. "
                  "Run 'python scripts/sync_notebook_metrics.py' to fix.")
            sys.exit(1)
        print("Notebook metrics match src/utils/metrics.py.")
    elif not any_drift:
        print("Already in sync, nothing to do.")


if __name__ == "__main__":
    main()
