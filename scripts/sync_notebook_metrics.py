"""
Keeps the mape/wape/mase copies inside the Kaggle notebooks byte-identical to
src/utils/metrics.py, which is the canonical source.

Why this exists: kaggle/notebooks/03_statistical_models.ipynb and
kaggle/notebooks/04_ml_models.ipynb run in an isolated Kaggle environment and can't
`import src.utils.metrics`, so they each carry their own copy of these three functions,
one per notebook cell. Copy-pasted code drifts silently - this script is what prevents
that, by treating src/utils/metrics.py as ground truth and mechanically overwriting the
matching notebook cell's source with it, delimited by "# BEGIN SYNCED METRICS: <name>" /
"# END SYNCED METRICS: <name>" markers that exist in all three files (each pair lives
entirely inside one cell in the notebooks).

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

from _notebook_utils import cell_source, read_notebook, set_cell_source, write_notebook

ROOT = Path(__file__).resolve().parents[1]
SOURCE_OF_TRUTH = ROOT / "src" / "utils" / "metrics.py"
NOTEBOOK_TARGETS = [
    ROOT / "kaggle" / "notebooks" / "03_statistical_models.ipynb",
    ROOT / "kaggle" / "notebooks" / "04_ml_models.ipynb",
]

MARKER_RE = re.compile(
    r"# BEGIN SYNCED METRICS: (\w+)\n.*?# END SYNCED METRICS: \1\n?",
    re.DOTALL,
)


def extract_blocks(text):
    """Returns {name: full_block_text_including_markers} found in a text blob."""
    return {m.group(1): m.group(0) for m in MARKER_RE.finditer(text)}


def sync_notebook(nb_path, canonical_blocks, check_only):
    nb = read_notebook(nb_path)
    changed_names = []
    diffs = []

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        original_src = cell_source(cell)
        cell_blocks = extract_blocks(original_src)
        if not cell_blocks:
            continue

        updated_src = original_src
        for name, cell_block in cell_blocks.items():
            if name not in canonical_blocks:
                continue
            canonical_block = canonical_blocks[name]
            if canonical_block.rstrip("\n") != cell_block.rstrip("\n"):
                changed_names.append(name)
                updated_src = updated_src.replace(cell_block, canonical_block)

        if updated_src != original_src:
            diffs.append("\n".join(difflib.unified_diff(
                original_src.splitlines(), updated_src.splitlines(),
                fromfile=f"{nb_path}::cell", tofile=f"{nb_path}::cell (synced)", lineterm="",
            )))
            if not check_only:
                set_cell_source(cell, updated_src)

    all_canonical_names = set(canonical_blocks)
    found_names = set()
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            found_names |= set(extract_blocks(cell_source(cell)))
    missing = all_canonical_names - found_names
    if missing:
        raise SystemExit(
            f"{nb_path}: missing SYNCED METRICS marker(s) for {sorted(missing)}. "
            "Add matching '# BEGIN/END SYNCED METRICS: <name>' markers before syncing."
        )

    if changed_names and check_only:
        print(f"DRIFT in {nb_path} for: {', '.join(sorted(set(changed_names)))}\n" + "\n".join(diffs) + "\n")
    elif changed_names:
        write_notebook(nb_path, nb)
        print(f"Synced {nb_path}: {', '.join(sorted(set(changed_names)))}")

    return bool(changed_names)


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
        if sync_notebook(target, canonical_blocks, check_only=args.check):
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
