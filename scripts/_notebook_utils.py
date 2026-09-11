"""Shared helpers for treating a .ipynb file's code cells as script text, used by
sync_notebook_metrics.py and smoke_test_notebook_fixes.py so neither has to re-implement
the same JSON <-> source plumbing."""
import json
from pathlib import Path


def read_notebook(path):
    return json.loads(Path(path).read_text())


def write_notebook(path, nb):
    Path(path).write_text(json.dumps(nb, indent=1) + "\n")


def cell_source(cell):
    return "".join(cell["source"])


def concatenated_source(nb):
    """All code cells joined in order, for tools (ast, regex) that want one script-like
    blob rather than cell-by-cell structure. A blank line between cells is enough to
    keep top-level statements from different cells from being mistaken for one."""
    return "\n\n".join(cell_source(c) for c in nb["cells"] if c["cell_type"] == "code")


def set_cell_source(cell, text):
    """text has no required trailing newline; mirrors how the converter originally
    built each cell's source list (each line but the last carries its own '\\n')."""
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
