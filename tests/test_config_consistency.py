import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config

# The Kaggle notebooks run in a separate environment and don't import configs/config.yaml
# directly - they hardcode these same values inline. This dict is a manual mirror of the
# notebook constants. If you change one of these in config.yaml, update the matching
# notebook cell AND this dict in the same commit, or the dashboard/grounding-check
# tolerances will silently disagree with the parameters that actually produced the results
# being displayed.
#
# (section, key) in config.yaml -> value hardcoded in the notebooks
EXPECTED_NOTEBOOK_PARAMS = {
    ("forecasting", "horizon_days"): 15,               # HORIZON in notebooks 03, 04
    ("forecasting", "n_folds"): 4,                      # N_FOLDS in notebooks 03, 04
    ("selection", "late_opening_threshold_days"): 400,  # LATE_OPENING_THRESHOLD_DAYS in 01
    ("selection", "sustained_activation", "window_days"): 30,       # window= default, first_sustained_activation, notebook 01
    ("selection", "sustained_activation", "min_active_days"): 20,   # min_active_days= default, first_sustained_activation, notebook 01
    ("anomaly_detection", "control_limit_k"): 2.5,      # k= in control_limit_flags, notebook 05
    ("anomaly_detection", "isolation_forest_contamination"): 0.05,  # contamination=, notebook 05
    ("anomaly_detection", "synthetic_injection", "n_anomalies"): 50,  # notebook 05
    # FAMILY_COST_PER_UNIT_ERROR in notebook 04 - was previously untracked here, so this
    # dict could drift from cost_per_unit_error in config.yaml without any test catching it.
    ("cost_per_unit_error",): {
        "GROCERY I": 0.75, "BEVERAGES": 0.75, "HOME CARE": 0.75,
        "PRODUCE": 0.50, "DAIRY": 0.50, "CLEANING": 0.75,
    },
}


def _get_nested(config, path):
    value = config
    for key in path:
        value = value[key]
    return value


def test_config_matches_notebook_constants():
    config = load_config()
    mismatches = []
    for path, expected in EXPECTED_NOTEBOOK_PARAMS.items():
        actual = _get_nested(config, path)
        if actual != expected:
            mismatches.append((path, actual, expected))
    assert not mismatches, (
        f"configs/config.yaml has drifted from the hardcoded notebook constants: {mismatches}. "
        "Update the relevant Kaggle notebook cell (and this test) if the change was intentional."
    )


def test_cost_of_error_file_key_present():
    # cost_of_error.json is written by notebook 04 and read by build_facts(); if this key
    # goes missing from config.yaml, build_facts() silently falls back to a hardcoded
    # default filename instead of failing loudly - this test makes the wiring explicit.
    config = load_config()
    assert "cost_of_error" in config["data"]["files"]
