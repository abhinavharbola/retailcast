"""
Smoke test for the fixes inside kaggle/notebooks/*.ipynb.

pytest can't reach these directly - the notebooks import Kaggle-only packages
(prophet, lightgbm, xgboost, mlflow, kaggle_secrets) and need the real Favorita
dataset to run end-to-end. This script pulls the *actual current* source of the
specific functions that were fixed straight out of the notebook JSON via `ast`
(not a hand-copied duplicate - if the notebook changes, this exercises whatever
is really there) and runs them against small synthetic data to check:

  - 05_anomaly_detection.ipynb / inject_synthetic_anomalies:
    the injected spike/drop baseline uses the series' OWN mean (store_nbr+family),
    not the global mean across all 60 series.
  - 05_anomaly_detection.ipynb / build_scaled_iso_features_from_clean_reference:
    IsolationForest eval features are scaled using clean (pre-injection) group
    stats, not eval_df's own contaminated stats.
  - 04_ml_models.ipynb / run_fold:
    raises a clear error on an unknown model_type instead of leaving `pred` unbound.

Usage: python scripts/smoke_test_notebook_fixes.py
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _notebook_utils import concatenated_source, read_notebook

ROOT = Path(__file__).resolve().parents[1]


def extract_functions(notebook_path, names):
    """Pulls specific top-level function defs out of a notebook's concatenated cell
    source via ast, so this test runs the actual current source, not a hand-copied
    duplicate."""
    nb = read_notebook(notebook_path)
    source = concatenated_source(nb)
    source = "\n".join(l for l in source.split("\n") if not l.strip().startswith("!"))
    tree = ast.parse(source)
    wanted = set(names)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            found[node.name] = ast.get_source_segment(source, node)
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"{notebook_path}: couldn't find function(s) {missing}. "
                          "Has the notebook been restructured?")
    return found


def check(label, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return condition


def check_injection_baseline():
    nb05 = ROOT / "kaggle" / "notebooks" / "05_anomaly_detection.ipynb"
    funcs = extract_functions(nb05, ["inject_synthetic_anomalies"])
    ns = {"np": np, "pd": pd}
    exec(funcs["inject_synthetic_anomalies"], ns)
    inject_synthetic_anomalies = ns["inject_synthetic_anomalies"]

    # A DAIRY row with sales=0 forces the baseline fallback. Mixed in with a much
    # larger-scale GROCERY I group so a global-mean fallback (the old bug) and a
    # per-series-mean fallback (the fix) produce very different magnitudes.
    df = pd.DataFrame([
        {"store_nbr": 1, "family": "DAIRY", "sales": 0, "forecast": 20},
        {"store_nbr": 1, "family": "DAIRY", "sales": 22, "forecast": 20},
        {"store_nbr": 1, "family": "DAIRY", "sales": 18, "forecast": 20},
        {"store_nbr": 1, "family": "GROCERY I", "sales": 500, "forecast": 500},
        {"store_nbr": 1, "family": "GROCERY I", "sales": 510, "forecast": 500},
        {"store_nbr": 1, "family": "GROCERY I", "sales": 490, "forecast": 500},
    ])
    # global mean across the mixed df is ~256.7; DAIRY's own mean is ~20.
    # Try seeds until the zero-sales DAIRY row (index 0) gets a "spike" injection -
    # spike is the direction that makes the two baselines diverge clearly (a "drop"
    # can look small under either baseline, so it wouldn't distinguish the bug).
    for seed in range(50):
        eval_df = inject_synthetic_anomalies(df, n_anomalies=len(df), seed=seed)
        row = eval_df.iloc[0]
        if row["is_synthetic_anomaly"] == 1 and row["sales_injected"] > row["sales"]:
            # a per-series (DAIRY ~20) baseline spikes to at most ~20*6=120;
            # a global-mean (~257) baseline would spike as high as ~257*6=1542.
            return check(
                f"injected DAIRY spike (seed={seed}) uses DAIRY's own scale, not the "
                f"60-series global mean (got sales_injected={row['sales_injected']:.1f}, "
                f"expected well under 200)",
                row["sales_injected"] < 200,
            )
    print("[SKIP] no seed in range produced a spike on the target row - widen the range")
    return True


def check_clean_reference_scaling():
    nb05 = ROOT / "kaggle" / "notebooks" / "05_anomaly_detection.ipynb"
    funcs = extract_functions(nb05, ["inject_synthetic_anomalies",
                                      "build_scaled_iso_features_from_clean_reference"])
    ns = {"np": np, "pd": pd}
    exec(funcs["inject_synthetic_anomalies"], ns)
    exec(funcs["build_scaled_iso_features_from_clean_reference"], ns)
    inject_synthetic_anomalies = ns["inject_synthetic_anomalies"]
    build_scaled = ns["build_scaled_iso_features_from_clean_reference"]

    rng = np.random.default_rng(0)
    rows = []
    for store in [1, 2]:
        for family in ["GROCERY I", "DAIRY"]:
            for _ in range(15):
                rows.append({
                    "store_nbr": store, "family": family,
                    "sales": max(0, rng.normal(100, 10)),
                    "forecast": max(0, rng.normal(100, 10)),
                })
    holdout_preds = pd.DataFrame(rows)
    holdout_preds["residual"] = holdout_preds["sales"] - holdout_preds["forecast"]
    eval_df = inject_synthetic_anomalies(holdout_preds, n_anomalies=10, seed=1)

    feats = build_scaled(
        eval_df, holdout_preds,
        value_cols={"sales_injected": "sales"},
    )
    expected_mean = holdout_preds.groupby(["store_nbr", "family"])["sales"].transform("mean")
    expected_std = holdout_preds.groupby(["store_nbr", "family"])["sales"].transform("std")
    expected = (eval_df["sales_injected"].values - expected_mean.values) / expected_std.values
    return check(
        "IsolationForest eval features are scaled from clean (pre-injection) group "
        "stats, matching a hand-computed reference",
        np.allclose(feats["sales_injected"].values, expected, equal_nan=True),
    )


def check_run_fold_raises():
    nb04 = ROOT / "kaggle" / "notebooks" / "04_ml_models.ipynb"
    funcs = extract_functions(nb04, ["run_fold"])
    ns = {"np": np, "pd": pd, "FEATURE_COLS": [], "TARGET": "sales", "CATEGORICAL_COLS": []}
    exec(funcs["run_fold"], ns)
    run_fold = ns["run_fold"]
    dummy = pd.DataFrame({"sales": [1, 2, 3]})
    try:
        run_fold("not_a_real_model", dummy, dummy)
        return check("run_fold raises on an unknown model_type", False)
    except ValueError:
        return check("run_fold raises ValueError on an unknown model_type", True)
    except Exception as e:
        return check(f"run_fold raised {type(e).__name__}, expected ValueError", False)


def main():
    results = [
        check_injection_baseline(),
        check_clean_reference_scaling(),
        check_run_fold_raises(),
    ]
    print()
    if all(results):
        print("All notebook smoke checks passed.")
    else:
        print("Some notebook smoke checks FAILED - see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
