import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.grounding_check import check_grounding, extract_claims


def test_extract_percentage():
    claims = extract_claims("The model achieved a MAPE of 15.3%.")
    assert any(c["kind"] == "percent" and abs(c["value"] - 15.3) < 1e-6 for c in claims)


def test_extract_currency():
    claims = extract_claims("Estimated cost: $206,144.55.")
    assert any(c["kind"] == "currency" and abs(c["value"] - 206144.55) < 1e-2 for c in claims)


def test_grounded_number_matches_within_tolerance():
    result = check_grounding("The holdout MASE was 0.63.", {"mase": 0.632})
    assert result["grounded_ratio"] == 1.0


def test_ungrounded_number_is_flagged():
    result = check_grounding("The holdout MASE was 4.2.", {"mase": 0.632})
    flagged = [c for c in result["claims"] if not c["grounded"]]
    assert len(flagged) == 1


def test_no_numbers_returns_full_ratio():
    result = check_grounding("This report has no numbers in it.", {"mase": 0.5})
    assert result["grounded_ratio"] == 1.0


def test_small_scale_hallucination_is_flagged():
    # Regression test: absolute_tolerance_default used to be 0.5, which is huge relative
    # to metrics like MASE/precision/recall that live in a 0-1 range. A claim this far off
    # (0.15 vs a true 0.632, a 76% error) was previously marked "grounded" purely because
    # the absolute difference (0.482) was under the old 0.5 tolerance.
    result = check_grounding(
        "The MASE was actually 0.15 this time, way better than reported.",
        {"best_ml_mase_holdout": 0.632},
    )
    flagged = [c for c in result["claims"] if not c["grounded"]]
    assert len(flagged) == 1


def test_small_scale_rounding_still_grounds():
    # Make sure the tightened tolerance didn't overcorrect: reasonable rounding on a
    # 0-1 scale fact should still pass.
    result = check_grounding("Recall came in around 0.86.", {"control_limit_recall": 0.858})
    assert result["grounded_ratio"] == 1.0