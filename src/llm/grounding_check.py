import re

from src.utils.config import CONFIG

NUMBER_PATTERN = re.compile(r"(?<![\w.])(\$?-?\d[\d,]*\.?\d*%?)(?![\w])")


def _normalize(token):
    kind = "number"
    t = token.strip()
    if t.startswith("$"):
        kind = "currency"
        t = t[1:]
    if t.endswith("%"):
        kind = "percent"
        t = t[:-1]
    t = t.replace(",", "")
    try:
        value = float(t)
    except ValueError:
        return None
    return value, kind


def extract_claims(text):
    claims = []
    for match in NUMBER_PATTERN.finditer(text):
        parsed = _normalize(match.group(1))
        if parsed is None:
            continue
        value, kind = parsed
        claims.append({"raw": match.group(1), "value": value, "kind": kind, "span": match.span()})
    return claims


def _within_tolerance(claim_value, fact_value, relative_tolerance, absolute_tolerance):
    if fact_value == 0:
        return abs(claim_value) <= absolute_tolerance
    rel_ok = abs(claim_value - fact_value) / abs(fact_value) <= relative_tolerance
    abs_ok = abs(claim_value - fact_value) <= absolute_tolerance
    return rel_ok or abs_ok


def check_grounding(text, facts, config=CONFIG):
    """
    Flags numbers in `text` that don't match any value in `facts` within tolerance.

    This is a first-pass safety net, not full claim verification:
    - It can't catch paraphrased claims with no literal number ("about half the error
      of a naive baseline") - those pass silently.
    - It can flag numbers that are correct but simply aren't in `facts` (dates, ranks,
      story framing like "the top 3 features").
    - Percent/currency signs anchor the comparison type but bare decimals (a MASE like
      0.63) are matched against everything in `facts`, since the LLM may drop context.
    - `absolute_tolerance_default` is an absolute error budget, not a percentage - it must
      stay small relative to the 0-1 scale most bare-number facts (MASE, precision, recall)
      live on. Larger numbers are still protected by `relative_tolerance`.

    Treat a low grounded_ratio as "needs human review," not "definitely wrong," and
    treat a high ratio as reassurance, not proof - it does not verify report structure,
    tone, or claims without numbers.
    """
    gc_cfg = config["grounding_check"]
    rel_tol = gc_cfg["relative_tolerance"]
    currency_tol = gc_cfg["absolute_tolerance_currency"]
    default_tol = gc_cfg["absolute_tolerance_default"]

    fact_values = [v for v in facts.values() if isinstance(v, (int, float))]
    claims = extract_claims(text)

    results = []
    for claim in claims:
        tol = currency_tol if claim["kind"] == "currency" else default_tol
        matched = any(_within_tolerance(claim["value"], fv, rel_tol, tol) for fv in fact_values)
        results.append({**claim, "grounded": matched})

    grounded_count = sum(1 for r in results if r["grounded"])
    total = len(results)
    return {
        "claims": results,
        "grounded_count": grounded_count,
        "total_claims": total,
        "grounded_ratio": grounded_count / total if total else 1.0,
    }