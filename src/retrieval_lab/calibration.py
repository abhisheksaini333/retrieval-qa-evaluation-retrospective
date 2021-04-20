"""Development-only decision calibration and inspectable frozen artifacts."""
from .core import _validate_rows, canonical_hash, calibrate_threshold, answer_scores, has_answer


def development_fingerprint(rows):
    _validate_rows(rows)
    fields = ["query_id", "question", "answers", "relevant_ids", "raw_answer", "confidence"]
    return canonical_hash([{key: row.get(key) for key in fields}
                           for row in sorted(rows, key=lambda row: row["query_id"])])


def bind_calibration(rows, dataset_fingerprint):
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
        raise ValueError("dataset fingerprint required")
    return {"schema_version": 1, "dataset_fingerprint": dataset_fingerprint,
            "development_fingerprint": development_fingerprint(rows),
            "calibration": calibrate_threshold(rows, split="dev")}


def verify_binding(artifact, rows, dataset_fingerprint):
    if (artifact.get("schema_version") != 1 or artifact.get("dataset_fingerprint") != dataset_fingerprint
            or artifact.get("development_fingerprint") != development_fingerprint(rows)):
        raise ValueError("calibration development-data binding mismatch")


def calibrate_risk(rows, *, max_risk=0.1, min_coverage=0.0, split="dev"):
    _validate_rows(rows)
    if (split != "dev" or type(max_risk) not in (int, float) or not 0 <= max_risk <= 1
            or type(min_coverage) not in (int, float) or not 0 <= min_coverage <= 1):
        raise ValueError("invalid development risk calibration configuration")
    candidates = []
    for threshold in sorted({0.0, 1.0000001, *(r["confidence"] for r in rows)}):
        accepted = [r for r in rows if has_answer(r["raw_answer"]) and r["confidence"] >= threshold]
        coverage = len(accepted) / len(rows)
        risk = (sum(1 - answer_scores(r["raw_answer"], r["answers"])[0] for r in accepted) / len(accepted)
                if accepted else None)
        if (risk is None or risk <= max_risk) and coverage >= min_coverage:
            candidates.append({"threshold": threshold, "risk": risk, "coverage": coverage})
    if not candidates:
        raise ValueError("no development threshold meets risk and coverage constraints")
    return {**max(candidates, key=lambda row: (row["coverage"], row["threshold"])),
            "objective": "selective_risk", "split": "dev", "max_risk": max_risk}


def threshold_diagnostics(rows):
    _validate_rows(rows)
    curve = []
    for threshold in sorted({0.0, 1.0000001, *(r["confidence"] for r in rows)}):
        counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for row in rows:
            accepted = has_answer(row["raw_answer"]) and row["confidence"] >= threshold
            counts[("tp" if accepted else "fn") if row["answers"] else ("fp" if accepted else "tn")] += 1
        positive, negative = counts["tp"] + counts["fn"], counts["tn"] + counts["fp"]
        balanced = (counts["tp"] / positive + counts["tn"] / negative) / 2 if positive and negative else None
        curve.append({"threshold": threshold, **counts, "accepted": counts["tp"] + counts["fp"],
                      "balanced_accuracy": balanced})
    return curve
