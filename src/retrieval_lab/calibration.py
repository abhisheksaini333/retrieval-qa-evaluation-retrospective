"""Development-only decision calibration and inspectable frozen artifacts."""
import re
from .core import _validate_rows, canonical_hash, calibrate_threshold, answer_scores, has_answer


def development_fingerprint(rows):
    _validate_rows(rows)
    fields = ["query_id", "question", "answers", "relevant_ids", "raw_answer", "confidence"]
    return canonical_hash([{key: row.get(key) for key in fields}
                           for row in sorted(rows, key=lambda row: row["query_id"])])


def bind_calibration(rows, dataset_fingerprint):
    if not isinstance(dataset_fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", dataset_fingerprint):
        raise ValueError("dataset fingerprint required")
    return {"schema_version": 1, "dataset_fingerprint": dataset_fingerprint,
            "development_fingerprint": development_fingerprint(rows),
            "calibration": calibrate_threshold(rows, split="dev")}


def verify_binding(artifact, rows, dataset_fingerprint):
    validate_calibration_artifact(artifact)
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


def save_calibration(path, artifact):
    import json
    from pathlib import Path
    path = Path(path)
    validate_calibration_artifact(artifact)
    payload = {"artifact": artifact, "sha256": canonical_hash(artifact)}
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def validate_calibration_artifact(artifact):
    import math
    if not isinstance(artifact, dict) or type(artifact.get("schema_version")) is not int or artifact["schema_version"] != 1:
        raise ValueError("unsupported calibration artifact schema")
    calibration = artifact.get("calibration", {})
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be an object")
    threshold = calibration.get("threshold")
    if (calibration.get("split") != "dev" or type(threshold) not in (int, float)
            or not math.isfinite(threshold) or not 0 <= threshold <= 1.0000001):
        raise ValueError("invalid development threshold artifact")
    for field in ["dataset_fingerprint", "development_fingerprint"]:
        if not isinstance(artifact.get(field), str) or not re.fullmatch(r"[a-f0-9]{64}", artifact[field]):
            raise ValueError("calibration fingerprint required")


def load_calibration(path):
    import json
    from pathlib import Path
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("sha256") != canonical_hash(payload.get("artifact")):
        raise ValueError("calibration checksum mismatch")
    validate_calibration_artifact(payload["artifact"])
    return payload["artifact"]


def cross_validate_calibration(rows, groups, *, folds=3, seed=0):
    import random
    from .core import score_predictions
    _validate_rows(rows)
    if set(groups) != {r["query_id"] for r in rows} or any(not isinstance(g, str) for g in groups.values()):
        raise ValueError("calibration groups must cover every development query")
    keys = sorted(set(groups.values()))
    if type(folds) is not int or not 2 <= folds <= len(keys) or type(seed) is not int:
        raise ValueError("invalid grouped fold configuration")
    random.Random(seed).shuffle(keys)
    results = []
    for fold in range(folds):
        held = set(keys[fold::folds])
        train = [r for r in rows if groups[r["query_id"]] not in held]
        validation = [r for r in rows if groups[r["query_id"]] in held]
        calibration = calibrate_threshold(train, split="dev")
        scored = score_predictions(validation, calibration["threshold"])[0]
        results.append({"fold": fold, "train_groups": sorted(set(keys) - held),
                        "validation_groups": sorted(held), "threshold": calibration["threshold"],
                        "metrics": scored})
    return results


def reliability_bins(rows, *, bins=10):
    _validate_rows(rows)
    if type(bins) is not int or not 1 <= bins <= 100:
        raise ValueError("reliability bin count must be in [1,100]")
    groups = [[] for _ in range(bins)]
    for row in rows:
        groups[min(bins - 1, int(row["confidence"] * bins))].append(row)
    result, ece = [], 0.0
    for index, group in enumerate(groups):
        score = sum(r["confidence"] for r in group) / len(group) if group else None
        accuracy = (sum(answer_scores(r["raw_answer"], r["answers"])[0] for r in group) / len(group)
                    if group else None)
        if group:
            ece += len(group) / len(rows) * abs(score - accuracy)
        result.append({"lower": index / bins, "upper": (index + 1) / bins, "count": len(group),
                       "mean_score": score, "accuracy": accuracy})
    return {"bins": result, "ece": ece, "target": "raw-answer exact match"}
