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
