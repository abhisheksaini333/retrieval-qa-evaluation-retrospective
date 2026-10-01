"""Bounded JSONL request validation and atomic batch prediction output."""
import json
from pathlib import Path
import tempfile
from .core import _jsonl, validate_identifier, validate_request


def validate_requests(rows):
    rows, seen = list(rows), set()
    if not rows or len(rows) > 10000:
        raise ValueError("prediction batch requires 1-10000 requests")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"query_id", "question"}:
            raise ValueError("prediction records require query_id and question")
        validate_identifier(row["query_id"])
        validate_request(row["question"], 1)
        if row["query_id"] in seen:
            raise ValueError("duplicate prediction query ID")
        seen.add(row["query_id"])
    return rows


def load_prediction_requests(path):
    return validate_requests(_jsonl(path, max_records=10000))


def batch_predict(bundle, requests, output, *, continue_on_error=False):
    if type(continue_on_error) is not bool:
        raise ValueError("continue_on_error must be a boolean")
    rows = validate_requests(requests)
    success_count = error_count = 0
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("prediction output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    with lock.open("x"):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
                temporary = Path(stream.name)
                for row in rows:
                    try:
                        result = {"query_id": row["query_id"], "prediction": bundle.predict(row["question"])}
                        success_count += 1
                    except (ValueError, RuntimeError) as error:
                        if not continue_on_error:
                            raise
                        result = {"query_id": row["query_id"], "error": {"type": type(error).__name__, "message": str(error)}}
                        error_count += 1
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
            if output.exists() or output.is_symlink():
                raise FileExistsError("prediction output appeared during evaluation")
            temporary.replace(output)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            lock.unlink()
    return {"request_count": len(rows), "success_count": success_count, "error_count": error_count, "output": str(output)}
