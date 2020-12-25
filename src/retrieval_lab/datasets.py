"""Dataset manifests, audit policies, and reproducible split construction."""
import json
from pathlib import Path


def validate_provenance(value, counts):
    if not isinstance(value, dict):
        raise ValueError("provenance must be an object")
    for key in ["kind", "license", "permission"]:
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"provenance {key} is required")
    if value.get("counts") != counts:
        raise ValueError("provenance counts do not match dataset")


def dataset_manifest(path):
    import hashlib
    from .core import canonical_hash
    files = {name: hashlib.sha256((Path(path) / name).read_bytes()).hexdigest()
             for name in ["corpus.jsonl", "dev.jsonl", "test.jsonl", "provenance.json"]}
    return {"schema_version": 1, "files": files, "dataset_sha256": canonical_hash(files)}


def verify_dataset_manifest(path, manifest):
    if dataset_manifest(path) != manifest:
        raise ValueError("dataset manifest mismatch")


def validate_split_policy(dev, test, documents, policy="document"):
    from .core import validate_splits, validate_evidence_isolation
    if policy not in {"document", "query"}:
        raise ValueError("split policy must be document or query")
    validate_splits(dev, test)
    if policy == "document":
        if {d for q in dev for d in q.relevant_ids} & {d for q in test for d in q.relevant_ids}:
            raise ValueError("gold document overlap across splits")
        validate_evidence_isolation(dev, test, documents)


def grouped_split(items, group_key, test_fraction=0.25, *, seed=0):
    import random
    if isinstance(test_fraction, bool) or not 0 < test_fraction < 1 or type(seed) is not int:
        raise ValueError("test fraction must be between zero and one and seed an integer")
    groups = {}
    for item in items:
        groups.setdefault(str(group_key(item)), []).append(item)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("at least two groups required")
    random.Random(seed).shuffle(keys)
    count = max(1, min(len(keys) - 1, round(len(keys) * test_fraction)))
    held_out = set(keys[:count])
    return ([item for key in sorted(groups) if key not in held_out for item in groups[key]],
            [item for key in sorted(groups) if key in held_out for item in groups[key]])
