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
