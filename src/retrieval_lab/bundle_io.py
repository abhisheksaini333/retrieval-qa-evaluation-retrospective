"""Bundle validation without loading transformer weights."""
import json
import math
import re
from pathlib import Path


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] not in (1, 2):
        raise ValueError("unsupported bundle schema")
    for field in ("corpus_sha256", "encoder_fingerprint", "reader_fingerprint", "embeddings_sha256"):
        if not isinstance(manifest.get(field), str) or not re.fullmatch(r"[a-f0-9]{64}", manifest[field]):
            raise ValueError(f"invalid manifest {field}")
    ids = manifest.get("doc_ids")
    if not isinstance(ids, list) or any(not isinstance(i, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", i) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid manifest document IDs")
    threshold = manifest.get("threshold")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1.0000001:
        raise ValueError("invalid manifest threshold")
    for field, maximum in [("reader_k", 100), ("encoder_max_tokens", 4096), ("reader_max_tokens", 4096)]:
        if type(manifest.get(field)) is not int or not 1 <= manifest[field] <= maximum:
            raise ValueError(f"invalid manifest {field}")
    for field in ("encoder_id", "reader_id", "encoder_revision", "reader_revision"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"invalid manifest {field}")
    return manifest


def read_manifest(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate manifest key: {key}")
            result[key] = value
        return result
    source = Path(path) / "manifest.json"
    if source.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("manifest exceeds size limit")
    return validate_manifest(json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique))


def encoder_configuration(manifest):
    from .config import EncoderConfig
    if manifest["reader_max_tokens"] != 384:
        raise ValueError("unsupported reader token configuration")
    if manifest["schema_version"] == 1:
        if manifest["encoder_max_tokens"] != 128:
            raise ValueError("unsupported legacy encoder token configuration")
        return EncoderConfig()
    values = manifest.get("encoder_config")
    if not isinstance(values, dict) or set(values) != {"max_tokens", "batch_size", "pooling"}:
        raise ValueError("invalid encoder configuration")
    config = EncoderConfig(**values)
    if config.max_tokens != manifest["encoder_max_tokens"]:
        raise ValueError("encoder token configuration mismatch")
    return config
