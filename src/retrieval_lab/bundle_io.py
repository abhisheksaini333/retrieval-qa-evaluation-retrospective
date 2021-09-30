"""Bundle validation without loading transformer weights."""
from contextlib import contextmanager
import shutil
import tempfile
import hashlib
import json
import math
import re
from pathlib import Path
from .core import canonical_hash


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
    validate_tree(path)
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


def model_inventory(path):
    root = Path(path)
    validate_tree(root)
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts)
    required = {"config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json"}
    if not required.issubset(files):
        raise ValueError(f"missing required model files: {sorted(required - set(files))}")
    for name in files:
        p = Path(name)
        if p.suffix not in {".json", ".txt", ".safetensors"} and p.name not in {"README.md", "LICENSE", "LICENSE.md"}:
            raise ValueError(f"unsupported model file: {name}")
    return files


@contextmanager
def atomic_directory(path):
    target = Path(path).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name + ".lock")
    lock.mkdir()
    staging = None
    try:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output already exists: {target}")
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        yield staging
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output appeared during save: {target}")
        staging.rename(target)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.rmdir()


def validate_output_location(output, sources):
    target = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if target == source or target in source.parents or source in target.parents:
            raise ValueError("output path overlaps an input path")
    return target


def validate_tree(path):
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("artifact root must be a directory, not a symlink")
    resolved = root.resolve()
    for item in root.rglob("*"):
        if ".cache" in item.relative_to(root).parts:
            continue
        if item.is_symlink() or resolved not in item.resolve().parents:
            raise ValueError(f"artifact symlink or containment violation: {item.name}")
        if not item.is_file() and not item.is_dir():
            raise ValueError("unsupported special artifact file")


def copy_verified_model(source, destination, expected_fingerprint):
    validate_tree(source)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".cache"))
    if directory_fingerprint(destination) != expected_fingerprint:
        raise ValueError("model source changed since it was loaded")


def inspect_embeddings(path, count, dimension=None, max_bytes=512 * 1024 * 1024):
    import numpy as np
    with Path(path).open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("unsupported numpy array version")
        header_size = stream.tell()
    if len(shape) != 2 or shape[0] != count or not 1 <= shape[1] <= 8192 or (dimension is not None and shape[1] != dimension):
        raise ValueError("index mismatch: embedding shape")
    if dtype not in (np.dtype("float32"), np.dtype("float64")) or fortran:
        raise ValueError("unsupported embedding dtype or ordering")
    size = math.prod(shape) * dtype.itemsize
    if size > max_bytes:
        raise ValueError("embedding allocation exceeds limit")
    if Path(path).stat().st_size != header_size + size:
        raise ValueError("embedding payload size mismatch")
    return {"shape": list(shape), "dtype": str(dtype), "bytes": size}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_files(path):
    path = Path(path)
    return {str(p.relative_to(path)): sha256_file(p) for p in sorted(path.rglob("*"))
            if p.is_file() and ".cache" not in p.parts}


def directory_fingerprint(path):
    files = directory_files(path)
    if not files:
        raise ValueError(f"model directory has no files: {path}")
    return canonical_hash(files)


def inspect_bundle(path):
    from .core import load_corpus, validate_index
    path = Path(path)
    manifest = read_manifest(path)
    encoder_configuration(manifest)
    documents = load_corpus(path / "corpus.jsonl")
    for role in ("encoder", "reader"):
        model_inventory(path / role)
        if directory_fingerprint(path / role) != manifest[f"{role}_fingerprint"]:
            raise ValueError(f"{role} model fingerprint mismatch")
    validate_index(manifest, documents, manifest["encoder_fingerprint"])
    if sha256_file(path / "embeddings.npy") != manifest["embeddings_sha256"]:
        raise ValueError("index mismatch: embedding file hash")
    header = inspect_embeddings(path / "embeddings.npy", len(documents))
    return {"valid": True, "manifest": manifest, "embeddings": header}
