"""Validated model execution configuration and input boundaries."""
from dataclasses import asdict, dataclass
from .core import canonical_hash


def validate_encoder_texts(texts):
    if isinstance(texts, str) or texts is None:
        raise ValueError("encoder input must be an iterable of strings")
    try:
        texts = list(texts)
    except TypeError as exc:
        raise ValueError("encoder input must be iterable") from exc
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("encoder texts must be nonempty strings")
    return texts




@dataclass(frozen=True)
class EncoderConfig:
    max_tokens: int = 128
    batch_size: int = 16
    pooling: str = "mean"

    def __post_init__(self):
        if (type(self.max_tokens) is not int or not 2 <= self.max_tokens <= 4096
                or type(self.batch_size) is not int or not 1 <= self.batch_size <= 1024
                or self.pooling not in {"mean", "cls"}):
            raise ValueError("invalid encoder configuration")

    @property
    def fingerprint(self):
        return canonical_hash(asdict(self))


def validate_embeddings(embeddings, count, dimension):
    import numpy as np
    if (not isinstance(embeddings, np.ndarray) or embeddings.shape != (count, dimension)
            or not np.issubdtype(embeddings.dtype, np.floating)
            or not np.isfinite(embeddings).all()):
        raise ValueError("index mismatch: embedding shape, dtype or finite values")
    if count and not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
        raise ValueError("index mismatch: embeddings must have unit norm")
    result = np.array(embeddings, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result
