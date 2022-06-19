"""Explicit text-processing policies with reproducible configuration."""
from dataclasses import asdict, dataclass
import re
import unicodedata
from .core import canonical_hash


@dataclass(frozen=True)
class RetrievalTokenizer:
    min_length: int = 1
    unicode_form: str = "NFKC"

    def __post_init__(self):
        if type(self.min_length) is not int or self.min_length < 1 or self.unicode_form not in {"NFC", "NFKC"}:
            raise ValueError("invalid retrieval tokenizer configuration")

    def __call__(self, text):
        if not isinstance(text, str) or any(0xD800 <= ord(char) <= 0xDFFF for char in text):
            raise ValueError("tokenizer expects Unicode text without surrogates")
        normalized = unicodedata.normalize(self.unicode_form, text).casefold()
        return [token for token in re.findall(r"\b\w+\b", normalized) if len(token) >= self.min_length]

    @property
    def fingerprint(self):
        return canonical_hash(asdict(self))
