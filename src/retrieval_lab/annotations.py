"""Extractive annotation utilities using Python character offsets."""
import re


def validate_span(text, answer, start, end):
    if not isinstance(text, str) or not isinstance(answer, str) or not answer:
        raise ValueError("text and nonempty answer strings required")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("invalid answer span bounds")
    if text[start:end] != answer:
        raise ValueError("answer span does not match source text")
    return start, end


def answer_spans(text, answer):
    if not isinstance(answer, str) or not answer:
        raise ValueError("nonempty answer required")
    return [(match.start(), match.start() + len(answer))
            for match in re.finditer(f"(?={re.escape(answer)})", text)]


def audit_truncation(tokenizer, texts, max_tokens):
    if type(max_tokens) is not int or max_tokens < 2:
        raise ValueError("token budget must be at least two")
    results = []
    for identifier, text in texts.items():
        if not isinstance(text, str):
            raise ValueError("truncation audit expects string texts")
        full = tokenizer(text, truncation=False)
        retained = tokenizer(text, truncation=True, max_length=max_tokens, return_offsets_mapping=True)
        results.append({"id": identifier, "token_count": len(full["input_ids"]),
                        "truncated": len(full["input_ids"]) > max_tokens,
                        "retained_char_end": max((end for _, end in retained["offset_mapping"]), default=0)})
    return results


from dataclasses import dataclass
from .core import Document, canonical_hash


@dataclass(frozen=True)
class DocumentChunk:
    document: Document
    parent_id: str
    start: int
    end: int


def chunk_document(document, *, max_chars=512, overlap=64):
    if (type(max_chars) is not int or max_chars < 1 or type(overlap) is not int
            or not 0 <= overlap < max_chars):
        raise ValueError("invalid character chunk size or overlap")
    result, start = [], 0
    prefix = canonical_hash({"id": document.doc_id, "text": document.text})[:24]
    while start < len(document.text):
        end = min(len(document.text), start + max_chars)
        identifier = f"chunk-{prefix}-{start}"
        result.append(DocumentChunk(Document(identifier, document.title, document.text[start:end]),
                                    document.doc_id, start, end))
        if end == len(document.text):
            break
        start = end - overlap
    return result
