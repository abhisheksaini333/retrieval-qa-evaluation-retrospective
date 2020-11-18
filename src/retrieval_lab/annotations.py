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
