"""Validated model execution configuration and input boundaries."""


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
