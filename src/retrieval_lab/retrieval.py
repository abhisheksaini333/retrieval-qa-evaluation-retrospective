"""Composable retrieval operations independent of model inference."""
import heapq


def stable_top_k(ranked, k):
    from .core import validate_request
    validate_request("top-k", k)
    return heapq.nsmallest(k, ranked, key=lambda pair: (-pair[1], pair[0]))


def validate_ranked(ranked, known_ids, k):
    import math
    from .core import validate_request
    validate_request("ranking", k)
    rows, seen = list(ranked), set()
    if len(rows) > k:
        raise ValueError("retriever returned more than requested k")
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("retriever results require ID/score pairs")
        identifier, score = row
        if identifier not in known_ids or identifier in seen:
            raise ValueError("retriever returned unknown or duplicate ID")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("retrieval score must be finite")
        seen.add(identifier)
    return rows
