"""Composable retrieval operations independent of model inference."""
import heapq


def stable_top_k(ranked, k):
    from .core import validate_request
    validate_request("top-k", k)
    return heapq.nsmallest(k, ranked, key=lambda pair: (-pair[1], pair[0]))
