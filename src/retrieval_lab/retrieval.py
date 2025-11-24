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
        if not isinstance(identifier, str) or not identifier.strip() or identifier not in known_ids or identifier in seen:
            raise ValueError("retriever returned unknown or duplicate ID")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("retrieval score must be finite")
        seen.add(identifier)
    return rows


def reciprocal_rank_fusion(rankings, *, weights=None, rank_constant=60, k=5):
    import math
    from collections import defaultdict
    rankings = [list(ranking) for ranking in rankings]
    weights = [1.0] * len(rankings) if weights is None else list(weights)
    if (len(weights) != len(rankings) or type(rank_constant) not in (int, float)
            or not math.isfinite(rank_constant) or rank_constant < 0
            or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights)):
        raise ValueError("invalid fusion configuration")
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for row in ranking:
            if not isinstance(row, (tuple, list)) or len(row) != 2 or not isinstance(row[0], str) or not row[0].strip() or type(row[1]) not in (int, float) or not math.isfinite(row[1]):
                raise ValueError("fusion components require nonempty IDs and finite score pairs")
        identifiers = [row[0] for row in ranking]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate IDs in fusion component")
        if weight:
            for rank, identifier in enumerate(identifiers, 1):
                scores[identifier] += weight / (rank_constant + rank)
    if any(not math.isfinite(score) for score in scores.values()):
        raise ValueError("fused scores must remain finite")
    return stable_top_k(scores.items(), k)


class FusionRetriever:
    def __init__(self, retrievers, weights=None, rank_constant=60):
        self.retrievers = list(retrievers)
        self.weights = None if weights is None else tuple(weights)
        self.rank_constant = rank_constant
        if not self.retrievers or any(not callable(getattr(engine, "search", None)) for engine in self.retrievers):
            raise ValueError("fusion requires searchable retrievers")
        reciprocal_rank_fusion([[] for _ in self.retrievers], weights=self.weights, rank_constant=rank_constant)

    def search(self, question, k=5):
        return reciprocal_rank_fusion([engine.search(question, k) for engine in self.retrievers],
                                      weights=self.weights, rank_constant=self.rank_constant, k=k)


def mmr_select(query, vectors, doc_ids, k=5, relevance_weight=0.5):
    import numpy as np
    from .core import validate_request
    validate_request("MMR", k)
    matrix, query = np.asarray(vectors, dtype=float), np.asarray(query, dtype=float)
    if (type(relevance_weight) not in (int, float) or not 0 <= relevance_weight <= 1
            or matrix.ndim != 2 or query.shape != (matrix.shape[1],)
            or len(doc_ids) != len(matrix) or len(set(doc_ids)) != len(doc_ids)
            or not np.isfinite(matrix).all() or not np.isfinite(query).all()):
        raise ValueError("invalid MMR inputs")
    if not matrix.shape[1]:
        raise ValueError("MMR requires nonempty vectors")
    scales = np.max(np.abs(matrix), axis=1)
    query_scale = np.max(np.abs(query))
    if (scales == 0).any() or query_scale == 0:
        raise ValueError("MMR requires nonzero vectors")
    matrix = matrix / scales[:, None]
    matrix = matrix / np.linalg.norm(matrix, axis=1)[:, None]
    query = query / query_scale
    relevance = matrix @ (query / np.linalg.norm(query))
    selected, remaining = [], set(range(len(matrix)))
    while remaining and len(selected) < k:
        def objective(index):
            redundancy = max((float(matrix[index] @ matrix[j]) for j in selected), default=0.0)
            return (float(relevance[index]) if not selected else
                    relevance_weight * float(relevance[index]) - (1 - relevance_weight) * redundancy)
        best = min(remaining, key=lambda i: (-objective(i), doc_ids[i]))
        selected.append(best)
        remaining.remove(best)
    return [(doc_ids[i], float(relevance[i])) for i in selected]


class CachedRetriever:
    def __init__(self, retriever, capacity=128):
        from collections import OrderedDict
        if type(capacity) is not int or capacity < 1:
            raise ValueError("cache capacity must be positive")
        self.retriever, self.capacity, self.cache = retriever, capacity, OrderedDict()

    def search(self, question, k=5):
        from .core import validate_request
        validate_request(question, k)
        identity = getattr(self.retriever, "fingerprint", None)
        if not isinstance(identity, str) or not identity:
            raise ValueError("cached retriever requires a current fingerprint")
        key = (identity, question, k)
        if key not in self.cache:
            rows = list(self.retriever.search(question, k))
            known = [row[0] for row in rows if isinstance(row, (list, tuple)) and len(row) == 2]
            self.cache[key] = tuple(tuple(row) for row in validate_ranked(rows, known, k))
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return list(self.cache[key])


def retrieve_batch(retriever, queries, k=5):
    from .core import validate_identifier, validate_request
    queries, seen = list(queries), set()
    for identifier, question in queries:
        validate_identifier(identifier)
        validate_request(question, k)
        if identifier in seen:
            raise ValueError("duplicate batch query ID")
        seen.add(identifier)
    rankings = (retriever.search_batch([question for _, question in queries], k)
                if hasattr(retriever, "search_batch") else [retriever.search(question, k) for _, question in queries])
    if len(rankings) != len(queries):
        raise ValueError("batch retriever changed query cardinality")
    return [{"query_id": identifier, "ranked": ranked} for (identifier, _), ranked in zip(queries, rankings)]
