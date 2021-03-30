"""Additional per-query ranking, support, uncertainty, and slice diagnostics."""
import math
import statistics


def citation_metrics(rows):
    accepted = [r for r in rows if r.get("answer", "").strip()]
    supported = [r for r in accepted if r.get("document_id") in r["relevant_ids"]]
    return {"citation_precision": len(supported) / len(accepted) if accepted else None,
            "attributed_answer_count": len(accepted),
            "supported_answer_em": sum(r["answer_em"] for r in supported) / len(rows) if rows else None}


def _ranking(ranked, k):
    from .core import validate_request
    validate_request("ranking metric", k)
    ranked = list(ranked)
    if any(not isinstance(d, str) or not d for d in ranked) or len(set(ranked)) != len(ranked):
        raise ValueError("ranked IDs must be nonempty and unique")
    return ranked[:k]


def precision_at_k(ranked, relevant, k):
    return len(set(_ranking(ranked, k)) & set(relevant)) / k


def ndcg_at_k(ranked, relevance, k):
    ranked = _ranking(ranked, k)
    if any(type(g) not in (int, float) or not math.isfinite(g) or not 0 <= g <= 32
           for g in relevance.values()):
        raise ValueError("relevance grades must be finite and in [0,32]")
    def discounted(grades):
        return sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))
    ideal = discounted(sorted(relevance.values(), reverse=True)[:k])
    return discounted([relevance.get(d, 0) for d in ranked]) / ideal if ideal else None


def average_precision(ranked, relevant, k=5):
    ranked, relevant = _ranking(ranked, k), set(relevant)
    if not relevant:
        return None
    hits, total = 0, 0.0
    for rank, identifier in enumerate(ranked, 1):
        if identifier in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def mean_average_precision(examples, k=5):
    values = [average_precision(ranked, relevant, k) for ranked, relevant in examples]
    defined = [value for value in values if value is not None]
    return statistics.mean(defined) if defined else None


def risk_coverage_curve(rows):
    from .core import _validate_rows, answer_scores, has_answer
    _validate_rows(rows)
    candidates = sorted([r for r in rows if has_answer(r["raw_answer"])],
                        key=lambda r: (-r["confidence"], r["query_id"]))
    points, accepted, wrong, area, previous = [], 0, 0, 0.0, 0.0
    for threshold in sorted({r["confidence"] for r in candidates}, reverse=True):
        group = [r for r in candidates if r["confidence"] == threshold]
        accepted += len(group)
        wrong += sum(1 - answer_scores(r["raw_answer"], r["answers"])[0] for r in group)
        coverage, risk = accepted / len(rows), wrong / accepted
        area += (coverage - previous) * risk
        previous = coverage
        points.append({"threshold": threshold, "coverage": coverage, "risk": risk, "accepted": accepted})
    return {"points": points, "aurc": area, "integration": "right-step over attainable coverage"}


def bootstrap_mean(values, *, repetitions=1000, confidence=0.95, seed=0):
    import random
    from .core import percentile
    values = list(values)
    if (not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or type(repetitions) is not int or not 2 <= repetitions <= 100000
            or type(seed) is not int or type(confidence) not in (float, int) or not 0 < confidence < 1):
        raise ValueError("invalid bootstrap samples or configuration")
    randomizer = random.Random(seed)
    samples = [statistics.mean(randomizer.choices(values, k=len(values))) for _ in range(repetitions)]
    tail = (1 - confidence) / 2
    return {"estimate": statistics.mean(values), "lower": percentile(samples, tail),
            "upper": percentile(samples, 1 - tail), "confidence": confidence,
            "repetitions": repetitions, "seed": seed, "sample_count": len(values)}
