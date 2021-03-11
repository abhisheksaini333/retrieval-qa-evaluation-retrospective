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
