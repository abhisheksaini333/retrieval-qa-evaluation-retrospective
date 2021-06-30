"""Reader result validation, answer selection, and oracle diagnostics."""
import math
from .annotations import validate_span


def validate_reader_result(document, result):
    if not isinstance(result, dict) or not {"answer", "score", "start", "end"} <= result.keys():
        raise ValueError("reader result is incomplete")
    score = result["score"]
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("reader score must be finite and in [0,1]")
    validate_span(document.text, result["answer"], result["start"], result["end"])
    return {"raw_answer": result["answer"], "confidence": float(score), "document_id": document.doc_id,
            "start": result["start"], "end": result["end"]}


def oracle_evaluate(reader, queries, documents):
    from .core import answer_scores
    lookup = {d.doc_id: d for d in documents}
    result = []
    for query in queries:
        if set(query.relevant_ids) - lookup.keys():
            raise ValueError("oracle references unknown gold document")
        predicted = reader.answer(query.question, [lookup[d] for d in query.relevant_ids])
        em, f1 = answer_scores(predicted["raw_answer"], query.answers)
        result.append({"query_id": query.query_id, "oracle_answer": predicted["raw_answer"],
                       "oracle_em": em, "oracle_f1": f1, "gold_context_ids": list(query.relevant_ids)})
    return result
