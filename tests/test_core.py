import json
import math

import pytest

from retrieval_lab.core import (
    BM25, Document, Query, calibrate_threshold, corpus_hash, load_corpus,
    load_queries, score_predictions, validate_index, validate_splits,
)


def docs():
    return [Document("d1", "Aster archive", "Aster records are retained for seven years."),
            Document("d2", "Cedar archive", "Cedar records are retained for three months.")]


def test_bm25_ranks_evidence_and_empty_retrieval():
    assert BM25(docs()).search("Aster retention records", 2)[0][0] == "d1"
    assert BM25([]).search("records", 3) == []
    assert BM25(docs()).search("unfindablexyz", 3) == []


@pytest.mark.parametrize("question,k", [("", 2), (None, 2), ("ok", 0), ("ok", -1)])
def test_retrieval_rejects_malformed_input(question, k):
    with pytest.raises(ValueError):
        BM25(docs()).search(question, k)


def test_duplicate_document_ids_fail(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text('\n'.join(json.dumps(vars(d)) for d in [docs()[0], docs()[0]]))
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(p)


@pytest.mark.parametrize("payload", ['not json', '{"doc_id":"a","title":"ok","text":3}',
                                       '{"doc_id":"a","title":"ok","text":""}'])
def test_malformed_corpus_fails(tmp_path, payload):
    p = tmp_path / "corpus.jsonl"
    p.write_text(payload)
    with pytest.raises(ValueError):
        load_corpus(p)


def test_query_requires_existing_evidence_and_literal_answer(tmp_path):
    p = tmp_path / "q.jsonl"
    base = dict(query_id="q1", question="How long?", relevant_ids=["d1"], answers=["twenty years"])
    p.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="answer"):
        load_queries(p, docs())
    base.update(relevant_ids=["unknown"], answers=["seven years"])
    p.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="unknown"):
        load_queries(p, docs())


def test_splits_reject_same_question_even_with_different_ids():
    a = Query("one", "How long?", ("d1",), ("seven years",))
    b = Query("two", "how LONG?", ("d1",), ("seven years",))
    with pytest.raises(ValueError, match="overlap"):
        validate_splits([a], [b])


def test_index_checks_order_corpus_and_encoder():
    manifest = {"corpus_sha256": corpus_hash(docs()), "doc_ids": ["d1", "d2"],
                "encoder_fingerprint": "abc"}
    validate_index(manifest, docs(), "abc")
    for documents, encoder in [(list(reversed(docs())), "abc"), (docs(), "xyz")]:
        with pytest.raises(ValueError, match="mismatch"):
            validate_index(manifest, documents, encoder)


def row(query_id, answer, gold, ranked, relevant, confidence=0.9):
    return {"query_id": query_id, "raw_answer": answer, "confidence": confidence,
            "answers": gold, "ranked_ids": ranked, "relevant_ids": relevant,
            "retrieval_ms": 1.0, "reader_ms": 2.0, "total_ms": 3.0}


def test_metric_denominators_partial_f1_and_error_attribution():
    rows = [row("a", "seven", ["seven years"], ["d1"], ["d1"]),
            row("b", "wrong", ["red"], ["d2"], ["d3"]),
            row("c", "noise", [], ["d2"], [], 0.1)]
    metrics, scored = score_predictions(rows, 0.5, reader_k=1)
    assert metrics["recall_at_1"] == 0.5
    assert metrics["mrr_at_5"] == 0.5
    assert metrics["answer_em"] == pytest.approx(1 / 3)
    assert metrics["answer_f1"] == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert [r["failure"] for r in scored] == ["reader_failure", "retrieval_miss", "correct_abstention"]
    assert metrics["latency_total_p95_ms"] == 3.0


def test_threshold_calibrates_only_development_and_favors_abstention_on_ties():
    rows = [row("a", "yes", ["yes"], ["d1"], ["d1"], 0.8),
            row("b", "noisy", [], ["d1"], [], 0.3)]
    result = calibrate_threshold(rows, split="dev")
    assert 0.3 < result["threshold"] <= 0.8
    assert result["objective_value"] == 1.0
    with pytest.raises(ValueError, match="dev"):
        calibrate_threshold(rows, split="test")
    with pytest.raises(ValueError):
        calibrate_threshold([], split="dev")
    assert math.isfinite(result["threshold"])


def test_mlflow_records_preserve_null_offsets_as_json_null():
    from retrieval_lab.tracking import json_records
    import pandas as pd
    frame = pd.DataFrame([{"answer": "years", "start": 2, "end": 7},
                          {"answer": "", "start": None, "end": None}])
    records = json_records(frame)
    assert records[1]["start"] is None
    assert json.loads(json.dumps(records, allow_nan=False))[1]["end"] is None
