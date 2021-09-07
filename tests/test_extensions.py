import json
import pytest


def test_strict_json_records(tmp_path):
    from retrieval_lab.core import _jsonl

    for text in ['{"id":1,"id":2}', '{"id":NaN}', '{"id":Infinity}']:
        path = tmp_path / "rows.jsonl"
        path.write_text(text)
        with pytest.raises(ValueError):
            _jsonl(path)


def test_jsonl_resource_limits(tmp_path):
    from retrieval_lab.core import _jsonl

    path = tmp_path / "rows.jsonl"
    path.write_text('{"a":1}\n{"a":2}\n')
    assert len(_jsonl(path, max_bytes=100, max_records=2)) == 2
    with pytest.raises(ValueError, match="byte"):
        _jsonl(path, max_bytes=4)
    with pytest.raises(ValueError, match="record"):
        _jsonl(path, max_records=1)


def test_jsonl_source_line_errors(tmp_path):
    from retrieval_lab.core import _jsonl

    path = tmp_path / "broken.jsonl"
    path.write_text('{"a":1}\n\nnot-json\n')
    with pytest.raises(ValueError, match=r"broken.jsonl:3"):
        _jsonl(path)
    path.write_text('{"a":1}\n[1]\n')
    with pytest.raises(ValueError, match=r"broken.jsonl:2"):
        _jsonl(path)


def test_identifier_contract(tmp_path):
    from retrieval_lab.core import load_corpus

    path = tmp_path / "corpus.jsonl"
    for identifier in [" a", "a ", "a\tb", "../a", "x" * 129]:
        path.write_text(json.dumps({"doc_id": identifier, "title": "Title", "text": "Body"}))
        with pytest.raises(ValueError, match="identifier"):
            load_corpus(path)


def test_in_memory_contracts():
    from retrieval_lab.core import Document, Query

    with pytest.raises(ValueError):
        Document("id", "title", "")
    with pytest.raises(ValueError):
        Query("q", "Question?", ("d",), ())
    with pytest.raises(ValueError):
        Query("q", "Question?", ("d", "d"), ("answer",))
    assert Query("q", "Question?", (), ()).answers == ()


def test_duplicate_query_normalization(tmp_path):
    from retrieval_lab.core import load_queries, question_key

    assert question_key("  ＨOW   long? ") == question_key("how long?")
    path = tmp_path / "q.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"query_id": q, "question": text, "relevant_ids": [], "answers": []})
            for q, text in [("a", "Who  owns it?"), ("b", "who owns it?")]
        )
    )
    with pytest.raises(ValueError, match="duplicate question"):
        load_queries(path, [])


def test_evidence_content_leakage():
    from retrieval_lab.core import Document, Query, validate_evidence_isolation

    docs = [Document("a", "One", "Same evidence."), Document("b", "Two", " same  evidence. ")]
    dev = [Query("x", "First?", ("a",), ("evidence",))]
    test = [Query("y", "Second?", ("b",), ("evidence",))]
    with pytest.raises(ValueError, match="content overlap"):
        validate_evidence_isolation(dev, test, docs)


def test_answer_span_offsets():
    from retrieval_lab.annotations import answer_spans, validate_span

    text = "red then red"
    assert answer_spans(text, "red") == [(0, 3), (9, 12)]
    assert validate_span(text, "red", 9, 12) == (9, 12)
    for bounds in [(1, 4), (-1, 3), (True, 3), (9, 15)]:
        with pytest.raises(ValueError):
            validate_span(text, "red", *bounds)


def test_provenance_count_validation():
    from retrieval_lab.datasets import validate_provenance

    counts = {"documents": 2, "development_queries": 1, "held_out_queries": 1}
    value = {"kind": "synthetic", "license": "MIT", "permission": "original", "counts": counts}
    validate_provenance(value, counts)
    with pytest.raises(ValueError, match="counts"):
        validate_provenance(value, {**counts, "documents": 3})
    with pytest.raises(ValueError):
        validate_provenance({**value, "license": ""}, counts)


def test_dataset_manifest_change_detection(tmp_path):
    from retrieval_lab.datasets import dataset_manifest, verify_dataset_manifest

    for name in ["corpus.jsonl", "dev.jsonl", "test.jsonl", "provenance.json"]:
        (tmp_path / name).write_text("{}")
    manifest = dataset_manifest(tmp_path)
    verify_dataset_manifest(tmp_path, manifest)
    (tmp_path / "provenance.json").write_text('{"license":"changed"}')
    with pytest.raises(ValueError, match="mismatch"):
        verify_dataset_manifest(tmp_path, manifest)


def test_split_isolation_policies():
    from retrieval_lab.datasets import validate_split_policy
    from retrieval_lab.core import Document, Query

    docs = [Document("a", "Title", "answer")]
    dev = [Query("a", "First?", ("a",), ("answer",))]
    test = [Query("b", "Second?", ("a",), ("answer",))]
    validate_split_policy(dev, test, docs, "query")
    with pytest.raises(ValueError):
        validate_split_policy(dev, test, docs, "document")
    with pytest.raises(ValueError):
        validate_split_policy(dev, test, docs, "unknown")


def test_grouped_split_reproducibility():
    from retrieval_lab.datasets import grouped_split

    items = [{"id": i, "group": str(i // 2)} for i in range(10)]
    dev, test = grouped_split(items, lambda x: x["group"], 0.4, seed=7)
    assert (dev, test) == grouped_split(items, lambda x: x["group"], 0.4, seed=7)
    assert not {x["group"] for x in dev} & {x["group"] for x in test}
    assert len(dev) + len(test) == 10 and dev and test
    with pytest.raises(ValueError):
        grouped_split(items, lambda x: x["group"], 1)


def test_generator_corpus_parity():
    from retrieval_lab.core import BM25, Document
    from retrieval_lab.models import QABundle
    import numpy as np

    docs = [Document("a", "Archive", "Receipts are retained.")]

    class EncoderDouble:
        dimension = 2

        def encode(self, texts):
            return np.tile([1.0, 0.0], (len(texts), 1))

    assert BM25(iter(docs)).search("receipts") == BM25(docs).search("receipts")
    bundle = QABundle(iter(docs), EncoderDouble(), None)
    assert bundle.lookup == {"a": docs[0]}
    assert bundle.index.embeddings.shape == (1, 2)


def test_bm25_hyperparameters():
    from retrieval_lab.core import BM25

    for kwargs in [{"k1": -1}, {"k1": 0}, {"b": 2}, {"b": float("nan")}, {"k1": True}]:
        with pytest.raises(ValueError):
            BM25([], **kwargs)
    assert BM25([], b=0).search("ok") == []


def test_unicode_retrieval_tokenizer():
    from retrieval_lab.core import BM25, Document
    from retrieval_lab.text import RetrievalTokenizer

    tokenizer = RetrievalTokenizer(min_length=2)
    assert tokenizer("ＡＳＴＥＲ Straße x") == ["aster", "strasse"]
    index = BM25([Document("a", "Aster", "Receipts")], tokenizer=tokenizer)
    assert index.search("ＡＳＴＥＲ")[0][0] == "a"
    assert tokenizer.fingerprint != RetrievalTokenizer(min_length=1).fingerprint


def test_bm25_explanations_reconcile():
    from retrieval_lab.core import BM25, Document

    index = BM25([Document("a", "Archive", "Archive receipt"), Document("b", "Other", "receipt")])
    result = index.explain("archive missing", "a")
    assert sum(x["contribution"] for x in result["terms"]) == pytest.approx(
        index.search("archive missing")[0][1]
    )
    assert result["terms"][1]["contribution"] == 0
    with pytest.raises(ValueError):
        index.explain("archive", "unknown")


def test_bm25_postings_candidates():
    from retrieval_lab.core import BM25, Document

    index = BM25(
        [
            Document("a", "First", "archive"),
            Document("b", "Second", "storage"),
            Document("c", "Third", "archive storage"),
        ]
    )
    assert index.candidate_ids("archive absent") == ["a", "c"]
    expected = sorted(
        [
            (d.doc_id, index.explain("archive", d.doc_id)["score"])
            for d in index.documents
            if index.explain("archive", d.doc_id)["score"] > 0
        ],
        key=lambda x: (-x[1], x[0]),
    )
    assert index.search("archive") == expected


def test_stable_topk_matches_full_sort():
    from retrieval_lab.retrieval import stable_top_k

    rows = [("z", 1.0), ("a", 1.0), ("c", 2.0), ("b", 0.0)]
    for k in [1, 2, 5]:
        assert stable_top_k(iter(rows), k) == sorted(rows, key=lambda x: (-x[1], x[0]))[:k]
    with pytest.raises(ValueError):
        stable_top_k(rows, 0)


def test_title_weighting():
    from retrieval_lab.core import BM25, Document

    docs = [Document("title", "needle", "noise noise"), Document("body", "noise", "needle needle")]
    assert BM25(docs, title_weight=5).search("needle")[0][0] == "title"
    assert BM25(docs, title_weight=0).search("needle")[0][0] == "body"
    with pytest.raises(ValueError):
        BM25(docs, title_weight=-1)


def test_retriever_result_contract():
    from retrieval_lab.retrieval import validate_ranked

    assert validate_ranked([("a", -0.2)], {"a"}, 2) == [("a", -0.2)]
    for ranked in [[("x", 1.0)], [("a", float("nan"))], [("a", 1.0), ("a", 2.0)], [("a", True)]]:
        with pytest.raises(ValueError):
            validate_ranked(ranked, {"a"}, 2)


def test_weighted_rank_fusion():
    from retrieval_lab.retrieval import reciprocal_rank_fusion

    result = reciprocal_rank_fusion(
        [[("a", 9), ("b", 1)], [("b", 0.1)]], weights=[1, 2], rank_constant=0, k=2
    )
    assert result == [("b", 2.5), ("a", 1.0)]
    assert reciprocal_rank_fusion([[]], k=2) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[("a", 1), ("a", 2)]])


def test_mmr_diversity_selection():
    from retrieval_lab.retrieval import mmr_select
    import numpy as np

    vectors = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]])
    assert [x[0] for x in mmr_select([1.0, 0.0], vectors, ["a", "b", "c"], 2, 1.0)] == ["a", "b"]
    assert [x[0] for x in mmr_select([1.0, 0.0], vectors, ["a", "b", "c"], 2, 0.0)] == ["a", "c"]
    with pytest.raises(ValueError):
        mmr_select([1.0, 0.0], vectors, ["a", "b", "c"], 2, -1.0)


def test_retrieval_cache_invalidation():
    from retrieval_lab.retrieval import CachedRetriever

    class Engine:
        fingerprint = "one"
        calls = 0

        def search(self, q, k=5):
            self.calls += 1
            return [("a", float(self.calls))]

    engine = Engine()
    cache = CachedRetriever(engine, capacity=1)
    first = cache.search("q")
    assert cache.search("q") == first and engine.calls == 1
    first.append(("bad", 0))
    assert len(cache.search("q")) == 1
    engine.fingerprint = "two"
    cache.search("q")
    assert engine.calls == 2
    cache.search("other")
    cache.search("q")
    assert engine.calls == 4


def test_batch_retrieval_identity():
    from retrieval_lab.retrieval import retrieve_batch
    from retrieval_lab.core import BM25, Document

    engine = BM25([Document("a", "Title", "archive")])
    queries = [("q2", "archive"), ("q1", "missing")]
    assert retrieve_batch(engine, queries) == [
        {"query_id": "q2", "ranked": engine.search("archive")},
        {"query_id": "q1", "ranked": []},
    ]
    with pytest.raises(ValueError):
        retrieve_batch(engine, [("q", "a"), ("q", "b")])


def metric_row(**changes):
    row = {
        "query_id": "q",
        "raw_answer": "seven years",
        "confidence": 0.9,
        "answers": ["seven years"],
        "ranked_ids": ["d"],
        "relevant_ids": ["d"],
        "retrieval_ms": 1.0,
        "reader_ms": 2.0,
        "total_ms": 3.0,
    }
    return {**row, **changes}


def test_evaluation_row_contracts():
    from retrieval_lab.core import score_predictions

    for changes in [
        {"confidence": True},
        {"total_ms": -1},
        {"reader_ms": float("nan")},
        {"answers": "seven years"},
        {"relevant_ids": ["d", "d"]},
        {"raw_answer": None},
    ]:
        with pytest.raises(ValueError):
            score_predictions([metric_row(**changes)], 0.5)
    with pytest.raises(ValueError):
        score_predictions([{"confidence": 0.5}], 0.5)


def test_evaluation_query_coverage():
    from retrieval_lab.core import score_predictions

    with pytest.raises(ValueError, match="duplicate"):
        score_predictions([metric_row(), metric_row()], 0.5)
    with pytest.raises(ValueError, match="coverage"):
        score_predictions([metric_row()], 0.5, expected_query_ids=["q", "missing"])
    assert score_predictions([metric_row()], 0.5, expected_query_ids=["q"])[0]["query_count"] == 1


def test_whitespace_abstention_consistency():
    from retrieval_lab.core import score_predictions

    metrics, rows = score_predictions([metric_row(raw_answer="  ", answers=[], relevant_ids=[])], 0.5)
    assert rows[0]["failure"] == "correct_abstention"
    assert rows[0]["answer"] == "" and rows[0]["abstained"]
    assert metrics["coverage"] == 0 and metrics["answer_em"] == 1


def test_unicode_answer_normalization():
    from retrieval_lab.core import answer_scores, normalize

    assert answer_scores("ＴＨＥ ＣＡＴ！", ["cat"], policy="unicode") == (1.0, 1.0)
    assert answer_scores("ＴＨＥ ＣＡＴ！", ["cat"]) == (0.0, 0.0)
    with pytest.raises(ValueError):
        normalize("answer", policy="unknown")
    assert normalize("The seven-years") == "sevenyears"


def test_metric_cutoff_configuration():
    from retrieval_lab.core import score_predictions, percentile

    values, _ = score_predictions([metric_row()], 0.5, cutoffs=(2, 4))
    assert values["recall_at_2"] == 1 and values["mrr_at_4"] == 1
    for kwargs in [{"reader_k": 0}, {"cutoffs": (0,)}, {"cutoffs": (1, 1)}]:
        with pytest.raises(ValueError):
            score_predictions([metric_row()], 0.5, **kwargs)
    for values, p in [([], 0.5), ([1], 2), ([float("nan")], 0.5)]:
        with pytest.raises(ValueError):
            percentile(values, p)


def test_undefined_metric_denominators():
    from retrieval_lab.core import score_predictions

    metrics, _ = score_predictions([metric_row(raw_answer="", answers=[], relevant_ids=[])], 0.5)
    assert metrics["recall_at_1"] is None and metrics["mrr_at_5"] is None
    assert metrics["selective_em"] is None
    assert metrics["retrieval_query_count"] == 0 and metrics["accepted_answer_count"] == 0


def test_answerability_subgroups():
    from retrieval_lab.core import score_predictions

    rows = [metric_row(), metric_row(query_id="n", raw_answer="", answers=[], relevant_ids=[])]
    metrics, _ = score_predictions(rows, 0.5)
    assert metrics["answerable_em"] == 1 and metrics["unanswerable_rejection_rate"] == 1
    assert metrics["unanswerable_count"] == 1 and metrics["answerable_count"] == 1


def test_citation_support_independent_of_answer():
    from retrieval_lab.metrics import citation_metrics

    rows = [
        {**metric_row(), "answer": "seven years", "answer_em": 1.0, "document_id": "wrong"},
        {**metric_row(query_id="q2"), "answer": "seven years", "answer_em": 1.0, "document_id": "d"},
    ]
    assert citation_metrics(rows) == {
        "citation_precision": 0.5,
        "attributed_answer_count": 2,
        "supported_answer_em": 0.5,
    }


def test_reference_answer_metric_conformance():
    from retrieval_lab.core import answer_scores, score_predictions

    assert answer_scores("the red red car", ["red car", "red red car"]) == (1.0, 1.0)
    assert answer_scores("red red", ["red blue"]) == (0.0, 0.5)
    assert answer_scores("", []) == (1.0, 1.0)
    assert answer_scores("extra", []) == (0.0, 0.0)
    row = metric_row(ranked_ids=["x", "y", "z", "w", "d"])
    result, _ = score_predictions([row], 0.5)
    assert result["recall_at_3"] == 0 and result["mrr_at_5"] == 0.2


def test_precision_at_cutoff():
    from retrieval_lab.metrics import precision_at_k

    assert precision_at_k(["a"], {"a"}, 3) == pytest.approx(1 / 3)
    assert precision_at_k(["x", "a"], {"a"}, 1) == 0
    with pytest.raises(ValueError):
        precision_at_k(["a", "a"], {"a"}, 2)


def test_graded_ndcg():
    from retrieval_lab.metrics import ndcg_at_k

    relevance = {"a": 3, "b": 1}
    assert ndcg_at_k(["a", "b"], relevance, 2) == 1
    assert 0 < ndcg_at_k(["b", "a"], relevance, 2) < 1
    assert ndcg_at_k(["a"], {"a": 0}, 1) is None
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a": -1}, 1)


def test_average_precision_multiple_gold_documents():
    from retrieval_lab.metrics import average_precision, mean_average_precision

    assert average_precision(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(5 / 6)
    assert average_precision(["a"], {"a", "b"}, 1) == 0.5
    assert mean_average_precision([(["a"], {"a"}), ([], set())], 3) == 1
    assert average_precision([], set(), 3) is None


def test_risk_coverage_ties():
    from retrieval_lab.metrics import risk_coverage_curve

    rows = [metric_row(confidence=0.8), metric_row(query_id="q2", raw_answer="wrong", confidence=0.8)]
    report = risk_coverage_curve(rows)
    assert len(report["points"]) == 1
    assert report["points"][0]["coverage"] == 1 and report["points"][0]["risk"] == 0.5
    assert report["aurc"] == 0.5


def test_seeded_bootstrap_intervals():
    from retrieval_lab.metrics import bootstrap_mean

    result = bootstrap_mean([0.0, 1.0, 1.0], repetitions=100, seed=12)
    assert result == bootstrap_mean([0.0, 1.0, 1.0], repetitions=100, seed=12)
    assert result["lower"] <= result["estimate"] <= result["upper"]
    assert bootstrap_mean([2.0, 2.0], repetitions=20)["lower"] == 2.0
    with pytest.raises(ValueError):
        bootstrap_mean([], repetitions=20)


def test_paired_query_comparison():
    from retrieval_lab.metrics import paired_comparison

    left = [{"query_id": "a", "answer_em": 1.0}, {"query_id": "b", "answer_em": 0.0}]
    right = [{"query_id": "b", "answer_em": 0.0}, {"query_id": "a", "answer_em": 0.0}]
    result = paired_comparison(left, right, repetitions=30)
    assert result["estimate"] == 0.5 and result["wins"] == 1 and result["ties"] == 1
    with pytest.raises(ValueError):
        paired_comparison(left, right[:1], repetitions=30)


def test_overlapping_slice_metrics():
    from retrieval_lab.metrics import slice_metrics

    rows = [metric_row(), metric_row(query_id="q2", raw_answer="wrong")]
    result = slice_metrics(rows, {"q": ["all", "easy"], "q2": ["all"]}, 0.5)
    assert result["all"]["query_count"] == 2 and result["all"]["answer_em"] == 0.5
    assert result["easy"]["query_count"] == 1 and result["easy"]["answer_em"] == 1
    with pytest.raises(ValueError):
        slice_metrics(rows, {"unknown": ["all"]}, 0.5)


def test_calibration_dataset_binding():
    from retrieval_lab.calibration import bind_calibration, verify_binding

    rows = [metric_row(), metric_row(query_id="n", answers=[], relevant_ids=[], confidence=0.1)]
    artifact = bind_calibration(rows, "dataset-a")
    verify_binding(artifact, rows, "dataset-a")
    with pytest.raises(ValueError):
        verify_binding(artifact, rows, "dataset-b")
    with pytest.raises(ValueError):
        verify_binding(artifact, [{**rows[0], "query_id": "test"}, rows[1]], "dataset-a")


def test_selective_risk_calibration():
    from retrieval_lab.calibration import calibrate_risk

    rows = [metric_row(confidence=0.8), metric_row(query_id="wrong", raw_answer="incorrect", confidence=0.2)]
    result = calibrate_risk(rows, max_risk=0.0, min_coverage=0.5)
    assert result["threshold"] == 0.8 and result["risk"] == 0 and result["coverage"] == 0.5
    with pytest.raises(ValueError):
        calibrate_risk(rows, max_risk=0, min_coverage=1)


def test_calibration_minimum_support():
    from retrieval_lab.core import calibrate_threshold

    rows = [metric_row(), metric_row(query_id="n", answers=[], relevant_ids=[], confidence=0.1)]
    with pytest.raises(ValueError, match="support"):
        calibrate_threshold(rows, split="dev", min_class_count=2)
    assert calibrate_threshold(rows, split="dev", min_class_count=1)["class_counts"] == {
        "answerable": 1,
        "unanswerable": 1,
    }


def test_threshold_confusion_totals():
    from retrieval_lab.calibration import threshold_diagnostics

    rows = [metric_row(confidence=0.9), metric_row(query_id="n", answers=[], relevant_ids=[], confidence=0.2)]
    curve = threshold_diagnostics(rows)
    for point in curve:
        assert sum(point[key] for key in ["tp", "tn", "fp", "fn"]) == 2
    assert [point["accepted"] for point in curve] == sorted(
        [point["accepted"] for point in curve], reverse=True
    )
    assert next(p for p in curve if p["threshold"] == 0.9)["balanced_accuracy"] == 1


def test_calibration_artifact_roundtrip(tmp_path):
    from retrieval_lab.calibration import bind_calibration, save_calibration, load_calibration

    rows = [metric_row(), metric_row(query_id="n", answers=[], relevant_ids=[], confidence=0.1)]
    artifact = bind_calibration(rows, "data")
    path = tmp_path / "threshold.json"
    save_calibration(path, artifact)
    assert load_calibration(path) == artifact
    payload = json.loads(path.read_text())
    payload["artifact"]["calibration"]["threshold"] = 0.123
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        load_calibration(path)


def test_grouped_calibration_folds():
    from retrieval_lab.calibration import cross_validate_calibration

    rows = [
        metric_row(
            query_id=f"q{i}",
            answers=["seven years"] if i % 2 else [],
            relevant_ids=["d"] if i % 2 else [],
            confidence=0.8 if i % 2 else 0.1,
        )
        for i in range(8)
    ]
    groups = {f"q{i}": str(i // 2) for i in range(8)}
    result = cross_validate_calibration(rows, groups, folds=2, seed=4)
    assert result == cross_validate_calibration(rows, groups, folds=2, seed=4)
    assert len(result) == 2
    for fold in result:
        assert not set(fold["train_groups"]) & set(fold["validation_groups"])


def test_empirical_score_reliability():
    from retrieval_lab.calibration import reliability_bins

    rows = [metric_row(confidence=0.9), metric_row(query_id="wrong", raw_answer="wrong", confidence=0.9)]
    result = reliability_bins(rows, bins=5)
    assert result["ece"] == pytest.approx(0.4)
    assert result["bins"][-1]["accuracy"] == 0.5 and result["bins"][0]["accuracy"] is None
    assert sum(b["count"] for b in result["bins"]) == 2


def test_encoder_batch_boundary():
    from retrieval_lab.config import validate_encoder_texts

    assert validate_encoder_texts([]) == []
    assert validate_encoder_texts(("one", "two")) == ["one", "two"]
    for value in ["a string", None, ["ok", None], [""]]:
        with pytest.raises(ValueError):
            validate_encoder_texts(value)


def test_encoder_config_and_cache_identity():
    from retrieval_lab.config import EncoderConfig
    from retrieval_lab.core import BM25, Document
    from retrieval_lab.retrieval import CachedRetriever

    assert EncoderConfig().fingerprint != EncoderConfig(max_tokens=64).fingerprint
    with pytest.raises(ValueError):
        EncoderConfig(pooling="unknown")
    docs = [Document("a", "Title", "archive")]
    index = BM25(docs)
    assert CachedRetriever(index).search("archive") == index.search("archive")
    assert index.fingerprint != BM25(docs, title_weight=2).fingerprint


def test_token_truncation_audit():
    from retrieval_lab.annotations import audit_truncation

    class TokenizerDouble:
        def __call__(self, text, **kwargs):
            n = min(len(text), kwargs.get("max_length", len(text))) if kwargs.get("truncation") else len(text)
            return {"input_ids": list(range(n)), "offset_mapping": [(i, i + 1) for i in range(n)]}

    result = audit_truncation(TokenizerDouble(), {"short": "cat", "long": "longer"}, 4)
    assert result[0]["truncated"] is False and result[1]["truncated"] is True
    assert result[1]["retained_char_end"] == 4 and result[1]["token_count"] == 6


def test_document_chunk_coverage():
    from retrieval_lab.annotations import chunk_document
    from retrieval_lab.core import Document

    doc = Document("a", "Title", "abcdefghijk")
    chunks = chunk_document(doc, max_chars=5, overlap=2)
    assert [(c.start, c.end) for c in chunks] == [(0, 5), (3, 8), (6, 11)]
    assert all(c.document.text == doc.text[c.start : c.end] and c.parent_id == "a" for c in chunks)
    assert len({c.document.doc_id for c in chunks}) == 3
    with pytest.raises(ValueError):
        chunk_document(doc, max_chars=5, overlap=5)


def test_parent_chunk_aggregation():
    from retrieval_lab.annotations import chunk_document, aggregate_chunks
    from retrieval_lab.core import Document

    chunks = chunk_document(Document("a", "Title", "abcdefgh"), max_chars=4, overlap=0)
    ranked = [(chunks[0].document.doc_id, 0.5), (chunks[1].document.doc_id, 0.9)]
    parents, trace = aggregate_chunks(ranked, chunks)
    assert parents == [("a", 0.9)] and trace["a"]["best_chunk_id"] == ranked[1][0]
    assert aggregate_chunks(ranked, chunks, policy="sum")[0] == [("a", 1.4)]
    with pytest.raises(ValueError):
        aggregate_chunks([("unknown", 1)], chunks)


def test_embedding_matrix_ownership():
    from retrieval_lab.config import validate_embeddings
    import numpy as np

    source = np.array([[1.0, 0.0]], dtype=np.float64)
    result = validate_embeddings(source, 1, 2)
    source[0, 0] = 9
    assert result[0, 0] == 1 and not result.flags.writeable
    for invalid in [
        np.array([[0.0, 0.0]]),
        np.array([[2.0, 0.0]]),
        np.array([[1, 0]]),
        np.array([[float("nan"), 0.0]]),
    ]:
        with pytest.raises(ValueError):
            validate_embeddings(invalid, 1, 2)


def test_scoped_cpu_threads():
    from retrieval_lab.runtime import cpu_threads
    import torch

    before = torch.get_num_threads()
    with pytest.raises(RuntimeError):
        with cpu_threads(1):
            assert torch.get_num_threads() == 1
            raise RuntimeError("stop")
    assert torch.get_num_threads() == before
    with pytest.raises(ValueError):
        with cpu_threads(0):
            pass


def test_dense_batch_query_parity():
    from retrieval_lab.models import DenseIndex
    from retrieval_lab.core import Document
    import numpy as np

    class EncoderDouble:
        dimension = 2
        calls = 0

        def encode(self, texts):
            self.calls += 1
            return np.array([[1.0, 0.0] if "cat" in t else [0.0, 1.0] for t in texts])

    encoder = EncoderDouble()
    index = DenseIndex([Document("cat", "Title", "cat"), Document("dog", "Title", "dog")], encoder)
    before = encoder.calls
    result = index.search_batch(["cat", "dog"], 1)
    assert encoder.calls == before + 1 and [r[0][0] for r in result] == ["cat", "dog"]
    assert result == [index.search("cat", 1), index.search("dog", 1)]


def test_reader_output_boundaries():
    from retrieval_lab.reading import validate_reader_result
    from retrieval_lab.core import Document

    doc = Document("d", "Title", "red car")
    base = {"answer": "red", "score": 0.8, "start": 0, "end": 3}
    assert validate_reader_result(doc, base)["document_id"] == "d"
    for changes in [{"score": float("nan")}, {"start": 1}, {"end": 50}, {"answer": "blue"}, {"start": True}]:
        with pytest.raises(ValueError):
            validate_reader_result(doc, {**base, **changes})


def test_batched_reader_attribution():
    from retrieval_lab.models import Reader
    from retrieval_lab.core import Document

    reader = Reader.__new__(Reader)
    reader.threads = 1
    calls = []

    def pipeline_double(inputs, **kwargs):
        calls.append(len(inputs))
        return [
            {"answer": item["context"], "score": 0.8, "start": 0, "end": len(item["context"])}
            for item in inputs
        ]

    reader.pipeline = pipeline_double
    docs = [Document("b", "Title", "blue"), Document("a", "Title", "red")]
    result = reader.answer_batch("Color?", docs, batch_size=2)
    assert calls == [2] and result["document_id"] == "a"
    assert result == reader.answer_batch("Color?", docs, batch_size=1)


def test_oracle_reader_uses_only_gold_context():
    from retrieval_lab.reading import oracle_evaluate
    from retrieval_lab.core import Document, Query

    calls = []

    class ReaderDouble:
        def answer(self, q, docs):
            calls.append([d.doc_id for d in docs])
            return {"raw_answer": docs[0].text if docs else ""}

    rows = oracle_evaluate(
        ReaderDouble(),
        [Query("q", "Color?", ("gold",), ("red",))],
        [Document("bad", "Title", "blue"), Document("gold", "Title", "red")],
    )
    assert calls == [["gold"]] and rows[0]["oracle_em"] == 1


def test_answer_selection_strategies():
    from retrieval_lab.reading import select_answer

    candidates = [{"document_id": "first", "confidence": 0.6}, {"document_id": "second", "confidence": 0.9}]
    assert select_answer(candidates, ["first", "second"], "max_score")["document_id"] == "second"
    assert select_answer(candidates, ["first", "second"], "top_ranked")["document_id"] == "first"
    with pytest.raises(ValueError):
        select_answer(candidates, ["first"], "max_score")


def test_reader_candidate_trace():
    from retrieval_lab.models import Reader
    from retrieval_lab.core import Document

    reader = Reader.__new__(Reader)
    reader.threads = 1
    reader.pipeline = lambda inputs, **kwargs: [
        {"answer": v["context"], "score": 0.7, "start": 0, "end": len(v["context"])} for v in inputs
    ]
    trace = reader.trace("Color?", [Document("a", "Title", "red"), Document("b", "Title", "blue")])
    assert trace["selected"]["document_id"] == "a"
    assert [r["document_id"] for r in trace["candidates"]] == ["a", "b"]
    assert sum(r["selected"] for r in trace["candidates"]) == 1


def test_cached_mutable_pairs_cannot_be_poisoned():
    from retrieval_lab.retrieval import CachedRetriever

    class Engine:
        fingerprint = "fixed"

        def search(self, question, k=5):
            return [["a", 1.0]]

    cache = CachedRetriever(Engine())
    first = cache.search("question")
    try:
        first[0][1] = 999.0
    except TypeError:
        pass
    assert cache.search("question")[0][1] == 1.0


def test_reader_singleton_pipeline_result():
    from retrieval_lab.models import Reader
    from retrieval_lab.core import Document

    reader = Reader.__new__(Reader)
    reader.threads = 1
    reader.pipeline = lambda *a, **kw: {"answer": "cat", "score": 0.9, "start": 0, "end": 3}
    assert reader.answer("Which animal?", [Document("a", "Animal", "cat")])["raw_answer"] == "cat"


def test_manifest_rejects_bad_configuration():
    from retrieval_lab.bundle_io import validate_manifest

    valid = {
        "schema_version": 1,
        "doc_ids": ["a"],
        "threshold": 0.5,
        "reader_k": 3,
        "encoder_max_tokens": 128,
        "reader_max_tokens": 384,
        **{
            k: "0" * 64
            for k in ["corpus_sha256", "encoder_fingerprint", "reader_fingerprint", "embeddings_sha256"]
        },
        **{k: "local" for k in ["encoder_id", "reader_id", "encoder_revision", "reader_revision"]},
    }
    assert validate_manifest(valid) == valid
    for key, value in [("threshold", True), ("reader_k", 0), ("doc_ids", ["a", "a"]), ("corpus_sha256", "x")]:
        with pytest.raises(ValueError):
            validate_manifest({**valid, key: value})


def test_manifest_restores_encoder_configuration():
    from retrieval_lab.bundle_io import encoder_configuration

    assert (
        encoder_configuration(
            {"schema_version": 1, "encoder_max_tokens": 128, "reader_max_tokens": 384}
        ).max_tokens
        == 128
    )
    custom = {
        "schema_version": 2,
        "encoder_max_tokens": 96,
        "reader_max_tokens": 384,
        "encoder_config": {"max_tokens": 96, "batch_size": 4, "pooling": "cls"},
    }
    assert encoder_configuration(custom).pooling == "cls"
    with pytest.raises(ValueError, match="mismatch"):
        encoder_configuration({**custom, "encoder_max_tokens": 128})


def test_model_inventory_requires_weights_and_tokenizer(tmp_path):
    from retrieval_lab.bundle_io import model_inventory

    for name in ["config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json"]:
        (tmp_path / name).write_text("{}")
    assert "model.safetensors" in model_inventory(tmp_path)
    (tmp_path / "model.py").write_text("pass")
    with pytest.raises(ValueError, match="unsupported"):
        model_inventory(tmp_path)
    (tmp_path / "model.py").unlink()
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing"):
        model_inventory(tmp_path)
