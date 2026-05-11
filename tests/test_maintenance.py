import json
import math
import pytest

def test_hash_rejects_nonstandard_numbers():
    from retrieval_lab.core import canonical_hash
    for value in (float('nan'),float('inf'),-float('inf')):
        with pytest.raises(ValueError):canonical_hash({'score':value})

def test_tokenizer_input_has_a_clear_text_contract():
    from retrieval_lab.text import RetrievalTokenizer
    tokenizer=RetrievalTokenizer()
    for value in (None,[],123,"bad\ud800"):
        with pytest.raises(ValueError):tokenizer(value)
    assert tokenizer("Ｃａｆé")==["café"]

def test_ranking_metrics_validate_relevance_identifiers():
    from retrieval_lab.metrics import precision_at_k,average_precision,ndcg_at_k
    for relevant in ("a",["a","a"],[""]):
        for metric in (precision_at_k,average_precision):
            with pytest.raises(ValueError):metric(["a"],relevant,1)
    with pytest.raises(ValueError):ndcg_at_k(["a"],{"":1},1)

def test_paired_comparison_rejects_invalid_values_before_arithmetic():
    from retrieval_lab.metrics import paired_comparison
    for value in (True,None,float('nan'),"1"):
        with pytest.raises(ValueError):paired_comparison([{"query_id":"q","answer_em":value}],[{"query_id":"q","answer_em":1}],repetitions=2)
    with pytest.raises(ValueError):paired_comparison([{"query_id":"q"}],[{"query_id":"q","answer_em":1}],repetitions=2)

def test_paired_comparison_rejects_blank_query_identity():
    from retrieval_lab.metrics import paired_comparison
    for identifier in ("",None," ",1):
        rows=[{"query_id":identifier,"answer_em":1}]
        with pytest.raises(ValueError):paired_comparison(rows,rows,repetitions=2)

def test_reciprocal_rank_cutoff_and_unanswerable_semantics():
    from retrieval_lab.metrics import reciprocal_rank
    assert reciprocal_rank(["a","b"],["b"],2)==.5
    assert reciprocal_rank(["a","b"],["b"],1)==0
    assert reciprocal_rank(["a"],[],1) is None

def test_hit_rate_excludes_undefined_unanswerable_samples():
    from retrieval_lab.metrics import hit_rate
    assert hit_rate([(["a"],["a"]),(["a"],["b"]),(["a"],[])],1)=={"hit_rate":.5,"samples":2}
    assert hit_rate([],1)=={"hit_rate":None,"samples":0}

def test_weighted_mean_validates_weights_and_preserves_scale():
    from retrieval_lab.metrics import weighted_mean
    assert weighted_mean([0,1],[1,3])==.75
    for weights in ([0,0],[1,-1],[True,1],[1]):
        with pytest.raises(ValueError):weighted_mean([0,1],weights)

def test_slice_labels_merge_normalized_equivalents():
    from retrieval_lab.metrics import slice_metrics
    from unittest.mock import patch
    rows=[{"query_id":"q"}]
    with patch("retrieval_lab.core._validate_rows"),patch("retrieval_lab.core.score_predictions",return_value=({"count":1},None)):
        assert list(slice_metrics(rows,{"q":[" Café ","Cafe\u0301"]},.5))==["Café"]

def test_calibration_validation_handles_invalid_container_shapes():
    from retrieval_lab.calibration import validate_calibration_artifact
    for artifact in ([],{"schema_version":True},{"schema_version":1,"calibration":None}):
        with pytest.raises(ValueError):validate_calibration_artifact(artifact)

def test_calibration_artifacts_require_content_hashes():
    from retrieval_lab.calibration import validate_calibration_artifact
    artifact={"schema_version":1,"calibration":{"split":"dev","threshold":.5},"dataset_fingerprint":"placeholder","development_fingerprint":"a"*64}
    with pytest.raises(ValueError):validate_calibration_artifact(artifact)
    artifact["dataset_fingerprint"]="b"*64
    validate_calibration_artifact(artifact)

def test_calibration_loader_rejects_duplicate_envelope_keys(tmp_path):
    from retrieval_lab.calibration import load_calibration
    from retrieval_lab.core import canonical_hash
    artifact={"schema_version":1,"calibration":{"split":"dev","threshold":.5},"dataset_fingerprint":"a"*64,"development_fingerprint":"b"*64}
    encoded=json.dumps({"artifact":artifact,"sha256":canonical_hash(artifact)})
    path=tmp_path/"calibration.json";path.write_text(encoded[:-1]+',"sha256":'+json.dumps(canonical_hash(artifact))+'}')
    with pytest.raises(ValueError,match="duplicate"):load_calibration(path)

def test_grouped_calibration_rejects_blank_labels():
    from retrieval_lab.calibration import cross_validate_calibration
    rows=[{"query_id":q,"raw_answer":"Paris","confidence":.9,"answers":["Paris"],"ranked_ids":["d"],"relevant_ids":["d"],"retrieval_ms":1,"reader_ms":2,"total_ms":3} for q in ("a","b")]
    with pytest.raises(ValueError,match="groups"):cross_validate_calibration(rows,{"a":" ","b":"b"},folds=2)

def test_grouped_split_requires_explicit_string_identities():
    from retrieval_lab.datasets import grouped_split
    for values in ([1,"1","b"],[" ","b"],[None,"b"]):
        with pytest.raises(ValueError):grouped_split(values,lambda value:value)
    left,right=grouped_split(["a","b"],lambda value:value)
    assert set(left+right)=={"a","b"}

def test_provenance_counts_reject_boolean_equivalence():
    from retrieval_lab.datasets import validate_provenance
    value={"kind":"synthetic","license":"MIT","permission":"original","counts":{"corpus":True}}
    with pytest.raises(ValueError):validate_provenance(value,{"corpus":1})

def test_fusion_rejects_malformed_component_rankings():
    from retrieval_lab.retrieval import reciprocal_rank_fusion
    for ranking in ([[]],[(None,1)],[("",1)],[("a",float("nan"))]):
        with pytest.raises(ValueError):reciprocal_rank_fusion([ranking])

def test_fusion_rejects_overflowed_combined_scores():
    from retrieval_lab.retrieval import reciprocal_rank_fusion
    with pytest.raises(ValueError,match="finite"):
        reciprocal_rank_fusion([[('a',1)],[('a',1)]],weights=[1e308,1e308],rank_constant=0)

def test_fusion_materializes_weights_for_repeated_searches():
    from retrieval_lab.retrieval import FusionRetriever
    class Engine:
        def search(self,question,k):return [("a",1)]
    retriever=FusionRetriever([Engine()],weights=iter([1]))
    assert retriever.search("q")==retriever.search("q")
    with pytest.raises(ValueError):FusionRetriever([Engine()],weights=[-1])

def test_mmr_normalizes_extreme_finite_vectors_stably():
    from retrieval_lab.retrieval import mmr_select
    for scale in (1e300,1e-300):
        result=mmr_select([scale,0],[[scale,0],[0,scale]],["a","b"],k=2)
        assert result[0]==("a",1.0)
        assert all(math.isfinite(score) for _,score in result)

def test_cached_retriever_does_not_store_invalid_rankings():
    from retrieval_lab.retrieval import CachedRetriever
    class Engine:
        fingerprint="a"
        def search(self,question,k):return [("a",float("nan"))]
    cached=CachedRetriever(Engine())
    with pytest.raises(ValueError):cached.search("question")
    assert not cached.cache

def test_batch_retrieval_rejects_invalid_per_query_rankings():
    from retrieval_lab.retrieval import retrieve_batch
    class Engine:
        def search_batch(self,questions,k):return [[("a",float("inf"))] for _ in questions]
    with pytest.raises(ValueError,match="query-a"):
        retrieve_batch(Engine(),[("query-a","question")])

def test_answer_span_search_validates_source_text():
    from retrieval_lab.annotations import answer_spans
    with pytest.raises(ValueError):answer_spans(None,"x")
    assert answer_spans("ééé","éé")==[(0,2),(1,3)]
