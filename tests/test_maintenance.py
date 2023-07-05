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
