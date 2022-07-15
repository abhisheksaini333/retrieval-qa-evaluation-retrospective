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
