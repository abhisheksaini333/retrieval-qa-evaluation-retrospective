import json
from pathlib import Path
import pytest



def test_strict_json_records(tmp_path):
    from retrieval_lab.core import _jsonl
    for text in ['{"id":1,"id":2}', '{"id":NaN}', '{"id":Infinity}']:
        path = tmp_path / 'rows.jsonl'; path.write_text(text)
        with pytest.raises(ValueError): _jsonl(path)



def test_jsonl_resource_limits(tmp_path):
    from retrieval_lab.core import _jsonl
    path = tmp_path / 'rows.jsonl'; path.write_text('{"a":1}\n{"a":2}\n')
    assert len(_jsonl(path, max_bytes=100, max_records=2)) == 2
    with pytest.raises(ValueError, match='byte'): _jsonl(path, max_bytes=4)
    with pytest.raises(ValueError, match='record'): _jsonl(path, max_records=1)



def test_jsonl_source_line_errors(tmp_path):
    from retrieval_lab.core import _jsonl
    path = tmp_path / 'broken.jsonl'; path.write_text('{"a":1}\n\nnot-json\n')
    with pytest.raises(ValueError, match=r'broken.jsonl:3'): _jsonl(path)
    path.write_text('{"a":1}\n[1]\n')
    with pytest.raises(ValueError, match=r'broken.jsonl:2'): _jsonl(path)



def test_identifier_contract(tmp_path):
    from retrieval_lab.core import load_corpus
    path = tmp_path / 'corpus.jsonl'
    for identifier in [' a', 'a ', 'a\tb', '../a', 'x'*129]:
        path.write_text(json.dumps({'doc_id':identifier,'title':'Title','text':'Body'}))
        with pytest.raises(ValueError, match='identifier'): load_corpus(path)



def test_in_memory_contracts():
    from retrieval_lab.core import Document, Query
    with pytest.raises(ValueError): Document('id','title','')
    with pytest.raises(ValueError): Query('q','Question?',('d',),())
    with pytest.raises(ValueError): Query('q','Question?',('d','d'),('answer',))
    assert Query('q','Question?',(),()).answers == ()



def test_duplicate_query_normalization(tmp_path):
    from retrieval_lab.core import load_queries, question_key
    assert question_key('  ＨOW   long? ') == question_key('how long?')
    path=tmp_path/'q.jsonl'
    path.write_text('\n'.join(json.dumps({'query_id':q,'question':text,'relevant_ids':[],'answers':[]}) for q,text in [('a','Who  owns it?'),('b','who owns it?')]))
    with pytest.raises(ValueError, match='duplicate question'): load_queries(path, [])



def test_evidence_content_leakage():
    from retrieval_lab.core import Document, Query, validate_evidence_isolation
    docs=[Document('a','One','Same evidence.'),Document('b','Two',' same  evidence. ')]
    dev=[Query('x','First?',('a',),('evidence',))];test=[Query('y','Second?',('b',),('evidence',))]
    with pytest.raises(ValueError,match='content overlap'):validate_evidence_isolation(dev,test,docs)



def test_answer_span_offsets():
    from retrieval_lab.annotations import answer_spans, validate_span
    text='red then red'
    assert answer_spans(text,'red') == [(0,3),(9,12)]
    assert validate_span(text,'red',9,12) == (9,12)
    for bounds in [(1,4),(-1,3),(True,3),(9,15)]:
        with pytest.raises(ValueError):validate_span(text,'red',*bounds)



def test_provenance_count_validation():
    from retrieval_lab.datasets import validate_provenance
    counts={'documents':2,'development_queries':1,'held_out_queries':1}
    value={'kind':'synthetic','license':'MIT','permission':'original','counts':counts}
    validate_provenance(value,counts)
    with pytest.raises(ValueError,match='counts'):validate_provenance(value,{**counts,'documents':3})
    with pytest.raises(ValueError):validate_provenance({**value,'license':''},counts)



def test_dataset_manifest_change_detection(tmp_path):
    from retrieval_lab.datasets import dataset_manifest, verify_dataset_manifest
    for name in ['corpus.jsonl','dev.jsonl','test.jsonl','provenance.json']:(tmp_path/name).write_text('{}')
    manifest=dataset_manifest(tmp_path);verify_dataset_manifest(tmp_path,manifest)
    (tmp_path/'provenance.json').write_text('{"license":"changed"}')
    with pytest.raises(ValueError,match='mismatch'):verify_dataset_manifest(tmp_path,manifest)



def test_split_isolation_policies():
    from retrieval_lab.datasets import validate_split_policy
    from retrieval_lab.core import Document, Query
    docs=[Document('a','Title','answer')]
    dev=[Query('a','First?',('a',),('answer',))];test=[Query('b','Second?',('a',),('answer',))]
    validate_split_policy(dev,test,docs,'query')
    with pytest.raises(ValueError):validate_split_policy(dev,test,docs,'document')
    with pytest.raises(ValueError):validate_split_policy(dev,test,docs,'unknown')



def test_grouped_split_reproducibility():
    from retrieval_lab.datasets import grouped_split
    items=[{'id':i,'group':str(i//2)} for i in range(10)]
    dev,test=grouped_split(items,lambda x:x['group'],0.4,seed=7)
    assert (dev,test)==grouped_split(items,lambda x:x['group'],0.4,seed=7)
    assert not {x['group'] for x in dev}&{x['group'] for x in test}
    assert len(dev)+len(test)==10 and dev and test
    with pytest.raises(ValueError):grouped_split(items,lambda x:x['group'],1)



def test_generator_corpus_parity():
    from retrieval_lab.core import BM25, Document
    from retrieval_lab.models import QABundle
    import numpy as np
    docs=[Document('a','Archive','Receipts are retained.')]
    class EncoderDouble:
        dimension=2
        def encode(self,texts):return np.tile([1.,0.],(len(texts),1))
    assert BM25(iter(docs)).search('receipts')==BM25(docs).search('receipts')
    bundle=QABundle(iter(docs),EncoderDouble(),None)
    assert bundle.lookup == {'a':docs[0]}
    assert bundle.index.embeddings.shape == (1,2)



def test_bm25_hyperparameters():
    from retrieval_lab.core import BM25
    for kwargs in [{'k1':-1},{'k1':0},{'b':2},{'b':float('nan')},{'k1':True}]:
        with pytest.raises(ValueError):BM25([],**kwargs)
    assert BM25([],b=0).search('ok') == []



def test_unicode_retrieval_tokenizer():
    from retrieval_lab.core import BM25,Document
    from retrieval_lab.text import RetrievalTokenizer
    tokenizer=RetrievalTokenizer(min_length=2)
    assert tokenizer('ＡＳＴＥＲ Straße x') == ['aster','strasse']
    index=BM25([Document('a','Aster','Receipts')],tokenizer=tokenizer)
    assert index.search('ＡＳＴＥＲ')[0][0]=='a'
    assert tokenizer.fingerprint != RetrievalTokenizer(min_length=1).fingerprint



def test_bm25_explanations_reconcile():
    from retrieval_lab.core import BM25,Document
    index=BM25([Document('a','Archive','Archive receipt'),Document('b','Other','receipt')])
    result=index.explain('archive missing','a')
    assert sum(x['contribution'] for x in result['terms']) == pytest.approx(index.search('archive missing')[0][1])
    assert result['terms'][1]['contribution']==0
    with pytest.raises(ValueError):index.explain('archive','unknown')



def test_bm25_postings_candidates():
    from retrieval_lab.core import BM25,Document
    index=BM25([Document('a','First','archive'),Document('b','Second','storage'),Document('c','Third','archive storage')])
    assert index.candidate_ids('archive absent') == ['a','c']
    expected=sorted([(d.doc_id,index.explain('archive',d.doc_id)['score']) for d in index.documents if index.explain('archive',d.doc_id)['score']>0],key=lambda x:(-x[1],x[0]))
    assert index.search('archive')==expected



def test_stable_topk_matches_full_sort():
    from retrieval_lab.retrieval import stable_top_k
    rows=[('z',1.),('a',1.),('c',2.),('b',0.)]
    for k in [1,2,5]:assert stable_top_k(iter(rows),k)==sorted(rows,key=lambda x:(-x[1],x[0]))[:k]
    with pytest.raises(ValueError):stable_top_k(rows,0)



def test_title_weighting():
    from retrieval_lab.core import BM25,Document
    docs=[Document('title','needle','noise noise'),Document('body','noise','needle needle')]
    assert BM25(docs,title_weight=5).search('needle')[0][0]=='title'
    assert BM25(docs,title_weight=0).search('needle')[0][0]=='body'
    with pytest.raises(ValueError):BM25(docs,title_weight=-1)



def test_retriever_result_contract():
    from retrieval_lab.retrieval import validate_ranked
    assert validate_ranked([('a',-0.2)],{'a'},2)==[('a',-0.2)]
    for ranked in [[('x',1.)],[('a',float('nan'))],[('a',1.),('a',2.)],[('a',True)]]:
        with pytest.raises(ValueError):validate_ranked(ranked,{'a'},2)



def test_weighted_rank_fusion():
    from retrieval_lab.retrieval import reciprocal_rank_fusion
    result=reciprocal_rank_fusion([[('a',9),('b',1)],[('b',0.1)]],weights=[1,2],rank_constant=0,k=2)
    assert result==[('b',2.5),('a',1.)]
    assert reciprocal_rank_fusion([[]],k=2)==[]
    with pytest.raises(ValueError):reciprocal_rank_fusion([[('a',1),('a',2)]])

