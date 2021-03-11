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



def test_mmr_diversity_selection():
    from retrieval_lab.retrieval import mmr_select
    import numpy as np
    vectors=np.array([[1.,0.],[0.99,0.1],[0.,1.]])
    assert [x[0] for x in mmr_select([1.,0.],vectors,['a','b','c'],2,1.)]==['a','b']
    assert [x[0] for x in mmr_select([1.,0.],vectors,['a','b','c'],2,0.)]==['a','c']
    with pytest.raises(ValueError):mmr_select([1.,0.],vectors,['a','b','c'],2,-1.)



def test_retrieval_cache_invalidation():
    from retrieval_lab.retrieval import CachedRetriever
    class Engine:
        fingerprint='one'; calls=0
        def search(self,q,k=5):self.calls+=1;return [('a',float(self.calls))]
    engine=Engine();cache=CachedRetriever(engine,capacity=1)
    first=cache.search('q');assert cache.search('q')==first and engine.calls==1
    first.append(('bad',0));assert len(cache.search('q'))==1
    engine.fingerprint='two';cache.search('q');assert engine.calls==2
    cache.search('other');cache.search('q');assert engine.calls==4



def test_batch_retrieval_identity():
    from retrieval_lab.retrieval import retrieve_batch
    from retrieval_lab.core import BM25,Document
    engine=BM25([Document('a','Title','archive')])
    queries=[('q2','archive'),('q1','missing')]
    assert retrieve_batch(engine,queries)==[{'query_id':'q2','ranked':engine.search('archive')},{'query_id':'q1','ranked':[]}]
    with pytest.raises(ValueError):retrieve_batch(engine,[('q','a'),('q','b')])



def metric_row(**changes):
    row={'query_id':'q','raw_answer':'seven years','confidence':0.9,'answers':['seven years'],'ranked_ids':['d'],'relevant_ids':['d'],'retrieval_ms':1.,'reader_ms':2.,'total_ms':3.}
    return {**row,**changes}


def test_evaluation_row_contracts():
    from retrieval_lab.core import score_predictions
    for changes in [{'confidence':True},{'total_ms':-1},{'reader_ms':float('nan')},{'answers':'seven years'},{'relevant_ids':['d','d']},{'raw_answer':None}]:
        with pytest.raises(ValueError):score_predictions([metric_row(**changes)],0.5)
    with pytest.raises(ValueError):score_predictions([{'confidence':0.5}],0.5)



def test_evaluation_query_coverage():
    from retrieval_lab.core import score_predictions
    with pytest.raises(ValueError,match='duplicate'):score_predictions([metric_row(),metric_row()],0.5)
    with pytest.raises(ValueError,match='coverage'):score_predictions([metric_row()],0.5,expected_query_ids=['q','missing'])
    assert score_predictions([metric_row()],0.5,expected_query_ids=['q'])[0]['query_count']==1



def test_whitespace_abstention_consistency():
    from retrieval_lab.core import score_predictions
    metrics,rows=score_predictions([metric_row(raw_answer='  ',answers=[],relevant_ids=[])],0.5)
    assert rows[0]['failure']=='correct_abstention'
    assert rows[0]['answer']=='' and rows[0]['abstained']
    assert metrics['coverage']==0 and metrics['answer_em']==1



def test_unicode_answer_normalization():
    from retrieval_lab.core import answer_scores,normalize
    assert answer_scores('ＴＨＥ ＣＡＴ！',['cat'],policy='unicode')==(1.,1.)
    assert answer_scores('ＴＨＥ ＣＡＴ！',['cat'])==(0.,0.)
    with pytest.raises(ValueError):normalize('answer',policy='unknown')
    assert normalize('The seven-years')=='sevenyears'



def test_metric_cutoff_configuration():
    from retrieval_lab.core import score_predictions,percentile
    values,_=score_predictions([metric_row()],0.5,cutoffs=(2,4))
    assert values['recall_at_2']==1 and values['mrr_at_4']==1
    for kwargs in [{'reader_k':0},{'cutoffs':(0,)},{'cutoffs':(1,1)}]:
        with pytest.raises(ValueError):score_predictions([metric_row()],0.5,**kwargs)
    for values,p in [([],0.5),([1],2),([float('nan')],0.5)]:
        with pytest.raises(ValueError):percentile(values,p)



def test_undefined_metric_denominators():
    from retrieval_lab.core import score_predictions
    metrics,_=score_predictions([metric_row(raw_answer='',answers=[],relevant_ids=[])],0.5)
    assert metrics['recall_at_1'] is None and metrics['mrr_at_5'] is None
    assert metrics['selective_em'] is None
    assert metrics['retrieval_query_count']==0 and metrics['accepted_answer_count']==0



def test_answerability_subgroups():
    from retrieval_lab.core import score_predictions
    rows=[metric_row(),metric_row(query_id='n',raw_answer='',answers=[],relevant_ids=[])]
    metrics,_=score_predictions(rows,0.5)
    assert metrics['answerable_em']==1 and metrics['unanswerable_rejection_rate']==1
    assert metrics['unanswerable_count']==1 and metrics['answerable_count']==1



def test_citation_support_independent_of_answer():
    from retrieval_lab.metrics import citation_metrics
    rows=[{**metric_row(),'answer':'seven years','answer_em':1.,'document_id':'wrong'},
          {**metric_row(query_id='q2'),'answer':'seven years','answer_em':1.,'document_id':'d'}]
    assert citation_metrics(rows)=={'citation_precision':0.5,'attributed_answer_count':2,'supported_answer_em':0.5}



def test_reference_answer_metric_conformance():
    from retrieval_lab.core import answer_scores,score_predictions
    assert answer_scores('the red red car',['red car','red red car'])==(1.,1.)
    assert answer_scores('red red',['red blue'])==(0.,0.5)
    assert answer_scores('',[])==(1.,1.)
    assert answer_scores('extra',[])==(0.,0.)
    row=metric_row(ranked_ids=['x','y','z','w','d'])
    result,_=score_predictions([row],0.5)
    assert result['recall_at_3']==0 and result['mrr_at_5']==0.2



def test_precision_at_cutoff():
    from retrieval_lab.metrics import precision_at_k
    assert precision_at_k(['a'],{'a'},3)==pytest.approx(1/3)
    assert precision_at_k(['x','a'],{'a'},1)==0
    with pytest.raises(ValueError):precision_at_k(['a','a'],{'a'},2)



def test_graded_ndcg():
    from retrieval_lab.metrics import ndcg_at_k
    relevance={'a':3,'b':1}
    assert ndcg_at_k(['a','b'],relevance,2)==1
    assert 0<ndcg_at_k(['b','a'],relevance,2)<1
    assert ndcg_at_k(['a'],{'a':0},1) is None
    with pytest.raises(ValueError):ndcg_at_k(['a'],{'a':-1},1)

