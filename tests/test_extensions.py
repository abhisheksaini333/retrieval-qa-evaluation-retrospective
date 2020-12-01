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

